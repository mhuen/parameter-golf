import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import math
import torch
import torch.nn.functional as F
import pytest
import numpy as np
from efficient_byte_tokenizer import EfficientByteTokenizer, ByteCategory
from byte_modules import UTF8Prior

tok = EfficientByteTokenizer()
V = tok.vocab_size  # 208

NEG_INF = float("-inf")


def _byte_to_tid(byte_val):
    """Find the token ID for a specific byte value."""
    for tid in range(tok.vocab_size):
        info = tok.token_info(tid)
        if info is not None and info.byte_value == byte_val:
            return tid
    raise ValueError(f"No token for byte {byte_val:#x}")


def _encode(text: str) -> torch.Tensor:
    """Encode text to (1, S) long tensor (no BOS)."""
    return torch.tensor(tok.encode(text), dtype=torch.long).unsqueeze(0)


def _make_prior():
    return UTF8Prior(tok)


def _cont_tids():
    """Return set of token IDs that are continuation bytes."""
    mask = tok.mask(ByteCategory.MB_CONTINUATION)
    return set(np.where(mask)[0].tolist())


# ---- Shape and basic properties ----


@torch.no_grad()
def test_cat_mask_shape():
    prior = _make_prior()
    ids = _encode("abc")
    cat_mask, _ = prior(ids)
    assert cat_mask.shape == (1, 3, 8)


@torch.no_grad()
def test_token_mask_shape():
    prior = _make_prior()
    ids = _encode("abc")
    _, token_mask = prior(ids)
    assert token_mask.shape == (1, 3, V)


@torch.no_grad()
def test_mask_values_are_zero_or_neginf():
    prior = _make_prior()
    ids = _encode("Hello café 日本語")
    cat_mask, token_mask = prior(ids)
    for name, mask in [("cat_mask", cat_mask), ("token_mask", token_mask)]:
        vals = mask.unique()
        for v in vals:
            assert v.item() == 0.0 or v.item() == NEG_INF, (
                f"{name} contains unexpected value {v.item()}"
            )


# ---- State: READY ----


@torch.no_grad()
def test_ascii_sequence_forbids_continuation():
    prior = _make_prior()
    ids = _encode("abc")
    _, token_mask = prior(ids)
    cont_ids = _cont_tids()
    for t in range(ids.shape[1]):
        for tid in cont_ids:
            assert token_mask[0, t, tid].item() == NEG_INF, (
                f"Position {t}: continuation token {tid} should be forbidden"
            )
        # Non-continuation tokens should be allowed
        for tid in range(V):
            if tid not in cont_ids:
                assert token_mask[0, t, tid].item() == 0.0, (
                    f"Position {t}: non-cont token {tid} should be allowed"
                )


@torch.no_grad()
def test_after_complete_two_byte_is_ready():
    prior = _make_prior()
    # Prepend BOS so the bounded lookback can find the lead byte
    raw = tok.encode("é")  # bytes: 0xC3, 0xA9
    ids = torch.tensor(
        [[tok.bos_id] + raw.tolist()], dtype=torch.long
    )
    _, token_mask = prior(ids)
    # Last position (0xA9) completes the 2-byte sequence -> READY
    cont_ids = _cont_tids()
    last = ids.shape[1] - 1  # position of 0xA9
    for tid in cont_ids:
        assert token_mask[0, last, tid].item() == NEG_INF, (
            f"After complete 2-byte: continuation token {tid} should be forbidden"
        )


# ---- State: EXPECT_CONT ----


@torch.no_grad()
def test_after_lead2_only_continuation_allowed():
    prior = _make_prior()
    tid_c3 = _byte_to_tid(0xC3)
    ids = torch.tensor([[tid_c3]], dtype=torch.long)
    _, token_mask = prior(ids)
    cont_ids = _cont_tids()
    for tid in range(V):
        if tid in cont_ids:
            assert token_mask[0, 0, tid].item() == 0.0, (
                f"After lead-2: continuation token {tid} should be allowed"
            )
        else:
            assert token_mask[0, 0, tid].item() == NEG_INF, (
                f"After lead-2: non-continuation token {tid} should be forbidden"
            )


@torch.no_grad()
def test_after_lead3_expects_continuation():
    prior = _make_prior()
    tid_e2 = _byte_to_tid(0xE2)
    ids = torch.tensor([[tid_e2]], dtype=torch.long)
    _, token_mask = prior(ids)
    cont_ids = _cont_tids()
    for tid in cont_ids:
        assert token_mask[0, 0, tid].item() == 0.0, (
            f"After lead-3: continuation token {tid} should be allowed"
        )
    # Non-continuation should be forbidden
    for tid in range(V):
        if tid not in cont_ids:
            assert token_mask[0, 0, tid].item() == NEG_INF


@torch.no_grad()
def test_mid_3byte_still_expects_cont():
    prior = _make_prior()
    tid_e2 = _byte_to_tid(0xE2)
    tid_82 = _byte_to_tid(0x82)
    # Prepend BOS so continuation byte's lookback can find lead at pos 1
    ids = torch.tensor([[tok.bos_id, tid_e2, tid_82]], dtype=torch.long)
    _, token_mask = prior(ids)
    cont_ids = _cont_tids()
    # At position 2 (0x82), still need 1 more continuation
    for tid in cont_ids:
        assert token_mask[0, 2, tid].item() == 0.0, (
            f"Mid 3-byte: continuation token {tid} should be allowed"
        )
    for tid in range(V):
        if tid not in cont_ids:
            assert token_mask[0, 2, tid].item() == NEG_INF


@torch.no_grad()
def test_cat_mask_expect_cont_only_multibyte():
    prior = _make_prior()
    tid_c3 = _byte_to_tid(0xC3)
    ids = torch.tensor([[tid_c3]], dtype=torch.long)
    cat_mask, _ = prior(ids)
    # Categories 0-6 should be -inf, category 7 (multibyte) should be 0
    for ci in range(7):
        assert cat_mask[0, 0, ci].item() == NEG_INF, (
            f"EXPECT_CONT: category {ci} should be -inf"
        )
    assert cat_mask[0, 0, 7].item() == 0.0, (
        "EXPECT_CONT: multibyte category (7) should be 0"
    )


# ---- State: UNSYNCED ----


@torch.no_grad()
def test_bos_is_unsynced_all_allowed():
    prior = _make_prior()
    ids = torch.tensor([[tok.bos_id]], dtype=torch.long)
    cat_mask, token_mask = prior(ids)
    assert (cat_mask == 0.0).all(), "BOS: cat_mask should be all zeros"
    assert (token_mask == 0.0).all(), "BOS: token_mask should be all zeros"


# ---- Special lead byte constraints ----


@torch.no_grad()
def test_after_e0_first_cont_range():
    prior = _make_prior()
    tid_e0 = _byte_to_tid(0xE0)
    ids = torch.tensor([[tid_e0]], dtype=torch.long)
    _, token_mask = prior(ids)
    for tid in range(V):
        info = tok.token_info(tid)
        if info is not None and info.byte_value is not None:
            bv = info.byte_value
            if 0x80 <= bv <= 0xBF:  # continuation byte range
                if 0xA0 <= bv <= 0xBF:
                    assert token_mask[0, 0, tid].item() == 0.0, (
                        f"After 0xE0: cont byte {bv:#x} (0xA0-0xBF) should be allowed"
                    )
                else:  # 0x80-0x9F
                    assert token_mask[0, 0, tid].item() == NEG_INF, (
                        f"After 0xE0: cont byte {bv:#x} (0x80-0x9F) should be forbidden"
                    )


@torch.no_grad()
def test_after_ed_first_cont_range():
    prior = _make_prior()
    tid_ed = _byte_to_tid(0xED)
    ids = torch.tensor([[tid_ed]], dtype=torch.long)
    _, token_mask = prior(ids)
    for tid in range(V):
        info = tok.token_info(tid)
        if info is not None and info.byte_value is not None:
            bv = info.byte_value
            if 0x80 <= bv <= 0xBF:
                if 0x80 <= bv <= 0x9F:
                    assert token_mask[0, 0, tid].item() == 0.0, (
                        f"After 0xED: cont byte {bv:#x} (0x80-0x9F) should be allowed"
                    )
                else:  # 0xA0-0xBF
                    assert token_mask[0, 0, tid].item() == NEG_INF, (
                        f"After 0xED: cont byte {bv:#x} (0xA0-0xBF) should be forbidden"
                    )


@torch.no_grad()
def test_after_f0_first_cont_range():
    prior = _make_prior()
    tid_f0 = _byte_to_tid(0xF0)
    ids = torch.tensor([[tid_f0]], dtype=torch.long)
    _, token_mask = prior(ids)
    for tid in range(V):
        info = tok.token_info(tid)
        if info is not None and info.byte_value is not None:
            bv = info.byte_value
            if 0x80 <= bv <= 0xBF:
                if 0x90 <= bv <= 0xBF:
                    assert token_mask[0, 0, tid].item() == 0.0, (
                        f"After 0xF0: cont byte {bv:#x} (0x90-0xBF) should be allowed"
                    )
                else:  # 0x80-0x8F
                    assert token_mask[0, 0, tid].item() == NEG_INF, (
                        f"After 0xF0: cont byte {bv:#x} (0x80-0x8F) should be forbidden"
                    )


# ---- Valid tokens never masked ----


@torch.no_grad()
def test_valid_utf8_never_masked():
    prior = _make_prior()
    test_strings = ["Hello", "café", "日本語", "𝄞"]
    for text in test_strings:
        ids = _encode(text)
        _, token_mask = prior(ids)
        S = ids.shape[1]
        for t in range(S - 1):
            next_tid = ids[0, t + 1].item()
            val = token_mask[0, t, next_tid].item()
            assert val == 0.0, (
                f"Text {text!r}, pos {t}: next token {next_tid} is masked "
                f"(value={val}), but it should be allowed"
            )


# ---- Cross-entropy improvement ----


@torch.no_grad()
def test_utf8_prior_improves_cross_entropy():
    prior = _make_prior()
    text = "Hello café 日本語"
    ids = _encode(text)
    B, S = ids.shape

    # Uniform log-probs
    uniform_logp = torch.full((B, S, V), math.log(1.0 / V))

    cat_mask, token_mask = prior(ids)

    # Apply token mask and renormalize
    constrained_logp = uniform_logp + token_mask
    constrained_logp = constrained_logp - torch.logsumexp(
        constrained_logp, dim=-1, keepdim=True
    )

    # Check that constrained CE <= uniform CE for positions with constraints
    uniform_ce = math.log(V)  # -log(1/V)
    for t in range(S - 1):
        next_tid = ids[0, t + 1].item()
        has_constraint = (token_mask[0, t] == NEG_INF).any().item()
        if has_constraint:
            constrained_nll = -constrained_logp[0, t, next_tid].item()
            assert constrained_nll <= uniform_ce + 1e-6, (
                f"Position {t}: constrained CE {constrained_nll:.4f} > "
                f"uniform CE {uniform_ce:.4f}"
            )


# ---- Causality ----


@torch.no_grad()
def test_causality():
    prior = _make_prior()
    ids_abc = _encode("abc")
    ids_abcd = _encode("abcd")

    cat_abc, tok_abc = prior(ids_abc)
    cat_abcd, tok_abcd = prior(ids_abcd)

    # First 3 positions should be identical
    assert torch.equal(cat_abc, cat_abcd[:, :3, :]), (
        "cat_mask: first 3 positions should be identical for 'abc' and 'abcd'"
    )
    assert torch.equal(tok_abc, tok_abcd[:, :3, :]), (
        "token_mask: first 3 positions should be identical for 'abc' and 'abcd'"
    )
