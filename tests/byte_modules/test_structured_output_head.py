"""Tests for StructuredOutputHead from byte_modules."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from efficient_byte_tokenizer import EfficientByteTokenizer
from byte_modules import (
    ByteLogitHierarchy,
    StructuredLogitsAdapter,
    StructuredLogitsScatter,
    StructuredOutputHead,
)

tok = EfficientByteTokenizer()
V = tok.vocab_size

MODEL_DIM = 32
LOGIT_SOFTCAP = 30.0


def _make_head(**kwargs):
    defaults = dict(
        model_dim=MODEL_DIM, vocab_size=V, tok=tok, logit_softcap=LOGIT_SOFTCAP
    )
    defaults.update(kwargs)
    return StructuredOutputHead(**defaults)


def _zero_init_head(**kwargs):
    head = _make_head(**kwargs)
    for h in head.heads:
        nn.init.zeros_(h.weight)
    return head


# --------------------------------------------------------------------------
# Normalization tests
# --------------------------------------------------------------------------


@torch.no_grad()
def test_log_probs_sum_to_one():
    head = _make_head()
    x = torch.randn(2, 5, MODEL_DIM)
    log_p = head(x)
    sums = log_p.exp().sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


@torch.no_grad()
def test_log_probs_sum_to_one_with_cat_prior():
    head = _make_head()
    x = torch.randn(2, 5, MODEL_DIM)
    cat_prior = torch.randn(2, 5, 8)
    # Mask out categories 0 and 1 with -inf
    cat_prior[..., 0] = float("-inf")
    cat_prior[..., 1] = float("-inf")
    log_p = head(x, cat_prior=cat_prior)
    sums = log_p.exp().sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


@torch.no_grad()
def test_log_probs_sum_to_one_with_token_prior():
    head = _make_head()
    x = torch.randn(2, 5, MODEL_DIM)
    token_prior = torch.zeros(2, 5, V)
    # Mask out some tokens with -inf
    token_prior[..., :50] = float("-inf")
    log_p = head(x, token_prior=token_prior)
    sums = log_p.exp().sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


@torch.no_grad()
def test_log_probs_sum_to_one_with_ngram():
    head = _make_head()
    x = torch.randn(2, 5, MODEL_DIM)
    ngram_logp = F.log_softmax(torch.randn(2, 5, V), dim=-1)
    log_p = head(x, ngram_logp=ngram_logp)
    sums = log_p.exp().sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


@torch.no_grad()
def test_log_probs_all_negative():
    head = _make_head()
    x = torch.randn(2, 5, MODEL_DIM)
    log_p = head(x)
    assert (log_p <= 1e-6).all(), "Log-probabilities should be <= 0"


# --------------------------------------------------------------------------
# Token coverage tests
# --------------------------------------------------------------------------


@torch.no_grad()
def test_every_token_gets_probability():
    head = _zero_init_head()
    x = torch.randn(2, 5, MODEL_DIM)
    log_p = head(x)
    assert torch.isfinite(log_p).all()


@torch.no_grad()
def test_zero_init_gives_input_independent_output():
    """With zero-init weights, the output is deterministic and independent of input.

    The hierarchy means tokens on different paths get different log-probs (not
    uniform), but the output should be identical for any input vector because
    all head logits are zero regardless of x.
    """
    head = _zero_init_head()
    x1 = torch.randn(2, 5, MODEL_DIM)
    x2 = torch.randn(2, 5, MODEL_DIM)
    log_p1 = head(x1)
    log_p2 = head(x2)
    # Output should be identical regardless of input
    assert torch.allclose(log_p1, log_p2, atol=1e-6)
    # All probs should be positive (finite log-probs)
    assert torch.isfinite(log_p1).all()
    # Still sums to 1
    sums = log_p1.exp().sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


# --------------------------------------------------------------------------
# Hierarchy correctness tests
# --------------------------------------------------------------------------


@torch.no_grad()
def test_leaf_masks_partition_vocab():
    """Leaf masks should cover every token at most once.

    Singleton groups (size <= 1) do not get a leaf head, so they have 0 in
    the leaf mask sum.  All other tokens should appear in exactly one leaf.
    """
    head = _make_head()
    hier = head.hierarchy
    leaf_mask_sum = torch.zeros(V)
    for i, is_leaf in enumerate(hier._is_leaf):
        if is_leaf:
            leaf_mask_sum += hier.level_masks[i]
    # Each entry should be 0 (singleton) or 1 (covered by exactly one leaf)
    assert ((leaf_mask_sum == 0) | (leaf_mask_sum == 1)).all()
    # Non-singleton tokens should cover the vast majority of the vocab
    assert leaf_mask_sum.sum() >= V - 10  # at most a handful of singletons


@torch.no_grad()
def test_leaf_masks_disjoint():
    head = _make_head()
    hier = head.hierarchy
    leaf_indices = [i for i, lf in enumerate(hier._is_leaf) if lf]
    for a_idx in range(len(leaf_indices)):
        for b_idx in range(a_idx + 1, len(leaf_indices)):
            i = leaf_indices[a_idx]
            j = leaf_indices[b_idx]
            overlap = (hier.level_masks[i] * hier.level_masks[j]).sum()
            assert overlap == 0, f"Leaf levels {i} and {j} overlap"


@torch.no_grad()
def test_level_indices_within_head_size():
    head = _make_head()
    hier = head.hierarchy
    for i, h in enumerate(head.heads):
        mask = hier.level_masks[i]
        active = mask > 0
        indices = hier.level_indices[i][active]
        assert indices.min() >= 0, f"Level {i}: negative index"
        assert indices.max() < h.out_features, (
            f"Level {i}: index {indices.max()} >= head size {h.out_features}"
        )


# --------------------------------------------------------------------------
# Margin matrix tests
# --------------------------------------------------------------------------


@torch.no_grad()
def test_margin_rows_sum_to_mask():
    head = _make_head()
    hier = head.hierarchy
    for i in range(len(head.heads)):
        margin = getattr(hier, f"margin_{i}")
        row_sums = margin.sum(dim=1)
        assert torch.allclose(row_sums, hier.level_masks[i]), (
            f"Level {i}: margin row sums != level mask"
        )


@torch.no_grad()
def test_margin_is_binary():
    head = _make_head()
    hier = head.hierarchy
    for i in range(len(head.heads)):
        margin = getattr(hier, f"margin_{i}")
        assert ((margin == 0.0) | (margin == 1.0)).all(), (
            f"Level {i}: margin contains non-binary values"
        )


@torch.no_grad()
def test_ngram_marginalization_preserves_total():
    head = _make_head()
    hier = head.hierarchy
    # Uniform ngram probs: 1/V for every token
    ngram_probs = torch.full((1, 1, V), 1.0 / V)
    for i in range(len(head.heads)):
        margin = getattr(hier, f"margin_{i}")
        level_probs = ngram_probs @ margin  # (1, 1, H_i)
        total = level_probs.sum(dim=-1)
        expected = hier.level_masks[i].sum() / V
        assert torch.allclose(total, expected.unsqueeze(0).unsqueeze(0), atol=1e-6), (
            f"Level {i}: marginalized total {total.item():.6f} != expected {expected:.6f}"
        )


# --------------------------------------------------------------------------
# Softcap test
# --------------------------------------------------------------------------


@torch.no_grad()
def test_softcap_applied():
    x = torch.randn(2, 5, MODEL_DIM) * 10.0  # large input
    head_small = _make_head(logit_softcap=1.0)
    head_large = _make_head(logit_softcap=1000.0)
    # Copy weights so comparison is fair
    head_large.load_state_dict(head_small.state_dict(), strict=False)

    log_p_small = head_small(x)
    log_p_large = head_large(x)

    range_small = log_p_small.max() - log_p_small.min()
    range_large = log_p_large.max() - log_p_large.min()
    assert range_small < range_large, (
        f"Small softcap range ({range_small:.4f}) should be < large softcap range ({range_large:.4f})"
    )


# --------------------------------------------------------------------------
# Cross-entropy comparison test
# --------------------------------------------------------------------------


@torch.no_grad()
def test_structured_nll_matches_manual():
    """With zero-init weights, NLL loss should match manual computation from log_p."""
    head = _zero_init_head()
    x = torch.randn(2, 5, MODEL_DIM)
    log_p = head(x)
    target = torch.randint(0, V, (2, 5))
    loss = F.nll_loss(log_p.view(-1, V), target.view(-1))
    # Manual: average of -log_p[b, s, target[b, s]]
    manual_loss = -log_p.view(-1, V)[torch.arange(10), target.view(-1)].mean()
    assert torch.allclose(loss, manual_loss, atol=1e-5), (
        f"NLL loss {loss.item():.6f} != manual {manual_loss.item():.6f}"
    )
    # Loss should be positive (log_p is negative)
    assert loss.item() > 0


# --------------------------------------------------------------------------
# Output shape test
# --------------------------------------------------------------------------


@torch.no_grad()
def test_output_shape():
    head = _make_head()
    B, S = 2, 5
    x = torch.randn(B, S, MODEL_DIM)
    log_p = head(x)
    assert log_p.shape == (B, S, V), f"Expected ({B}, {S}, {V}), got {log_p.shape}"


# --------------------------------------------------------------------------
# ByteLogitHierarchy tests
# --------------------------------------------------------------------------


def _make_hierarchy():
    return ByteLogitHierarchy(vocab_size=V, tok=tok, logit_softcap=LOGIT_SOFTCAP)


@torch.no_grad()
def test_hierarchy_level_sizes():
    hier = _make_hierarchy()
    assert hier.num_levels > 0
    assert len(hier.level_sizes) == hier.num_levels
    assert hier.total_slots == sum(hier.level_sizes)


@torch.no_grad()
def test_hierarchy_assemble_sums_to_one():
    hier = _make_hierarchy()
    B, S = 2, 5
    per_level = [torch.randn(B, S, size) for size in hier.level_sizes]
    log_p = hier.assemble(per_level)
    assert log_p.shape == (B, S, V)
    sums = log_p.exp().sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


@torch.no_grad()
def test_hierarchy_assemble_with_priors():
    hier = _make_hierarchy()
    B, S = 2, 5
    per_level = [torch.randn(B, S, size) for size in hier.level_sizes]
    cat_prior = torch.randn(B, S, hier.level_sizes[0])
    token_prior = torch.zeros(B, S, V)
    token_prior[..., :50] = float("-inf")
    ngram_logp = F.log_softmax(torch.randn(B, S, V), dim=-1)
    log_p = hier.assemble(
        per_level, cat_prior=cat_prior, token_prior=token_prior, ngram_logp=ngram_logp
    )
    sums = log_p.exp().sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


# --------------------------------------------------------------------------
# StructuredLogitsAdapter tests
# --------------------------------------------------------------------------


@torch.no_grad()
def test_adapter_shape_and_normalization():
    hier = _make_hierarchy()
    adapter = StructuredLogitsAdapter(hier)
    B, S = 2, 5
    flat = torch.randn(B, S, hier.total_slots)
    log_p = adapter(flat)
    assert log_p.shape == (B, S, V)
    sums = log_p.exp().sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


@torch.no_grad()
def test_adapter_matches_head():
    """StructuredOutputHead and StructuredLogitsAdapter produce identical output
    when given the same per-level logits."""
    head = _make_head()
    hier = head.hierarchy
    adapter = head.adapter
    B, S = 2, 5
    x = torch.randn(B, S, MODEL_DIM)
    # Get the flat logits that the head would produce
    flat = torch.cat([h(x) for h in head.heads], dim=-1)
    # Compare adapter output to head output
    log_p_adapter = adapter(flat)
    log_p_head = head(x)
    assert torch.allclose(log_p_adapter, log_p_head, atol=1e-6)


# --------------------------------------------------------------------------
# assemble_logits tests
# --------------------------------------------------------------------------


@torch.no_grad()
def test_assemble_logits_shape():
    hier = _make_hierarchy()
    B, S = 2, 5
    per_level = [torch.randn(B, S, hs) for hs in hier.level_sizes]
    flat = hier.assemble_logits(per_level)
    assert flat.shape == (B, S, V)


@torch.no_grad()
def test_assemble_logits_scatter_correctness():
    """Each token's flat logit should equal the sum of its per-level slot logits."""
    hier = _make_hierarchy()
    B, S = 1, 1
    per_level = [torch.randn(B, S, hs) for hs in hier.level_sizes]
    flat = hier.assemble_logits(per_level)

    # Manually compute expected value for each token
    for t in range(V):
        expected = 0.0
        for i, logits in enumerate(per_level):
            mask_val = hier.level_masks[i, t].item()
            if mask_val > 0:
                idx = hier.level_indices[i, t].item()
                level_logit = logits[0, 0, idx].item()
                if hier._is_leaf[i]:
                    level_logit = LOGIT_SOFTCAP * math.tanh(level_logit / LOGIT_SOFTCAP)
                expected += level_logit
        assert abs(flat[0, 0, t].item() - expected) < 1e-5, (
            f"Token {t}: got {flat[0, 0, t].item()}, expected {expected}"
        )


@torch.no_grad()
def test_assemble_logits_zero_input():
    """All-zero per-level logits should produce all-zero flat logits."""
    hier = _make_hierarchy()
    B, S = 2, 5
    per_level = [torch.zeros(B, S, hs) for hs in hier.level_sizes]
    flat = hier.assemble_logits(per_level)
    assert torch.allclose(flat, torch.zeros_like(flat))


@torch.no_grad()
def test_assemble_logits_with_cat_prior():
    hier = _make_hierarchy()
    B, S = 2, 5
    per_level = [torch.randn(B, S, hs) for hs in hier.level_sizes]
    cat_prior = torch.randn(B, S, hier.level_sizes[0])
    flat_no_prior = hier.assemble_logits(per_level)
    flat_with_prior = hier.assemble_logits(per_level, cat_prior=cat_prior)
    # The difference should only be in the category contribution
    assert not torch.allclose(flat_no_prior, flat_with_prior)
    assert flat_with_prior.shape == (B, S, V)


def test_assemble_logits_gradient_flow():
    """Gradients should flow through assemble_logits."""
    hier = _make_hierarchy()
    B, S = 1, 3
    per_level = [torch.randn(B, S, hs, requires_grad=True) for hs in hier.level_sizes]
    flat = hier.assemble_logits(per_level)
    loss = flat.sum()
    loss.backward()
    for i, logits in enumerate(per_level):
        assert logits.grad is not None, f"No gradient for level {i}"
        assert logits.grad.abs().sum() > 0, f"Zero gradient for level {i}"


# --------------------------------------------------------------------------
# StructuredLogitsScatter tests
# --------------------------------------------------------------------------


@torch.no_grad()
def test_scatter_shape_and_matches_assemble_logits():
    hier = _make_hierarchy()
    scatter = StructuredLogitsScatter(hier)
    B, S = 2, 5
    flat_slots = torch.randn(B, S, hier.total_slots)
    result = scatter(flat_slots)
    assert result.shape == (B, S, V)
    # Should match calling assemble_logits directly
    chunks = flat_slots.split(hier.level_sizes, dim=-1)
    expected = hier.assemble_logits(list(chunks))
    assert torch.allclose(result, expected)
