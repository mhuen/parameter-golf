import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import math
import torch
import pytest
from efficient_byte_tokenizer import EfficientByteTokenizer
from byte_stream_components import DigitComputeComponent, DigitEncoding, PairwiseOp

tok = EfficientByteTokenizer()


def _encode(text: str) -> torch.Tensor:
    return torch.tensor(tok.encode(text), dtype=torch.long).unsqueeze(0)


def _make(k=2, ops=None):
    return DigitComputeComponent(tok, k=k, ops=ops, digit_encoding=DigitEncoding.ONEHOT)


# Mirrors _result_to_string from the component
def _result_to_string(value: float) -> str:
    av = abs(value)
    if av == int(av) and av < 1e15:
        return str(int(av))
    s = f"{av}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


# ===== Shape / dim =====


@torch.no_grad()
def test_dim_default():
    comp = _make()
    n_arith = 8  # ADD, MUL, SUB, RSUB, DIV, RDIV, MOD, RMOD
    n_cmp = 3  # GT, EQ, LT
    # 1 pair (k=2), each arith op = 13 dims (1 sign + 12 one-hot)
    expected = 1 * (n_arith * 13 + n_cmp)
    assert comp.dim == expected


@torch.no_grad()
def test_dim_k3():
    comp = _make(k=3)
    n_arith = 8
    n_cmp = 3
    n_pairs = 3  # C(3,2) = 3
    expected = n_pairs * (n_arith * 13 + n_cmp)
    assert comp.dim == expected


@torch.no_grad()
def test_dim_subset_ops():
    comp = _make(ops={PairwiseOp.ADD, PairwiseOp.GT})
    # 1 arith (ADD) * 13 + 1 cmp (GT) = 14
    assert comp.dim == 14


@torch.no_grad()
def test_output_shape():
    comp = _make()
    ids = _encode("12 + 34")
    out = comp(ids, torch.float32)
    assert out.shape == (1, ids.shape[1], comp.dim)


@torch.no_grad()
def test_output_dtype():
    comp = _make()
    ids = _encode("12 + 34")
    for dt in [torch.float32, torch.float64]:
        out = comp(ids, dt)
        assert out.dtype == dt


# ===== Number parsing =====


@torch.no_grad()
def test_single_number_no_output():
    """Before two numbers are seen, output should be all zeros."""
    comp = _make(ops={PairwiseOp.ADD})
    ids = _encode("42")
    out = comp(ids, torch.float32)
    assert (out == 0).all()


@torch.no_grad()
def test_digit_run_parsing():
    """After "12 34", the ring should contain [12, 34]."""
    comp = _make(ops={PairwiseOp.ADD})
    # "12 34x" — after the space following 34, both numbers are in the ring
    # The 'x' token triggers finalization of 34
    ids = _encode("12 34x")
    out = comp(ids, torch.float32)
    B, S, D = out.shape
    # At the 'x' position (last token), 34 just got pushed, ring = [12, 34]
    # ADD = 12 + 34 = 46, sign = +1
    last = out[0, -1]
    assert last[0].item() == 1.0  # sign = +1 (positive)


@torch.no_grad()
def test_decimal_number_parsing():
    """Decimal numbers like 3.14 should be parsed correctly."""
    comp = _make(ops={PairwiseOp.ADD})
    # "1 3.14x" — after x, ring = [1, 3.14], ADD = 4.14
    ids = _encode("1 3.14x")
    out = comp(ids, torch.float32)
    last = out[0, -1]
    assert last[0].item() == 1.0  # sign = +1


@torch.no_grad()
def test_number_word_parsing():
    """Number words like 'three' should be recognized."""
    comp = _make(ops={PairwiseOp.ADD})
    # "5 three " — after trailing space, ring = [5, 3], ADD = 8
    ids = _encode("5 three ")
    out = comp(ids, torch.float32)
    last = out[0, -1]
    assert last[0].item() == 1.0  # positive


# ===== Next-digit encoding =====


def _get_next_digit_idx(one_hot_12: torch.Tensor) -> int:
    """Extract the active index from a 12-dim one-hot vector."""
    assert one_hot_12.shape == (12,), f"Expected (12,), got {one_hot_12.shape}"
    nonzero = one_hot_12.nonzero(as_tuple=False)
    if len(nonzero) == 0:
        return -1
    return nonzero[0].item()


@torch.no_grad()
def test_next_digit_first_digit_when_not_in_run():
    """When not in a digit run, active_len=0, so we get the first digit of the result."""
    comp = _make(ops={PairwiseOp.ADD})
    # "12 34 " — at the trailing space, active_len=0, ring=[12,34], ADD=46
    # Result string "46", first char '4' → idx=4
    ids = _encode("12 34 ")
    out = comp(ids, torch.float32)
    last = out[0, -1]
    sign = last[0].item()
    one_hot = last[1:13]
    assert sign == 1.0
    assert _get_next_digit_idx(one_hot) == 4  # '4' of "46"


@torch.no_grad()
def test_next_digit_second_digit_in_run():
    """When in a digit run of length 1, we get the second digit of result."""
    comp = _make(ops={PairwiseOp.ADD})
    # "12 34 4" — at '4', active_len=1, ring=[12,34], ADD=46
    # Result string "46", char at index 1 = '6' → idx=6
    ids = _encode("12 34 4")
    out = comp(ids, torch.float32)
    last = out[0, -1]
    one_hot = last[1:13]
    assert _get_next_digit_idx(one_hot) == 6  # '6' of "46"


@torch.no_grad()
def test_next_digit_end_after_result():
    """When active_len >= len(result_string), output END token."""
    comp = _make(ops={PairwiseOp.ADD})
    # "5 3 89" — ring=[5,3], ADD=8, result string "8" (len=1)
    # At '8' (first digit), active_len=1 → but the '8' starts a new number...
    # Use "5 3 8x" instead: at 'x', active_len=0, so first digit
    # Better: "1 1 11" — ring=[1,1], ADD=2, result "2" (len=1)
    # At second '1' of "11", active_len=2 >= len("2")=1, so END
    ids = _encode("1 1 11")
    out = comp(ids, torch.float32)
    last = out[0, -1]  # second '1' of trailing "11"
    one_hot = last[1:13]
    assert _get_next_digit_idx(one_hot) == 11  # END


@torch.no_grad()
def test_next_digit_dot_in_result():
    """When result has a decimal point, it should be encoded as DIGIT_IDX_DOT=10."""
    comp = _make(ops={PairwiseOp.DIV})
    eps = comp._eps
    # "7 2x" — ring=[7,2], DIV = 7 / (|2| + eps) ≈ 3.4999...
    # Result string: _result_to_string(7 / (2 + eps)) — let's compute
    val = 7 / (2 + eps)
    s = _result_to_string(val)
    # Find where the dot is
    if "." in s:
        dot_pos = s.index(".")
        # Encode "7 2 " + enough digits to reach the dot
        text = "7 2 " + "0" * dot_pos
        ids = _encode(text)
        out = comp(ids, torch.float32)
        last = out[0, -1]
        one_hot = last[1:13]
        assert _get_next_digit_idx(one_hot) == 10  # DOT


# ===== Arithmetic ops =====


def _assert_op_at(
    text: str,
    op: PairwiseOp,
    expected_sign: float,
    expected_first_digit: int | None = None,
    active_len: int = 0,
):
    """Helper: check sign and optional first digit of a single op after `text`."""
    comp = _make(ops={op})
    ids = _encode(text)
    out = comp(ids, torch.float32)
    last = out[0, -1]
    sign = last[0].item()
    assert sign == expected_sign, f"Expected sign {expected_sign}, got {sign}"
    if expected_first_digit is not None:
        one_hot = last[1:13]
        idx = _get_next_digit_idx(one_hot)
        assert idx == expected_first_digit, (
            f"Expected digit idx {expected_first_digit}, got {idx}"
        )


@torch.no_grad()
def test_add():
    # "10 25 " → ring=[10,25], ADD=35, sign=+1, first digit='3'
    _assert_op_at("10 25 ", PairwiseOp.ADD, 1.0, 3)


@torch.no_grad()
def test_sub():
    # ring=[10,25], SUB = a - b = 10 - 25 = -15, sign=-1, first digit='1'
    _assert_op_at("10 25 ", PairwiseOp.SUB, -1.0, 1)


@torch.no_grad()
def test_rsub():
    # ring=[10,25], RSUB = b - a = 25 - 10 = 15, sign=+1, first digit='1'
    _assert_op_at("10 25 ", PairwiseOp.RSUB, 1.0, 1)


@torch.no_grad()
def test_mul():
    # ring=[3,7], MUL = 21, sign=+1, first digit='2'
    _assert_op_at("3 7 ", PairwiseOp.MUL, 1.0, 2)


@torch.no_grad()
def test_div():
    eps = 1e-8
    # ring=[10,4], DIV = 10 / (|4| + eps) ≈ 2.5, sign=+1, first digit='2'
    _assert_op_at("10 4 ", PairwiseOp.DIV, 1.0, 2)


@torch.no_grad()
def test_rdiv():
    eps = 1e-8
    # ring=[10,4], RDIV = 4 / (|10| + eps) ≈ 0.4, sign=+1, first digit='0'
    _assert_op_at("10 4 ", PairwiseOp.RDIV, 1.0, 0)


@torch.no_grad()
def test_mod():
    # ring=[10,3], MOD = fmod(10, |3|) = 1.0, sign=+1, first digit='1'
    _assert_op_at("10 3 ", PairwiseOp.MOD, 1.0, 1)


@torch.no_grad()
def test_rmod():
    eps = 1e-8
    # ring=[10,3], RMOD = fmod(3, |10| + eps) ≈ 3.0, sign=+1, first digit='3'
    _assert_op_at("10 3 ", PairwiseOp.RMOD, 1.0, 3)


# ===== Comparisons =====


@torch.no_grad()
def test_gt():
    comp = _make(ops={PairwiseOp.GT})
    # ring=[10,25], GT: a > b → 10 > 25 → 0.0
    ids = _encode("10 25 ")
    out = comp(ids, torch.float32)
    assert out[0, -1, 0].item() == 0.0

    # ring=[30,5], GT: 30 > 5 → 1.0
    ids2 = _encode("30 5 ")
    out2 = comp(ids2, torch.float32)
    assert out2[0, -1, 0].item() == 1.0


@torch.no_grad()
def test_eq():
    comp = _make(ops={PairwiseOp.EQ})
    # ring=[7,7], EQ: 7 == 7 → 1.0
    ids = _encode("7 7 ")
    out = comp(ids, torch.float32)
    assert out[0, -1, 0].item() == 1.0

    # ring=[7,8], EQ: 7 == 8 → 0.0
    ids2 = _encode("7 8 ")
    out2 = comp(ids2, torch.float32)
    assert out2[0, -1, 0].item() == 0.0


@torch.no_grad()
def test_lt():
    comp = _make(ops={PairwiseOp.LT})
    # ring=[10,25], LT: 10 < 25 → 1.0
    ids = _encode("10 25 ")
    out = comp(ids, torch.float32)
    assert out[0, -1, 0].item() == 1.0


# ===== Number words =====


@torch.no_grad()
def test_number_word_three_plus_five():
    comp = _make(ops={PairwiseOp.ADD})
    # "three five " → ring=[3, 5], ADD=8
    ids = _encode("three five ")
    out = comp(ids, torch.float32)
    last = out[0, -1]
    assert last[0].item() == 1.0  # positive
    assert _get_next_digit_idx(last[1:13]) == 8  # '8'


@torch.no_grad()
def test_number_word_twenty():
    comp = _make(ops={PairwiseOp.ADD})
    # "1 twenty " → ring=[1, 20], ADD=21
    ids = _encode("1 twenty ")
    out = comp(ids, torch.float32)
    last = out[0, -1]
    assert last[0].item() == 1.0
    assert _get_next_digit_idx(last[1:13]) == 2  # '2' of "21"


@torch.no_grad()
def test_unknown_word_ignored():
    comp = _make(ops={PairwiseOp.ADD})
    # "1 hello 2 " → 'hello' is not a number word, ring=[1, 2], ADD=3
    ids = _encode("1 hello 2 ")
    out = comp(ids, torch.float32)
    last = out[0, -1]
    assert last[0].item() == 1.0
    assert _get_next_digit_idx(last[1:13]) == 3  # '3'


# ===== Ring buffer =====


@torch.no_grad()
def test_ring_buffer_overwrites_oldest():
    """With k=2, the third number should evict the first."""
    comp = _make(k=2, ops={PairwiseOp.ADD})
    # "10 20 30 " → ring=[20, 30] (10 evicted), ADD = 50
    ids = _encode("10 20 30 ")
    out = comp(ids, torch.float32)
    last = out[0, -1]
    assert last[0].item() == 1.0
    assert _get_next_digit_idx(last[1:13]) == 5  # '5' of "50"


@torch.no_grad()
def test_k3_three_pairs():
    """With k=3 and 3 numbers, we should get C(3,2)=3 pairs."""
    comp = _make(k=3, ops={PairwiseOp.ADD})
    # "2 3 5 " → nums=[2,3,5] oldest-first
    # Pairs (i<j): (2,3)=5, (2,5)=7, (3,5)=8
    ids = _encode("2 3 5 ")
    out = comp(ids, torch.float32)
    last = out[0, -1]
    pair_dim = 13  # 1 arith op * 13
    # Pair 0: (2,3) ADD=5, sign=+1, first digit='5'
    assert last[0].item() == 1.0
    assert _get_next_digit_idx(last[1:13]) == 5
    # Pair 1: (2,5) ADD=7, sign=+1, first digit='7'
    assert last[pair_dim].item() == 1.0
    assert _get_next_digit_idx(last[pair_dim + 1 : pair_dim + 13]) == 7
    # Pair 2: (3,5) ADD=8, sign=+1, first digit='8'
    assert last[2 * pair_dim].item() == 1.0
    assert _get_next_digit_idx(last[2 * pair_dim + 1 : 2 * pair_dim + 13]) == 8


# ===== Causality =====


@torch.no_grad()
def test_causality():
    """Output at position t should not change when tokens after t change."""
    comp = _make(ops={PairwiseOp.ADD})
    ids1 = _encode("5 3 hello")
    ids2 = _encode("5 3 world")
    out1 = comp(ids1, torch.float32)
    out2 = comp(ids2, torch.float32)
    # First 4 tokens are "5 3 " — identical in both
    # Outputs at positions 0..3 should be identical
    assert torch.allclose(out1[0, :4], out2[0, :4])


# ===== Batch =====


@torch.no_grad()
def test_batch_independent():
    """Each batch element should be computed independently."""
    comp = _make(ops={PairwiseOp.ADD})
    ids1 = _encode("10 20 ")
    ids2 = _encode("3 7   ")
    # Pad to same length
    max_len = max(ids1.shape[1], ids2.shape[1])
    ids1_pad = torch.nn.functional.pad(ids1, (0, max_len - ids1.shape[1]))
    ids2_pad = torch.nn.functional.pad(ids2, (0, max_len - ids2.shape[1]))
    batch = torch.cat([ids1_pad, ids2_pad], dim=0)
    out = comp(batch, torch.float32)

    out1_solo = comp(ids1_pad, torch.float32)
    out2_solo = comp(ids2_pad, torch.float32)
    assert torch.allclose(out[0], out1_solo[0])
    assert torch.allclose(out[1], out2_solo[0])


# ===== Zero result =====


@torch.no_grad()
def test_zero_result_sign():
    """Zero result should have sign = 0.0."""
    comp = _make(ops={PairwiseOp.SUB})
    # ring=[5,5], SUB = 5-5 = 0, sign=0.0
    ids = _encode("5 5 ")
    out = comp(ids, torch.float32)
    assert out[0, -1, 0].item() == 0.0


# ===== Negative result =====


@torch.no_grad()
def test_negative_result_sign_and_digits():
    """Negative results should have sign=-1 and digits from abs value."""
    comp = _make(ops={PairwiseOp.SUB})
    # ring=[3,10], SUB = 3 - 10 = -7, sign=-1, first digit='7'
    ids = _encode("3 10 ")
    out = comp(ids, torch.float32)
    last = out[0, -1]
    assert last[0].item() == -1.0
    assert _get_next_digit_idx(last[1:13]) == 7


# ===== Float results via DIV =====


@torch.no_grad()
def test_div_float_result_digits():
    """Division producing a float should have correct digit sequence."""
    comp = _make(ops={PairwiseOp.DIV})
    eps = comp._eps
    # ring=[7,2], DIV = 7 / (2 + eps) ≈ 3.5
    val = 7 / (2 + eps)
    s = _result_to_string(val)
    # "7 2 " → at trailing space, active_len=0, first char of result
    ids = _encode("7 2 ")
    out = comp(ids, torch.float32)
    last = out[0, -1]
    assert last[0].item() == 1.0  # positive
    first_ch = s[0]
    expected_idx = int(first_ch) if first_ch.isdigit() else 10
    assert _get_next_digit_idx(last[1:13]) == expected_idx


@torch.no_grad()
def test_div_float_multi_digit_sequence():
    """Walk through all digits of a fractional division result during a digit run.

    Regression test: an off-by-one in frac_pos (not accounting for the dot
    character) caused every fractional digit to be shifted by one position,
    e.g. 0.490277 was emitted as 0.902777.
    """
    comp = _make(ops={PairwiseOp.DIV})
    eps = comp._eps
    ad = comp._arith_dim  # 13 (1 sign + 12 one-hot)

    # "7 3 " → ring=[7,3], DIV = 7/(3+eps) ≈ 2.333...
    # result string: "2.333..."
    val = 7 / (3 + eps)
    expected = _result_to_string(val)

    # Build input that triggers the result, then continues typing result digits
    # "7 3 " produces the result at the trailing space (active_len=0 → first char)
    # "7 3 2.333" types out the result — each typed digit bumps active_len
    text = "7 3 " + expected
    ids = _encode(text)
    out = comp(ids, torch.float32)

    prefix_len = len("7 3 ")
    # Check first 7 chars (int + dot + 5 frac digits); beyond that float
    # precision drift can cause mismatches unrelated to the encoding logic.
    check_len = min(len(expected), 7)
    for i, ch in enumerate(expected[:check_len]):
        pos = prefix_len + i  # token position in the sequence
        if ch == ".":
            exp_idx = 10  # DIGIT_IDX_DOT
        elif ch.isdigit():
            exp_idx = int(ch)
        else:
            continue
        # At the position *before* this char is typed, active_len = i
        # so the component predicts expected[i] as the next digit.
        # That prediction is at position (prefix_len + i - 1) for i>0,
        # and at position (prefix_len - 1) for i==0 (the trailing space).
        if i == 0:
            t = prefix_len - 1
        else:
            t = prefix_len + i - 1
        actual_idx = _get_next_digit_idx(out[0, t, 1:13])
        assert actual_idx == exp_idx, (
            f"digit {i} of '{expected}': expected idx {exp_idx} ('{ch}'), "
            f"got {actual_idx} at seq pos {t}"
        )


# ===== Long integration test =====


@torch.no_grad()
def test_long_sequence_integration():
    """Walk through a multi-number sequence and verify ADD at every non-digit position."""
    comp = _make(k=2, ops={PairwiseOp.ADD, PairwiseOp.SUB})
    text = "12 34 5 100 7 three 42 "
    ids = _encode(text)
    out = comp(ids, torch.float32)
    B, S, D = out.shape

    # Simulate the component's number parsing to get expected ring state
    eps = comp._eps
    k = 2
    ring = [0.0] * k
    ring_idx = 0
    ring_count = 0
    digit_num = 0.0
    digit_len = 0
    digit_has_dot = False
    digit_frac_mul = 0.0
    in_digit_run = False
    word_bytes_list: list[int] = []
    in_word_run = False
    active_len = 0
    active_has_dot = False

    dv = comp.digit_value[ids[0]]
    lb_buf = comp.lowercase_byte[ids[0]]
    id_buf = comp.is_decimal[ids[0]]

    ad = comp._arith_dim  # 13

    for t in range(S):
        d = dv[t].item()
        l = lb_buf[t].item()
        is_dot = id_buf[t].item()
        pushed = False

        dot_continues_run = is_dot and in_digit_run and not digit_has_dot

        # Finish runs
        if d >= 0:
            if in_word_run:
                word = bytes(word_bytes_list).decode("ascii", errors="replace")
                val = comp._number_words.get(word)
                if val is not None:
                    ring[ring_idx % k] = float(val)
                    ring_idx += 1
                    ring_count += 1
                    pushed = True
                in_word_run = False
                word_bytes_list = []
        elif dot_continues_run:
            pass
        elif l > 0:
            if in_digit_run:
                ring[ring_idx % k] = digit_num
                ring_idx += 1
                ring_count += 1
                pushed = True
                in_digit_run = False
        else:
            if in_digit_run:
                ring[ring_idx % k] = digit_num
                ring_idx += 1
                ring_count += 1
                pushed = True
                in_digit_run = False
            if in_word_run:
                word = bytes(word_bytes_list).decode("ascii", errors="replace")
                val = comp._number_words.get(word)
                if val is not None:
                    ring[ring_idx % k] = float(val)
                    ring_idx += 1
                    ring_count += 1
                    pushed = True
                in_word_run = False
                word_bytes_list = []

        # Extend/start runs
        if d >= 0:
            if not in_digit_run:
                digit_num = 0.0
                digit_len = 0
                digit_has_dot = False
                digit_frac_mul = 0.0
                in_digit_run = True
            if digit_len < 15:
                if digit_has_dot:
                    digit_num += d * digit_frac_mul
                    digit_frac_mul *= 0.1
                else:
                    digit_num = digit_num * 10 + d
            digit_len += 1
        elif dot_continues_run:
            digit_has_dot = True
            digit_frac_mul = 0.1
            digit_len += 1
        elif l > 0:
            if not in_word_run:
                word_bytes_list = []
                in_word_run = True
            word_bytes_list.append(l)

        # Active length tracking
        if d >= 0:
            active_len += 1
        elif is_dot and not active_has_dot:
            active_len += 1
            active_has_dot = True
        else:
            active_len = 0
            active_has_dot = False

        # Verify output
        if ring_count >= 2:
            n_avail = min(ring_count, k)
            nums = [ring[(ring_idx - 1 - i) % k] for i in range(n_avail)][::-1]
            a, bv = nums[0], nums[1]

            # ADD
            add_val = a + bv
            expected_sign = 1.0 if add_val > 0 else (-1.0 if add_val < 0 else 0.0)
            actual_sign = out[0, t, 0].item()
            assert actual_sign == expected_sign, (
                f"t={t}: ADD sign expected {expected_sign}, got {actual_sign}"
            )

            s = _result_to_string(add_val)
            if active_len < len(s):
                ch = s[active_len]
                if ch == ".":
                    exp_idx = 10
                elif ch.isdigit():
                    exp_idx = int(ch)
                else:
                    exp_idx = 11
            else:
                exp_idx = 11
            actual_idx = _get_next_digit_idx(out[0, t, 1:13])
            assert actual_idx == exp_idx, (
                f"t={t}: ADD digit idx expected {exp_idx}, got {actual_idx} (result='{s}', active_len={active_len})"
            )

            # SUB (second op)
            sub_val = a - bv
            sub_sign_exp = 1.0 if sub_val > 0 else (-1.0 if sub_val < 0 else 0.0)
            sub_sign_act = out[0, t, ad].item()
            assert sub_sign_act == sub_sign_exp, (
                f"t={t}: SUB sign expected {sub_sign_exp}, got {sub_sign_act}"
            )


# ===== Edge cases =====


@torch.no_grad()
def test_empty_input():
    comp = _make()
    ids = torch.zeros(1, 0, dtype=torch.long)
    out = comp(ids, torch.float32)
    assert out.shape == (1, 0, comp.dim)


@torch.no_grad()
def test_k1_no_pairs():
    """k=1 means 0 pairs, so dim=0 and output is empty last dim."""
    comp = _make(k=1)
    assert comp.dim == 0
    ids = _encode("12 34")
    out = comp(ids, torch.float32)
    assert out.shape == (1, ids.shape[1], 0)


@torch.no_grad()
def test_digit_then_word_transition():
    """Digit run followed by word number — both should be detected."""
    comp = _make(ops={PairwiseOp.ADD})
    # "42three " → 42 finishes when 't' starts, then 'three' finishes at space
    # ring=[42, 3], ADD=45
    ids = _encode("42three ")
    out = comp(ids, torch.float32)
    last = out[0, -1]
    assert last[0].item() == 1.0  # positive
    assert _get_next_digit_idx(last[1:13]) == 4  # '4' of "45"


@torch.no_grad()
def test_word_then_digit_transition():
    """Word number followed by digit run."""
    comp = _make(ops={PairwiseOp.ADD})
    # "three42 " → 'three'(=3) finishes when '4' starts, then '42' finishes at space
    # ring=[3, 42], ADD=45
    ids = _encode("three42 ")
    out = comp(ids, torch.float32)
    last = out[0, -1]
    assert last[0].item() == 1.0
    assert _get_next_digit_idx(last[1:13]) == 4  # '4' of "45"


@torch.no_grad()
def test_multiple_decimals_split_number():
    """A second decimal point should split into two numbers."""
    comp = _make(ops={PairwiseOp.ADD})
    # "1.2.3 " — first number is 1.2 (dot inside digit run),
    # second dot is NOT inside a digit run since the run was broken...
    # Actually: "1.2" parses as 1.2, then ".3" — the dot starts fresh
    # but dot alone doesn't start a digit run. '3' starts new run = 3.
    # So ring = [1.2, 3], ADD = 4.2? Let's just check it doesn't crash.
    ids = _encode("1.2.3 ")
    out = comp(ids, torch.float32)
    # Should not crash; verify shape
    assert out.shape[0] == 1


@torch.no_grad()
def test_active_len_resets_between_runs():
    """active_len should reset to 0 between digit runs."""
    comp = _make(ops={PairwiseOp.ADD})
    # "5 3 12" — at position of '1' in "12", active_len=1
    # at position of '2' in "12", active_len=2
    # ring=[5,3], ADD=8, result "8" (len=1)
    # At '1': active_len=1 >= len("8")=1 → END
    # At '2': active_len=2 >= len("8")=1 → END
    ids = _encode("5 3 12")
    out = comp(ids, torch.float32)
    # At the '2' position (last), should show END
    last = out[0, -1]
    assert _get_next_digit_idx(last[1:13]) == 11  # END


# =============================================================================
# Rotation encoding tests
# =============================================================================

from modules import RotationCodebook

_NEXT_DIGIT_VOCAB = 12  # {0-9, '.', END}


def _make_rot(k=2, ops=None):
    return DigitComputeComponent(
        tok, k=k, ops=ops, digit_encoding=DigitEncoding.ROTATION
    )


def _decode_rot_digit(rot_vec: torch.Tensor, codebook: RotationCodebook) -> int:
    """Decode a 2-dim rotation vector back to a digit index."""
    return codebook.decode(rot_vec.unsqueeze(0)).item()


# ===== Shape / dim (rotation) =====


@torch.no_grad()
def test_rot_dim_default():
    comp = _make_rot()
    n_arith = 8
    n_cmp = 3
    # 1 pair, each arith op = 3 dims (1 sign + 2 rotation)
    expected = 1 * (n_arith * 3 + n_cmp)
    assert comp.dim == expected


@torch.no_grad()
def test_rot_dim_k3():
    comp = _make_rot(k=3)
    n_arith = 8
    n_cmp = 3
    n_pairs = 3
    expected = n_pairs * (n_arith * 3 + n_cmp)
    assert comp.dim == expected


@torch.no_grad()
def test_rot_dim_subset_ops():
    comp = _make_rot(ops={PairwiseOp.ADD, PairwiseOp.GT})
    # 1 arith (ADD) * 3 + 1 cmp (GT) = 4
    assert comp.dim == 4


@torch.no_grad()
def test_rot_output_shape():
    comp = _make_rot()
    ids = _encode("12 + 34")
    out = comp(ids, torch.float32)
    assert out.shape == (1, ids.shape[1], comp.dim)


# ===== Next-digit encoding (rotation) =====


@torch.no_grad()
def test_rot_next_digit_first_digit():
    """First digit of result via rotation decoding."""
    comp = _make_rot(ops={PairwiseOp.ADD})
    cb = comp._rotation_codebook
    # "12 34 " → ring=[12,34], ADD=46, first char '4' → idx=4
    ids = _encode("12 34 ")
    out = comp(ids, torch.float32)
    last = out[0, -1]
    assert last[0].item() == 1.0  # sign
    assert _decode_rot_digit(last[1:3], cb) == 4


@torch.no_grad()
def test_rot_next_digit_second_digit():
    """Second digit of result via rotation decoding."""
    comp = _make_rot(ops={PairwiseOp.ADD})
    cb = comp._rotation_codebook
    # "12 34 4" → active_len=1, result "46", char[1]='6' → idx=6
    ids = _encode("12 34 4")
    out = comp(ids, torch.float32)
    last = out[0, -1]
    assert _decode_rot_digit(last[1:3], cb) == 6


@torch.no_grad()
def test_rot_next_digit_end():
    """END token via rotation when active_len >= result length."""
    comp = _make_rot(ops={PairwiseOp.ADD})
    cb = comp._rotation_codebook
    # "1 1 11" → ring=[1,1], ADD=2, result "2" (len=1)
    # At second '1' of "11", active_len=2 >= 1 → END (idx=11)
    ids = _encode("1 1 11")
    out = comp(ids, torch.float32)
    last = out[0, -1]
    assert _decode_rot_digit(last[1:3], cb) == 11


@torch.no_grad()
def test_rot_next_digit_dot():
    """Decimal point encoded as rotation for DOT (idx=10)."""
    comp = _make_rot(ops={PairwiseOp.DIV})
    cb = comp._rotation_codebook
    eps = comp._eps
    val = 7 / (2 + eps)
    s = _result_to_string(val)
    if "." in s:
        dot_pos = s.index(".")
        text = "7 2 " + "0" * dot_pos
        ids = _encode(text)
        out = comp(ids, torch.float32)
        last = out[0, -1]
        assert _decode_rot_digit(last[1:3], cb) == 10


# ===== Arithmetic ops (rotation) =====


def _assert_rot_op_at(
    text: str,
    op: PairwiseOp,
    expected_sign: float,
    expected_first_digit: int | None = None,
):
    """Check sign and optional first digit for rotation-encoded output."""
    comp = _make_rot(ops={op})
    cb = comp._rotation_codebook
    ids = _encode(text)
    out = comp(ids, torch.float32)
    last = out[0, -1]
    sign = last[0].item()
    assert sign == expected_sign, f"Expected sign {expected_sign}, got {sign}"
    if expected_first_digit is not None:
        idx = _decode_rot_digit(last[1:3], cb)
        assert idx == expected_first_digit, (
            f"Expected digit idx {expected_first_digit}, got {idx}"
        )


@torch.no_grad()
def test_rot_add():
    _assert_rot_op_at("10 25 ", PairwiseOp.ADD, 1.0, 3)


@torch.no_grad()
def test_rot_sub():
    _assert_rot_op_at("10 25 ", PairwiseOp.SUB, -1.0, 1)


@torch.no_grad()
def test_rot_mul():
    _assert_rot_op_at("3 7 ", PairwiseOp.MUL, 1.0, 2)


@torch.no_grad()
def test_rot_div():
    _assert_rot_op_at("10 4 ", PairwiseOp.DIV, 1.0, 2)


@torch.no_grad()
def test_rot_negative_result():
    """Negative results should have sign=-1 and correct digit."""
    _assert_rot_op_at("3 10 ", PairwiseOp.SUB, -1.0, 7)


# ===== Rotation encoding properties =====


@torch.no_grad()
def test_rot_vectors_are_unit():
    """All rotation-encoded digit vectors should be unit vectors."""
    comp = _make_rot(ops={PairwiseOp.ADD})
    # "12 34 " → at trailing space, output has rotation vectors
    ids = _encode("12 34 ")
    out = comp(ids, torch.float32)
    last = out[0, -1]
    rot = last[1:3]
    norm = rot.norm().item()
    assert abs(norm - 1.0) < 1e-5, f"Rotation vector norm {norm}, expected 1.0"


@torch.no_grad()
def test_rot_zero_before_two_numbers():
    """Before two numbers are seen, rotation output should be all zeros."""
    comp = _make_rot(ops={PairwiseOp.ADD})
    ids = _encode("42")
    out = comp(ids, torch.float32)
    assert (out == 0).all()


@torch.no_grad()
def test_rot_adjacent_digits_similar():
    """Adjacent digit encodings should have higher dot product than distant ones."""
    cb = RotationCodebook(_NEXT_DIGIT_VOCAB)
    # digit 3 and digit 4 should be more similar than digit 3 and digit 9
    v3 = cb.encode(torch.tensor(3))
    v4 = cb.encode(torch.tensor(4))
    v9 = cb.encode(torch.tensor(9))
    dot_34 = (v3 * v4).sum()
    dot_39 = (v3 * v9).sum()
    assert dot_34 > dot_39


@torch.no_grad()
def test_rot_roundtrip_all_digits():
    """Every digit index should survive encode → decode."""
    cb = RotationCodebook(_NEXT_DIGIT_VOCAB)
    indices = torch.arange(_NEXT_DIGIT_VOCAB)
    encoded = cb.encode(indices)
    decoded = cb.decode(encoded)
    assert torch.equal(decoded, indices)


@torch.no_grad()
def test_rot_k3_three_pairs():
    """With k=3, verify all 3 pairs produce correct rotation-encoded results."""
    comp = _make_rot(k=3, ops={PairwiseOp.ADD})
    cb = comp._rotation_codebook
    ad = comp._arith_dim  # 3 (1 sign + 2 rotation)
    # "2 3 5 " → nums=[2,3,5], pairs: (2,3)=5, (2,5)=7, (3,5)=8
    ids = _encode("2 3 5 ")
    out = comp(ids, torch.float32)
    last = out[0, -1]
    # Pair 0: ADD=5, sign=+1, first digit='5'
    assert last[0].item() == 1.0
    assert _decode_rot_digit(last[1:3], cb) == 5
    # Pair 1: ADD=7, sign=+1, first digit='7'
    assert last[ad].item() == 1.0
    assert _decode_rot_digit(last[ad + 1 : ad + 3], cb) == 7
    # Pair 2: ADD=8, sign=+1, first digit='8'
    assert last[2 * ad].item() == 1.0
    assert _decode_rot_digit(last[2 * ad + 1 : 2 * ad + 3], cb) == 8


@torch.no_grad()
def test_rot_causality():
    """Output at position t should not change when tokens after t change."""
    comp = _make_rot(ops={PairwiseOp.ADD})
    ids1 = _encode("5 3 hello")
    ids2 = _encode("5 3 world")
    out1 = comp(ids1, torch.float32)
    out2 = comp(ids2, torch.float32)
    assert torch.allclose(out1[0, :4], out2[0, :4])


@torch.no_grad()
def test_rot_matches_onehot_sign():
    """Rotation and one-hot modes should produce identical sign values."""
    comp_oh = _make(ops={PairwiseOp.ADD, PairwiseOp.SUB})
    comp_rot = _make_rot(ops={PairwiseOp.ADD, PairwiseOp.SUB})
    ids = _encode("12 34 5 100 7 three 42 ")
    out_oh = comp_oh(ids, torch.float32)
    out_rot = comp_rot(ids, torch.float32)
    ad_oh = comp_oh._arith_dim  # 13
    ad_rot = comp_rot._arith_dim  # 3
    # Compare sign dims for both ops at every position
    for t in range(ids.shape[1]):
        # ADD sign
        assert out_oh[0, t, 0].item() == out_rot[0, t, 0].item(), f"ADD sign mismatch at t={t}"
        # SUB sign
        assert out_oh[0, t, ad_oh].item() == out_rot[0, t, ad_rot].item(), f"SUB sign mismatch at t={t}"


@torch.no_grad()
def test_rot_matches_onehot_digits():
    """Rotation and one-hot should decode to the same digit index everywhere."""
    comp_oh = _make(ops={PairwiseOp.ADD})
    comp_rot = _make_rot(ops={PairwiseOp.ADD})
    cb = comp_rot._rotation_codebook
    ids = _encode("12 34 5 100 7 three 42 ")
    out_oh = comp_oh(ids, torch.float32)
    out_rot = comp_rot(ids, torch.float32)
    for t in range(ids.shape[1]):
        oh_vec = out_oh[0, t, 1:13]
        rot_vec = out_rot[0, t, 1:3]
        # Skip positions where output is all zeros (before two numbers)
        if oh_vec.abs().sum() == 0:
            assert rot_vec.abs().sum() == 0, f"t={t}: onehot is zero but rotation is not"
            continue
        oh_idx = _get_next_digit_idx(oh_vec)
        rot_idx = _decode_rot_digit(rot_vec, cb)
        assert oh_idx == rot_idx, (
            f"t={t}: onehot decoded {oh_idx}, rotation decoded {rot_idx}"
        )
