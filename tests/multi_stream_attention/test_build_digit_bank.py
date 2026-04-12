"""Unit tests for CausalArithmeticMultiStreamAttention._build_digit_bank.

Tests the non-differentiable digit bank computation that converts arithmetic
results into digit sequences.  The implementation uses pure tensor ops
(fixed-point digit extraction in float64) for torch.compile fullgraph
compatibility.
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import pytest

from efficient_byte_tokenizer import EfficientByteTokenizer
from modules import RMSNorm, CenterLastDim
from multi_streams import (
    StreamType,
    StreamID,
    StreamConfig,
    MultiStreamConfig,
)
from multi_stream_attention import CausalArithmeticMultiStreamAttention

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOT = CausalArithmeticMultiStreamAttention.DIGIT_IDX_DOT  # 10
END = CausalArithmeticMultiStreamAttention.DIGIT_IDX_END  # 11
MAX_LEN = CausalArithmeticMultiStreamAttention._MAX_RESULT_LEN  # 20

# Op indices in the 5-op bank
OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MOD = range(5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_arith_attn(n_max: int = 4) -> CausalArithmeticMultiStreamAttention:
    """Build a minimal CausalArithmeticMultiStreamAttention for testing."""
    tok = EfficientByteTokenizer()
    logit_id = StreamID(StreamType.LOGIT)
    context_id = StreamID(StreamType.CONTEXT)
    stream_config = MultiStreamConfig(
        streams=[
            StreamConfig(logit_id, dim=tok.vocab_size, read_only=False, input_norm_types=[CenterLastDim, RMSNorm]),
            StreamConfig(context_id, dim=16, read_only=False, input_norm_types=[RMSNorm]),
        ]
    )
    return CausalArithmeticMultiStreamAttention(
        stream_config=stream_config,
        tok=tok,
        n_max=n_max,
        d_head=8,
    )


def _ref_digit_seq(value: float) -> list[int]:
    """Reference: fixed-point digit extraction matching the tensor algorithm.

    Extracts integer digits via floor-divide by powers of 10, fractional
    digits via multiply-by-powers-of-10, with trailing-zero stripping
    limited to float64 significant precision (~15 digits total).
    """
    av = min(abs(value), 1e15)
    int_part = int(math.floor(av))
    frac_part = av - int_part

    # Number of integer digits (min 1)
    n_int = max(1, int(math.floor(math.log10(max(int_part, 1)))) + 1)
    n_int = max(1, min(n_int, MAX_LEN))

    seq = [END] * MAX_LEN

    # Integer digits
    for p in range(MAX_LEN):
        power = n_int - 1 - p
        if power >= 0:
            seq[p] = int(math.floor(int_part / (10 ** power))) % 10

    # Fractional digits
    F = min(MAX_LEN - 2, 18)
    frac_digits = [
        int(math.floor(frac_part * (10 ** (k + 1)))) % 10 for k in range(F)
    ]

    # Significant precision limit and trailing-zero stripping
    max_sig = max(0, min(15 - n_int, F))
    sig = frac_digits[:max_sig]
    n_frac = 0
    for k in range(len(sig) - 1, -1, -1):
        if sig[k] != 0:
            n_frac = k + 1
            break

    if n_frac > 0 and n_int < MAX_LEN:
        seq[n_int] = DOT
        for k in range(n_frac):
            pos = n_int + 1 + k
            if pos < MAX_LEN:
                seq[pos] = frac_digits[k]

    return seq


def _ref_sign(value: float) -> float:
    if value > 0:
        return 1.0
    elif value < 0:
        return -1.0
    return 0.0


def _ref_ops(a: float, bv: float, eps: float) -> list[float]:
    """Reference: compute the 5 arithmetic operations."""
    b_abs = abs(bv) + eps
    return [a + bv, a - bv, a * bv, a / b_abs, math.fmod(a, b_abs)]


# ---------------------------------------------------------------------------
# Tests: _build_digit_bank
# ---------------------------------------------------------------------------


class TestBuildDigitBankShapes:
    """Output shapes and value ranges."""

    @pytest.fixture
    def attn(self):
        return _make_arith_attn(n_max=4)

    def test_shapes(self, attn):
        B, N = 2, 4
        values = torch.tensor([[1.0, 2.0, 3.0, 0.0], [5.0, 10.0, 0.0, 0.0]])
        mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
        digit_seqs, signs = attn._build_digit_bank(values, mask)

        P = attn._n_pairs  # C(4,2) = 6
        assert digit_seqs.shape == (B, P, attn.n_ops, MAX_LEN)
        assert signs.shape == (B, P, attn.n_ops)

    def test_digit_value_range(self, attn):
        values = torch.tensor([[3.0, 7.0, 2.0, 5.0]])
        mask = torch.ones(1, 4, dtype=torch.bool)
        digit_seqs, signs = attn._build_digit_bank(values, mask)

        assert (digit_seqs >= 0).all() and (digit_seqs <= END).all()

    def test_sign_value_range(self, attn):
        values = torch.tensor([[3.0, -7.0, 2.0, 5.0]])
        mask = torch.ones(1, 4, dtype=torch.bool)
        _, signs = attn._build_digit_bank(values, mask)

        assert ((signs == -1) | (signs == 0) | (signs == 1)).all()


class TestBuildDigitBankIntegerOps:
    """Verify digit sequences for clean integer arithmetic."""

    @pytest.fixture
    def attn(self):
        return _make_arith_attn(n_max=4)

    def _run(self, attn, a, b):
        """Helper: build digit bank for a single (a, b) pair and return
        (digit_seqs[pair0], signs[pair0]) shaped (n_ops, MAX_LEN) and (n_ops,).
        """
        values = torch.tensor([[a, b, 0.0, 0.0]])
        mask = torch.tensor([[True, True, False, False]])
        digit_seqs, signs = attn._build_digit_bank(values, mask)
        return digit_seqs[0, 0], signs[0, 0]  # pair (0,1)

    def test_addition(self, attn):
        seqs, signs = self._run(attn, 3.0, 7.0)
        assert seqs[OP_ADD].tolist() == _ref_digit_seq(10.0)
        assert signs[OP_ADD].item() == 1.0

    def test_subtraction_positive(self, attn):
        seqs, signs = self._run(attn, 7.0, 3.0)
        assert seqs[OP_SUB].tolist() == _ref_digit_seq(4.0)
        assert signs[OP_SUB].item() == 1.0

    def test_subtraction_negative(self, attn):
        seqs, signs = self._run(attn, 3.0, 7.0)
        assert seqs[OP_SUB].tolist() == _ref_digit_seq(-4.0)
        assert signs[OP_SUB].item() == -1.0

    def test_subtraction_zero(self, attn):
        seqs, signs = self._run(attn, 5.0, 5.0)
        assert seqs[OP_SUB].tolist() == _ref_digit_seq(0.0)
        assert signs[OP_SUB].item() == 0.0

    def test_multiplication(self, attn):
        seqs, signs = self._run(attn, 3.0, 7.0)
        assert seqs[OP_MUL].tolist() == _ref_digit_seq(21.0)
        assert signs[OP_MUL].item() == 1.0

    def test_multiplication_by_zero(self, attn):
        seqs, signs = self._run(attn, 0.0, 7.0)
        assert seqs[OP_MUL].tolist() == _ref_digit_seq(0.0)
        assert signs[OP_MUL].item() == 0.0

    def test_large_product(self, attn):
        seqs, signs = self._run(attn, 999.0, 999.0)
        assert seqs[OP_MUL].tolist() == _ref_digit_seq(998001.0)
        assert signs[OP_MUL].item() == 1.0


class TestBuildDigitBankDivMod:
    """Division and modulo (these involve eps and can produce fractions)."""

    @pytest.fixture
    def attn(self):
        return _make_arith_attn(n_max=4)

    def _run(self, attn, a, b):
        values = torch.tensor([[a, b, 0.0, 0.0]])
        mask = torch.tensor([[True, True, False, False]])
        digit_seqs, signs = attn._build_digit_bank(values, mask)
        return digit_seqs[0, 0], signs[0, 0]

    def test_exact_division(self, attn):
        """10 / (5+eps) ≈ 2.0 — verify against reference."""
        eps = attn.eps
        expected_val = 10.0 / (5.0 + eps)
        seqs, signs = self._run(attn, 10.0, 5.0)
        assert seqs[OP_DIV].tolist() == _ref_digit_seq(expected_val)
        assert signs[OP_DIV].item() == _ref_sign(expected_val)

    def test_fractional_division(self, attn):
        """10 / (3+eps) — non-terminating decimal, verify against reference."""
        eps = attn.eps
        expected_val = 10.0 / (3.0 + eps)
        seqs, signs = self._run(attn, 10.0, 3.0)
        assert seqs[OP_DIV].tolist() == _ref_digit_seq(expected_val)

    def test_mod_exact(self, attn):
        """10 % (3+eps) — verify against reference."""
        eps = attn.eps
        expected_val = math.fmod(10.0, 3.0 + eps)
        seqs, signs = self._run(attn, 10.0, 3.0)
        assert seqs[OP_MOD].tolist() == _ref_digit_seq(expected_val)
        assert signs[OP_MOD].item() == _ref_sign(expected_val)

    def test_division_of_zero(self, attn):
        """0 / (5+eps) = 0."""
        seqs, signs = self._run(attn, 0.0, 5.0)
        eps = attn.eps
        expected_val = 0.0 / (5.0 + eps)
        assert seqs[OP_DIV].tolist() == _ref_digit_seq(expected_val)
        assert signs[OP_DIV].item() == 0.0


class TestBuildDigitBankMasking:
    """Invalid pairs (masked numbers) produce all-END sequences."""

    @pytest.fixture
    def attn(self):
        return _make_arith_attn(n_max=4)

    def test_fully_masked(self, attn):
        """All numbers masked → every pair is all-END, signs=0."""
        values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        mask = torch.zeros(1, 4, dtype=torch.bool)
        digit_seqs, signs = attn._build_digit_bank(values, mask)

        assert (digit_seqs == END).all()
        assert (signs == 0).all()

    def test_one_valid(self, attn):
        """Only one valid number → no valid pairs → all END."""
        values = torch.tensor([[5.0, 7.0, 0.0, 0.0]])
        mask = torch.tensor([[True, False, False, False]])
        digit_seqs, signs = attn._build_digit_bank(values, mask)

        assert (digit_seqs == END).all()
        assert (signs == 0).all()

    def test_partial_mask(self, attn):
        """Pairs involving a masked number are END; valid pairs are computed."""
        values = torch.tensor([[2.0, 3.0, 99.0, 0.0]])
        mask = torch.tensor([[True, True, False, False]])
        digit_seqs, signs = attn._build_digit_bank(values, mask)

        # Pair (0,1) is valid → add=5
        assert digit_seqs[0, 0, OP_ADD].tolist() == _ref_digit_seq(5.0)

        # Pair (0,2), (1,2), and any pair with index 3 are invalid
        for p in range(1, attn._n_pairs):
            i, j = attn.pair_i[p].item(), attn.pair_j[p].item()
            if not (mask[0, i] and mask[0, j]):
                assert (digit_seqs[0, p] == END).all(), f"pair ({i},{j}) should be all END"
                assert (signs[0, p] == 0).all()


class TestBuildDigitBankNegativeInputs:
    """Negative input values are handled correctly."""

    @pytest.fixture
    def attn(self):
        return _make_arith_attn(n_max=4)

    def test_negative_plus_positive(self, attn):
        values = torch.tensor([[-3.0, 7.0, 0.0, 0.0]])
        mask = torch.tensor([[True, True, False, False]])
        digit_seqs, signs = attn._build_digit_bank(values, mask)

        # add: -3 + 7 = 4
        assert digit_seqs[0, 0, OP_ADD].tolist() == _ref_digit_seq(4.0)
        assert signs[0, 0, OP_ADD].item() == 1.0

        # sub: -3 - 7 = -10
        assert digit_seqs[0, 0, OP_SUB].tolist() == _ref_digit_seq(-10.0)
        assert signs[0, 0, OP_SUB].item() == -1.0

    def test_both_negative(self, attn):
        eps = attn.eps
        values = torch.tensor([[-4.0, -6.0, 0.0, 0.0]])
        mask = torch.tensor([[True, True, False, False]])
        digit_seqs, signs = attn._build_digit_bank(values, mask)

        # add: -4 + -6 = -10
        assert digit_seqs[0, 0, OP_ADD].tolist() == _ref_digit_seq(-10.0)
        assert signs[0, 0, OP_ADD].item() == -1.0

        # mul: -4 * -6 = 24
        assert digit_seqs[0, 0, OP_MUL].tolist() == _ref_digit_seq(24.0)
        assert signs[0, 0, OP_MUL].item() == 1.0

        # div: -4 / (6 + eps)
        div_val = -4.0 / (6.0 + eps)
        assert digit_seqs[0, 0, OP_DIV].tolist() == _ref_digit_seq(div_val)
        assert signs[0, 0, OP_DIV].item() == _ref_sign(div_val)


class TestBuildDigitBankBatch:
    """Batch independence and multi-pair correctness."""

    @pytest.fixture
    def attn(self):
        return _make_arith_attn(n_max=4)

    def test_batch_independence(self, attn):
        """Different batches produce independent, correct results."""
        values = torch.tensor([[2.0, 3.0, 0.0, 0.0], [10.0, 5.0, 0.0, 0.0]])
        mask = torch.tensor([[True, True, False, False], [True, True, False, False]])
        digit_seqs, signs = attn._build_digit_bank(values, mask)

        # Batch 0: 2+3=5
        assert digit_seqs[0, 0, OP_ADD].tolist() == _ref_digit_seq(5.0)
        # Batch 1: 10+5=15
        assert digit_seqs[1, 0, OP_ADD].tolist() == _ref_digit_seq(15.0)

    def test_multiple_valid_pairs(self, attn):
        """With 3 valid numbers, all 3 pairs are correct."""
        values = torch.tensor([[2.0, 3.0, 5.0, 0.0]])
        mask = torch.tensor([[True, True, True, False]])
        digit_seqs, signs = attn._build_digit_bank(values, mask)

        # Pair ordering for n_max=4: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        pi = attn.pair_i.tolist()
        pj = attn.pair_j.tolist()
        pair_idx = {(pi[p], pj[p]): p for p in range(attn._n_pairs)}

        p01 = pair_idx[(0, 1)]
        p02 = pair_idx[(0, 2)]
        p12 = pair_idx[(1, 2)]
        assert digit_seqs[0, p01, OP_ADD].tolist() == _ref_digit_seq(5.0)   # 2+3
        assert digit_seqs[0, p02, OP_ADD].tolist() == _ref_digit_seq(7.0)   # 2+5
        assert digit_seqs[0, p12, OP_ADD].tolist() == _ref_digit_seq(8.0)   # 3+5
        assert digit_seqs[0, p12, OP_MUL].tolist() == _ref_digit_seq(15.0)  # 3*5


class TestBuildDigitBankComprehensive:
    """Exhaustive verification: every pair × op against reference."""

    @pytest.fixture
    def attn(self):
        return _make_arith_attn(n_max=4)

    @pytest.mark.parametrize(
        "vals",
        [
            [7.0, 2.0, -4.5, 100.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 7.0, 11.0],
            [999.0, 1.0, -1.0, 0.5],
            [0.25, 0.75, 1.5, 10.0],
        ],
        ids=["mixed", "zeros", "primes", "edge", "fractions"],
    )
    def test_all_pairs_all_ops(self, attn, vals):
        eps = attn.eps
        values = torch.tensor([vals])
        mask = torch.ones(1, 4, dtype=torch.bool)
        digit_seqs, signs = attn._build_digit_bank(values, mask)

        pi = attn.pair_i.tolist()
        pj = attn.pair_j.tolist()

        for p, (i, j) in enumerate(zip(pi, pj)):
            a, bv = vals[i], vals[j]
            for o, result in enumerate(_ref_ops(a, bv, eps)):
                expected_seq = _ref_digit_seq(result)
                expected_sign = _ref_sign(result)
                actual_seq = digit_seqs[0, p, o].tolist()
                actual_sign = signs[0, p, o].item()

                assert actual_seq == expected_seq, (
                    f"vals={vals} pair ({i},{j}) op {o}: "
                    f"result={result}, expected {expected_seq[:6]}…, got {actual_seq[:6]}…"
                )
                assert actual_sign == expected_sign, (
                    f"vals={vals} pair ({i},{j}) op {o}: "
                    f"sign expected {expected_sign}, got {actual_sign}"
                )
