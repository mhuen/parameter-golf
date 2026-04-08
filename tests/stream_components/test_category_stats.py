"""Tests for ByteCategoryStatsComponent from byte_modules."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
import torch

from efficient_byte_tokenizer import EfficientByteTokenizer
from byte_stream_components import ByteCategoryStatsComponent

tok = EfficientByteTokenizer()


def _encode(text: str) -> torch.Tensor:
    import numpy as np

    ids = np.concatenate([[tok.bos_id], tok.encode(text)])
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


class TestByteCategoryStatsComponent:
    @torch.no_grad()
    def test_shape(self):
        comp = ByteCategoryStatsComponent(tok)
        ids = _encode("hello")
        out = comp(ids, dtype=torch.float32)
        assert out.shape == (1, ids.shape[1], 6)

    @torch.no_grad()
    def test_dim(self):
        comp = ByteCategoryStatsComponent(tok)
        assert comp.dim == 6

    @torch.no_grad()
    def test_fractions_sum_le_one(self):
        comp = ByteCategoryStatsComponent(tok)
        ids = _encode("Hello, World! 123")
        out = comp(ids, dtype=torch.float32)
        sums = out.sum(dim=-1)  # (B, S)
        assert (sums <= 1.0 + 1e-5).all()

    @torch.no_grad()
    def test_all_letters(self):
        comp = ByteCategoryStatsComponent(tok)
        # "abc" → BOS(0), a(1), b(2), c(3)
        # Letter fraction at pos 3 = 3/4
        ids = _encode("abc")
        out = comp(ids, dtype=torch.float32)
        # _STATS_CATEGORIES order: DIGIT, LETTER, SEPARATOR, PUNCTUATION, SYMBOL, MULTIBYTE
        letter_frac = out[0, 3, 1].item()  # letter is index 1
        assert letter_frac == pytest.approx(3.0 / 4.0)

    @torch.no_grad()
    def test_mixed(self):
        comp = ByteCategoryStatsComponent(tok)
        # "a1" → BOS(0), a(1), 1(2)
        # At pos 2: letter=1/3, digit=1/3
        ids = _encode("a1")
        out = comp(ids, dtype=torch.float32)
        digit_frac = out[0, 2, 0].item()  # digit is index 0
        letter_frac = out[0, 2, 1].item()  # letter is index 1
        assert digit_frac == pytest.approx(1.0 / 3.0)
        assert letter_frac == pytest.approx(1.0 / 3.0)

    @torch.no_grad()
    def test_causality(self):
        comp = ByteCategoryStatsComponent(tok)
        ids_short = _encode("abc")
        ids_long = _encode("abc123")
        out_short = comp(ids_short, dtype=torch.float32)
        out_long = comp(ids_long, dtype=torch.float32)
        torch.testing.assert_close(
            out_short[0, : ids_short.shape[1]],
            out_long[0, : ids_short.shape[1]],
        )
