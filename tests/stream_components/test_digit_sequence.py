"""Tests for DigitSequenceComponent from byte_modules."""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
import torch

from efficient_byte_tokenizer import EfficientByteTokenizer
from byte_modules import DigitSequenceComponent

tok = EfficientByteTokenizer()


def _encode(text: str) -> torch.Tensor:
    import numpy as np

    ids = np.concatenate([[tok.bos_id], tok.encode(text)])
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


class TestDigitSequenceComponent:

    @torch.no_grad()
    def test_shape(self):
        comp = DigitSequenceComponent(tok, id_freqs=1)
        ids = _encode("a1b")
        out = comp(ids, dtype=torch.float32)
        assert out.shape == (1, ids.shape[1], 4)

    @torch.no_grad()
    def test_dim(self):
        comp = DigitSequenceComponent(tok, id_freqs=1)
        assert comp.dim == 4

    @torch.no_grad()
    def test_digit_state(self):
        comp = DigitSequenceComponent(tok, id_freqs=1)
        # "a1b" → BOS(0), a(1), 1(2), b(3)
        ids = _encode("a1b")
        out = comp(ids, dtype=torch.float32)
        assert out[0, 2, 0].item() == pytest.approx(1.0)   # '1' is digit
        assert out[0, 1, 0].item() == pytest.approx(-1.0)  # 'a' is not
        assert out[0, 3, 0].item() == pytest.approx(-1.0)  # 'b' is not

    @torch.no_grad()
    def test_non_digit_pos_minus_one(self):
        comp = DigitSequenceComponent(tok, id_freqs=1)
        ids = _encode("a1b")
        out = comp(ids, dtype=torch.float32)
        # dim 1 = -1 for non-digit positions
        assert out[0, 0, 1].item() == pytest.approx(-1.0)  # BOS
        assert out[0, 1, 1].item() == pytest.approx(-1.0)  # 'a'
        assert out[0, 3, 1].item() == pytest.approx(-1.0)  # 'b'

    @torch.no_grad()
    def test_number_id_increments(self):
        comp = DigitSequenceComponent(tok, id_freqs=1)
        # "1a2" → BOS(0), 1(1), a(2), 2(3) — two separate digit runs
        ids = _encode("1a2")
        out = comp(ids, dtype=torch.float32)
        # number_id sin/cos (dims 2-3) should differ between the two runs
        id_first = out[0, 1, 2:]   # digit '1', run 1
        id_second = out[0, 3, 2:]  # digit '2', run 2
        assert not torch.allclose(id_first, id_second)

    @torch.no_grad()
    def test_consecutive_digits(self):
        comp = DigitSequenceComponent(tok, id_freqs=1)
        # "123" → BOS(0), 1(1), 2(2), 3(3) — single digit run
        ids = _encode("123")
        out = comp(ids, dtype=torch.float32)
        # All three digits should be in-digit
        for pos in [1, 2, 3]:
            assert out[0, pos, 0].item() == pytest.approx(1.0)
        # Same run → same number_id
        torch.testing.assert_close(out[0, 1, 2:], out[0, 2, 2:])
        torch.testing.assert_close(out[0, 1, 2:], out[0, 3, 2:])
