"""Tests for MultiByteStateComponent from byte_modules."""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
import torch

from efficient_byte_tokenizer import EfficientByteTokenizer
from byte_modules import MultiByteStateComponent

tok = EfficientByteTokenizer()


def _encode(text: str) -> torch.Tensor:
    return torch.tensor(tok.encode(text), dtype=torch.long).unsqueeze(0)


def _make(**kwargs):
    defaults = dict(id_freqs=1, base=10000.0)
    defaults.update(kwargs)
    return MultiByteStateComponent(tok, **defaults)


class TestMultiByteStateComponent:

    @torch.no_grad()
    def test_shape(self):
        comp = _make()
        ids = _encode("hello")
        out = comp(ids, dtype=torch.float32)
        assert out.shape == (1, ids.shape[1], 4)

    @torch.no_grad()
    def test_dim(self):
        comp = _make()
        assert comp.dim == 4

    @torch.no_grad()
    def test_ascii_not_multibyte(self):
        comp = _make()
        ids = _encode("abc")
        out = comp(ids, dtype=torch.float32)
        # a(0) b(1) c(2)
        # All ASCII letter positions should have in_sequence=-1, remaining=-1
        for pos in [0, 1, 2]:
            assert out[0, pos, 0].item() == pytest.approx(-1.0)
            assert out[0, pos, 1].item() == pytest.approx(-1.0)

    @torch.no_grad()
    def test_two_byte_char(self):
        comp = _make()
        ids = _encode("\u00e9")  # "é" → 0xC3 0xA9
        out = comp(ids, dtype=torch.float32)
        # lead_2(0), continuation(1)
        # Lead byte (pos 0): in_sequence=+1, remaining=+1 (1 continuation expected)
        assert out[0, 0, 0].item() == pytest.approx(1.0)
        assert out[0, 0, 1].item() == pytest.approx(1.0)
        # Continuation byte (pos 1): in_sequence=+1, remaining=0
        assert out[0, 1, 0].item() == pytest.approx(1.0)
        assert out[0, 1, 1].item() == pytest.approx(0.0)

    @torch.no_grad()
    def test_codepoint_id_increments(self):
        comp = _make()
        ids = _encode("\u00e9\u00e0")  # "éà" → 4 bytes
        out = comp(ids, dtype=torch.float32)
        # lead_2(0), cont(1), lead_2(2), cont(3)
        # Codepoint ID (dims 2-3) should be same within each char
        torch.testing.assert_close(out[0, 0, 2:4], out[0, 1, 2:4])
        torch.testing.assert_close(out[0, 2, 2:4], out[0, 3, 2:4])
        # But differ between the two characters (ID increments at each lead byte)
        assert not torch.allclose(out[0, 0, 2:4], out[0, 2, 2:4], atol=1e-5)

    @torch.no_grad()
    def test_mixed_ascii_and_multibyte(self):
        comp = _make()
        ids = _encode("a\u00e9")  # "aé" → a, 0xC3, 0xA9
        out = comp(ids, dtype=torch.float32)
        # 'a' at pos 0: not multibyte
        assert out[0, 0, 0].item() == pytest.approx(-1.0)
        # Lead byte of 'é' at pos 1: in multibyte
        assert out[0, 1, 0].item() == pytest.approx(1.0)
        # Continuation byte at pos 2: in multibyte
        assert out[0, 2, 0].item() == pytest.approx(1.0)
