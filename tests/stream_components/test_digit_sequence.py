"""Tests for DigitSequenceComponent from byte_modules."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
import torch

from efficient_byte_tokenizer import EfficientByteTokenizer
from byte_stream_components import DigitSequenceComponent

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
        assert out.shape == (1, ids.shape[1], 5)

    @torch.no_grad()
    def test_dim(self):
        comp = DigitSequenceComponent(tok, id_freqs=1)
        assert comp.dim == 5

    @torch.no_grad()
    def test_digit_state(self):
        comp = DigitSequenceComponent(tok, id_freqs=1)
        # "a1b" → BOS(0), a(1), 1(2), b(3)
        ids = _encode("a1b")
        out = comp(ids, dtype=torch.float32)
        assert out[0, 2, 0].item() == pytest.approx(1.0)  # '1' is digit
        assert out[0, 1, 0].item() == pytest.approx(-1.0)  # 'a' is not
        assert out[0, 3, 0].item() == pytest.approx(-1.0)  # 'b' is not

    @torch.no_grad()
    def test_non_digit_pos_zero(self):
        comp = DigitSequenceComponent(tok, id_freqs=1)
        ids = _encode("a1b")
        out = comp(ids, dtype=torch.float32)
        # dims 1-2 = (0, 0) for non-digit positions (rotation zeroed out)
        for pos in [0, 1, 3]:  # BOS, 'a', 'b'
            assert out[0, pos, 1].item() == pytest.approx(0.0)
            assert out[0, pos, 2].item() == pytest.approx(0.0)

    @torch.no_grad()
    def test_number_id_increments(self):
        comp = DigitSequenceComponent(tok, id_freqs=1)
        # "1a2" → BOS(0), 1(1), a(2), 2(3) — two separate digit runs
        ids = _encode("1a2")
        out = comp(ids, dtype=torch.float32)
        # number_id sin/cos (dims 3-4) should differ between the two runs
        id_first = out[0, 1, 3:]  # digit '1', run 1
        id_second = out[0, 3, 3:]  # digit '2', run 2
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
        # Same run → same number_id (dims 3+)
        torch.testing.assert_close(out[0, 1, 3:], out[0, 2, 3:])
        torch.testing.assert_close(out[0, 1, 3:], out[0, 3, 3:])

    @torch.no_grad()
    def test_digit_pos_rotation_distinct(self):
        """Each digit position gets a distinct rotation vector."""
        comp = DigitSequenceComponent(tok, id_freqs=1)
        ids = _encode("12345")
        out = comp(ids, dtype=torch.float32)
        # Positions 1..5 are digits at run positions 0..4
        rot_vecs = out[0, 1:6, 1:3]  # (5, 2)
        # Each position should be a unit vector
        norms = rot_vecs.norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones(5), atol=1e-5, rtol=1e-5)
        # Adjacent positions should be more similar than distant ones
        dot_01 = (rot_vecs[0] * rot_vecs[1]).sum()
        dot_04 = (rot_vecs[0] * rot_vecs[4]).sum()
        assert dot_01 > dot_04

    @torch.no_grad()
    def test_bos_reset(self):
        """Number IDs reset at document boundary."""
        import numpy as np

        comp = DigitSequenceComponent(tok, id_freqs=1)
        doc1 = tok.encode("12 34")
        doc2 = tok.encode("56 78")
        multi = np.concatenate([[tok.bos_id], doc1, [tok.bos_id], doc2])
        ids_multi = torch.tensor(multi, dtype=torch.long).unsqueeze(0)
        out_multi = comp(ids_multi, dtype=torch.float32)

        solo = np.concatenate([[tok.bos_id], doc2])
        ids_solo = torch.tensor(solo, dtype=torch.long).unsqueeze(0)
        out_solo = comp(ids_solo, dtype=torch.float32)

        doc2_start = 1 + len(doc1)
        torch.testing.assert_close(
            out_multi[0, doc2_start:], out_solo[0], atol=1e-5, rtol=1e-5
        )
