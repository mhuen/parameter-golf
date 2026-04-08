"""Tests for ColumnPositionComponent from byte_modules."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
import torch

from efficient_byte_tokenizer import EfficientByteTokenizer
from byte_stream_components import ColumnPositionComponent

tok = EfficientByteTokenizer()


def _encode(text: str) -> torch.Tensor:
    import numpy as np

    ids = np.concatenate([[tok.bos_id], tok.encode(text)])
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


class TestColumnPositionComponent:
    @torch.no_grad()
    def test_shape(self):
        comp = ColumnPositionComponent(tok, num_freqs=3)
        ids = _encode("hello")
        out = comp(ids, dtype=torch.float32)
        assert out.shape == (1, ids.shape[1], 6)

    @torch.no_grad()
    def test_dim(self):
        comp = ColumnPositionComponent(tok, num_freqs=3)
        assert comp.dim == 6

    @torch.no_grad()
    def test_column_resets_at_newline(self):
        comp = ColumnPositionComponent(tok, num_freqs=3)
        # "ab\ncd" → BOS(0), a(1), b(2), \n(3), c(4), d(5)
        # Columns: [0, 1, 2, 0, 1, 2]
        # So positions 1 & 4 (both col=1) and 2 & 5 (both col=2) should match.
        ids = _encode("ab\ncd")
        out = comp(ids, dtype=torch.float32)
        torch.testing.assert_close(out[0, 1], out[0, 4])  # col 1
        torch.testing.assert_close(out[0, 2], out[0, 5])  # col 2

    @torch.no_grad()
    def test_monotonic_without_newline(self):
        comp = ColumnPositionComponent(tok, num_freqs=3)
        ids = _encode("abcde")
        out = comp(ids, dtype=torch.float32)
        # All positions after BOS should have distinct encodings
        for i in range(1, ids.shape[1]):
            for j in range(i + 1, ids.shape[1]):
                assert not torch.allclose(out[0, i], out[0, j])

    @torch.no_grad()
    def test_causality(self):
        comp = ColumnPositionComponent(tok, num_freqs=3)
        ids_short = _encode("abc")
        ids_long = _encode("abcde")
        out_short = comp(ids_short, dtype=torch.float32)
        out_long = comp(ids_long, dtype=torch.float32)
        # First len(short) positions must be identical
        torch.testing.assert_close(
            out_short[0, : ids_short.shape[1]],
            out_long[0, : ids_short.shape[1]],
        )

    @torch.no_grad()
    def test_bos_reset(self):
        """Column position resets at document boundary."""
        import numpy as np

        comp = ColumnPositionComponent(tok, num_freqs=3)
        doc1 = tok.encode("abc\ndef")
        doc2 = tok.encode("gh\nij")
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
