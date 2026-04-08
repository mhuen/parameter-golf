"""Tests for BoundaryComponent from byte_modules."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
import torch

from efficient_byte_tokenizer import EfficientByteTokenizer
from byte_stream_components import BoundaryComponent

tok = EfficientByteTokenizer()


def _encode(text: str) -> torch.Tensor:
    return torch.tensor(tok.encode(text), dtype=torch.long).unsqueeze(0)


def _make(**kwargs):
    defaults = dict(
        word_pos_freqs=3,
        word_id_freqs=2,
        sent_pos_freqs=3,
        sent_id_freqs=2,
        para_pos_freqs=2,
        para_id_freqs=1,
        base=10000.0,
    )
    defaults.update(kwargs)
    return BoundaryComponent(tok, **defaults)


class TestBoundaryComponent:
    @torch.no_grad()
    def test_shape(self):
        comp = _make()
        ids = _encode("hello world")
        out = comp(ids, dtype=torch.float32)
        assert out.shape == (1, ids.shape[1], comp.dim)

    @torch.no_grad()
    def test_dim(self):
        comp = _make()
        # dim = 2*(3+2+3+2+para_pos+para_id)
        # para terms depend on whether newline exists in tokenizer vocab
        expected_base = 2 * (3 + 2 + 3 + 2)
        if comp.para_pos_freqs > 0:
            expected_base += 2 * (2 + 1)
        assert comp.dim == expected_base

    @torch.no_grad()
    def test_word_boundary_at_space(self):
        comp = _make()
        ids = _encode("ab cd")
        out = comp(ids, dtype=torch.float32)
        # a(0) b(1) ' '(2) c(3) d(4)
        # word_id is encoded first (id before pos in _encode_boundary call order).
        # word_id_freqs=2 → dims 0-3 are word_id sin/cos.
        # word_id increments at separators (space at pos 2).
        # Features at pos 1 (in "ab", before sep) vs pos 3 (in "cd", after sep)
        # should differ in the word_id dimensions.
        assert not torch.allclose(out[0, 1, :4], out[0, 3, :4], atol=1e-5)

    @torch.no_grad()
    def test_sentence_boundary(self):
        comp = _make()
        ids = _encode("a.b")
        out = comp(ids, dtype=torch.float32)
        # a(0) .(1) b(2)
        # Sentence features should differ before and after '.'
        # sent_id dims come after word dims: word_id(4) + word_pos(6) = offset 10
        word_dims = 2 * (comp.word_id_freqs + comp.word_pos_freqs)
        sent_id_start = word_dims
        sent_id_end = sent_id_start + 2 * comp.sent_id_freqs
        # 'a' at pos 0 vs 'b' at pos 2: sentence_id should have incremented
        assert not torch.allclose(
            out[0, 0, sent_id_start:sent_id_end],
            out[0, 2, sent_id_start:sent_id_end],
            atol=1e-5,
        )

    @torch.no_grad()
    def test_position_within_word_resets(self):
        comp = _make()
        ids = _encode("ab cd")
        out = comp(ids, dtype=torch.float32)
        # a(0) b(1) ' '(2) c(3) d(4)
        # word_pos dims come after word_id dims
        word_id_dims = 2 * comp.word_id_freqs
        word_pos_start = word_id_dims
        word_pos_end = word_pos_start + 2 * comp.word_pos_freqs
        # 'b' (pos 1) is at position 1 from start of first word (distance from
        # pos 0), 'c' (pos 3) is at position 1 from start of second word
        # (distance from separator at pos 2). Both should have matching features.
        torch.testing.assert_close(
            out[0, 1, word_pos_start:word_pos_end],
            out[0, 3, word_pos_start:word_pos_end],
        )

    @torch.no_grad()
    def test_causality(self):
        comp = _make()
        ids_short = _encode("hello")
        ids_long = _encode("hello world")
        out_short = comp(ids_short, dtype=torch.float32)
        out_long = comp(ids_long, dtype=torch.float32)
        S = ids_short.shape[1]
        torch.testing.assert_close(out_short[0, :S], out_long[0, :S])

    @torch.no_grad()
    def test_dtype(self):
        comp = _make()
        ids = _encode("test")
        out_f32 = comp(ids, dtype=torch.float32)
        out_f16 = comp(ids, dtype=torch.float16)
        assert out_f32.dtype == torch.float32
        assert out_f16.dtype == torch.float16
