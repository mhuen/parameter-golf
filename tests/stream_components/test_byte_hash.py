"""Tests for ByteHashComponent from byte_modules."""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
import torch

from efficient_byte_tokenizer import EfficientByteTokenizer
from byte_modules import ByteHashComponent, HashBoundary

tok = EfficientByteTokenizer()


def _encode(text: str) -> torch.Tensor:
    return torch.tensor(tok.encode(text), dtype=torch.long).unsqueeze(0)


def _make(boundary=HashBoundary.WORD, track_hits=True, num_hashes=2):
    return ByteHashComponent(
        tok,
        window=12,
        num_hashes=num_hashes,
        boundary=boundary,
        track_hits=track_hits,
        hit_bucket_size=251,
    )


class TestByteHashComponent:

    @torch.no_grad()
    def test_shape(self):
        comp = _make()
        ids = _encode("hello world")
        out = comp(ids, dtype=torch.float32)
        assert out.shape == (1, ids.shape[1], 6)

    @torch.no_grad()
    def test_dim(self):
        comp = _make()
        assert comp.dim == 6

    @torch.no_grad()
    def test_dim_no_hits(self):
        comp = _make(track_hits=False)
        assert comp.dim == 4

    @torch.no_grad()
    def test_identical_words_same_hash(self):
        comp = _make(boundary=HashBoundary.WORD)
        ids = _encode("cat cat")
        out = comp(ids, dtype=torch.float32)
        # "cat cat": c(0) a(1) t(2) ' '(3) c(4) a(5) t(6)
        # First "cat" ends at position 2 (t), second "cat" ends at position 6 (t)
        # Hash features are dims 0-3
        torch.testing.assert_close(out[0, 2, :4], out[0, 6, :4])

    @torch.no_grad()
    def test_different_words_different_hash(self):
        comp = _make(boundary=HashBoundary.WORD)
        ids = _encode("cat dog")
        out = comp(ids, dtype=torch.float32)
        # c(0) a(1) t(2) ' '(3) d(4) o(5) g(6)
        # First word ends at pos 2 (t), second word ends at pos 6 (g)
        assert not torch.allclose(out[0, 2, :4], out[0, 6, :4])

    @torch.no_grad()
    def test_boundary_word(self):
        comp = _make(boundary=HashBoundary.WORD)
        ids = _encode("ab cd")
        out = comp(ids, dtype=torch.float32)
        # a(0) b(1) ' '(2) c(3) d(4)
        # At the space (separator, pos 2), hash should be inactive (lookback=-1).
        # With inactive lookback, gathered bytes are all zero → hash angle = 0.
        space_pos = 2
        # Hash angle = 0 → sin=0, cos=1 for each hash
        for h in range(2):
            assert out[0, space_pos, 2 * h + 0].item() == pytest.approx(0.0, abs=1e-5)
            assert out[0, space_pos, 2 * h + 1].item() == pytest.approx(1.0, abs=1e-5)

    @torch.no_grad()
    def test_boundary_digit(self):
        comp = _make(boundary=HashBoundary.DIGIT)
        ids = _encode("a12b")
        out = comp(ids, dtype=torch.float32)
        # a(0) 1(1) 2(2) b(3)
        # Non-digit positions (0, 3) should have inactive hash (lookback -1)
        # → hash features at sin=0, cos=1 (angle=0)
        for pos in [0, 3]:
            for h in range(2):
                assert out[0, pos, 2 * h + 0].item() == pytest.approx(0.0, abs=1e-5)
                assert out[0, pos, 2 * h + 1].item() == pytest.approx(1.0, abs=1e-5)
        # Digit positions (1, 2) should have non-trivial features
        for pos in [1, 2]:
            hash_feats = out[0, pos, :4]
            inactive = torch.tensor([0.0, 1.0, 0.0, 1.0])
            assert not torch.allclose(hash_feats, inactive, atol=1e-5)

    @torch.no_grad()
    def test_boundary_none(self):
        comp = _make(boundary=None)
        ids = _encode("hello")
        out = comp(ids, dtype=torch.float32)
        # With no boundary, every position has a hash (no inactive positions)
        assert out.shape == (1, ids.shape[1], 6)
        # Non-BOS positions should have non-trivial features
        for pos in range(1, ids.shape[1]):
            hash_feats = out[0, pos, :4]
            inactive = torch.tensor([0.0, 1.0, 0.0, 1.0])
            assert not torch.allclose(hash_feats, inactive, atol=1e-5)

    @torch.no_grad()
    def test_boundary_codepoint(self):
        comp = _make(boundary=HashBoundary.CODEPOINT)
        ids = _encode("abc")
        out = comp(ids, dtype=torch.float32)
        # Pure ASCII: all positions are non-multibyte → inactive hash
        for pos in range(ids.shape[1]):
            for h in range(2):
                assert out[0, pos, 2 * h + 0].item() == pytest.approx(0.0, abs=1e-5)
                assert out[0, pos, 2 * h + 1].item() == pytest.approx(1.0, abs=1e-5)

    @torch.no_grad()
    def test_causality(self):
        comp = _make()
        ids_short = _encode("hello")
        ids_long = _encode("hello world")
        out_short = comp(ids_short, dtype=torch.float32)
        out_long = comp(ids_long, dtype=torch.float32)
        # Output at prefix positions should be identical
        S = ids_short.shape[1]
        torch.testing.assert_close(out_short[0, :S], out_long[0, :S])

    @torch.no_grad()
    def test_batch_consistency(self):
        comp = _make()
        ids = _encode("test")
        ids_batch = ids.expand(2, -1)
        out = comp(ids_batch, dtype=torch.float32)
        torch.testing.assert_close(out[0], out[1])
