"""Tests for ByteCategoryComponent from byte_modules."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
import torch

from efficient_byte_tokenizer import EfficientByteTokenizer
from byte_stream_components import ByteCategoryComponent, DiscreteEncoding

tok = EfficientByteTokenizer()


def _encode(text: str) -> torch.Tensor:
    import numpy as np

    ids = np.concatenate([[tok.bos_id], tok.encode(text)])
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


class TestByteCategoryComponent:
    @torch.no_grad()
    def test_shape(self):
        comp = ByteCategoryComponent(tok)
        ids = _encode("hello")
        out = comp(ids, dtype=torch.float32)
        assert out.shape == (1, ids.shape[1], 2)

    @torch.no_grad()
    def test_dim(self):
        comp = ByteCategoryComponent(tok)
        assert comp.dim == 2

    @torch.no_grad()
    def test_unit_vectors(self):
        comp = ByteCategoryComponent(tok)
        ids = _encode("test")
        out = comp(ids, dtype=torch.float32)
        norms = out.norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)

    @torch.no_grad()
    def test_binary_mode(self):
        comp = ByteCategoryComponent(tok, encoding=DiscreteEncoding.BINARY)
        assert comp.dim == 3
        ids = _encode("test")
        out = comp(ids, dtype=torch.float32)
        assert out.shape == (1, ids.shape[1], 3)

    @torch.no_grad()
    def test_same_category_same_embedding(self):
        comp = ByteCategoryComponent(tok)
        ids = _encode("12")  # BOS, '1', '2' — both digits
        out = comp(ids, dtype=torch.float32)
        torch.testing.assert_close(out[0, 1], out[0, 2])

    @torch.no_grad()
    def test_different_category_different_embedding(self):
        comp = ByteCategoryComponent(tok)
        ids = _encode("1a")  # BOS, '1' (digit), 'a' (letter)
        out = comp(ids, dtype=torch.float32)
        assert not torch.allclose(out[0, 1], out[0, 2])

    @torch.no_grad()
    def test_binary_same_category_same_embedding(self):
        comp = ByteCategoryComponent(tok, encoding=DiscreteEncoding.BINARY)
        ids = _encode("12")
        out = comp(ids, dtype=torch.float32)
        torch.testing.assert_close(out[0, 1], out[0, 2])

    @torch.no_grad()
    def test_binary_different_category_different_embedding(self):
        comp = ByteCategoryComponent(tok, encoding=DiscreteEncoding.BINARY)
        ids = _encode("1a")
        out = comp(ids, dtype=torch.float32)
        assert not torch.allclose(out[0, 1], out[0, 2])

    @torch.no_grad()
    def test_dtype(self):
        comp = ByteCategoryComponent(tok)
        ids = _encode("hi")
        out = comp(ids, dtype=torch.float16)
        assert out.dtype == torch.float16
