"""Tests for ByteCategoryComponent from byte_modules."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
import torch

from efficient_byte_tokenizer import EfficientByteTokenizer
from byte_stream_components import ByteCategoryComponent

tok = EfficientByteTokenizer()


def _encode(text: str) -> torch.Tensor:
    import numpy as np

    ids = np.concatenate([[tok.bos_id], tok.encode(text)])
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


class TestByteCategoryComponent:
    @torch.no_grad()
    def test_shape(self):
        comp = ByteCategoryComponent(tok, embed_dim=4)
        ids = _encode("hello")
        out = comp(ids, dtype=torch.float32)
        assert out.shape == (1, ids.shape[1], 4)

    @torch.no_grad()
    def test_dim(self):
        comp = ByteCategoryComponent(tok, embed_dim=4)
        assert comp.dim == 4

    @torch.no_grad()
    def test_custom_embed_dim(self):
        comp = ByteCategoryComponent(tok, embed_dim=8)
        assert comp.dim == 8
        ids = _encode("test")
        out = comp(ids, dtype=torch.float32)
        assert out.shape == (1, ids.shape[1], 8)

    @torch.no_grad()
    def test_same_category_same_embedding(self):
        comp = ByteCategoryComponent(tok, embed_dim=4)
        ids = _encode("12")  # BOS, '1', '2' — both digits
        out = comp(ids, dtype=torch.float32)
        # positions 1 and 2 are both digits → same category → same embedding
        torch.testing.assert_close(out[0, 1], out[0, 2])

    @torch.no_grad()
    def test_different_category_different_embedding(self):
        comp = ByteCategoryComponent(tok, embed_dim=4)
        ids = _encode("1a")  # BOS, '1' (digit), 'a' (letter)
        out = comp(ids, dtype=torch.float32)
        # digit vs letter — random init makes equality astronomically unlikely
        assert not torch.allclose(out[0, 1], out[0, 2])

    @torch.no_grad()
    def test_dtype(self):
        comp = ByteCategoryComponent(tok, embed_dim=4)
        ids = _encode("hi")
        out = comp(ids, dtype=torch.float16)
        assert out.dtype == torch.float16
