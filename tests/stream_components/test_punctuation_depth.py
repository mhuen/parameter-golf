"""Tests for PunctuationDepthComponent from byte_modules."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
import torch

from efficient_byte_tokenizer import EfficientByteTokenizer
from byte_stream_components import PunctuationDepthComponent

tok = EfficientByteTokenizer()


def _encode(text: str) -> torch.Tensor:
    return torch.tensor(tok.encode(text), dtype=torch.long).unsqueeze(0)


@torch.no_grad()
def test_shape():
    comp = PunctuationDepthComponent(tok)
    ids = _encode("hello")
    out = comp(ids, dtype=torch.float32)
    assert out.shape == (1, ids.shape[1], 2)


@torch.no_grad()
def test_dim():
    comp = PunctuationDepthComponent(tok)
    assert comp.dim == 2


@torch.no_grad()
def test_bracket_depth():
    comp = PunctuationDepthComponent(tok)
    ids = _encode("(())")
    out = comp(ids, dtype=torch.float32)
    # Positions 0,1,2,3 are '(', '(', ')', ')'
    # Depth: 1, 2, 1, 0
    assert out[0, 0, 0].item() == pytest.approx(1.0)
    assert out[0, 1, 0].item() == pytest.approx(2.0)
    assert out[0, 2, 0].item() == pytest.approx(1.0)
    assert out[0, 3, 0].item() == pytest.approx(0.0)


@torch.no_grad()
def test_quote_toggle():
    comp = PunctuationDepthComponent(tok)
    ids = _encode('a"b"c')
    out = comp(ids, dtype=torch.float32)
    # Positions 0,1,2,3,4 are a, ", b, ", c
    # Quote cumsum: 0, 1, 1, 2, 2
    # Quote state: -1, +1, +1, -1, -1
    assert out[0, 0, 1].item() == pytest.approx(-1.0)  # a: outside
    assert out[0, 1, 1].item() == pytest.approx(1.0)  # ": inside (odd)
    assert out[0, 2, 1].item() == pytest.approx(1.0)  # b: inside (odd)
    assert out[0, 3, 1].item() == pytest.approx(-1.0)  # ": outside (even)
    assert out[0, 4, 1].item() == pytest.approx(-1.0)  # c: outside (even)


@torch.no_grad()
def test_unbalanced_negative():
    comp = PunctuationDepthComponent(tok)
    ids = _encode("))")
    out = comp(ids, dtype=torch.float32)
    # Positions 0,1 are ')', ')'
    # Depth: -1, -2
    assert out[0, 0, 0].item() == pytest.approx(-1.0)
    assert out[0, 1, 0].item() == pytest.approx(-2.0)
