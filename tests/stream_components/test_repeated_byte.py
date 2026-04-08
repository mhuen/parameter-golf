"""Tests for RepeatedByteComponent from byte_modules."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import math

import pytest
import torch

from efficient_byte_tokenizer import EfficientByteTokenizer
from byte_stream_components import RepeatedByteComponent

tok = EfficientByteTokenizer()


def _encode(text: str) -> torch.Tensor:
    return torch.tensor(tok.encode(text), dtype=torch.long).unsqueeze(0)


@torch.no_grad()
def test_shape():
    comp = RepeatedByteComponent()
    ids = _encode("hello")
    out = comp(ids, dtype=torch.float32)
    assert out.shape == (1, ids.shape[1], 2)


@torch.no_grad()
def test_dim():
    comp = RepeatedByteComponent()
    assert comp.dim == 2


@torch.no_grad()
def test_repeated():
    comp = RepeatedByteComponent()
    ids = _encode("aab")
    out = comp(ids, dtype=torch.float32)
    # Positions 0,1,2 are a, a, b
    # is_repeat: first 'a' starts run -> -1, second 'a' matches -> +1, 'b' differs -> -1
    assert out[0, 0, 0].item() == pytest.approx(-1.0)
    assert out[0, 1, 0].item() == pytest.approx(1.0)
    assert out[0, 2, 0].item() == pytest.approx(-1.0)


@torch.no_grad()
def test_all_same():
    comp = RepeatedByteComponent()
    ids = _encode("aaa")
    out = comp(ids, dtype=torch.float32)
    # Positions 0,1,2 are a, a, a
    assert out[0, 0, 0].item() == pytest.approx(-1.0)  # first position always -1
    assert out[0, 1, 0].item() == pytest.approx(1.0)  # matches previous
    assert out[0, 2, 0].item() == pytest.approx(1.0)  # matches previous


@torch.no_grad()
def test_run_length_values():
    comp = RepeatedByteComponent()
    ids = _encode("aaa")
    out = comp(ids, dtype=torch.float32)
    # Run length on token IDs for "aaa": all same token
    # Position 0: 0 (start), position 1: log1p(1), position 2: log1p(2)
    assert out[0, 0, 1].item() == pytest.approx(0.0)
    assert out[0, 1, 1].item() == pytest.approx(math.log1p(1))
    assert out[0, 2, 1].item() == pytest.approx(math.log1p(2))
