"""Tests for CaseComponent from byte_modules."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import math

import pytest
import torch

from efficient_byte_tokenizer import EfficientByteTokenizer
from byte_stream_components import CaseComponent

tok = EfficientByteTokenizer()


def _encode(text: str) -> torch.Tensor:
    return torch.tensor(tok.encode(text), dtype=torch.long).unsqueeze(0)


@torch.no_grad()
def test_shape():
    comp = CaseComponent(tok)
    ids = _encode("hello")
    out = comp(ids, dtype=torch.float32)
    assert out.shape == (1, ids.shape[1], 2)


@torch.no_grad()
def test_dim():
    comp = CaseComponent(tok)
    assert comp.dim == 2


@torch.no_grad()
def test_case_states():
    comp = CaseComponent(tok)
    ids = _encode("AbC")
    out = comp(ids, dtype=torch.float32)
    # Positions 0,1,2 are A, b, C
    assert out[0, 0, 0].item() == pytest.approx(1.0)  # A -> upper
    assert out[0, 1, 0].item() == pytest.approx(-1.0)  # b -> lower
    assert out[0, 2, 0].item() == pytest.approx(1.0)  # C -> upper


@torch.no_grad()
def test_run_length_resets():
    comp = CaseComponent(tok)
    ids = _encode("AAbb")
    out = comp(ids, dtype=torch.float32)
    # Positions 0,1,2,3 are A, A, b, b
    # Case categories: upper(1), upper(1), lower(2), lower(2)
    # Run lengths: 0, log1p(1), 0, log1p(1)
    assert out[0, 0, 1].item() == pytest.approx(0.0)
    assert out[0, 1, 1].item() == pytest.approx(math.log1p(1))
    assert out[0, 2, 1].item() == pytest.approx(0.0)
    assert out[0, 3, 1].item() == pytest.approx(math.log1p(1))


@torch.no_grad()
def test_non_letter_zero():
    comp = CaseComponent(tok)
    ids = _encode("1")
    out = comp(ids, dtype=torch.float32)
    # Position 0 is digit '1' -> case_state should be 0
    assert out[0, 0, 0].item() == pytest.approx(0.0)
