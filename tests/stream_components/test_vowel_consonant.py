"""Tests for VowelConsonantComponent from byte_modules."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import math

import pytest
import torch

from efficient_byte_tokenizer import EfficientByteTokenizer
from byte_stream_components import VowelConsonantComponent

tok = EfficientByteTokenizer()


def _encode(text: str) -> torch.Tensor:
    return torch.tensor(tok.encode(text), dtype=torch.long).unsqueeze(0)


@torch.no_grad()
def test_shape():
    comp = VowelConsonantComponent(tok)
    ids = _encode("hello")
    out = comp(ids, dtype=torch.float32)
    assert out.shape == (1, ids.shape[1], 2)


@torch.no_grad()
def test_dim():
    comp = VowelConsonantComponent(tok)
    assert comp.dim == 2


@torch.no_grad()
def test_vowel_consonant_states():
    comp = VowelConsonantComponent(tok)
    ids = _encode("abc")
    out = comp(ids, dtype=torch.float32)
    # Positions 0,1,2 are a, b, c
    assert out[0, 0, 0].item() == pytest.approx(1.0)  # a -> vowel
    assert out[0, 1, 0].item() == pytest.approx(-1.0)  # b -> consonant
    assert out[0, 2, 0].item() == pytest.approx(-1.0)  # c -> consonant


@torch.no_grad()
def test_run_length():
    comp = VowelConsonantComponent(tok)
    ids = _encode("bba")
    out = comp(ids, dtype=torch.float32)
    # Positions 0,1,2 are b, b, a
    # vc categories: consonant(2), consonant(2), vowel(1)
    # Run lengths: 0, log1p(1), 0
    assert out[0, 0, 1].item() == pytest.approx(0.0)
    assert out[0, 1, 1].item() == pytest.approx(math.log1p(1))
    assert out[0, 2, 1].item() == pytest.approx(0.0)


@torch.no_grad()
def test_non_letter():
    comp = VowelConsonantComponent(tok)
    ids = _encode("1")
    out = comp(ids, dtype=torch.float32)
    # Position 0 is digit '1' -> vc_state should be 0
    assert out[0, 0, 0].item() == pytest.approx(0.0)
