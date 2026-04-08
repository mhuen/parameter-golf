"""Tests for _run_length helper from byte_modules."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import math

import pytest
import torch
from torch import Tensor

from byte_stream_components import _run_length


@torch.no_grad()
def test_all_same():
    x = torch.tensor([[0, 0, 0, 0]])
    out = _run_length(x)
    expected = torch.tensor([[0.0, math.log1p(1), math.log1p(2), math.log1p(3)]])
    assert out.shape == (1, 4)
    torch.testing.assert_close(out, expected)


@torch.no_grad()
def test_alternating():
    x = torch.tensor([[0, 1, 0, 1]])
    out = _run_length(x)
    expected = torch.zeros(1, 4)
    torch.testing.assert_close(out, expected)


@torch.no_grad()
def test_mixed():
    x = torch.tensor([[0, 0, 1, 1, 1, 0]])
    out = _run_length(x)
    expected = torch.tensor(
        [[0.0, math.log1p(1), 0.0, math.log1p(1), math.log1p(2), 0.0]]
    )
    torch.testing.assert_close(out, expected)


@torch.no_grad()
def test_batch():
    x = torch.tensor(
        [
            [0, 0, 0, 1],
            [1, 0, 0, 0],
        ]
    )
    out = _run_length(x)
    expected = torch.tensor(
        [
            [0.0, math.log1p(1), math.log1p(2), 0.0],
            [0.0, 0.0, math.log1p(1), math.log1p(2)],
        ]
    )
    assert out.shape == (2, 4)
    torch.testing.assert_close(out, expected)
