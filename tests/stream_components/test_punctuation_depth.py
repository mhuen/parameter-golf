"""Tests for PunctuationDepthComponent from byte_modules."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import math

import pytest
import torch

from efficient_byte_tokenizer import EfficientByteTokenizer
from byte_stream_components import PunctuationDepthComponent
from modules import ContinuousRotation

tok = EfficientByteTokenizer()


def _encode(text: str) -> torch.Tensor:
    return torch.tensor(tok.encode(text), dtype=torch.long).unsqueeze(0)


@torch.no_grad()
def test_shape():
    comp = PunctuationDepthComponent(tok)
    ids = _encode("hello")
    out = comp(ids, dtype=torch.float32)
    assert out.shape == (1, ids.shape[1], 3)


@torch.no_grad()
def test_dim():
    comp = PunctuationDepthComponent(tok)
    assert comp.dim == 3


@torch.no_grad()
def test_bracket_depth():
    """Depth encoded as 2D rotation via ContinuousRotation(cap)."""
    comp = PunctuationDepthComponent(tok)
    enc = ContinuousRotation(min_val=-3.0, max_val=8.0, mode="cap")
    ids = _encode("(())")
    out = comp(ids, dtype=torch.float32)
    # Positions 0,1,2,3 are '(', '(', ')', ')'
    # Depth: 1, 2, 1, 0
    for pos, depth in [(0, 1), (1, 2), (2, 1), (3, 0)]:
        expected = enc.encode(torch.tensor(depth))
        torch.testing.assert_close(out[0, pos, :2], expected, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_quote_toggle():
    comp = PunctuationDepthComponent(tok)
    ids = _encode('a"b"c')
    out = comp(ids, dtype=torch.float32)
    # Positions 0,1,2,3,4 are a, ", b, ", c
    # Quote cumsum: 0, 1, 1, 2, 2
    # Quote state (dim 2): -1, +1, +1, -1, -1
    assert out[0, 0, 2].item() == pytest.approx(-1.0)  # a: outside
    assert out[0, 1, 2].item() == pytest.approx(1.0)  # ": inside (odd)
    assert out[0, 2, 2].item() == pytest.approx(1.0)  # b: inside (odd)
    assert out[0, 3, 2].item() == pytest.approx(-1.0)  # ": outside (even)
    assert out[0, 4, 2].item() == pytest.approx(-1.0)  # c: outside (even)


@torch.no_grad()
def test_unbalanced_negative():
    """Negative depths are captured by the rotation encoding."""
    comp = PunctuationDepthComponent(tok)
    enc = ContinuousRotation(min_val=-3.0, max_val=8.0, mode="cap")
    ids = _encode("))")
    out = comp(ids, dtype=torch.float32)
    # Positions 0,1 are ')', ')'
    # Depth: -1, -2
    for pos, depth in [(0, -1), (1, -2)]:
        expected = enc.encode(torch.tensor(depth))
        torch.testing.assert_close(out[0, pos, :2], expected, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_depth_distinct_vectors():
    """Different depths produce distinct rotation vectors."""
    enc = ContinuousRotation(min_val=-3.0, max_val=8.0, mode="cap")
    depths = torch.arange(-3, 9)
    vecs = enc.encode(depths)
    # All consecutive pairs should be distinct (cos decreasing over arc=π)
    cos_vals = vecs[:, 0]
    diffs = cos_vals[1:] - cos_vals[:-1]
    assert (diffs < 0).all(), "cos should decrease monotonically over arc=π"


@torch.no_grad()
def test_depth_capped():
    """Depths beyond [min_depth, max_depth] clamp to boundary angles."""
    enc = ContinuousRotation(min_val=-3.0, max_val=8.0, mode="cap")
    vec_low = enc.encode(torch.tensor(-10))
    vec_min = enc.encode(torch.tensor(-3))
    torch.testing.assert_close(vec_low, vec_min)
    vec_high = enc.encode(torch.tensor(20))
    vec_max = enc.encode(torch.tensor(8))
    torch.testing.assert_close(vec_high, vec_max)


@torch.no_grad()
def test_rotation_unit_norm():
    """Rotation encoding should produce unit vectors."""
    enc = ContinuousRotation(min_val=-3.0, max_val=8.0, mode="cap")
    depths = torch.arange(-5, 12)
    vecs = enc.encode(depths)
    norms = vecs.norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_bos_reset():
    """Bracket depth and quote state reset at document boundary."""
    import numpy as np

    comp = PunctuationDepthComponent(tok)
    doc1 = tok.encode('(("hello"')
    doc2 = tok.encode('(a"b")')
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


@torch.no_grad()
def test_standardizable_mask():
    comp = PunctuationDepthComponent(tok)
    mask = comp.standardizable_mask
    # Dims 0-1 are rotation (not standardizable), dim 2 is quote_state (standardizable)
    assert mask == (False, False, True)
