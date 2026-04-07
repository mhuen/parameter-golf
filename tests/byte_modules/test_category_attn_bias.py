import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import pytest
from efficient_byte_tokenizer import EfficientByteTokenizer
from byte_modules import CategoryAttnBias, NUM_BYTE_CATEGORIES

tok = EfficientByteTokenizer()


def _encode(text: str) -> torch.Tensor:
    return torch.tensor(tok.encode(text), dtype=torch.long).unsqueeze(0)


def _make_bias():
    return CategoryAttnBias(tok)


# --- Tests ---


@torch.no_grad()
def test_cat_ids_shape():
    bias = _make_bias()
    ids = _encode("hello world")
    cat_ids = bias.get_cat_ids(ids)
    assert cat_ids.shape == ids.shape
    assert cat_ids.min() >= 0
    assert cat_ids.max() <= 7


@torch.no_grad()
def test_digit_maps_to_digit_category():
    bias = _make_bias()
    ids = _encode("0123456789")
    cat_ids = bias.get_cat_ids(ids)
    assert (cat_ids == 2).all(), f"Expected all DIGIT (2), got {cat_ids}"


@torch.no_grad()
def test_letter_maps_to_letter_category():
    bias = _make_bias()
    ids = _encode("abcABC")
    cat_ids = bias.get_cat_ids(ids)
    assert (cat_ids == 3).all(), f"Expected all LETTER (3), got {cat_ids}"


@torch.no_grad()
def test_different_categories_differ():
    bias = _make_bias()
    ids = _encode("a1 ")
    cat_ids = bias.get_cat_ids(ids)
    assert cat_ids[0, 0] != cat_ids[0, 1]  # letter != digit
    assert cat_ids[0, 1] != cat_ids[0, 2]  # digit != separator
    assert cat_ids[0, 0] != cat_ids[0, 2]  # letter != separator


@torch.no_grad()
def test_precompute_bias_shape():
    bias = _make_bias()
    ids = _encode("test")
    out = bias.precompute_bias(ids, dtype=torch.float32)
    B, S = ids.shape
    assert out.shape == (B, 1, S, 8)


@torch.no_grad()
def test_precompute_is_one_hot():
    bias = _make_bias()
    ids = _encode("a1 !")
    out = bias.precompute_bias(ids, dtype=torch.float32)
    sums = out.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums))


@torch.no_grad()
def test_precompute_dtype():
    bias = _make_bias()
    ids = _encode("x")
    out = bias.precompute_bias(ids, dtype=torch.float16)
    assert out.dtype == torch.float16


@torch.no_grad()
def test_bias_forward_shape():
    bias = _make_bias()
    ids = _encode("hello")
    H = 4
    cat_oh = bias.precompute_bias(ids, dtype=torch.float32)
    cat_attn_logits = torch.randn(H, 8, 8)
    out = bias.bias_forward(cat_oh, cat_attn_logits)
    B, S = ids.shape
    assert out.shape == (B, H, S, S)


@torch.no_grad()
def test_bias_equals_logits_lookup():
    bias = _make_bias()
    ids = _encode("a1")
    H = 4
    cat_ids = bias.get_cat_ids(ids)
    cat_a = cat_ids[0, 0].item()
    cat_1 = cat_ids[0, 1].item()

    cat_attn_logits = torch.randn(H, 8, 8)
    cat_oh = bias.precompute_bias(ids, dtype=torch.float32)
    out = bias.bias_forward(cat_oh, cat_attn_logits)

    for h in range(H):
        # bias[b, h, i, j] == cat_attn_logits[h, cat(i), cat(j)]
        assert torch.isclose(out[0, h, 0, 1], cat_attn_logits[h, cat_a, cat_1])
        assert torch.isclose(out[0, h, 0, 0], cat_attn_logits[h, cat_a, cat_a])


@torch.no_grad()
def test_same_category_same_bias():
    bias = _make_bias()
    ids = _encode("ab")
    H = 3
    LETTER = 3
    cat_attn_logits = torch.randn(H, 8, 8)
    cat_oh = bias.precompute_bias(ids, dtype=torch.float32)
    out = bias.bias_forward(cat_oh, cat_attn_logits)

    for h in range(H):
        val_01 = out[0, h, 0, 1]
        val_10 = out[0, h, 1, 0]
        expected = cat_attn_logits[h, LETTER, LETTER]
        assert torch.isclose(val_01, val_10)
        assert torch.isclose(val_01, expected)
