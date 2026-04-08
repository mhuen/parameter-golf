import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
import torch

from efficient_byte_tokenizer import EfficientByteTokenizer
from byte_modules import StructuralBoundaryBias

tok = EfficientByteTokenizer()


def _encode(text: str) -> torch.Tensor:
    return torch.tensor(tok.encode(text), dtype=torch.long).unsqueeze(0)


@pytest.fixture
def mod():
    return StructuralBoundaryBias(tok)


# --- Boundary ID tests ---


@torch.no_grad()
def test_word_id_increments_at_separator(mod):
    ids = _encode("ab cd ef")
    word_id, _, _ = mod.get_boundary_ids(ids)
    expected = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 2]])
    assert torch.equal(word_id, expected)


@torch.no_grad()
def test_sentence_id_increments_at_period(mod):
    ids = _encode("a.b")
    _, sentence_id, _ = mod.get_boundary_ids(ids)
    expected = torch.tensor([[0, 1, 1]])
    assert torch.equal(sentence_id, expected)


@torch.no_grad()
def test_sentence_id_at_exclamation_question(mod):
    ids = _encode("a!b?c")
    _, sentence_id, _ = mod.get_boundary_ids(ids)
    # '!' at pos 1 increments, '?' at pos 3 increments
    expected = torch.tensor([[0, 1, 1, 2, 2]])
    assert torch.equal(sentence_id, expected)


@torch.no_grad()
def test_paragraph_id_increments_at_newline(mod):
    ids = _encode("a\nb")
    _, _, paragraph_id = mod.get_boundary_ids(ids)
    assert paragraph_id is not None
    expected = torch.tensor([[0, 1, 1]])
    assert torch.equal(paragraph_id, expected)


# --- Precompute bias tests ---


@torch.no_grad()
def test_precompute_shape(mod):
    ids = _encode("hello world")
    features = mod.precompute_bias(ids)
    B, F, S, S2 = features.shape
    assert B == 1
    assert F == mod.num_features
    assert S == ids.shape[1]
    assert S2 == ids.shape[1]


@torch.no_grad()
def test_same_word_feature_is_binary(mod):
    ids = _encode("ab cd ef")
    features = mod.precompute_bias(ids)
    same_word = features[:, 0]
    assert torch.all((same_word == 0.0) | (same_word == 1.0))


@torch.no_grad()
def test_same_word_is_symmetric(mod):
    ids = _encode("ab cd ef")
    features = mod.precompute_bias(ids)
    for f in range(features.shape[1]):
        feat = features[0, f]
        assert torch.equal(feat, feat.T)


@torch.no_grad()
def test_same_word_correctness(mod):
    ids = _encode("ab cd")
    features = mod.precompute_bias(ids)
    same_word = features[0, 0]
    # a(0) and b(1) share word_id=0
    assert same_word[0, 1].item() == 1.0
    assert same_word[1, 0].item() == 1.0
    # a(0) and c(3) have different word_ids (0 vs 1)
    assert same_word[0, 3].item() == 0.0


@torch.no_grad()
def test_hierarchy_same_word_implies_same_sentence(mod):
    ids = _encode("ab cd")
    features = mod.precompute_bias(ids)
    same_word = features[0, 0]
    same_sentence = features[0, 1]
    # Where same_word=1, same_sentence must be 1
    assert torch.all(same_sentence[same_word == 1.0] == 1.0)
    # same_sentence can be 1 where same_word is 0 (cross-word, same sentence)
    cross_word_same_sent = (same_sentence == 1.0) & (same_word == 0.0)
    assert cross_word_same_sent.any()


# --- bias_forward tests ---


@torch.no_grad()
def test_bias_forward_shape(mod):
    ids = _encode("ab cd")
    features = mod.precompute_bias(ids)
    H = 4
    weights = torch.randn(H, mod.num_features)
    out = mod.bias_forward(features, weights)
    assert out.shape == (1, H, ids.shape[1], ids.shape[1])


@torch.no_grad()
def test_bias_forward_einsum(mod):
    ids = _encode("ab cd")
    features = mod.precompute_bias(ids)
    H = 4
    weights = torch.randn(H, mod.num_features)
    out = mod.bias_forward(features, weights)
    expected = torch.einsum("bfqk,hf->bhqk", features, weights)
    assert torch.allclose(out, expected)


# --- struct_cat_ids tests ---


@torch.no_grad()
def test_struct_cat_ids_shape(mod):
    ids = _encode("ab cd ef")
    cat_ids = mod.get_struct_cat_ids(ids)
    assert cat_ids.shape == ids.shape
    assert cat_ids.dtype == torch.long


@torch.no_grad()
def test_struct_cat_ids_range(mod):
    ids = _encode("ab cd ef")
    cat_ids = mod.get_struct_cat_ids(ids)
    assert cat_ids.min().item() >= 0
    assert cat_ids.max().item() < mod.num_struct_cats
