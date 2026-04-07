"""Tests for DistanceBias from byte_modules."""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import pytest

from byte_modules import DistanceBias


@torch.no_grad()
def test_small_distances_identity():
    db = DistanceBias(num_buckets=32, max_distance=128)
    max_exact = 32 // 2  # 16
    for d in range(max_exact):
        assert db.distance_to_bucket[d].item() == d


@torch.no_grad()
def test_large_distances_log_spaced():
    db = DistanceBias(num_buckets=32, max_distance=128)
    max_exact = 16
    buckets = db.distance_to_bucket[max_exact : 129]
    # All should be in [16, 31]
    assert (buckets >= max_exact).all()
    assert (buckets <= 31).all()
    # Log-spacing means multiple distances map to the same bucket
    unique_buckets = buckets.unique()
    assert len(unique_buckets) < len(buckets), "expected merging in log-spaced region"


@torch.no_grad()
def test_bucket_monotonicity():
    db = DistanceBias(num_buckets=32, max_distance=128)
    b = db.distance_to_bucket
    diffs = b[1:] - b[:-1]
    assert (diffs >= 0).all(), "distance_to_bucket must be monotonically non-decreasing"


@torch.no_grad()
def test_bucket_range():
    db = DistanceBias(num_buckets=32, max_distance=128)
    assert db.distance_to_bucket.min().item() >= 0
    assert db.distance_to_bucket.max().item() <= 31


@torch.no_grad()
def test_max_distance_maps_to_last_bucket():
    db = DistanceBias(num_buckets=32, max_distance=128)
    assert db.distance_to_bucket[128].item() == 31


@torch.no_grad()
def test_precompute_shape():
    db = DistanceBias()
    seq_len = 20
    bucket_ids = db.precompute_bias(seq_len, device="cpu")
    assert bucket_ids.shape == (seq_len, seq_len)
    assert bucket_ids.dtype == torch.long


@torch.no_grad()
def test_causal_only_forward():
    db = DistanceBias()
    seq_len = 16
    bucket_ids = db.precompute_bias(seq_len, device="cpu")
    # For j > i (future positions), rel_dist is clamped to 0, so bucket = 0
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert bucket_ids[i, j].item() == 0


@torch.no_grad()
def test_diagonal_is_zero():
    db = DistanceBias()
    seq_len = 24
    bucket_ids = db.precompute_bias(seq_len, device="cpu")
    for i in range(seq_len):
        assert bucket_ids[i, i].item() == 0


@torch.no_grad()
def test_bias_forward_shape():
    db = DistanceBias()
    seq_len = 10
    H = 4
    bucket_ids = db.precompute_bias(seq_len, device="cpu")
    weights = torch.randn(H, db.num_buckets)
    out = db.bias_forward(bucket_ids, weights)
    assert out.shape == (1, H, seq_len, seq_len)


@torch.no_grad()
def test_bias_forward_values():
    db = DistanceBias(num_buckets=32, max_distance=128)
    seq_len = 8
    H = 2
    bucket_ids = db.precompute_bias(seq_len, device="cpu")
    weights = torch.randn(H, db.num_buckets)
    out = db.bias_forward(bucket_ids, weights)
    # Spot-check several positions
    for h in range(H):
        for i in range(seq_len):
            for j in range(seq_len):
                expected = weights[h, bucket_ids[i, j]].item()
                assert out[0, h, i, j].item() == pytest.approx(expected)


@torch.no_grad()
def test_pos_bin_ids_shape():
    db = DistanceBias()
    seq_len = 30
    ids = db.get_pos_bin_ids(seq_len, device="cpu")
    assert ids.shape == (1, seq_len)
    assert ids.dtype == torch.long


@torch.no_grad()
def test_pos_bin_ids_range():
    db = DistanceBias(num_buckets=32, max_distance=128)
    ids = db.get_pos_bin_ids(200, device="cpu")
    assert ids.min().item() >= 0
    assert ids.max().item() <= 31


@torch.no_grad()
def test_pos_bin_ids_monotonic():
    db = DistanceBias(num_buckets=32, max_distance=128)
    ids = db.get_pos_bin_ids(200, device="cpu").squeeze(0)
    diffs = ids[1:] - ids[:-1]
    assert (diffs >= 0).all(), "pos_bin_ids should be monotonically non-decreasing"
    # Small positions should identity-map
    for p in range(16):
        assert ids[p].item() == p
    # Large positions should saturate at num_buckets-1
    assert ids[-1].item() == 31
