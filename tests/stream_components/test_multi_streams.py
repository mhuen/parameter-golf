"""Tests for multi_streams: SinCosPositionComponent, DocBoundaryComponent,
CompositeStream, MultiStreamBuilder."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
import torch
from torch import Tensor

from multi_streams import (
    StreamType,
    StreamID,
    StreamDef,
    StreamConfig,
    MultiStreamConfig,
    SinCosPositionComponent,
    DocBoundaryComponent,
    CompositeStream,
    MultiStreamBuilder,
)
from modules import sincos_encode


# ── SinCosPositionComponent ──────────────────────────────────────────────


class TestSinCosPositionComponent:
    @torch.no_grad()
    def test_shape(self):
        comp = SinCosPositionComponent(num_freqs=8)
        ids = torch.zeros(2, 10, dtype=torch.long)
        out = comp(ids, dtype=torch.float32)
        assert out.shape == (2, 10, 16)

    @torch.no_grad()
    def test_dtype(self):
        comp = SinCosPositionComponent()
        ids = torch.zeros(1, 5, dtype=torch.long)
        out = comp(ids, dtype=torch.float16)
        assert out.dtype == torch.float16

    @torch.no_grad()
    def test_content_independent(self):
        comp = SinCosPositionComponent()
        ids_a = torch.randint(0, 256, (1, 8))
        ids_b = torch.randint(0, 256, (1, 8))
        out_a = comp(ids_a, dtype=torch.float32)
        out_b = comp(ids_b, dtype=torch.float32)
        assert torch.equal(out_a, out_b)

    @torch.no_grad()
    def test_values_match_sincos_encode(self):
        num_freqs, base, S = 8, 10000.0, 12
        comp = SinCosPositionComponent(num_freqs=num_freqs, base=base)
        ids = torch.zeros(3, S, dtype=torch.long)
        out = comp(ids, dtype=torch.float32)
        expected = sincos_encode(torch.arange(S), num_freqs, base)  # (S, dim)
        expected = expected.unsqueeze(0).expand(3, -1, -1)
        assert torch.allclose(out, expected, atol=1e-6)

    @torch.no_grad()
    def test_causality(self):
        comp = SinCosPositionComponent()
        t = 6
        ids_short = torch.zeros(1, t, dtype=torch.long)
        ids_long = torch.zeros(1, t + 5, dtype=torch.long)
        out_short = comp(ids_short, dtype=torch.float32)
        out_long = comp(ids_long, dtype=torch.float32)
        assert torch.equal(out_short, out_long[:, :t, :])


# ── DocBoundaryComponent ─────────────────────────────────────────────────


class TestDocBoundaryComponent:
    @torch.no_grad()
    def test_shape(self):
        comp = DocBoundaryComponent(bos_id=1, num_freqs=1)
        ids = torch.tensor([[5, 5, 5, 5]], dtype=torch.long)
        out = comp(ids, dtype=torch.float32)
        assert out.shape == (1, 4, 2)

    @torch.no_grad()
    def test_bos_increments_doc_id(self):
        comp = DocBoundaryComponent(bos_id=1, num_freqs=1)
        # bos at positions 0 and 3 → doc_ids = [1,1,1, 2,2,2]
        ids = torch.tensor([[1, 5, 5, 1, 5, 5]], dtype=torch.long)
        out = comp(ids, dtype=torch.float32)
        # Encoding for doc 1 (positions 0-2) should differ from doc 2 (positions 3-5)
        enc_doc1 = out[0, 0, :]  # first token of doc 1
        enc_doc2 = out[0, 3, :]  # first token of doc 2
        assert not torch.equal(enc_doc1, enc_doc2)
        # Within same doc, encoding is constant
        assert torch.equal(out[0, 0, :], out[0, 1, :])
        assert torch.equal(out[0, 0, :], out[0, 2, :])
        assert torch.equal(out[0, 3, :], out[0, 4, :])
        assert torch.equal(out[0, 3, :], out[0, 5, :])

    @torch.no_grad()
    def test_no_bos_constant(self):
        comp = DocBoundaryComponent(bos_id=1, num_freqs=1)
        ids = torch.tensor([[5, 5, 5, 5, 5]], dtype=torch.long)
        out = comp(ids, dtype=torch.float32)
        # All doc_ids are 0 → encoding is constant across positions
        for i in range(1, 5):
            assert torch.equal(out[0, 0, :], out[0, i, :])

    @torch.no_grad()
    def test_causality(self):
        comp = DocBoundaryComponent(bos_id=1, num_freqs=1)
        ids_short = torch.tensor([[1, 5, 5, 1]], dtype=torch.long)
        ids_long = torch.tensor([[1, 5, 5, 1, 5, 5, 1]], dtype=torch.long)
        out_short = comp(ids_short, dtype=torch.float32)
        out_long = comp(ids_long, dtype=torch.float32)
        assert torch.equal(out_short, out_long[:, :4, :])


# ── CompositeStream ──────────────────────────────────────────────────────


class TestCompositeStream:
    @torch.no_grad()
    def test_dim_sum(self):
        c1 = SinCosPositionComponent(num_freqs=4)  # dim=8
        c2 = SinCosPositionComponent(num_freqs=6)  # dim=12
        cs = CompositeStream([c1, c2])
        assert cs.dim == 20

    @torch.no_grad()
    def test_output_is_cat(self):
        c1 = SinCosPositionComponent(num_freqs=4)
        c2 = SinCosPositionComponent(num_freqs=6)
        cs = CompositeStream([c1, c2])
        ids = torch.zeros(2, 7, dtype=torch.long)
        dtype = torch.float32
        out = cs(ids, dtype)
        expected = torch.cat([c1(ids, dtype), c2(ids, dtype)], dim=-1)
        assert torch.equal(out, expected)

    @torch.no_grad()
    def test_empty_not_allowed(self):
        cs = CompositeStream([])
        assert cs.dim == 0


# ── MultiStreamBuilder ───────────────────────────────────────────────────


class TestMultiStreamBuilder:
    @torch.no_grad()
    def test_config_with_structural(self):
        builder = MultiStreamBuilder(
            stream_defs=[
                StreamDef(name=StreamID(StreamType.LOGIT), dim=256),
                StreamDef(
                    name=StreamID(StreamType.STRUCTURAL),
                    read_only=True,
                    components=[
                        SinCosPositionComponent(num_freqs=8),
                        DocBoundaryComponent(bos_id=1, num_freqs=1),
                    ],
                ),
            ],
        )
        cfg = builder.config
        assert len(cfg.streams) == 2
        assert cfg.streams[0].name == StreamID(StreamType.LOGIT)
        assert cfg.streams[0].read_only is False
        assert cfg.streams[1].name == StreamID(StreamType.STRUCTURAL)
        assert cfg.streams[1].read_only is True
        assert cfg.streams[1].dim == 18  # 8*2 + 1*2

    @torch.no_grad()
    def test_config_without_structural(self):
        builder = MultiStreamBuilder(
            stream_defs=[StreamDef(name=StreamID(StreamType.LOGIT), dim=256)],
        )
        cfg = builder.config
        assert len(cfg.streams) == 1
        assert cfg.streams[0].name == StreamID(StreamType.LOGIT)

    @torch.no_grad()
    def test_forward_returns_streams(self):
        builder = MultiStreamBuilder(
            stream_defs=[
                StreamDef(name=StreamID(StreamType.LOGIT), dim=32),
                StreamDef(
                    name=StreamID(StreamType.STRUCTURAL),
                    read_only=True,
                    components=[
                        SinCosPositionComponent(num_freqs=4),
                    ],
                ),
            ],
        )
        B, S = 2, 10
        ids = torch.zeros(B, S, dtype=torch.long)
        writable = torch.randn(B, S, 32)
        streams, _, _ = builder(ids, dtype=torch.float32, logit=writable)
        assert set(streams.keys()) == {
            StreamID(StreamType.LOGIT),
            StreamID(StreamType.STRUCTURAL),
        }
        assert streams[StreamID(StreamType.LOGIT)].shape == (B, S, 32)
        assert streams[StreamID(StreamType.STRUCTURAL)].shape == (B, S, 8)

    @torch.no_grad()
    def test_writable_passthrough(self):
        builder = MultiStreamBuilder(
            stream_defs=[
                StreamDef(name=StreamID(StreamType.LOGIT), dim=16),
                StreamDef(
                    name=StreamID(StreamType.STRUCTURAL),
                    read_only=True,
                    components=[
                        SinCosPositionComponent(),
                    ],
                ),
            ],
        )
        ids = torch.zeros(1, 5, dtype=torch.long)
        writable = torch.randn(1, 5, 16)
        streams, _, _ = builder(ids, logit=writable)
        assert torch.equal(streams[StreamID(StreamType.LOGIT)], writable)
