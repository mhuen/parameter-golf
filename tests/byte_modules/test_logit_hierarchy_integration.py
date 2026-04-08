"""Tests for logit_hierarchy integration in MultiStreamMLP and MultiStreamCausalConv."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
import torch
import torch.nn as nn

from efficient_byte_tokenizer import EfficientByteTokenizer
from byte_modules import ByteLogitHierarchy
from multi_streams import (
    StreamType,
    StreamID,
    StreamConfig,
    MultiStreamConfig,
)
from multi_stream_attention import (
    MultiStreamMLP,
    MultiStreamCausalConv,
    MultiStreamCausalConvLayers,
    MultiStreamBlock,
)

tok = EfficientByteTokenizer()
V = tok.vocab_size
LOGIT_SOFTCAP = 30.0

LOGIT_DIM = V
CONTEXT_DIM = 16
HIDDEN_DIM = 64
B, S = 2, 10


def _make_hierarchy():
    return ByteLogitHierarchy(vocab_size=V, tok=tok, logit_softcap=LOGIT_SOFTCAP)


def _make_stream_config():
    return MultiStreamConfig(
        streams=[
            StreamConfig(StreamID(StreamType.LOGIT), dim=LOGIT_DIM),
            StreamConfig(StreamID(StreamType.CONTEXT), dim=CONTEXT_DIM),
            StreamConfig(
                StreamID(StreamType.TOKENS), dim=V, read_only=True
            ),
        ]
    )


def _make_streams(cfg: MultiStreamConfig, device="cpu"):
    streams = {}
    for s in cfg.streams:
        streams[s.name] = torch.randn(B, S, s.dim, device=device)
    return streams


# --------------------------------------------------------------------------
# MultiStreamMLP with logit_hierarchy
# --------------------------------------------------------------------------


@torch.no_grad()
def test_mlp_hierarchy_output_shape():
    """MLP with hierarchy still produces vocab_size-dim logit stream output."""
    hier = _make_hierarchy()
    cfg = _make_stream_config()
    mlp = MultiStreamMLP(cfg, hidden_dim=HIDDEN_DIM, logit_hierarchy=hier)
    streams = _make_streams(cfg)
    out = mlp(streams)
    # Logit stream should be vocab_size (not total_slots)
    logit_out = out[StreamID(StreamType.LOGIT)]
    assert logit_out.shape == (B, S, V)
    # Context stream should be unchanged dim
    context_out = out[StreamID(StreamType.CONTEXT)]
    assert context_out.shape == (B, S, CONTEXT_DIM)
    # Read-only stream should not be in output
    assert StreamID(StreamType.TOKENS) not in out


@torch.no_grad()
def test_mlp_hierarchy_vs_no_hierarchy_shapes():
    """With and without hierarchy, output shapes are identical."""
    hier = _make_hierarchy()
    cfg = _make_stream_config()
    mlp_hier = MultiStreamMLP(cfg, hidden_dim=HIDDEN_DIM, logit_hierarchy=hier)
    mlp_flat = MultiStreamMLP(cfg, hidden_dim=HIDDEN_DIM)
    streams = _make_streams(cfg)
    out_hier = mlp_hier(streams)
    out_flat = mlp_flat(streams)
    for sid in out_hier:
        assert out_hier[sid].shape == out_flat[sid].shape


def test_mlp_hierarchy_gradient_flow():
    """Gradients flow through the hierarchy assembly back to MLP parameters."""
    hier = _make_hierarchy()
    cfg = _make_stream_config()
    mlp = MultiStreamMLP(cfg, hidden_dim=HIDDEN_DIM, logit_hierarchy=hier)
    streams = _make_streams(cfg)
    out = mlp(streams)
    loss = out[StreamID(StreamType.LOGIT)].sum()
    loss.backward()
    # Check that proj_value for the logit stream has gradients
    logit_key = StreamConfig(StreamID(StreamType.LOGIT), dim=LOGIT_DIM).key
    for name, param in mlp.proj_value[logit_key].named_parameters():
        assert param.grad is not None, f"No gradient for proj_value.{name}"
        assert param.grad.abs().sum() > 0, f"Zero gradient for proj_value.{name}"


@torch.no_grad()
def test_mlp_hierarchy_proj_dim():
    """The logit stream proj_value should project to total_slots, not vocab_size."""
    hier = _make_hierarchy()
    cfg = _make_stream_config()
    mlp = MultiStreamMLP(cfg, hidden_dim=HIDDEN_DIM, logit_hierarchy=hier)
    logit_key = StreamConfig(StreamID(StreamType.LOGIT), dim=LOGIT_DIM).key
    # proj_value for logit stream should output total_slots
    weight = mlp.proj_value[logit_key].weight
    assert weight.shape[0] == hier.total_slots


@torch.no_grad()
def test_mlp_hierarchy_zero_init_produces_zero():
    """MLP with zero-initialized weights and hierarchy should produce zero logit deltas."""
    hier = _make_hierarchy()
    cfg = _make_stream_config()
    mlp = MultiStreamMLP(
        cfg, hidden_dim=HIDDEN_DIM, logit_hierarchy=hier, gated_output=False
    )
    # Zero-init all parameters
    for p in mlp.parameters():
        nn.init.zeros_(p)
    streams = _make_streams(cfg)
    out = mlp(streams)
    logit_out = out[StreamID(StreamType.LOGIT)]
    assert torch.allclose(logit_out, torch.zeros_like(logit_out))


# --------------------------------------------------------------------------
# MultiStreamCausalConv with logit_hierarchy
# --------------------------------------------------------------------------


@torch.no_grad()
def test_conv_hierarchy_output_shape():
    hier = _make_hierarchy()
    cfg = _make_stream_config()
    conv = MultiStreamCausalConv(cfg, logit_hierarchy=hier)
    streams = _make_streams(cfg)
    out = conv(streams)
    logit_out = out[StreamID(StreamType.LOGIT)]
    assert logit_out.shape == (B, S, V)


@torch.no_grad()
def test_conv_hierarchy_proj_dim():
    hier = _make_hierarchy()
    cfg = _make_stream_config()
    conv = MultiStreamCausalConv(cfg, logit_hierarchy=hier)
    logit_key = StreamConfig(StreamID(StreamType.LOGIT), dim=LOGIT_DIM).key
    weight = conv.proj_value[logit_key].weight
    assert weight.shape[0] == hier.total_slots


def test_conv_hierarchy_gradient_flow():
    hier = _make_hierarchy()
    cfg = _make_stream_config()
    conv = MultiStreamCausalConv(cfg, logit_hierarchy=hier)
    streams = _make_streams(cfg)
    out = conv(streams)
    loss = out[StreamID(StreamType.LOGIT)].sum()
    loss.backward()
    logit_key = StreamConfig(StreamID(StreamType.LOGIT), dim=LOGIT_DIM).key
    for name, param in conv.proj_value[logit_key].named_parameters():
        assert param.grad is not None, f"No gradient for proj_value.{name}"


# --------------------------------------------------------------------------
# MultiStreamCausalConvLayers with logit_hierarchy
# --------------------------------------------------------------------------


@torch.no_grad()
def test_conv_layers_hierarchy_output_shape():
    hier = _make_hierarchy()
    cfg = _make_stream_config()
    conv_layers = MultiStreamCausalConvLayers(
        cfg, num_layers=2, logit_hierarchy=hier
    )
    streams = _make_streams(cfg)
    out = conv_layers(streams)
    logit_out = out[StreamID(StreamType.LOGIT)]
    assert logit_out.shape == (B, S, V)


# --------------------------------------------------------------------------
# MultiStreamBlock with logit_hierarchy
# --------------------------------------------------------------------------


@torch.no_grad()
def test_block_hierarchy_output_shape():
    hier = _make_hierarchy()
    cfg = _make_stream_config()
    block = MultiStreamBlock(
        multi_head_dim=32,
        num_heads=1,
        num_kv_heads=1,
        stream_config=cfg,
        logit_hierarchy=hier,
    )
    streams = _make_streams(cfg)
    out = block(streams)
    logit_out = out[StreamID(StreamType.LOGIT)]
    assert logit_out.shape == (B, S, V)


def test_block_hierarchy_training_step():
    """A full forward+backward through 2 stacked blocks with hierarchy.

    Uses 2 blocks so the context stream from block 0 feeds into block 1's
    attention, giving all block-0 parameters a gradient path to the loss.
    Block 1's context-stream output projections are structurally disconnected
    from loss (nothing reads context after the last block), so we only
    verify block 0 fully and spot-check block 1's logit-path.
    """
    hier = _make_hierarchy()
    cfg = _make_stream_config()
    block_kwargs = dict(
        multi_head_dim=32,
        num_heads=1,
        num_kv_heads=1,
        stream_config=cfg,
        logit_hierarchy=hier,
    )
    blocks = nn.ModuleList([MultiStreamBlock(**block_kwargs) for _ in range(2)])
    streams = _make_streams(cfg)
    for block in blocks:
        streams = block(streams)
    logit_out = streams[StreamID(StreamType.LOGIT)]
    targets = torch.randint(0, V, (B, S))
    loss = torch.nn.functional.cross_entropy(
        logit_out.reshape(-1, V), targets.reshape(-1)
    )
    loss.backward()
    assert loss.isfinite()
    # Block 0: all parameters should receive gradients (outputs feed block 1)
    for name, param in blocks[0].named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for blocks.0.{name}"
    # Block 1: logit-stream hierarchy path should receive gradients
    logit_key = StreamConfig(StreamID(StreamType.LOGIT), dim=LOGIT_DIM).key
    for name, param in blocks[1].mlp.proj_value[logit_key].named_parameters():
        assert param.grad is not None, f"No gradient for blocks.1.mlp.proj_value.{name}"
        assert param.grad.abs().sum() > 0
