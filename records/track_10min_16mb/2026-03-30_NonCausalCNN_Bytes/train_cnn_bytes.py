"""
Non-Causal CNN Byte-Level Next-Token Predictor.

Takes a fixed-length context (B, s) of byte token IDs, processes through
configurable Conv1d blocks with dimensionality reduction, and predicts
the s+1-th token.  Sequences shorter than s are left-padded with pad_id=0.

Uses EfficientByteTokenizer (vocab ~208, pad_id=0, bos_id=1).

Layer design inspired by TFScripts (github.com/icecube/TFScripts):
  - Configurable pipeline: conv -> std-dev repair -> bias -> activation
    -> normalization -> residual -> pooling -> dropout
  - Residual connections handle channel and spatial dimension mismatches
  - Std-dev repair preserves activation variance through the network

Env-var config (see Hyperparameters class for full list):
    LAYER_CONFIGS='[{"num_filters":128,"kernel_size":7},...]'  (JSON)
    TRAIN_SEQ_LEN=512
    EMBED_DIM=128
    GLOBAL_POOL=adaptive_avg   # adaptive_avg, adaptive_max, last, flatten
    WINDOW_STRIDE=256          # sliding window stride for data loading
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import datetime
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# Import from the project root
_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from efficient_byte_tokenizer import ByteCategory, EfficientByteTokenizer

# -----------------------------
# HYPERPARAMETERS
# -----------------------------


class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_byte260")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    run_id = (
        os.environ.get("RUN_ID", str(uuid.uuid4()))
        + f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_max_tokens = int(os.environ.get("VAL_MAX_TOKENS", 0))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # CNN-specific
    embed_dim = int(os.environ.get("EMBED_DIM", 128))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    global_pool = os.environ.get("GLOBAL_POOL", "flatten")
    layer_configs_json = os.environ.get("LAYER_CONFIGS", "")
    head_dims_json = os.environ.get(
        "HEAD_DIMS", "[512, 512]"
    )  # e.g. "[512,256]" or empty=one hidden layer matching final conv channels
    window_stride = int(os.environ.get("WINDOW_STRIDE", 0))  # 0 = seq_len // 2

    # Optimizer
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    conv_lr = float(os.environ.get("CONV_LR", 0.001))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(
        os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85)
    )
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    # Byte tokenizer config
    discard_unused_bytes = bool(int(os.environ.get("DISCARD_UNUSED_BYTES", "1")))
    fold = os.environ.get("FOLD", "")


# -----------------------------
# MUON OPTIMIZER
# -----------------------------


def zeropower_via_newtonschulz5(
    G: Tensor, steps: int = 10, eps: float = 1e-7
) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        momentum: float,
        backend_steps: int,
        nesterov: bool = True,
    ):
        super().__init__(
            params,
            dict(
                lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr, momentum, backend_steps, nesterov = (
                group["lr"],
                group["momentum"],
                group["backend_steps"],
                group["nesterov"],
            )
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(
                total_params, device=params[0].device, dtype=torch.bfloat16
            )
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                p.data.add_(
                    updates_flat[curr : curr + p.numel()].reshape(p.shape).to(p.dtype),
                    alpha=-lr,
                )
                curr += p.numel()
        return loss


# -----------------------------
# BYTE-LEVEL BPB EVAL
# -----------------------------


def build_byte_bpb_lut(tok: EfficientByteTokenizer, device: torch.device):
    base_bytes = np.zeros(tok.vocab_size, dtype=np.int16)
    base_bytes[tok.n_special :] = 1
    return torch.tensor(base_bytes, dtype=torch.int16, device=device)


def load_validation_tokens(pattern, seq_len, max_tokens=0):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    raw_bytes = np.concatenate([load_data_shard(file) for file in files])
    if max_tokens > 0:
        raw_bytes = raw_bytes[: max_tokens + 1]
    usable = ((len(raw_bytes) - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return raw_bytes[: usable + 1]


# -----------------------------
# INT8 QUANTIZATION
# -----------------------------

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "scale_factor,residual_scale",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0


def tensor_nbytes(t):
    return int(t.numel()) * int(t.element_size())


def keep_float_tensor(name, t, passthrough_orig_dtypes):
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def quantize_float_tensor(t):
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(
            torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None]
        )
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = (
            torch.clamp(torch.round(clipped / scale[:, None]), -127, 127)
            .to(torch.int8)
            .contiguous()
        )
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = (
        float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item())
        if t32.numel()
        else 0.0
    )
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = (
        torch.clamp(
            torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127
        )
        .to(torch.int8)
        .contiguous()
    )
    return q, scale


def quantize_state_dict_int8(state_dict):
    quantized, scales, dtypes, passthrough = {}, {}, {}, {}
    passthrough_orig_dtypes, qmeta = {}, {}
    stats = dict.fromkeys(
        (
            "param_count",
            "num_tensors",
            "num_float_tensors",
            "num_nonfloat_tensors",
            "baseline_tensor_bytes",
            "int8_payload_bytes",
        ),
        0,
    )
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue
        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
    obj = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int8(obj):
    out = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (
                (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1))))
                .to(dtype=dtype)
                .contiguous()
            )
        else:
            out[name] = (q.float() * float(s.item())).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# -----------------------------
# DATA LOADING
# -----------------------------


def load_data_shard(file):
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return tokens_np


def remap_shard_tokens(
    shard_tokens: np.ndarray, tok: EfficientByteTokenizer
) -> torch.Tensor:
    """Convert byte260 shard tokens to EfficientByteTokenizer IDs.

    BOS tokens at document boundaries are preserved.
    """
    remapped = tok.remap_byte260_shard(shard_tokens)
    remapped = tok.filter_stream(remapped)
    return torch.from_numpy(remapped)


class TokenStream:
    def __init__(self, pattern, tok: EfficientByteTokenizer):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.tok = tok
        self.file_idx = 0
        self.tokens = remap_shard_tokens(load_data_shard(self.files[0]), tok)
        self.pos = 0

    def _advance_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = remap_shard_tokens(
            load_data_shard(self.files[self.file_idx]), self.tok
        )
        self.pos = 0

    def take(self, n):
        chunks = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class ContextTargetLoader:
    """Produces (context, target) pairs with sliding window for non-causal prediction.

    From a contiguous token stream, extracts overlapping windows of length (seq_len+1).
    context = window[:seq_len], target = window[seq_len].

    stride controls step between consecutive windows:
      stride=1        -> maximum overlap (most token-efficient)
      stride=seq_len  -> no overlap (each token used as target once)
    """

    def __init__(self, pattern, tok, rank, world_size, device, seq_len, stride):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.seq_len = seq_len
        self.stride = stride
        self.stream = TokenStream(pattern, tok)
        # Buffer for sliding window extraction
        self._buf = torch.empty(0, dtype=torch.int16)

    def _fill_buffer(self, min_tokens):
        """Ensure buffer has at least min_tokens available."""
        need = min_tokens - self._buf.numel()
        if need > 0:
            new = self.stream.take(need)
            self._buf = torch.cat([self._buf, new]) if self._buf.numel() > 0 else new

    def next_batch(self, batch_size_per_rank):
        """Returns (context, target): shapes (B, seq_len) and (B,)."""
        s = self.seq_len
        stride = self.stride

        per_rank_seqs = batch_size_per_rank
        total_seqs = per_rank_seqs * self.world_size
        total_tokens = (s + 1) + (total_seqs - 1) * stride

        # Fill buffer with contiguous tokens covering all ranks
        self._fill_buffer(total_tokens)
        buf = self._buf[:total_tokens]
        self._buf = self._buf[total_seqs * stride :]

        # Extract this rank's sliding windows
        rank_offset = self.rank * per_rank_seqs * stride
        contexts = []
        targets = []
        for i in range(per_rank_seqs):
            start = rank_offset + i * stride
            window = buf[start : start + s + 1]
            contexts.append(window[:s])
            targets.append(window[s])

        context = torch.stack(contexts).to(
            device=self.device, dtype=torch.int64, non_blocking=True
        )
        target = torch.tensor(
            [t.item() for t in targets],
            device=self.device,
            dtype=torch.int64,
        )

        return context, target


# -----------------------------
# CNN MODEL COMPONENTS
# -----------------------------


@dataclass
class ConvLayerConfig:
    """Configuration for a single convolutional block.

    Each block follows the TFScripts pipeline:
      conv -> std-dev repair -> bias -> activation -> norm -> residual -> pool -> dropout
    """

    num_filters: int = 128
    kernel_size: int = 3
    stride: int = 1
    padding: str = "same"  # "same", "valid", or an integer string like "2"
    dilation: int = 1
    groups: int = 1
    activation: str = "relu"  # "relu", "gelu", "silu", "relu_squared", "none"
    use_residual: bool = True
    use_scale_factor: bool = True  # learned residual scale (TFScripts AddResidual)
    scale_factor_init: float = 0.001  # initial std for learned scale factor
    repair_std_deviation: bool = True  # TFScripts-style variance correction
    use_bias: bool = True
    norm: str = "none"  # "batch", "group", "layer", "rms", "none"
    norm_groups: int = 8
    pool_type: str = "none"  # "max", "avg", "none"
    pool_size: int = 2
    dropout: float = 0.0


# Default layer configuration for s=512
DEFAULT_LAYER_CONFIGS = [
    ConvLayerConfig(
        num_filters=128,
        kernel_size=7,
        stride=1,
        padding="same",
        activation="relu",
        use_residual=True,
        repair_std_deviation=True,
    ),
    ConvLayerConfig(
        num_filters=128,
        kernel_size=5,
        stride=2,
        activation="relu",
        use_residual=True,
        repair_std_deviation=True,
    ),
    ConvLayerConfig(
        num_filters=256,
        kernel_size=5,
        stride=1,
        padding="same",
        activation="relu",
        use_residual=True,
        repair_std_deviation=True,
    ),
    ConvLayerConfig(
        num_filters=256,
        kernel_size=3,
        stride=2,
        activation="relu",
        use_residual=True,
        repair_std_deviation=True,
    ),
    ConvLayerConfig(
        num_filters=256,
        kernel_size=3,
        stride=1,
        padding="same",
        activation="relu",
        use_residual=True,
        repair_std_deviation=True,
    ),
    ConvLayerConfig(
        num_filters=512,
        kernel_size=3,
        stride=2,
        activation="relu",
        use_residual=True,
        repair_std_deviation=True,
    ),
    ConvLayerConfig(
        num_filters=512,
        kernel_size=3,
        stride=1,
        padding="same",
        activation="relu",
        use_residual=True,
        repair_std_deviation=True,
    ),
    ConvLayerConfig(
        num_filters=512,
        kernel_size=3,
        stride=2,
        activation="relu",
        use_residual=True,
        repair_std_deviation=True,
    ),
    ConvLayerConfig(
        num_filters=512,
        kernel_size=3,
        stride=2,
        activation="relu",
        use_residual=True,
        repair_std_deviation=True,
    ),
]


def _resolve_activation(name: str):
    """Return an activation function from its string name."""
    if name == "relu":
        return F.relu
    elif name == "gelu":
        return F.gelu
    elif name == "silu":
        return F.silu
    elif name == "relu_squared":

        def relu_squared(x):
            return F.relu(x).square()

        return relu_squared
    elif name == "none":
        return lambda x: x
    else:
        raise ValueError(f"Unknown activation: {name}")


def _compute_same_padding(kernel_size: int, stride: int, dilation: int) -> int:
    """Compute padding for 'same' output length: ceil(input_len / stride).

    For stride=1 this gives exact 'same' padding.
    For stride>1 this gives the padding that preserves ceil(L/stride) output length.
    """
    effective_ks = dilation * (kernel_size - 1) + 1
    return (effective_ks - 1) // 2


class AddResidual(nn.Module):
    """Residual addition with channel/spatial mismatch handling.

    Inspired by TFScripts core.py AddResidual:
    - Channel mismatch: slice or concat to match dimensions
    - Spatial mismatch: strided slice of skip connection
    - Scale factor: learned (init near zero) or divide by sqrt(2)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        pool_size: int = 1,
        use_scale_factor: bool = True,
        scale_factor_init: float = 0.001,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.pool_size = pool_size
        self.use_scale_factor = use_scale_factor
        # Total spatial downsampling factor
        self.spatial_factor = stride * pool_size

        if use_scale_factor:
            self.scale_factor = nn.Parameter(
                torch.randn(out_channels) * scale_factor_init
            )
        else:
            self.register_parameter("scale_factor", None)

    def forward(self, identity: Tensor, residual: Tensor) -> Tensor:
        """Add identity (skip) to residual, handling mismatches.

        Args:
            identity: (B, C_in, L_in) — the input to the block (skip connection)
            residual: (B, C_out, L_out) — the block's output before residual add
        """
        skip = identity

        # Handle spatial mismatch from stride/pooling
        if self.spatial_factor > 1:
            skip = skip[:, :, :: self.spatial_factor]
            # Ensure lengths match (may differ by 1 due to rounding)
            min_len = min(skip.shape[2], residual.shape[2])
            skip = skip[:, :, :min_len]
            residual = residual[:, :, :min_len]

        # Handle channel mismatch
        c_in = self.in_channels
        c_out = self.out_channels

        if c_in == c_out:
            combined = skip
        elif c_in > c_out:
            # More input channels than output: slice input to match
            # Scale up to preserve variance: we're keeping c_out of c_in channels
            combined = skip[:, :c_out, :] * math.sqrt(c_in / c_out)
        else:
            # Fewer input channels: place skip into expanded tensor, scale to
            # preserve variance (only c_in of c_out channels carry signal)
            combined = torch.zeros_like(residual)
            combined[:, :c_in, :] = skip * math.sqrt(c_out / c_in)

        # Apply scale and add
        if self.use_scale_factor:
            sf = self.scale_factor.to(dtype=residual.dtype)[None, :, None]
            return combined + sf * residual
        else:
            return (combined + residual) / math.sqrt(2.0)


class ConfigurableConvBlock(nn.Module):
    """Single configurable conv block following the TFScripts pipeline.

    Pipeline: conv -> std-dev repair -> bias -> std-dev repair ->
              activation -> normalization -> residual -> pooling -> dropout

    Data format: (B, C, L) throughout.
    """

    def __init__(self, in_channels: int, config: ConvLayerConfig):
        super().__init__()
        self.config = config
        self.in_channels = in_channels

        # Compute padding
        if config.padding == "same":
            pad = _compute_same_padding(
                config.kernel_size, config.stride, config.dilation
            )
        elif config.padding == "valid":
            pad = 0
        else:
            pad = int(config.padding)

        # 1. Conv1d (bias handled separately for std-dev repair)
        self.conv = nn.Conv1d(
            in_channels,
            config.num_filters,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=pad,
            dilation=config.dilation,
            groups=config.groups,
            bias=False,
        )

        # When repair_std_deviation is enabled, we use unit-variance weight init
        # (TFScripts style). The repair factor 1/sqrt(fan_in) then normalizes the
        # conv output back to unit variance. Without repair, we keep PyTorch's
        # default Kaiming init which already accounts for fan_in.
        fan_in = config.kernel_size * (in_channels // config.groups)
        if config.repair_std_deviation:
            nn.init.normal_(self.conv.weight, std=1.0)

        # 2. Separate bias parameter
        if config.use_bias:
            self.bias = nn.Parameter(torch.zeros(config.num_filters))
        else:
            self.register_parameter("bias", None)

        # 3. Activation
        self.activation_fn = _resolve_activation(config.activation)

        # 4. Normalization
        self.norm = self._build_norm(config)

        # 5. Residual connection
        # Note: pooling happens AFTER the residual add in forward(), so
        # AddResidual only needs to account for stride, not pool_size.
        if config.use_residual:
            self.residual = AddResidual(
                in_channels,
                config.num_filters,
                stride=config.stride,
                pool_size=1,
                use_scale_factor=config.use_scale_factor,
                scale_factor_init=config.scale_factor_init,
            )
        else:
            self.residual = None

        # 6. Pooling
        if config.pool_type == "max":
            self.pool = nn.MaxPool1d(config.pool_size)
        elif config.pool_type == "avg":
            self.pool = nn.AvgPool1d(config.pool_size)
        else:
            self.pool = None

        # 7. Dropout
        if config.dropout > 0:
            self.drop = nn.Dropout(config.dropout)
        else:
            self.drop = None

        # Pre-compute std-dev repair constants
        self._conv_std_repair = 1.0 / math.sqrt(fan_in)
        self._bias_std_repair = 1.0 / math.sqrt(2.0)

    def _build_norm(self, config: ConvLayerConfig):
        nf = config.num_filters
        if config.norm == "batch":
            return nn.BatchNorm1d(nf)
        elif config.norm == "group":
            return nn.GroupNorm(min(config.norm_groups, nf), nf)
        elif config.norm == "layer":
            return nn.GroupNorm(1, nf)  # LayerNorm equivalent for (B, C, L)
        elif config.norm == "rms":
            # RMS norm over channel dim for conv format
            return _RMSNorm1d(nf)
        elif config.norm == "none":
            return None
        else:
            raise ValueError(f"Unknown norm type: {config.norm}")

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        # Conv
        out = self.conv(x)

        # Std-dev repair after conv
        if self.config.repair_std_deviation:
            out = out * self._conv_std_repair

        # Bias
        if self.bias is not None:
            out = out + self.bias[None, :, None]
            # Std-dev repair after bias addition
            if self.config.repair_std_deviation:
                out = out * self._bias_std_repair

        # Activation
        out = self.activation_fn(out)

        # Normalization
        if self.norm is not None:
            out = self.norm(out)

        # Residual
        if self.residual is not None:
            out = self.residual(identity, out)

        # Pooling
        if self.pool is not None:
            out = self.pool(out)

        # Dropout
        if self.drop is not None:
            out = self.drop(out)

        return out


class _RMSNorm1d(nn.Module):
    """RMS normalization over the channel dimension for (B, C, L) tensors."""

    def __init__(self, num_features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, L)
        rms = x.float().pow(2).mean(dim=1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms.to(x.dtype)) * self.weight[None, :, None].to(x.dtype)


# -----------------------------
# TOP-LEVEL MODEL
# -----------------------------


class CNNNextTokenPredictor(nn.Module):
    """Non-causal CNN that predicts the next token from a fixed-length context.

    Input:  (B, s) integer token IDs (left-padded with pad_id=0)
    Output: (B, vocab_size) logits for the s+1-th token
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        layer_configs: list[ConvLayerConfig],
        logit_softcap: float,
        tied_embed_init_std: float,
        global_pool: str = "adaptive_avg",
        seq_len: int = 512,
        head_dims: list[int] | None = None,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.logit_softcap = logit_softcap
        self.global_pool_type = global_pool
        self.seq_len = seq_len

        # Embedding (pad_id=0 gets zero embedding via padding_idx)
        self.tok_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=tied_embed_init_std)
        # Re-zero the padding embedding
        with torch.no_grad():
            self.tok_emb.weight[0].zero_()

        # Build conv stack from configs
        blocks = []
        in_ch = embed_dim
        for cfg in layer_configs:
            blocks.append(ConfigurableConvBlock(in_ch, cfg))
            in_ch = cfg.num_filters
        self.conv_stack = nn.ModuleList(blocks)
        self.final_channels = in_ch

        # Compute final spatial dim for "flatten" mode
        if global_pool == "flatten":
            self._flat_dim = self._compute_output_spatial_dim(seq_len) * in_ch
        else:
            self._flat_dim = in_ch

        # MLP head: configurable hidden layers
        # head_dims=None or [] -> single hidden layer matching final conv channels
        # head_dims=[512, 256] -> Linear->GELU->Linear->GELU->Linear(->vocab)
        if head_dims is None or len(head_dims) == 0:
            head_dims = [in_ch]
        head_layers: list[nn.Module] = []
        prev_dim = self._flat_dim
        for hd in head_dims:
            head_layers.append(nn.Linear(prev_dim, hd))
            head_layers.append(nn.GELU())
            prev_dim = hd
        head_layers.append(nn.Linear(prev_dim, vocab_size))
        self.head = nn.Sequential(*head_layers)
        # Zero-init last linear for stable training start
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def _compute_output_spatial_dim(self, input_len: int) -> int:
        """Trace through conv stack to compute final spatial dimension."""
        L = input_len
        for cfg in [b.config for b in self.conv_stack]:
            if cfg.padding == "same":
                pad = _compute_same_padding(cfg.kernel_size, cfg.stride, cfg.dilation)
            elif cfg.padding == "valid":
                pad = 0
            else:
                pad = int(cfg.padding)
            eff_ks = cfg.dilation * (cfg.kernel_size - 1) + 1
            L = (L + 2 * pad - eff_ks) // cfg.stride + 1
            if cfg.pool_type != "none":
                L = L // cfg.pool_size
        return max(L, 1)

    def forward(self, input_ids: Tensor, target_ids: Tensor | None = None) -> Tensor:
        """
        Args:
            input_ids: (B, s) context token IDs
            target_ids: (B,) target token IDs (optional, for loss computation)

        Returns:
            If target_ids provided: scalar cross-entropy loss
            Otherwise: (B, vocab_size) logits
        """
        # Embedding + RMS norm to unit variance (required for std-dev repair)
        x = self.tok_emb(input_ids)  # (B, s, embed_dim)
        x = F.rms_norm(x, (x.size(-1),))
        x = x.transpose(1, 2)  # (B, embed_dim, s) for conv1d

        # Conv stack
        for block in self.conv_stack:
            x = block(x)

        # Spatial collapse -> (B, C_final) or (B, C_final * L_final)
        # Scale by sqrt(L) after avg pooling to preserve unit variance
        if self.global_pool_type == "adaptive_avg":
            spatial_len = x.shape[2]
            x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
            x = x * math.sqrt(spatial_len)
        elif self.global_pool_type == "adaptive_max":
            x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        elif self.global_pool_type == "last":
            x = x[:, :, -1]  # rightmost = most recent context (due to left-padding)
        elif self.global_pool_type == "flatten":
            x = x.reshape(x.shape[0], -1)
        else:
            raise ValueError(f"Unknown global_pool: {self.global_pool_type}")

        # MLP head -> logits
        logits = self.head(x)  # (B, vocab_size)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)

        if target_ids is not None:
            return F.cross_entropy(logits.float(), target_ids, reduction="mean")
        return logits


# -----------------------------
# VALIDATION
# -----------------------------


def eval_val_cnn(
    args,
    model,
    rank,
    world_size,
    device,
    val_tokens,
    base_bytes_lut,
):
    """Evaluate CNN model on validation set with single-token prediction."""
    seq_len = args.train_seq_len
    stride = args.window_stride if args.window_stride > 0 else seq_len // 2

    # Number of windows we can extract
    total_tokens = val_tokens.numel()
    num_windows = max(1, (total_tokens - seq_len - 1) // stride + 1)

    # Distribute across ranks
    win_start = (num_windows * rank) // world_size
    win_end = (num_windows * (rank + 1)) // world_size

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    batch_size = max(1, args.val_batch_size // (seq_len + 1))
    model.eval()

    with torch.inference_mode():
        i = win_start
        while i < win_end:
            batch_end = min(i + batch_size, win_end)

            contexts = []
            targets = []
            for w in range(i, batch_end):
                start = w * stride
                end = start + seq_len + 1
                if end > total_tokens:
                    break
                window = val_tokens[start:end]
                contexts.append(window[:seq_len])
                targets.append(window[seq_len].item())

            if not contexts:
                i = batch_end
                continue

            context = torch.stack(contexts).to(
                device=device, dtype=torch.int64, non_blocking=True
            )
            target = torch.tensor(targets, device=device, dtype=torch.int64)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(context, target).detach()

            n = float(len(targets))
            val_loss_sum += batch_loss.to(torch.float64) * n
            val_count += n
            # Each predicted token = 1 byte (byte-level tokenizer)
            tgt_bytes = base_bytes_lut[target].to(torch.float64).sum()
            val_byte_count += tgt_bytes

            i = batch_end

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_count
    # BPB: each prediction covers 1 token which may be 0 or 1 bytes
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_count.item() / max(val_byte_count.item(), 1.0)
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


# -----------------------------
# UTILITIES
# -----------------------------


def restore_low_dim_params_to_fp32(module):
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (
                param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)
            ) and param.dtype != torch.float32:
                param.data = param.data.float()


def _build_tokenizer(args: Hyperparameters) -> EfficientByteTokenizer:
    fold_cats: frozenset[ByteCategory] = frozenset()
    if args.fold:
        fold_cats = frozenset(
            ByteCategory(c.strip()) for c in args.fold.split(",") if c.strip()
        )
    return EfficientByteTokenizer(
        discard_unused_bytes=args.discard_unused_bytes,
        fold=fold_cats if fold_cats else None,
    )


def _parse_layer_configs(json_str: str) -> list[ConvLayerConfig]:
    """Parse JSON layer configs into ConvLayerConfig list.

    Accepts either:
      - Empty string (returns DEFAULT_LAYER_CONFIGS)
      - A file path ending in .json
      - An inline JSON array string
    """
    s = json_str.strip()
    if not s:
        return list(DEFAULT_LAYER_CONFIGS)
    # If it looks like a file path, read from file
    if s.endswith(".json") or os.sep in s:
        with open(s, encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raw = json.loads(s)
    if not isinstance(raw, list):
        raise ValueError("LAYER_CONFIGS must be a JSON array of objects")
    return [
        ConvLayerConfig(**{k: v for k, v in item.items() if not k.startswith("_")})
        for item in raw
    ]


# -----------------------------
# TRAINING
# -----------------------------


def main():
    global zeropower_via_newtonschulz5
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if "GRAD_ACCUM_STEPS" in os.environ:
        grad_accum_steps = int(os.environ["GRAD_ACCUM_STEPS"])
    else:
        if 8 % world_size != 0:
            raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
        grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    run_dir = f"models/{args.run_id}"
    logfile = None
    if master_process:
        os.makedirs(run_dir, exist_ok=True)
        logfile = f"{run_dir}/log.txt"
        print(logfile)

    def log0(msg, console=True):
        if not master_process:
            return
        if console:
            print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        ).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # --- Byte tokenizer setup ---
    tok = _build_tokenizer(args)
    vocab_size = tok.vocab_size
    log0(f"tokenizer: {tok.describe()}")

    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))

    # Remap validation tokens
    raw_val_bytes = load_validation_tokens(
        args.val_files, args.train_seq_len, args.val_max_tokens
    )
    val_tokens = remap_shard_tokens(raw_val_bytes, tok)

    base_bytes_lut = build_byte_bpb_lut(tok, device)
    log0(f"val_bpb:enabled tokenizer_kind=efficient_byte vocab_size={vocab_size}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")

    # --- Parse layer configs ---
    layer_configs = _parse_layer_configs(args.layer_configs_json)
    log0(f"cnn_layers: {len(layer_configs)} blocks")
    for i, cfg in enumerate(layer_configs):
        log0(
            f"  layer {i}: filters={cfg.num_filters} ks={cfg.kernel_size} "
            f"stride={cfg.stride} pad={cfg.padding} act={cfg.activation} "
            f"residual={cfg.use_residual} repair_std={cfg.repair_std_deviation} "
            f"norm={cfg.norm} pool={cfg.pool_type}"
        )

    # --- Parse head dims ---
    head_dims: list[int] | None = None
    if args.head_dims_json.strip():
        head_dims = json.loads(args.head_dims_json)
        assert isinstance(head_dims, list) and all(
            isinstance(d, int) for d in head_dims
        )
    log0(f"head_dims: {head_dims or '(default: match final conv channels)'}")

    # --- Sliding window stride ---
    window_stride = (
        args.window_stride if args.window_stride > 0 else args.train_seq_len // 2
    )
    log0(f"window_stride: {window_stride}")

    # --- Build model ---
    base_model = (
        CNNNextTokenPredictor(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            layer_configs=layer_configs,
            logit_softcap=args.logit_softcap,
            tied_embed_init_std=args.tied_embed_init_std,
            global_pool=args.global_pool,
            seq_len=args.train_seq_len,
            head_dims=head_dims,
        )
        .to(device)
        .bfloat16()
    )

    # Keep conv weights and linear weights in float32 for precision
    for module in base_model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            module.float()
    restore_low_dim_params_to_fp32(base_model)

    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model = (
        DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False)
        if distributed
        else compiled_model
    )

    # --- Optimizer setup ---
    # Conv1d weights -> Adam, Linear 2D in head -> Muon, rest -> Adam
    conv_weight_ids = {
        id(p)
        for m in base_model.modules()
        if isinstance(m, nn.Conv1d)
        for p in m.parameters()
    }

    head_linear_params = []
    head_scalar_params = []
    for m in base_model.head.modules():
        if isinstance(m, nn.Linear):
            head_linear_params.append(m.weight)
            if m.bias is not None:
                head_scalar_params.append(m.bias)

    head_linear_ids = {id(p) for p in head_linear_params}

    conv_params = [
        p
        for m in base_model.modules()
        if isinstance(m, nn.Conv1d)
        for p in m.parameters()
    ]
    scalar_params = []
    for n, p in base_model.named_parameters():
        pid = id(p)
        if (
            pid in conv_weight_ids
            or pid in head_linear_ids
            or pid == id(base_model.tok_emb.weight)
        ):
            continue
        if pid in {id(hp) for hp in head_scalar_params}:
            scalar_params.append(p)
        else:
            scalar_params.append(p)

    # Embedding optimizer
    optimizer_tok = torch.optim.Adam(
        [
            {
                "params": [base_model.tok_emb.weight],
                "lr": args.embed_lr,
                "base_lr": args.embed_lr,
            }
        ],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )

    # Muon for 2D head linear weights
    optimizer_muon = Muon(
        head_linear_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr

    # Adam for conv weights
    optimizer_conv = (
        torch.optim.Adam(
            [{"params": conv_params, "lr": args.conv_lr, "base_lr": args.conv_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        if conv_params
        else None
    )

    # Adam for all scalar/1D params (biases, norms, scale factors, head biases)
    optimizer_scalar = (
        torch.optim.Adam(
            [
                {
                    "params": scalar_params,
                    "lr": args.scalar_lr,
                    "base_lr": args.scalar_lr,
                }
            ],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        if scalar_params
        else None
    )

    optimizers = [optimizer_tok, optimizer_muon]
    if optimizer_conv is not None:
        optimizers.append(optimizer_conv)
    if optimizer_scalar is not None:
        optimizers.append(optimizer_scalar)

    # Verify every trainable parameter is in exactly one optimizer
    optimized_ids: set[int] = set()
    for opt in optimizers:
        for group in opt.param_groups:
            for p in group["params"]:
                assert id(p) not in optimized_ids, "parameter in multiple optimizers"
                optimized_ids.add(id(p))
    all_param_ids = {id(p) for p in base_model.parameters()}
    untrained = all_param_ids - optimized_ids
    assert not untrained, (
        f"{len(untrained)} parameters not in any optimizer: "
        + ", ".join(
            n for n, p in base_model.named_parameters() if id(p) not in optimized_ids
        )
    )

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(
        f"embed_lr:{args.embed_lr} conv_lr:{args.conv_lr} "
        f"head_lr(muon):{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"global_pool:{args.global_pool} embed_dim:{args.embed_dim}")
    log0(f"seed:{args.seed}")

    # --- Compute batch size in sequences ---
    # Each sequence consumes `stride` tokens from the stream (plus seq_len context for first)
    # For grad accumulation: each micro-step gets some sequences
    seqs_per_micro_step = max(
        1,
        args.train_batch_tokens // (args.train_seq_len * grad_accum_steps * world_size),
    )
    log0(f"seqs_per_micro_step:{seqs_per_micro_step}")

    train_loader = ContextTargetLoader(
        args.train_files,
        tok,
        rank,
        world_size,
        device,
        args.train_seq_len,
        window_stride,
    )

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = (
        1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    )

    def lr_mul(step, elapsed_ms):
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return (
                max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
                if warmdown_start <= step < args.iterations
                else 1.0
            )
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return (
            remaining_ms / max(warmdown_ms, 1e-9)
            if remaining_ms <= warmdown_ms
            else 1.0
        )

    # --- Warmup (torch.compile warmup) ---
    if args.warmup_steps > 0:
        initial_model_state = {
            n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()
        }
        initial_optimizer_states = [
            copy.deepcopy(opt.state_dict()) for opt in optimizers
        ]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = (
                        micro_step == grad_accum_steps - 1
                    )
                x, y = train_loader.next_batch(seqs_per_micro_step)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if (
                args.warmup_steps <= 20
                or (warmup_step + 1) % 10 == 0
                or warmup_step + 1 == args.warmup_steps
            ):
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = ContextTargetLoader(
            args.train_files,
            tok,
            rank,
            world_size,
            device,
            args.train_seq_len,
            window_stride,
        )

    # --- Main training loop ---
    training_time_ms = 0.0
    stop_after_step = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == args.iterations or (
            stop_after_step is not None and step >= stop_after_step
        )
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val_cnn(
                args,
                model,
                rank,
                world_size,
                device,
                val_tokens,
                base_bytes_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)

        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(seqs_per_micro_step)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        # Muon momentum warmup
        frac = (
            min(step / args.muon_momentum_warmup_steps, 1.0)
            if args.muon_momentum_warmup_steps > 0
            else 1.0
        )
        for group in optimizer_muon.param_groups:
            group["momentum"] = (
                1 - frac
            ) * args.muon_momentum_warmup_start + frac * args.muon_momentum

        # LR schedule
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)

        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (
            step <= 10
            or step % args.train_log_every == 0
            or stop_after_step is not None
        ):
            tl = train_loss.item()
            # BPB for single-token prediction: loss / ln(2) * tokens_per_byte
            # For byte tokenizer, each non-special target = 1 byte
            tgt_bytes = base_bytes_lut[y].to(torch.float64).sum().item()
            tpb = float(y.numel()) / max(tgt_bytes, 1.0)
            log0(
                f"step:{step}/{args.iterations} train_loss:{tl:.4f} "
                f"train_bpb:{tl / math.log(2.0) * tpb:.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms "
                f"step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        reached_cap = (
            max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        )
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    # --- Serialization ---
    if master_process:
        torch.save(base_model.state_dict(), f"{run_dir}/model.pt")
        log0(f"Serialized model: {os.path.getsize(f'{run_dir}/model.pt')} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_blob = zlib.compress(quant_buf.getvalue(), level=9)
    if master_process:
        with open(f"{run_dir}/model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        qfb = os.path.getsize(f"{run_dir}/model.int8.ptz")
        ratio = quant_stats["baseline_tensor_bytes"] / max(
            quant_stats["int8_payload_bytes"], 1
        )
        log0(f"Serialized model int8+zlib: {qfb} bytes (payload_ratio:{ratio:.2f}x)")

    # --- Quantized model eval ---
    if distributed:
        dist.barrier()
    with open(f"{run_dir}/model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    base_model.load_state_dict(
        dequantize_state_dict_int8(
            torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
        ),
        strict=True,
    )
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val_cnn(
        args,
        model,
        rank,
        world_size,
        device,
        val_tokens,
        base_bytes_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(
        f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}"
    )
    log0(f"run_id: {args.run_id} | run_dir: {run_dir}/")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
