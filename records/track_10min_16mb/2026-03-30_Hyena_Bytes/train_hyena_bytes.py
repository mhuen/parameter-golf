"""
Hyena GPT — byte-level tokenizer variant.

Replaces transformer attention with Hyena operators (long convolutions +
multiplicative gating via FFT). Keeps ALBERT-style weight sharing, U-Net
skip connections, Muon optimizer, and all other training infrastructure
from the Universal Transformer variant.

Tokenizer configs (via env vars):
    DISCARD_UNUSED_BYTES=1  (default) 206 used bytes, vocab=208
    DISCARD_UNUSED_BYTES=0           all 256 bytes, vocab=258
    FOLD=uppercase                   fold uppercase->lowercase, vocab=182

Data: expects raw UTF-8 byte shards (byte values 0-255 as uint16, no special tokens).
Byte values are remapped on-the-fly to EfficientByteTokenizer IDs at load time.
"""

from __future__ import annotations

import copy
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
from collections import Counter

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# Import from the project root — add it to sys.path if running from records dir.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from efficient_byte_tokenizer import ByteCategory, EfficientByteTokenizer

# -----------------------------
# HYPERPARAMETERS
# -----------------------------


class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_bytes")
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

    # Model
    num_layers = int(os.environ.get("NUM_LAYERS", 24))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    structured_output_logits = bool(
        int(os.environ.get("STRUCTURED_OUTPUT_LOGITS", "0"))
    )
    utf8_prior = bool(int(os.environ.get("UTF8_PRIOR", "0")))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    block_pattern = os.environ.get(
        "BLOCK_PATTERN",
        "0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7",
    )

    # Hyena-specific
    short_conv_kernel = int(os.environ.get("SHORT_CONV_KERNEL", 7))
    num_poles = int(os.environ.get("NUM_POLES", 2))
    filter_lr = float(os.environ.get("FILTER_LR", 0.01))

    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
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
    fold = os.environ.get("FOLD", "")  # comma-separated ByteCategory values


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
    """Build a simple LUT: each non-special token = 1 byte, special tokens = 0 bytes."""
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


def eval_val(
    args,
    model,
    rank,
    world_size,
    device,
    grad_accum_steps,
    val_tokens,
    base_bytes_lut,
):
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError("VAL_BATCH_SIZE too small")
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(
                device=device, dtype=torch.int64, non_blocking=True
            )
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "hyena_scale,mlp_scale,resid_mix,skip_weight,skip_weights,log_amplitude,log_decay,frequency,phase",
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

# Byte shard format: raw UTF-8 byte values (0-255) stored as uint16, no special tokens.
# We use tok.remap_byte_array() + tok.filter_stream() to convert to token IDs.


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
    raw_bytes: np.ndarray, tok: EfficientByteTokenizer
) -> torch.Tensor:
    """Convert raw byte values to EfficientByteTokenizer IDs."""
    remapped = tok.remap_byte_array(raw_bytes)
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


class DistributedTokenLoader:
    def __init__(self, pattern, tok, rank, world_size, device):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(pattern, tok)

    def next_batch(self, global_tokens, seq_len, grad_accum_steps):
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        if local_tokens < seq_len:
            raise ValueError("TRAIN_BATCH_TOKENS too small for this world_size/seq_len")
        local_seqs = local_tokens // seq_len
        n = local_seqs * seq_len + 1
        for _ in range(self.rank):
            self.stream.take(local_seqs * seq_len)
        local = self.stream.take(n)
        for _ in range(self.rank + 1, self.world_size):
            self.stream.take(local_seqs * seq_len)
        x = (
            local[:-1]
            .reshape(local_seqs, seq_len)
            .to(device=self.device, dtype=torch.int64, non_blocking=True)
        )
        y = (
            local[1:]
            .reshape(local_seqs, seq_len)
            .to(device=self.device, dtype=torch.int64, non_blocking=True)
        )
        return x, y


# -----------------------------
# HYENA MODULES
# -----------------------------


class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(
            x,
            self.weight.to(x.dtype),
            self.bias.to(x.dtype) if self.bias is not None else None,
        )


def restore_low_dim_params_to_fp32(module):
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (
                param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)
            ) and param.dtype != torch.float32:
                param.data = param.data.float()


class ExponentialDecayFilter(nn.Module):
    """Implicit long convolution filter via exponential decay + oscillation.

    h[t] = sum_p amplitude_p * exp(-decay_p * t) * cos(frequency_p * t + phase_p)

    Parameters are per-channel, with num_poles components per channel.
    """

    def __init__(self, dim, num_poles=2):
        super().__init__()
        self.dim = dim
        self.num_poles = num_poles
        self.log_amplitude = nn.Parameter(torch.zeros(dim, num_poles))
        # Init log_decay so softplus gives ~0.01-0.05 (moderate decay)
        self.log_decay = nn.Parameter(torch.full((dim, num_poles), -3.0))
        self.frequency = nn.Parameter(torch.randn(dim, num_poles) * 0.1)
        self.phase = nn.Parameter(torch.zeros(dim, num_poles))

    def filter(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)  # (L,)
        amplitude = self.log_amplitude.exp()  # (D, P)
        decay = F.softplus(self.log_decay)  # (D, P)
        # (D, P, 1) * (1, 1, L) -> (D, P, L)
        envelope = amplitude.unsqueeze(-1) * torch.exp(
            -decay.unsqueeze(-1) * t[None, None, :]
        )
        oscillation = torch.cos(
            self.frequency.unsqueeze(-1) * t[None, None, :] + self.phase.unsqueeze(-1)
        )
        h = (envelope * oscillation).sum(dim=1)  # (D, L)
        return h.to(dtype=dtype)


def fft_causal_conv(x, h):
    """Causal convolution via FFT.

    Args:
        x: (B, D, L) input signal
        h: (D, L) causal filter (h[0] is the current-time weight)

    Returns:
        (B, D, L) causal convolution output
    """
    orig_dtype = x.dtype
    L = x.size(-1)
    fft_size = 2 * L
    # CUDA FFT requires float32 (bfloat16 not supported)
    X_f = torch.fft.rfft(x.float(), n=fft_size, dim=-1)
    H_f = torch.fft.rfft(h.float(), n=fft_size, dim=-1)
    Y_f = X_f * H_f.unsqueeze(0)
    y = torch.fft.irfft(Y_f, n=fft_size, dim=-1)[..., :L]
    return y.to(dtype=orig_dtype)


class HyenaOperator(nn.Module):
    """Order-2 Hyena operator: one multiplicative gating stage.

    v = short_conv(proj_v(x))       -- "value"
    x1 = short_conv(proj_x1(x))     -- "gate"
    y = x1 * long_conv(v)           -- gated long convolution
    out = out_proj(y)
    """

    def __init__(self, dim, short_conv_kernel=7, num_poles=2):
        super().__init__()
        self.proj_v = CastedLinear(dim, dim, bias=False)
        self.proj_x1 = CastedLinear(dim, dim, bias=False)

        self.short_conv_v = nn.Conv1d(
            dim, dim, short_conv_kernel, groups=dim, bias=False
        )
        self.short_conv_x1 = nn.Conv1d(
            dim, dim, short_conv_kernel, groups=dim, bias=False
        )
        self.short_pad = short_conv_kernel - 1

        self.long_filter = ExponentialDecayFilter(dim, num_poles=num_poles)

        self.out_proj = CastedLinear(dim, dim, bias=False)
        self.out_proj._zero_init = True

    def forward(self, x):
        B, L, D = x.shape

        v = self.proj_v(x)
        x1 = self.proj_x1(x)

        # Short causal convolutions (B, L, D) -> (B, D, L) -> conv -> (B, L, D)
        v = F.pad(v.transpose(1, 2), (self.short_pad, 0))
        v = F.silu(self.short_conv_v(v)).transpose(1, 2)

        x1 = F.pad(x1.transpose(1, 2), (self.short_pad, 0))
        x1 = F.silu(self.short_conv_x1(x1)).transpose(1, 2)

        # Long causal convolution via FFT
        h = self.long_filter.filter(L, x.device, v.dtype)  # (D, L)
        y = fft_causal_conv(v.transpose(1, 2), h).transpose(1, 2)  # (B, L, D)

        # Multiplicative gating
        out = x1 * y

        return self.out_proj(out)


class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class SharedHyenaBlock(nn.Module):
    """Shared Hyena block: Hyena operator + MLP with norms."""

    def __init__(self, dim, short_conv_kernel=7, mlp_mult=2, num_poles=2):
        super().__init__()
        self.hyena_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.hyena = HyenaOperator(dim, short_conv_kernel, num_poles)
        self.mlp = MLP(dim, mlp_mult)

    def forward(self, x, hyena_scale, mlp_scale):
        n = self.hyena_norm(x)
        x = x + hyena_scale.to(dtype=x.dtype)[None, None, :] * self.hyena(n)
        x = x + mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class LayerScalars(nn.Module):
    """Per-layer scalars: hyena_scale, mlp_scale, resid_mix."""

    def __init__(self, dim):
        super().__init__()
        self.hyena_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(
            torch.stack((torch.ones(dim), torch.zeros(dim))).float()
        )


class UTF8Prior(nn.Module):
    """Precomputed UTF-8 structural prior masks (no learnable parameters)."""

    BT_BOS = 0
    BT_PAD = 1
    BT_ASCII = 2
    BT_LEAD_2 = 3
    BT_LEAD_3 = 4
    BT_LEAD_4 = 5
    BT_CONT = 6

    ST_READY = 0
    ST_EXPECT_CONT = 1
    ST_UNSYNCED = 2

    def __init__(self, tok, num_categories=8, multibyte_cat_idx=7):
        super().__init__()
        V = tok.vocab_size
        self.num_categories = num_categories
        NEG_INF = float("-inf")

        token_byte_type = torch.zeros(V, dtype=torch.long)
        token_byte_value = torch.zeros(V, dtype=torch.long)
        for tid in range(V):
            info = tok.token_info(tid)
            if info is None:
                token_byte_type[tid] = self.BT_BOS if tid == tok.bos_id else self.BT_PAD
            elif info.has(ByteCategory.MB_CONTINUATION):
                token_byte_type[tid] = self.BT_CONT
                token_byte_value[tid] = info.byte_value
            elif info.has(ByteCategory.MB_LEAD_2):
                token_byte_type[tid] = self.BT_LEAD_2
                token_byte_value[tid] = info.byte_value
            elif info.has(ByteCategory.MB_LEAD_3):
                token_byte_type[tid] = self.BT_LEAD_3
                token_byte_value[tid] = info.byte_value
            elif info.has(ByteCategory.MB_LEAD_4):
                token_byte_type[tid] = self.BT_LEAD_4
                token_byte_value[tid] = info.byte_value
            else:
                token_byte_type[tid] = self.BT_ASCII
                token_byte_value[tid] = info.byte_value
        self.register_buffer("token_byte_type", token_byte_type)
        self.register_buffer("token_byte_value", token_byte_value)

        lead_expected = torch.zeros(7, dtype=torch.long)
        lead_expected[self.BT_LEAD_2] = 1
        lead_expected[self.BT_LEAD_3] = 2
        lead_expected[self.BT_LEAD_4] = 3
        self.register_buffer("lead_expected", lead_expected)

        state_cat_mask = torch.zeros(3, num_categories)
        for ci in range(num_categories):
            if ci != multibyte_cat_idx:
                state_cat_mask[self.ST_EXPECT_CONT, ci] = NEG_INF
        self.register_buffer("state_cat_mask", state_cat_mask)

        cont_np = tok.mask(ByteCategory.MB_CONTINUATION)
        state_token_mask = torch.zeros(3, V)
        for tid in range(V):
            if cont_np[tid]:
                state_token_mask[self.ST_READY, tid] = NEG_INF
        for tid in range(V):
            if not cont_np[tid]:
                state_token_mask[self.ST_EXPECT_CONT, tid] = NEG_INF
        self.register_buffer("state_token_mask", state_token_mask)

        special_e0 = state_token_mask[self.ST_EXPECT_CONT].clone()
        special_ed = state_token_mask[self.ST_EXPECT_CONT].clone()
        special_f0 = state_token_mask[self.ST_EXPECT_CONT].clone()
        for tid in range(V):
            info = tok.token_info(tid)
            if info and info.has(ByteCategory.MB_CONTINUATION):
                bv = info.byte_value
                if bv < 0xA0:
                    special_e0[tid] = NEG_INF
                if bv > 0x9F:
                    special_ed[tid] = NEG_INF
                if bv < 0x90:
                    special_f0[tid] = NEG_INF
        self.register_buffer("special_e0", special_e0)
        self.register_buffer("special_ed", special_ed)
        self.register_buffer("special_f0", special_f0)

    def forward(self, input_ids):
        B, S = input_ids.shape
        device = input_ids.device

        byte_type = self.token_byte_type[input_ids]
        byte_val = self.token_byte_value[input_ids]

        is_cont = byte_type == self.BT_CONT
        c1 = is_cont
        c2 = torch.zeros(B, S, device=device, dtype=torch.bool)
        c2[:, 1:] = is_cont[:, 1:] & is_cont[:, :-1]
        c3 = torch.zeros(B, S, device=device, dtype=torch.bool)
        c3[:, 2:] = is_cont[:, 2:] & is_cont[:, 1:-1] & is_cont[:, :-2]
        cont_count = c1.long() + c2.long() + c3.long()

        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        lead_pos = (positions - cont_count).clamp(min=0)
        lead_type = byte_type.gather(1, lead_pos)

        non_cont_remaining = self.lead_expected[byte_type]
        cont_remaining = self.lead_expected[lead_type] - cont_count
        remaining = torch.where(is_cont, cont_remaining, non_cont_remaining)

        is_valid_lead = (
            (lead_type == self.BT_LEAD_2)
            | (lead_type == self.BT_LEAD_3)
            | (lead_type == self.BT_LEAD_4)
        )
        is_special = (byte_type == self.BT_BOS) | (byte_type == self.BT_PAD)
        unsynced = is_special | (is_cont & ((lead_pos <= 0) | ~is_valid_lead))
        state = torch.where(
            unsynced,
            self.ST_UNSYNCED,
            torch.where(remaining > 0, self.ST_EXPECT_CONT, self.ST_READY),
        )

        cat_mask = self.state_cat_mask[state]
        token_mask = self.state_token_mask[state]

        is_e0 = (byte_type == self.BT_LEAD_3) & (byte_val == 0xE0)
        is_ed = (byte_type == self.BT_LEAD_3) & (byte_val == 0xED)
        is_f0 = (byte_type == self.BT_LEAD_4) & (byte_val == 0xF0)
        token_mask = torch.where(is_e0.unsqueeze(-1), self.special_e0, token_mask)
        token_mask = torch.where(is_ed.unsqueeze(-1), self.special_ed, token_mask)
        token_mask = torch.where(is_f0.unsqueeze(-1), self.special_f0, token_mask)

        return cat_mask, token_mask


class StructuredOutputHead(nn.Module):
    """Hierarchical softmax output head based on ByteCategory tree."""

    def __init__(self, model_dim, vocab_size, tok, logit_softcap):
        super().__init__()
        self.logit_softcap = logit_softcap
        self.vocab_size = vocab_size

        def _ids(cat):
            arr = tok.ids(cat)
            return set(arr.tolist()) if len(arr) > 0 else set()

        cat_defs = [
            ("bos", ByteCategory.BOS),
            ("pad", ByteCategory.PAD),
            ("digit", ByteCategory.DIGIT),
            ("letter", ByteCategory.LETTER),
            ("separator", ByteCategory.SEPARATOR),
            ("punctuation", ByteCategory.PUNCTUATION),
            ("symbol", ByteCategory.SYMBOL),
            ("multibyte", ByteCategory.MULTIBYTE),
        ]
        cat_id_sets = {name: _ids(cat) for name, cat in cat_defs}

        upper_ids = _ids(ByteCategory.UPPERCASE)
        lower_ids = _ids(ByteCategory.LOWERCASE)
        vowel_ids = _ids(ByteCategory.VOWEL)
        consonant_ids = _ids(ByteCategory.CONSONANT)
        mb_cont_ids = _ids(ByteCategory.MB_CONTINUATION)
        mb_leading_ids = _ids(ByteCategory.MB_LEADING)
        mb_lead2_ids = _ids(ByteCategory.MB_LEAD_2)
        mb_lead3_ids = _ids(ByteCategory.MB_LEAD_3)
        mb_lead4_ids = _ids(ByteCategory.MB_LEAD_4)

        heads = []
        all_indices = []
        all_masks = []
        self._is_leaf: list[bool] = []

        def _add_level(head_size, token_to_idx, active_tids, leaf):
            idx = torch.zeros(vocab_size, dtype=torch.long)
            mask = torch.zeros(vocab_size, dtype=torch.float32)
            for tid, j in token_to_idx.items():
                idx[tid] = j
            for tid in active_tids:
                mask[tid] = 1.0
            head = CastedLinear(model_dim, head_size, bias=False)
            head._zero_init = True
            heads.append(head)
            all_indices.append(idx)
            all_masks.append(mask)
            self._is_leaf.append(leaf)

        cat_map: dict[int, int] = {}
        all_tids: set[int] = set()
        for ci, (name, _) in enumerate(cat_defs):
            for tid in cat_id_sets[name]:
                cat_map[tid] = ci
                all_tids.add(tid)
        _add_level(len(cat_defs), cat_map, all_tids, leaf=False)

        letter_ids = cat_id_sets["letter"]
        if upper_ids and lower_ids:
            _add_level(
                2,
                {tid: (0 if tid in upper_ids else 1) for tid in letter_ids},
                letter_ids,
                leaf=False,
            )

        if upper_ids and (vowel_ids & upper_ids) and (consonant_ids & upper_ids):
            _add_level(
                2,
                {tid: (0 if tid in vowel_ids else 1) for tid in upper_ids},
                upper_ids,
                leaf=False,
            )

        if lower_ids and (vowel_ids & lower_ids) and (consonant_ids & lower_ids):
            _add_level(
                2,
                {tid: (0 if tid in vowel_ids else 1) for tid in lower_ids},
                lower_ids,
                leaf=False,
            )

        mb_ids = cat_id_sets["multibyte"]
        if mb_cont_ids and mb_leading_ids:
            _add_level(
                2,
                {tid: (0 if tid in mb_cont_ids else 1) for tid in mb_ids},
                mb_ids,
                leaf=False,
            )

        lead_tids = mb_lead2_ids | mb_lead3_ids | mb_lead4_ids
        if len(lead_tids) > 1:
            lead_map: dict[int, int] = {}
            for tid in mb_lead2_ids:
                lead_map[tid] = 0
            for tid in mb_lead3_ids:
                lead_map[tid] = 1
            for tid in mb_lead4_ids:
                lead_map[tid] = 2
            _add_level(3, lead_map, lead_tids, leaf=False)

        leaf_token_ids: list[int] = []

        def _add_leaf(group):
            if len(group) > 1:
                _add_level(
                    len(group),
                    {tid: i for i, tid in enumerate(sorted(group))},
                    group,
                    leaf=True,
                )
            leaf_token_ids.extend(sorted(group))

        _add_leaf(cat_id_sets["bos"])
        _add_leaf(cat_id_sets["pad"])
        _add_leaf(cat_id_sets["digit"])
        _add_leaf(cat_id_sets["separator"])
        _add_leaf(upper_ids & vowel_ids)
        _add_leaf(upper_ids & consonant_ids)
        _add_leaf(lower_ids & vowel_ids)
        _add_leaf(lower_ids & consonant_ids)
        _add_leaf(cat_id_sets["punctuation"])
        _add_leaf(cat_id_sets["symbol"])
        _add_leaf(mb_cont_ids)
        _add_leaf(mb_lead2_ids)
        _add_leaf(mb_lead3_ids)
        _add_leaf(mb_lead4_ids)

        leaf_counts = Counter(leaf_token_ids)
        duplicates = {tid: cnt for tid, cnt in leaf_counts.items() if cnt > 1}
        if duplicates:
            raise ValueError(
                f"StructuredOutputHead: tokens appear in multiple leaves: {duplicates}"
            )
        leaf_set = set(leaf_token_ids)
        expected = set(range(vocab_size))
        missing = expected - leaf_set
        extra = leaf_set - expected
        if missing or extra:
            raise ValueError(
                f"StructuredOutputHead leaf coverage error: "
                f"missing token IDs {sorted(missing)}, "
                f"extra token IDs {sorted(extra)}"
            )

        self.heads = nn.ModuleList(heads)
        self.register_buffer("level_indices", torch.stack(all_indices))
        self.register_buffer("level_masks", torch.stack(all_masks))

    def forward(self, x, cat_prior=None, token_prior=None):
        B, S, _ = x.shape
        log_p = torch.zeros(B, S, self.vocab_size, device=x.device, dtype=x.dtype)
        for i, head in enumerate(self.heads):
            logits = head(x)
            if i == 0 and cat_prior is not None:
                logits = logits + cat_prior
            if self._is_leaf[i]:
                logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
            lp = F.log_softmax(logits, dim=-1)
            log_p = log_p + lp[..., self.level_indices[i]] * self.level_masks[i]
        if token_prior is not None:
            log_p = log_p + token_prior
            log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        return log_p


class HyenaGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_layers,
        model_dim,
        mlp_mult,
        tie_embeddings,
        tied_embed_init_std,
        logit_softcap,
        block_pattern="",
        short_conv_kernel=7,
        num_poles=2,
        structured_output_logits=False,
        utf8_prior=False,
        tok=None,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.num_layers = num_layers
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(
            torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32)
        )

        if block_pattern.strip():
            self.block_map = [int(x) for x in block_pattern.strip().split(",")]
            if len(self.block_map) != num_layers:
                raise ValueError(
                    f"BLOCK_PATTERN has {len(self.block_map)} entries but NUM_LAYERS={num_layers}"
                )
        else:
            self.block_map = [0] * num_layers

        num_blocks = max(self.block_map) + 1
        self.shared_blocks = nn.ModuleList(
            [
                SharedHyenaBlock(model_dim, short_conv_kernel, mlp_mult, num_poles)
                for _ in range(num_blocks)
            ]
        )

        self.layer_scalars = nn.ModuleList(
            [LayerScalars(model_dim) for _ in range(num_layers)]
        )

        self.final_norm = RMSNorm()
        self.structured_output_logits = structured_output_logits
        if structured_output_logits:
            assert not tie_embeddings, (
                "structured_output_logits requires tie_embeddings=False"
            )
            assert tok is not None, "structured_output_logits requires tok"
            self.structured_head = StructuredOutputHead(
                model_dim, vocab_size, tok, logit_softcap
            )
            self.lm_head = None
        else:
            self.structured_head = None
            self.lm_head = (
                None
                if tie_embeddings
                else CastedLinear(model_dim, vocab_size, bias=False)
            )
            if self.lm_head is not None:
                self.lm_head._zero_init = True
        if utf8_prior:
            assert tok is not None, "utf8_prior requires tok"
            self.utf8_prior_mod = UTF8Prior(tok)
        else:
            self.utf8_prior_mod = None
        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)) and getattr(
                module, "_zero_init", False
            ):
                nn.init.zeros_(module.weight)

    def forward(self, input_ids, target_ids):
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []

        for i in range(self.num_encoder_layers):
            ls = self.layer_scalars[i]
            mix = ls.resid_mix.to(dtype=x.dtype)
            x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
            x = self.shared_blocks[self.block_map[i]](x, ls.hyena_scale, ls.mlp_scale)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = (
                    x
                    + self.skip_weights[i].to(dtype=x.dtype)[None, None, :]
                    * skips.pop()
                )
            layer_idx = self.num_encoder_layers + i
            ls = self.layer_scalars[layer_idx]
            mix = ls.resid_mix.to(dtype=x.dtype)
            x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
            x = self.shared_blocks[self.block_map[layer_idx]](
                x, ls.hyena_scale, ls.mlp_scale
            )
        x = self.final_norm(x)

        cat_prior, token_prior = None, None
        if self.utf8_prior_mod is not None:
            cat_prior, token_prior = self.utf8_prior_mod(input_ids)
        if self.structured_output_logits:
            log_p = self.structured_head(
                x, cat_prior=cat_prior, token_prior=token_prior
            )
            return F.nll_loss(
                log_p.float().reshape(-1, log_p.size(-1)),
                target_ids.reshape(-1),
                reduction="mean",
            )
        if self.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        if token_prior is not None:
            logits = logits + token_prior
        return F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            reduction="mean",
        )


# -----------------------------
# TRAINING
# -----------------------------

BOS_ID = 1

# Filter parameter names — these are 2D but should NOT go to Muon
FILTER_PARAM_PATTERNS = ("log_amplitude", "log_decay", "frequency", "phase")


def _build_tokenizer(args: Hyperparameters) -> EfficientByteTokenizer:
    """Construct EfficientByteTokenizer from env-var config."""
    fold_cats: frozenset[ByteCategory] = frozenset()
    if args.fold:
        fold_cats = frozenset(
            ByteCategory(c.strip()) for c in args.fold.split(",") if c.strip()
        )
    return EfficientByteTokenizer(
        discard_unused_bytes=args.discard_unused_bytes,
        fold=fold_cats if fold_cats else None,
    )


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

    raw_val_bytes = load_validation_tokens(
        args.val_files, args.train_seq_len, args.val_max_tokens
    )
    val_tokens = remap_shard_tokens(raw_val_bytes, tok)

    base_bytes_lut = build_byte_bpb_lut(tok, device)
    log0(f"val_bpb:enabled tokenizer_kind=efficient_byte vocab_size={vocab_size}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")

    if args.structured_output_logits:
        args.tie_embeddings = False
        log0("structured_output_logits:enabled (forcing tie_embeddings=False)")
    if args.utf8_prior:
        log0("utf8_prior:enabled")

    base_model = (
        HyenaGPT(
            vocab_size=vocab_size,
            num_layers=args.num_layers,
            model_dim=args.model_dim,
            mlp_mult=args.mlp_mult,
            tie_embeddings=args.tie_embeddings,
            tied_embed_init_std=args.tied_embed_init_std,
            logit_softcap=args.logit_softcap,
            block_pattern=args.block_pattern,
            short_conv_kernel=args.short_conv_kernel,
            num_poles=args.num_poles,
            structured_output_logits=args.structured_output_logits,
            utf8_prior=args.utf8_prior,
            tok=tok if (args.structured_output_logits or args.utf8_prior) else None,
        )
        .to(device)
        .bfloat16()
    )
    for module in base_model.modules():
        if isinstance(module, (CastedLinear, nn.Conv1d)):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model = (
        DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False)
        if distributed
        else compiled_model
    )

    # Optimizer: shared_blocks 2D (non-filter, non-conv) -> Muon, filter -> Adam, rest -> Adam
    conv_weight_ids = {
        id(p)
        for m in base_model.modules()
        if isinstance(m, nn.Conv1d)
        for p in m.parameters()
    }
    filter_param_ids = set()
    for n, p in base_model.named_parameters():
        if any(pat in n for pat in FILTER_PARAM_PATTERNS):
            filter_param_ids.add(id(p))

    shared_blocks_named = list(base_model.shared_blocks.named_parameters())
    matrix_params = [
        p
        for n, p in shared_blocks_named
        if p.ndim == 2
        and id(p) not in conv_weight_ids
        and id(p) not in filter_param_ids
        and not any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    filter_params = [p for n, p in shared_blocks_named if id(p) in filter_param_ids]
    scalar_params = [
        p
        for n, p in shared_blocks_named
        if (p.ndim != 2 or any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS))
        and id(p) not in conv_weight_ids
        and id(p) not in filter_param_ids
    ]
    conv_params = [p for n, p in shared_blocks_named if id(p) in conv_weight_ids]

    for ls in base_model.layer_scalars:
        for p in ls.parameters():
            scalar_params.append(p)
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_filter = torch.optim.Adam(
        [{"params": filter_params, "lr": args.filter_lr, "base_lr": args.filter_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar, optimizer_filter]
    if conv_params:
        optimizer_conv = torch.optim.Adam(
            [{"params": conv_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.append(optimizer_conv)
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [
                {
                    "params": [base_model.lm_head.weight],
                    "lr": args.head_lr,
                    "base_lr": args.head_lr,
                }
            ],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)
    if base_model.structured_head is not None:
        structured_params = list(base_model.structured_head.parameters())
        if structured_params:
            optimizer_structured = torch.optim.Adam(
                [
                    {
                        "params": structured_params,
                        "lr": args.head_lr,
                        "base_lr": args.head_lr,
                    }
                ],
                betas=(args.beta1, args.beta2),
                eps=args.adam_eps,
                fused=True,
            )
            optimizers.insert(1, optimizer_structured)

    # Verify every trainable parameter is in exactly one optimizer.
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
    num_blocks = len(base_model.shared_blocks)
    log0(f"model_params:{n_params}")
    log0(
        f"architecture:hyena order:2 short_conv_kernel:{args.short_conv_kernel} num_poles:{args.num_poles}"
    )
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr} filter_lr:{args.filter_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"num_blocks:{num_blocks} block_map:{base_model.block_map}")
    log0(f"seed:{args.seed}")

    train_loader = DistributedTokenLoader(
        args.train_files, tok, rank, world_size, device
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
                x, y = train_loader.next_batch(
                    args.train_batch_tokens, args.train_seq_len, grad_accum_steps
                )
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
        train_loader = DistributedTokenLoader(
            args.train_files, tok, rank, world_size, device
        )

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
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(
                args.train_batch_tokens, args.train_seq_len, grad_accum_steps
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = (
            min(step / args.muon_momentum_warmup_steps, 1.0)
            if args.muon_momentum_warmup_steps > 0
            else 1.0
        )
        for group in optimizer_muon.param_groups:
            group["momentum"] = (
                1 - frac
            ) * args.muon_momentum_warmup_start + frac * args.muon_momentum
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
            tgt_bytes = base_bytes_lut[y.reshape(-1)].to(torch.float64).sum().item()
            tpb = float(y.numel()) / max(tgt_bytes, 1.0)
            log0(
                f"step:{step}/{args.iterations} train_loss:{tl:.4f} train_bpb:{tl / math.log(2.0) * tpb:.4f} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
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
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

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
    q_val_loss, q_val_bpb = eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(
        f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}"
    )
    log0(f"run_id: {args.run_id} | run_dir: {run_dir}/")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
