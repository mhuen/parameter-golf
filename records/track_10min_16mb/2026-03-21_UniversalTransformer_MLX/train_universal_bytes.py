"""
Universal Transformer GPT — byte-level tokenizer variant.

Same model architecture as train_universal.py, but uses EfficientByteTokenizer
instead of SentencePiece. Each token represents exactly one UTF-8 byte, so
BPB = bits_per_token (no subword-to-byte accounting needed).

Tokenizer configs (via env vars):
    DISCARD_UNUSED_BYTES=1  (default) 206 used bytes, vocab=208
    DISCARD_UNUSED_BYTES=0           all 256 bytes, vocab=258
    FOLD=uppercase                   fold uppercase->lowercase, vocab=182

Data: expects PureByteTokenizer .bin shards (4 special + 256 bytes = 260 vocab).
Token IDs are remapped on-the-fly to EfficientByteTokenizer IDs at load time.
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
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model
    # vocab_size is set dynamically from the tokenizer (see main()).
    num_layers = int(os.environ.get("NUM_LAYERS", 27))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    block_pattern = os.environ.get(
        "BLOCK_PATTERN",
        "0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8",
    )

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

    conv_kernel_size = int(os.environ.get("CONV_KERNEL_SIZE", 4))
    conv_groups = int(os.environ.get("CONV_GROUPS", 0))
    conv_shared = bool(int(os.environ.get("CONV_SHARED", "0")))
    conv_enabled = bool(int(os.environ.get("CONV_ENABLED", "1")))
    conv_lr = float(os.environ.get("CONV_LR", 0.01))

    rope_dim_fraction = float(os.environ.get("ROPE_DIM_FRACTION", 1.0))

    pack_doc_mask = bool(int(os.environ.get("PACK_DOC_MASK", "0")))

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
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    if max_tokens > 0:
        tokens = tokens[: max_tokens + 1]
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


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
            doc_mask = build_doc_mask(x, BOS_ID) if args.pack_doc_mask else None
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y, doc_mask=doc_mask).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            # For byte-level: each non-special token is exactly 1 byte.
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
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,conv_scale",
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

# PureByteTokenizer shard format: 4 special tokens (pad=0, bos=1, eos=2, unk=3),
# byte values offset by 4. We need to remap to EfficientByteTokenizer IDs.
_PURE_BYTE_OFFSET = 4
_PURE_BYTE_BOS = 1


def _build_shard_remap_lut(tok: EfficientByteTokenizer) -> np.ndarray:
    """Build a LUT that maps PureByteTokenizer token IDs -> EfficientByteTokenizer IDs.

    PureByteTokenizer layout: [pad=0, bos=1, eos=2, unk=3, byte_0=4, ..., byte_255=259]
    The LUT has 260 entries.
    """
    lut = np.full(260, tok.pad_id, dtype=np.uint16)  # unmapped -> pad
    lut[_PURE_BYTE_BOS] = tok.bos_id  # BOS -> BOS
    # Map each byte: PureByteTokenizer ID (byte+4) -> EfficientByteTokenizer ID
    for byte_val in range(256):
        eff_id = int(tok._byte_to_id[byte_val])
        if eff_id != 0xFFFF:  # not a drop marker
            lut[_PURE_BYTE_OFFSET + byte_val] = eff_id
        else:
            lut[_PURE_BYTE_OFFSET + byte_val] = tok.pad_id
    return lut


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
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def remap_shard_tokens(tokens: torch.Tensor, remap_lut: np.ndarray) -> torch.Tensor:
    """Remap PureByteTokenizer IDs to EfficientByteTokenizer IDs using a numpy LUT."""
    tokens_np = tokens.numpy()
    remapped = remap_lut[tokens_np]
    return torch.from_numpy(remapped)


class TokenStream:
    def __init__(self, pattern, remap_lut: np.ndarray):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.remap_lut = remap_lut
        self.file_idx = 0
        self.tokens = remap_shard_tokens(load_data_shard(self.files[0]), remap_lut)
        self.pos = 0

    def _advance_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = remap_shard_tokens(
            load_data_shard(self.files[self.file_idx]), self.remap_lut
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
    def __init__(self, pattern, remap_lut, rank, world_size, device):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(pattern, remap_lut)

    def next_batch(self, global_tokens, seq_len, grad_accum_steps):
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        if local_tokens < seq_len:
            raise ValueError("TRAIN_BATCH_TOKENS too small for this world_size/seq_len")
        local_seqs = local_tokens // seq_len
        n = local_seqs * seq_len + 1
        # Skip other ranks' tokens.
        for _ in range(self.rank):
            self.stream.take(local_seqs * seq_len)
        local = self.stream.take(n)
        # Skip remaining ranks' tokens.
        for _ in range(self.rank + 1, self.world_size):
            self.stream.take(local_seqs * seq_len)
        x = local[:-1].reshape(local_seqs, seq_len).to(device=self.device, dtype=torch.int64, non_blocking=True)
        y = local[1:].reshape(local_seqs, seq_len).to(device=self.device, dtype=torch.int64, non_blocking=True)
        return x, y


# -----------------------------
# TRANSFORMER MODULES (Universal Transformer)
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


class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0, rope_dim_fraction=1.0):
        super().__init__()
        rope_dims = max(2, 2 * (int(dim * rope_dim_fraction) // 2))
        inv_freq = 1.0 / (
            base ** (torch.arange(0, rope_dims, 2, dtype=torch.float32) / rope_dims)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, seq_len, device, dtype):
        if (
            self._cos_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x, cos, sin):
    rope_dims = cos.size(-1) * 2
    if rope_dims >= x.size(-1):
        half = x.size(-1) // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
    x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
    half = rope_dims // 2
    x1, x2 = x_rope[..., :half], x_rope[..., half:]
    x_rotated = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
    return torch.cat((x_rotated, x_pass), dim=-1)


def build_doc_mask(input_ids: Tensor, bos_id: int) -> Tensor:
    """Build a causal block-diagonal attention mask from BOS document boundaries.
    Tokens only attend to earlier tokens within the same document."""
    bsz, seq_len = input_ids.shape
    doc_ids = (input_ids == bos_id).cumsum(dim=1)  # (B, S)
    same_doc = doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)  # (B, S, S)
    causal = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device)
    )
    return (same_doc & causal).unsqueeze(1)  # (B, 1, S, S)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        num_kv_heads,
        rope_base,
        qk_gain_init,
        rope_dim_fraction=1.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(
            torch.full((num_heads,), qk_gain_init, dtype=torch.float32)
        )
        self.rotary = Rotary(
            self.head_dim, base=rope_base, rope_dim_fraction=rope_dim_fraction
        )

    def forward(self, x, doc_mask=None):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)
        q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        if doc_mask is not None:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=doc_mask,
                enable_gqa=(self.num_kv_heads != self.num_heads),
            )
        else:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                is_causal=True,
                enable_gqa=(self.num_kv_heads != self.num_heads),
            )
        return self.proj(y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim))


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


class GatedCausalConv(nn.Module):
    """Gated causal conv1d for local n-gram mixing.
    gate = sigmoid(conv_gate(x)), value = SiLU(conv_value(x)), out = gate * value.
    Uses causal (left) padding so output[t] only depends on input[t-k+1..t]."""

    def __init__(self, dim, kernel_size=4, groups=0):
        super().__init__()
        groups = dim if groups <= 0 else groups
        if dim % groups != 0:
            raise ValueError(
                f"model_dim ({dim}) must be divisible by conv_groups ({groups})"
            )
        self.pad = kernel_size - 1
        self.conv_gate = nn.Conv1d(dim, dim, kernel_size, groups=groups, bias=False)
        self.conv_value = nn.Conv1d(dim, dim, kernel_size, groups=groups, bias=False)
        self.conv_value._zero_init = True

    def forward(self, x):
        h = x.transpose(1, 2)
        h = F.pad(h, (self.pad, 0))
        gate = torch.sigmoid(self.conv_gate(h))
        value = F.silu(self.conv_value(h))
        return (gate * value).transpose(1, 2)


class SharedBlock(nn.Module):
    """Shared transformer block: attention + MLP with norms. No per-layer scalars."""

    def __init__(
        self,
        dim,
        num_heads,
        num_kv_heads,
        mlp_mult,
        rope_base,
        qk_gain_init,
        rope_dim_fraction=1.0,
        conv_kernel_size=0,
        conv_groups=0,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim, num_heads, num_kv_heads, rope_base, qk_gain_init, rope_dim_fraction
        )
        self.mlp = MLP(dim, mlp_mult)
        self.conv = (
            GatedCausalConv(dim, conv_kernel_size, conv_groups)
            if conv_kernel_size > 0
            else None
        )
        self.conv_norm = RMSNorm()

    def forward(
        self,
        x,
        attn_scale,
        mlp_scale,
        conv=None,
        conv_scale=None,
        doc_mask=None,
    ):
        conv_mod = conv if conv is not None else self.conv
        if conv_mod is not None and conv_scale is not None:
            x = x + conv_scale.to(dtype=x.dtype)[None, None, :] * conv_mod(
                self.conv_norm(x)
            )
        n = self.attn_norm(x)
        attn_out = self.attn(n, doc_mask=doc_mask)
        x = x + attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class LayerScalars(nn.Module):
    """Per-layer scalars: attn_scale, mlp_scale, conv_scale, resid_mix.
    Optionally holds a per-layer GatedCausalConv when conv is not shared."""

    def __init__(self, dim, conv_kernel_size=0, conv_groups=0):
        super().__init__()
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(
            torch.stack((torch.ones(dim), torch.zeros(dim))).float()
        )
        self.conv = (
            GatedCausalConv(dim, conv_kernel_size, conv_groups)
            if conv_kernel_size > 0
            else None
        )
        self.conv_scale = (
            nn.Parameter(torch.ones(dim, dtype=torch.float32))
            if conv_kernel_size > 0
            else None
        )


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_layers,
        model_dim,
        num_heads,
        num_kv_heads,
        mlp_mult,
        tie_embeddings,
        tied_embed_init_std,
        logit_softcap,
        rope_base,
        qk_gain_init,
        block_pattern="",
        rope_dim_fraction=1.0,
        conv_kernel_size=0,
        conv_groups=0,
        conv_shared=False,
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

        shared_conv_ks = conv_kernel_size if conv_shared else 0
        per_layer_conv_ks = conv_kernel_size if not conv_shared else 0

        num_blocks = max(self.block_map) + 1
        self.shared_blocks = nn.ModuleList(
            [
                SharedBlock(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    rope_dim_fraction=rope_dim_fraction,
                    conv_kernel_size=shared_conv_ks,
                    conv_groups=conv_groups,
                )
                for _ in range(num_blocks)
            ]
        )

        self.layer_scalars = nn.ModuleList(
            [
                LayerScalars(
                    model_dim,
                    conv_kernel_size=per_layer_conv_ks,
                    conv_groups=conv_groups,
                )
                for _ in range(num_layers)
            ]
        )
        self.conv_enabled = conv_kernel_size > 0
        if conv_shared and conv_kernel_size > 0:
            self.shared_conv_scales = nn.ParameterList(
                [
                    nn.Parameter(torch.ones(model_dim, dtype=torch.float32))
                    for _ in range(num_layers)
                ]
            )
        else:
            self.shared_conv_scales = None

        self.final_norm = RMSNorm()
        self.lm_head = (
            None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        )
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)) and getattr(
                module, "_zero_init", False
            ):
                nn.init.zeros_(module.weight)

    def _get_conv_args(self, ls, layer_idx):
        if not self.conv_enabled:
            return None, None
        if ls.conv is not None:
            return ls.conv, ls.conv_scale
        return None, self.shared_conv_scales[layer_idx]

    def forward(self, input_ids, target_ids, doc_mask=None):
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []

        for i in range(self.num_encoder_layers):
            ls = self.layer_scalars[i]
            mix = ls.resid_mix.to(dtype=x.dtype)
            x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
            conv, conv_scale = self._get_conv_args(ls, i)
            x = self.shared_blocks[self.block_map[i]](
                x,
                ls.attn_scale,
                ls.mlp_scale,
                conv=conv,
                conv_scale=conv_scale,
                doc_mask=doc_mask,
            )
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
            conv, conv_scale = self._get_conv_args(ls, layer_idx)
            x = self.shared_blocks[self.block_map[layer_idx]](
                x,
                ls.attn_scale,
                ls.mlp_scale,
                conv=conv,
                conv_scale=conv_scale,
                doc_mask=doc_mask,
            )
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            reduction="mean",
        )


# -----------------------------
# TRAINING
# -----------------------------

BOS_ID = 1


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
    from torch.backends.cuda import (
        enable_cudnn_sdp,
        enable_flash_sdp,
        enable_math_sdp,
        enable_mem_efficient_sdp,
    )

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(args.pack_doc_mask)
    enable_math_sdp(args.pack_doc_mask)

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
    remap_lut = _build_shard_remap_lut(tok)
    log0(f"tokenizer: {tok.describe()}")

    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))

    # Remap validation tokens
    raw_val_tokens = load_validation_tokens(
        args.val_files, args.train_seq_len, args.val_max_tokens
    )
    val_tokens = remap_shard_tokens(raw_val_tokens, remap_lut)

    base_bytes_lut = build_byte_bpb_lut(tok, device)
    log0(f"val_bpb:enabled tokenizer_kind=efficient_byte vocab_size={vocab_size}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")

    base_model = (
        GPT(
            vocab_size=vocab_size,
            num_layers=args.num_layers,
            model_dim=args.model_dim,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            mlp_mult=args.mlp_mult,
            tie_embeddings=args.tie_embeddings,
            tied_embed_init_std=args.tied_embed_init_std,
            logit_softcap=args.logit_softcap,
            rope_base=args.rope_base,
            qk_gain_init=args.qk_gain_init,
            block_pattern=args.block_pattern,
            rope_dim_fraction=args.rope_dim_fraction,
            conv_kernel_size=args.conv_kernel_size if args.conv_enabled else 0,
            conv_groups=args.conv_groups,
            conv_shared=args.conv_shared,
        )
        .to(device)
        .bfloat16()
    )
    for module in base_model.modules():
        if isinstance(module, (CastedLinear, nn.Conv1d)):
            module.float()
        if isinstance(module, Rotary):
            module.inv_freq.data = module.inv_freq.data.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model = (
        DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False)
        if distributed
        else compiled_model
    )

    # Optimizer: shared_blocks 2D (non-conv) -> Muon, everything else -> Adam
    conv_weight_ids = {
        id(p)
        for m in base_model.modules()
        if isinstance(m, nn.Conv1d)
        for p in m.parameters()
    }
    shared_blocks_named = list(base_model.shared_blocks.named_parameters())
    matrix_params = [
        p
        for n, p in shared_blocks_named
        if p.ndim == 2 and not any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for n, p in shared_blocks_named
        if (p.ndim != 2 or any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS))
        and id(p) not in conv_weight_ids
    ]
    conv_params = [p for n, p in shared_blocks_named if id(p) in conv_weight_ids]
    for ls in base_model.layer_scalars:
        for p in ls.parameters():
            if id(p) in conv_weight_ids:
                conv_params.append(p)
            else:
                scalar_params.append(p)
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    if base_model.shared_conv_scales is not None:
        for p in base_model.shared_conv_scales:
            scalar_params.append(p)

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
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if conv_params:
        optimizer_conv = torch.optim.Adam(
            [{"params": conv_params, "lr": args.conv_lr, "base_lr": args.conv_lr}],
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

    n_params = sum(p.numel() for p in base_model.parameters())
    num_blocks = len(base_model.shared_blocks)
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(
        f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads} pack_doc_mask:{args.pack_doc_mask}"
    )
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"num_blocks:{num_blocks} block_map:{base_model.block_map}")
    log0(f"seed:{args.seed}")

    train_loader = DistributedTokenLoader(
        args.train_files, remap_lut, rank, world_size, device
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
                doc_mask = build_doc_mask(x, BOS_ID) if args.pack_doc_mask else None
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    warmup_loss = model(x, y, doc_mask=doc_mask)
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
            args.train_files, remap_lut, rank, world_size, device
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
            doc_mask = build_doc_mask(x, BOS_ID) if args.pack_doc_mask else None
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y, doc_mask=doc_mask)
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
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
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
