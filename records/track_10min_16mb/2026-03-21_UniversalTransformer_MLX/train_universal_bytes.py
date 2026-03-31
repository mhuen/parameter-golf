"""
Universal Transformer GPT — byte-level tokenizer variant.

Same model architecture as train_universal.py, but uses EfficientByteTokenizer
instead of SentencePiece. Each token represents exactly one UTF-8 byte, so
BPB = bits_per_token (no subword-to-byte accounting needed).

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
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model
    # vocab_size is set dynamically from the tokenizer (see main()).
    num_layers = int(os.environ.get("NUM_LAYERS", 27))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    structured_output_logits = bool(
        int(os.environ.get("STRUCTURED_OUTPUT_LOGITS", "0"))
    )
    utf8_prior = bool(int(os.environ.get("UTF8_PRIOR", "0")))
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
    muon_gram_ns = bool(int(os.environ.get("MUON_GRAM_NS", "0")))
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

    pack_doc_mask = bool(
        int(os.environ.get("PACK_DOC_MASK", "0"))
    )  # broken with byte shards — see NotImplementedError below
    use_fa4 = bool(int(os.environ.get("USE_FA4", "0")))

    # Learnable category attention mask: off, bias, or lora
    catmask_mode = os.environ.get("CATMASK_MODE", "off")  # off | bias | lora
    catmask_lr = float(os.environ.get("CATMASK_LR", 0.04))
    catmask_rank = int(os.environ.get("CATMASK_RANK", 1))
    catmask_lora_v = bool(int(os.environ.get("CATMASK_LORA_V", "0")))

    # Byte tokenizer config
    discard_unused_bytes = bool(int(os.environ.get("DISCARD_UNUSED_BYTES", "1")))
    fold = os.environ.get("FOLD", "")  # comma-separated ByteCategory values


# -----------------------------
# Flash Attention 4 (optional, requires Hopper/Blackwell + flash-attn-4 package)
try:
    from flash_attn.cute import flash_attn_func as _fa4_func
    from flash_attn.cute import flash_attn_varlen_func as _fa4_varlen_func

    _FA4_AVAILABLE = True
except ImportError:
    _FA4_AVAILABLE = False


# MUON OPTIMIZER
# -----------------------------
# Gram Newton-Schulz: https://dao-lab.ai/blog/2026/gram-newton-schulz/

# Try to import Dao-AILab's optimized Gram Newton-Schulz (requires Hopper/Blackwell + CUDA 12.9+)
try:
    from gram_newton_schulz import GramNewtonSchulz, POLAR_EXPRESS_COEFFICIENTS

    _gram_ns_op = GramNewtonSchulz(
        ns_coefficients=POLAR_EXPRESS_COEFFICIENTS,
        gram_newton_schulz_reset_iterations=[2],
    )
    print("Imported optimized Gram Newton-Schulz from Dao-AILab.")
    _GRAM_NS_LIB = True
except ImportError:
    _GRAM_NS_LIB = False
    print(
        "Could not import optimized Gram Newton-Schulz from Dao-AILab; falling back to pure PyTorch implementation."
    )

# Polar Express coefficients for pure-PyTorch Gram NS fallback.
_GRAM_NS_COEFFS = [
    (8.123737, -22.232240, 16.373715),
    (4.026529, -2.776323, 0.514551),
    (3.870284, -2.739120, 0.520999),
    (3.253351, -2.343223, 0.481420),
    (2.300652, -1.668904, 0.418807),
]


def _zeropower_standard_ns5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
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


def _zeropower_gram_ns5(G: Tensor, eps: float = 1e-7) -> Tensor:
    """Stabilized Gram Newton-Schulz: iterates on the n×n Gram matrix instead of
    the full n×m rectangle.  ~42-58 % fewer FLOPs for rectangular matrices.
    Falls back to standard NS for square matrices (no benefit)."""
    if G.size(0) == G.size(1):
        return _zeropower_standard_ns5(G, eps=eps)
    X = G.half()  # Gram NS uses fp16, not bf16
    X /= X.norm() + eps
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    n = X.size(0)
    R = X @ X.T
    Q = torch.eye(n, device=X.device, dtype=X.dtype)
    for t in range(5):
        if t == 2:  # restart after iteration 2 for numerical stability
            X = Q @ X
            R = X @ X.T
            Q = torch.eye(n, device=X.device, dtype=X.dtype)
        a, b, c = _GRAM_NS_COEFFS[t]
        Z = b * R + c * R @ R
        Q = Q @ Z + a * Q
        RZ = R @ Z + a * R
        R = Z @ RZ + a * RZ
    X = Q @ X
    return X.T if transposed else X


def zeropower_via_newtonschulz5(
    G: Tensor, steps: int = 10, eps: float = 1e-7, gram_ns: bool = False
) -> Tensor:
    if gram_ns and G.size(0) != G.size(1):
        if _GRAM_NS_LIB:
            return _gram_ns_op(G)
        return _zeropower_gram_ns5(G, eps=eps)
    return _zeropower_standard_ns5(G, steps=steps, eps=eps)


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        momentum: float,
        backend_steps: int,
        nesterov: bool = True,
        gram_ns: bool = False,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                backend_steps=backend_steps,
                nesterov=nesterov,
                gram_ns=gram_ns,
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
            lr, momentum, backend_steps, nesterov, gram_ns = (
                group["lr"],
                group["momentum"],
                group["backend_steps"],
                group["nesterov"],
                group["gram_ns"],
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
                    g = zeropower_via_newtonschulz5(
                        g, steps=backend_steps, gram_ns=gram_ns
                    )
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
            if args.use_fa4 and _FA4_AVAILABLE and args.pack_doc_mask:
                cu_seqlens, max_seqlen = build_cu_seqlens(x, BOS_ID, pad_to=cu_seqlens_budget)
            else:
                cu_seqlens, max_seqlen = None, None
            doc_mask = (
                build_doc_mask(x, BOS_ID)
                if args.pack_doc_mask and cu_seqlens is None
                else None
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(
                    x,
                    y,
                    doc_mask=doc_mask,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                ).detach()
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
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,conv_scale,cat_attn_logits,cat_lora_down,cat_q_up,cat_k_up,cat_v_up",
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
    """Convert raw byte values to EfficientByteTokenizer IDs.

    Uses tok.remap_byte_array() for the LUT lookup, then tok.filter_stream()
    to apply the OtherTokenStrategy (e.g. drop unused bytes).
    """
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
        # Skip other ranks' tokens.
        for _ in range(self.rank):
            self.stream.take(local_seqs * seq_len)
        local = self.stream.take(n)
        # Skip remaining ranks' tokens.
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


def build_cu_seqlens(input_ids: Tensor, bos_id: int, pad_to: int = 0) -> tuple[Tensor, int]:
    """BOS boundaries → cu_seqlens for flash_attn_varlen_func. O(S) memory.

    Args:
        pad_to: if > 0, pad cu_seqlens to this fixed length with zero-length
                phantom segments (terminal value repeated). Keeps tensor shape
                constant for torch.compile with dynamic=False.

    Returns:
        cu_seqlens: (num_docs+1,) or (pad_to,) int32 cumulative sequence lengths.
        max_seqlen: S (safe upper bound; avoids GPU→CPU sync).
    """
    B, S = input_ids.shape
    flat = input_ids.reshape(-1)
    total = B * S
    bos_pos = torch.where(flat == bos_id)[0].to(torch.int32)
    batch_starts = torch.arange(0, total, S, device=input_ids.device, dtype=torch.int32)
    all_starts = torch.unique(torch.cat([bos_pos, batch_starts]), sorted=True)
    cu_seqlens = torch.cat(
        [all_starts, torch.tensor([total], device=input_ids.device, dtype=torch.int32)]
    )
    if pad_to > 0 and cu_seqlens.size(0) < pad_to:
        padding = cu_seqlens.new_full((pad_to - cu_seqlens.size(0),), total)
        cu_seqlens = torch.cat([cu_seqlens, padding])
    return cu_seqlens, S


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        num_kv_heads,
        rope_base,
        qk_gain_init,
        rope_dim_fraction=1.0,
        use_fa4=False,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if use_fa4 and not _FA4_AVAILABLE:
            raise RuntimeError("USE_FA4=1 but flash-attn-4 is not installed")
        self.use_fa4 = use_fa4
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

    def forward(
        self,
        x,
        doc_mask=None,
        attn_bias=None,
        q_delta=None,
        k_delta=None,
        v_delta=None,
        cu_seqlens=None,
        max_seqlen=None,
    ):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)
        # Category-conditional LoRA deltas (applied before reshape/norm)
        if q_delta is not None:
            q = q + q_delta
        if k_delta is not None:
            k = k + k_delta
        if v_delta is not None:
            v = v + v_delta
        q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        if cu_seqlens is not None:
            # FA4 varlen path: (B,H,S,D) → (B*S, H, D)
            q_fa = q.transpose(1, 2).reshape(
                bsz * seqlen, self.num_heads, self.head_dim
            )
            k_fa = k.transpose(1, 2).reshape(
                bsz * seqlen, self.num_kv_heads, self.head_dim
            )
            v_fa = v.transpose(1, 2).reshape(
                bsz * seqlen, self.num_kv_heads, self.head_dim
            )
            y = _fa4_varlen_func(
                q_fa, k_fa, v_fa,
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                causal=True,
            )[0].reshape(bsz, seqlen, dim)
        elif self.use_fa4 and attn_bias is None:
            # FA4 simple causal (no attn_bias support — catmask_mode=bias uses SDPA)
            y = _fa4_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=True
            )[0].reshape(bsz, seqlen, dim)
        elif doc_mask is not None or attn_bias is not None:
            # Build explicit causal + bias mask for non-flash path
            mask = torch.zeros(1, 1, seqlen, seqlen, device=x.device, dtype=q.dtype)
            causal = torch.triu(
                torch.full(
                    (seqlen, seqlen), float("-inf"), device=x.device, dtype=q.dtype
                ),
                diagonal=1,
            )
            mask = mask + causal
            if doc_mask is not None:
                mask = mask + torch.where(doc_mask, 0.0, float("-inf"))
            if attn_bias is not None:
                mask = mask + attn_bias
            y = (
                F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=mask,
                    enable_gqa=(self.num_kv_heads != self.num_heads),
                )
                .transpose(1, 2)
                .contiguous()
                .reshape(bsz, seqlen, dim)
            )
        else:
            y = (
                F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    is_causal=True,
                    enable_gqa=(self.num_kv_heads != self.num_heads),
                )
                .transpose(1, 2)
                .contiguous()
                .reshape(bsz, seqlen, dim)
            )
        return self.proj(y)


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


NUM_BYTE_CATEGORIES = 8
_BYTE_CATEGORY_DEFS = [
    ByteCategory.BOS,
    ByteCategory.PAD,
    ByteCategory.DIGIT,
    ByteCategory.LETTER,
    ByteCategory.SEPARATOR,
    ByteCategory.PUNCTUATION,
    ByteCategory.SYMBOL,
    ByteCategory.MULTIBYTE,
]


class CategoryAttnBias(nn.Module):
    """Token→category LUT shared by both catmask modes.

    - bias mode: computes (B, H, S, S) additive attention bias from (H, C, C) logits.
    - lora mode: provides cat_ids for the LoRA gather (computation in SharedBlock).
    """

    def __init__(self, tok: EfficientByteTokenizer):
        super().__init__()
        V = tok.vocab_size
        token_to_cat = torch.zeros(V, dtype=torch.long)
        for ci, cat in enumerate(_BYTE_CATEGORY_DEFS):
            mask = tok.mask(cat)
            for tid in range(V):
                if mask[tid]:
                    token_to_cat[tid] = ci
        self.register_buffer("token_to_cat", token_to_cat)

    def get_cat_ids(self, input_ids: Tensor) -> Tensor:
        """(B, S) token IDs → (B, S) category indices."""
        return self.token_to_cat[input_ids]

    def bias_forward(self, input_ids: Tensor, cat_attn_logits: Tensor) -> Tensor:
        """Compute (B, H, S, S) additive attention bias (bias mode only)."""
        cat_ids = self.token_to_cat[input_ids]  # (B, S)
        cat_oh = F.one_hot(cat_ids, NUM_BYTE_CATEGORIES).to(
            dtype=cat_attn_logits.dtype
        )  # (B, S, C)
        # (B, 1, S, C) @ (1, H, C, C) -> (B, H, S, C)
        q_contrib = cat_oh.unsqueeze(1) @ cat_attn_logits.unsqueeze(0)
        # (B, H, S, C) @ (B, 1, C, S) -> (B, H, S, S)
        return q_contrib @ cat_oh.unsqueeze(1).transpose(-1, -2)


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
        use_fa4=False,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim,
            num_heads,
            num_kv_heads,
            rope_base,
            qk_gain_init,
            rope_dim_fraction,
            use_fa4=use_fa4,
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
        attn_bias=None,
        cat_lora=None,
        cu_seqlens=None,
        max_seqlen=None,
    ):
        conv_mod = conv if conv is not None else self.conv
        if conv_mod is not None and conv_scale is not None:
            x = x + conv_scale.to(dtype=x.dtype)[None, None, :] * conv_mod(
                self.conv_norm(x)
            )
        n = self.attn_norm(x)
        # Compute category-conditional LoRA deltas from normed input
        q_delta, k_delta, v_delta = None, None, None
        if cat_lora is not None:
            cat_oh, down, q_up, k_up, v_up = cat_lora
            low = n @ down.to(dtype=n.dtype).T  # (B, S, r)

            # One-hot matmul: (B,S,C) @ (C, dim*r) -> (B,S, dim*r), then contract with low
            def _cat_lora_delta(up, low):
                C, out_dim, r = up.shape
                selected = cat_oh @ up.to(dtype=n.dtype).reshape(
                    C, -1
                )  # (B, S, out_dim*r)
                if r == 1:
                    return selected * low  # (B, S, out_dim) * (B, S, 1) broadcast
                return (
                    selected.reshape(-1, out_dim, r) @ low.reshape(-1, r, 1)
                ).reshape(cat_oh.shape[0], cat_oh.shape[1], out_dim)

            q_delta = _cat_lora_delta(q_up, low)
            k_delta = _cat_lora_delta(k_up, low)
            if v_up is not None:
                v_delta = _cat_lora_delta(v_up, low)
        attn_out = self.attn(
            n,
            doc_mask=doc_mask,
            attn_bias=attn_bias,
            q_delta=q_delta,
            k_delta=k_delta,
            v_delta=v_delta,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        x = x + attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class LayerScalars(nn.Module):
    """Per-layer scalars: attn_scale, mlp_scale, conv_scale, resid_mix.
    Optionally holds a per-layer GatedCausalConv when conv is not shared."""

    def __init__(
        self,
        dim,
        num_heads=0,
        num_kv_heads=0,
        conv_kernel_size=0,
        conv_groups=0,
        catmask_mode="off",
        catmask_rank=1,
        catmask_lora_v=False,
    ):
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
        # Catmask: bias mode — per-head (C, C) attention score bias
        self.cat_attn_logits = (
            nn.Parameter(
                torch.zeros(
                    num_heads,
                    NUM_BYTE_CATEGORIES,
                    NUM_BYTE_CATEGORIES,
                    dtype=torch.float32,
                )
            )
            if catmask_mode == "bias"
            else None
        )
        # Catmask: lora mode — category-conditional low-rank Q/K update
        C = NUM_BYTE_CATEGORIES
        head_dim = dim // num_heads if num_heads > 0 else 0
        kv_dim = num_kv_heads * head_dim
        r = catmask_rank
        if catmask_mode == "lora":
            # down: shared projection (random init — gradient flows via zero-init up)
            self.cat_lora_down = nn.Parameter(
                torch.randn(r, dim, dtype=torch.float32) * (1.0 / dim**0.5)
            )
            # up: per-category projections (zero init — output is zero at start)
            self.cat_q_up = nn.Parameter(torch.zeros(C, dim, r, dtype=torch.float32))
            self.cat_k_up = nn.Parameter(torch.zeros(C, kv_dim, r, dtype=torch.float32))
            self.cat_v_up = (
                nn.Parameter(torch.zeros(C, kv_dim, r, dtype=torch.float32))
                if catmask_lora_v
                else None
            )
        else:
            self.cat_lora_down = None
            self.cat_q_up = None
            self.cat_k_up = None
            self.cat_v_up = None


class UTF8Prior(nn.Module):
    """Precomputed UTF-8 structural prior masks (no learnable parameters).

    Given input_ids, computes position-dependent masks that zero out tokens
    impossible under UTF-8 encoding rules.  All operations are parallel
    (bounded lookback of 3, no sequential scan).

    Two masks are returned:
      cat_mask   (B, S, num_categories) — additive 0/-inf for structured head level-0
      token_mask (B, S, V)              — additive 0/-inf for final logits/log-probs
    """

    # Byte-type enum (plain ints — torch.compile friendly)
    BT_BOS = 0
    BT_PAD = 1
    BT_ASCII = 2  # digit, letter, separator, punctuation, symbol
    BT_LEAD_2 = 3
    BT_LEAD_3 = 4
    BT_LEAD_4 = 5
    BT_CONT = 6

    # State enum
    ST_READY = 0  # expect ASCII / leading / special (no continuation)
    ST_EXPECT_CONT = 1  # expect continuation byte
    ST_UNSYNCED = 2  # unknown state (chunk boundary) — no constraint

    def __init__(self, tok, num_categories=8, multibyte_cat_idx=7):
        super().__init__()
        V = tok.vocab_size
        self.num_categories = num_categories
        NEG_INF = float("-inf")

        # ---- token_id → byte type & byte value ----
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

        # ---- lead type → expected continuation count ----
        lead_expected = torch.zeros(7, dtype=torch.long)
        lead_expected[self.BT_LEAD_2] = 1
        lead_expected[self.BT_LEAD_3] = 2
        lead_expected[self.BT_LEAD_4] = 3
        self.register_buffer("lead_expected", lead_expected)

        # ---- state → category mask (3, num_categories) ----
        state_cat_mask = torch.zeros(3, num_categories)
        # READY: all categories valid (no constraint)
        # EXPECT_CONT: only multibyte (idx 7) valid
        for ci in range(num_categories):
            if ci != multibyte_cat_idx:
                state_cat_mask[self.ST_EXPECT_CONT, ci] = NEG_INF
        # UNSYNCED: all valid
        self.register_buffer("state_cat_mask", state_cat_mask)

        # ---- state → token mask (3, V) ----
        cont_np = tok.mask(ByteCategory.MB_CONTINUATION)
        state_token_mask = torch.zeros(3, V)
        # READY: forbid continuation bytes
        for tid in range(V):
            if cont_np[tid]:
                state_token_mask[self.ST_READY, tid] = NEG_INF
        # EXPECT_CONT: only continuation bytes valid
        for tid in range(V):
            if not cont_np[tid]:
                state_token_mask[self.ST_EXPECT_CONT, tid] = NEG_INF
        # UNSYNCED: all valid
        self.register_buffer("state_token_mask", state_token_mask)

        # ---- special lead byte constraints (first cont only) ----
        # After 0xE0: first cont must be 0xA0–0xBF (prevent overlong 3-byte)
        # After 0xED: first cont must be 0x80–0x9F (prevent surrogates)
        # After 0xF0: first cont must be 0x90–0xBF (prevent overlong 4-byte)
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
        """Compute UTF-8 structural prior masks from input_ids.

        The mask at position t constrains the *prediction* at position t
        (i.e. target token t), based on the UTF-8 state after consuming
        input_ids[t].  Only causal information (positions ≤ t) is used.

        Args:
            input_ids: (B, S) token IDs with BOS at position 0.

        Returns:
            cat_mask:   (B, S, num_categories) additive mask (0.0 or -inf)
            token_mask: (B, S, V) additive mask (0.0 or -inf)
        """
        B, S = input_ids.shape
        device = input_ids.device

        # Step 1: classify each input token
        byte_type = self.token_byte_type[input_ids]  # (B, S) long
        byte_val = self.token_byte_value[input_ids]  # (B, S) long

        # Step 2: continuation count via bounded lookback (max 3)
        is_cont = byte_type == self.BT_CONT  # (B, S) bool
        c1 = is_cont
        c2 = torch.zeros(B, S, device=device, dtype=torch.bool)
        c2[:, 1:] = is_cont[:, 1:] & is_cont[:, :-1]
        c3 = torch.zeros(B, S, device=device, dtype=torch.bool)
        c3[:, 2:] = is_cont[:, 2:] & is_cont[:, 1:-1] & is_cont[:, :-2]
        cont_count = c1.long() + c2.long() + c3.long()  # (B, S), 0-3

        # Step 3: find lead byte via lookback
        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        lead_pos = (positions - cont_count).clamp(min=0)
        lead_type = byte_type.gather(1, lead_pos)

        # Step 4: remaining continuations expected
        non_cont_remaining = self.lead_expected[byte_type]
        cont_remaining = self.lead_expected[lead_type] - cont_count
        remaining = torch.where(is_cont, cont_remaining, non_cont_remaining)

        # Step 5: state assignment
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
        )  # (B, S) long

        # Step 6: gather base masks by state
        cat_mask = self.state_cat_mask[state]  # (B, S, num_categories)
        token_mask = self.state_token_mask[state]  # (B, S, V)

        # Step 7: special lead byte refinements (first cont after E0/ED/F0)
        # These apply when byte_type[t] is LEAD_3/LEAD_4 and byte value is special
        is_e0 = (byte_type == self.BT_LEAD_3) & (byte_val == 0xE0)
        is_ed = (byte_type == self.BT_LEAD_3) & (byte_val == 0xED)
        is_f0 = (byte_type == self.BT_LEAD_4) & (byte_val == 0xF0)
        token_mask = torch.where(is_e0.unsqueeze(-1), self.special_e0, token_mask)
        token_mask = torch.where(is_ed.unsqueeze(-1), self.special_ed, token_mask)
        token_mask = torch.where(is_f0.unsqueeze(-1), self.special_f0, token_mask)

        return cat_mask, token_mask


class StructuredOutputHead(nn.Module):
    """Hierarchical softmax output head based on ByteCategory tree.

    Decomposes token log-probability as a sum of log-normalized levels:
      log p(token) = log p(category) + sum(log fraction_sub_i) + log fraction_leaf

    Each level is independently log-softmax normalized.  Softcap is applied
    only to leaf-level logits.
    """

    def __init__(self, model_dim, vocab_size, tok, logit_softcap):
        super().__init__()
        self.logit_softcap = logit_softcap
        self.vocab_size = vocab_size

        # ---- Gather token-ID sets from the tokenizer ----
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

        # ---- Build levels (heads + index/mask buffers) ----
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

        # Level: category (8-way, all tokens)
        cat_map: dict[int, int] = {}
        all_tids: set[int] = set()
        for ci, (name, _) in enumerate(cat_defs):
            for tid in cat_id_sets[name]:
                cat_map[tid] = ci
                all_tids.add(tid)
        _add_level(len(cat_defs), cat_map, all_tids, leaf=False)

        # Level: letter case (upper=0, lower=1)
        letter_ids = cat_id_sets["letter"]
        if upper_ids and lower_ids:
            _add_level(
                2,
                {tid: (0 if tid in upper_ids else 1) for tid in letter_ids},
                letter_ids,
                leaf=False,
            )

        # Level: vowel/consonant for uppercase (vowel=0, consonant=1)
        if upper_ids and (vowel_ids & upper_ids) and (consonant_ids & upper_ids):
            _add_level(
                2,
                {tid: (0 if tid in vowel_ids else 1) for tid in upper_ids},
                upper_ids,
                leaf=False,
            )

        # Level: vowel/consonant for lowercase (vowel=0, consonant=1)
        if lower_ids and (vowel_ids & lower_ids) and (consonant_ids & lower_ids):
            _add_level(
                2,
                {tid: (0 if tid in vowel_ids else 1) for tid in lower_ids},
                lower_ids,
                leaf=False,
            )

        # Level: multibyte type (continuation=0, leading=1)
        mb_ids = cat_id_sets["multibyte"]
        if mb_cont_ids and mb_leading_ids:
            _add_level(
                2,
                {tid: (0 if tid in mb_cont_ids else 1) for tid in mb_ids},
                mb_ids,
                leaf=False,
            )

        # Level: multibyte lead type (lead2=0, lead3=1, lead4=2)
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

        # Leaf levels (skip groups with <=1 token — singletons need no leaf head)
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

        # Assert every token appears in exactly one leaf group
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

        # ---- Store as module attributes ----
        self.heads = nn.ModuleList(heads)
        self.register_buffer(
            "level_indices", torch.stack(all_indices)
        )  # (num_levels, V)
        self.register_buffer("level_masks", torch.stack(all_masks))  # (num_levels, V)

    def forward(self, x, cat_prior=None, token_prior=None):
        """Return (B, S, V) log-probabilities assembled from the hierarchy.

        Args:
            x: (B, S, D) hidden states.
            cat_prior: (B, S, num_categories) additive mask for level-0 logits.
            token_prior: (B, S, V) additive mask for final log-probs.
        """
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
            # Renormalize: nll_loss expects valid log-probs summing to 1
            log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        return log_p


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
        structured_output_logits=False,
        utf8_prior=False,
        catmask_mode="off",
        catmask_rank=1,
        catmask_lora_v=False,
        tok=None,
        use_fa4=False,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.num_layers = num_layers
        self.catmask_mode = catmask_mode
        self.use_fa4 = use_fa4
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
                    use_fa4=use_fa4,
                )
                for _ in range(num_blocks)
            ]
        )

        self.layer_scalars = nn.ModuleList(
            [
                LayerScalars(
                    model_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    conv_kernel_size=per_layer_conv_ks,
                    conv_groups=conv_groups,
                    catmask_mode=catmask_mode,
                    catmask_rank=catmask_rank,
                    catmask_lora_v=catmask_lora_v,
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
        # UTF-8 structural prior (no learnable params, just buffers)
        if utf8_prior:
            assert tok is not None, "utf8_prior requires tok"
            self.utf8_prior_mod = UTF8Prior(tok)
        else:
            self.utf8_prior_mod = None
        # Learnable category attention (token→category LUT, no learnable params here)
        if catmask_mode != "off":
            assert tok is not None, "catmask requires tok"
            self.cat_attn_mod = CategoryAttnBias(tok)
        else:
            self.cat_attn_mod = None
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

    def _get_catmask_args(self, ls, input_ids, cat_oh):
        """Return (attn_bias, cat_lora) — one or both will be None."""
        if self.cat_attn_mod is None:
            return None, None
        if self.catmask_mode == "bias" and ls.cat_attn_logits is not None:
            return self.cat_attn_mod.bias_forward(input_ids, ls.cat_attn_logits), None
        if self.catmask_mode == "lora" and ls.cat_lora_down is not None:
            return None, (
                cat_oh,
                ls.cat_lora_down,
                ls.cat_q_up,
                ls.cat_k_up,
                ls.cat_v_up,
            )
        return None, None

    def forward(
        self, input_ids, target_ids, doc_mask=None, cu_seqlens=None, max_seqlen=None
    ):
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []

        # Precompute category one-hot once for all layers (lora mode)
        cat_oh = None
        if self.catmask_mode == "lora" and self.cat_attn_mod is not None:
            cat_ids = self.cat_attn_mod.get_cat_ids(input_ids)
            cat_oh = F.one_hot(cat_ids, NUM_BYTE_CATEGORIES).to(dtype=x.dtype)

        for i in range(self.num_encoder_layers):
            ls = self.layer_scalars[i]
            mix = ls.resid_mix.to(dtype=x.dtype)
            x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
            conv, conv_scale = self._get_conv_args(ls, i)
            attn_bias, cat_lora = self._get_catmask_args(ls, input_ids, cat_oh)
            x = self.shared_blocks[self.block_map[i]](
                x,
                ls.attn_scale,
                ls.mlp_scale,
                conv=conv,
                conv_scale=conv_scale,
                doc_mask=doc_mask,
                attn_bias=attn_bias,
                cat_lora=cat_lora,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
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
            attn_bias, cat_lora = self._get_catmask_args(ls, input_ids, cat_oh)
            x = self.shared_blocks[self.block_map[layer_idx]](
                x,
                ls.attn_scale,
                ls.mlp_scale,
                conv=conv,
                conv_scale=conv_scale,
                doc_mask=doc_mask,
                attn_bias=attn_bias,
                cat_lora=cat_lora,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        x = self.final_norm(x)
        # Compute UTF-8 prior masks (purely from input_ids, causal)
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
    global zeropower_via_newtonschulz5, _zeropower_standard_ns5, _zeropower_gram_ns5
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    _zeropower_standard_ns5 = torch.compile(_zeropower_standard_ns5)
    if args.muon_gram_ns and not _GRAM_NS_LIB:
        _zeropower_gram_ns5 = torch.compile(_zeropower_gram_ns5)

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

    _needs_explicit_mask = args.pack_doc_mask or args.catmask_mode == "bias"
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(_needs_explicit_mask)
    enable_math_sdp(_needs_explicit_mask)

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

    # Remap validation tokens: raw byte values -> EfficientByteTokenizer IDs
    raw_val_bytes = load_validation_tokens(
        args.val_files, args.train_seq_len, args.val_max_tokens
    )
    val_tokens = remap_shard_tokens(raw_val_bytes, tok)

    base_bytes_lut = build_byte_bpb_lut(tok, device)
    log0(f"val_bpb:enabled tokenizer_kind=efficient_byte vocab_size={vocab_size}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")

    if args.pack_doc_mask:
        raise NotImplementedError(
            "PACK_DOC_MASK is broken with byte shards: BOS tokens are stripped "
            "during remap_byte_array (byte value 1 is in _UNUSED_BYTES) so "
            "build_doc_mask never finds document boundaries. The mask collapses "
            "to a plain causal mask, adding overhead with no effect."
        )
    if args.structured_output_logits:
        args.tie_embeddings = False
        log0("structured_output_logits:enabled (forcing tie_embeddings=False)")
    if args.utf8_prior:
        log0("utf8_prior:enabled")
    if args.catmask_mode != "off":
        log0(
            f"catmask:mode={args.catmask_mode} rank={args.catmask_rank} lr={args.catmask_lr}"
        )

    _needs_tok = (
        args.structured_output_logits or args.utf8_prior or args.catmask_mode != "off"
    )
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
            structured_output_logits=args.structured_output_logits,
            utf8_prior=args.utf8_prior,
            catmask_mode=args.catmask_mode,
            catmask_rank=args.catmask_rank,
            catmask_lora_v=args.catmask_lora_v,
            tok=tok if _needs_tok else None,
            use_fa4=args.use_fa4,
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
    compiled_model = torch.compile(
        base_model, dynamic=False, fullgraph=not args.use_fa4,
    )
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
    catmask_param_ids = set()
    for ls in base_model.layer_scalars:
        for attr in (
            "cat_attn_logits",
            "cat_lora_down",
            "cat_q_up",
            "cat_k_up",
            "cat_v_up",
        ):
            p = getattr(ls, attr, None)
            if p is not None:
                catmask_param_ids.add(id(p))
    catmask_params = []
    for ls in base_model.layer_scalars:
        for p in ls.parameters():
            if id(p) in conv_weight_ids:
                conv_params.append(p)
            elif id(p) in catmask_param_ids:
                catmask_params.append(p)
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
        gram_ns=args.muon_gram_ns,
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
    if catmask_params:
        optimizer_catmask = torch.optim.Adam(
            [
                {
                    "params": catmask_params,
                    "lr": args.catmask_lr,
                    "base_lr": args.catmask_lr,
                }
            ],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.append(optimizer_catmask)
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
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(
        f"sdp_backends:cudnn=False flash=True mem_efficient={_needs_explicit_mask} math={_needs_explicit_mask}"
    )
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
    if args.use_fa4:
        assert _FA4_AVAILABLE, "USE_FA4=1 but flash-attn-4 is not installed"
        if args.catmask_mode == "bias":
            raise RuntimeError(
                "USE_FA4=1 is incompatible with CATMASK_MODE=bias (requires dense attn_bias). "
                "Use CATMASK_MODE=lora or CATMASK_MODE=off for FA4."
            )
        log0("Using Flash Attention 4 (FA4)")
    cu_seqlens_budget = args.train_batch_tokens // 64 + 2 if args.use_fa4 else 0

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
        # Step-based warmdown
        warmdown_start = max(args.iterations - args.warmdown_iters, 0)
        step_mul = (
            max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
            if warmdown_start <= step < args.iterations
            else 1.0
        )
        # Time-based warmdown
        if max_wallclock_ms is not None:
            step_ms = elapsed_ms / max(step, 1)
            warmdown_ms = args.warmdown_iters * step_ms
            remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
            time_mul = (
                remaining_ms / max(warmdown_ms, 1e-9)
                if remaining_ms <= warmdown_ms
                else 1.0
            )
            return min(step_mul, time_mul)
        return step_mul

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
                if args.use_fa4 and _FA4_AVAILABLE and args.pack_doc_mask:
                    cu_seqlens, max_seqlen = build_cu_seqlens(x, BOS_ID, pad_to=cu_seqlens_budget)
                else:
                    cu_seqlens, max_seqlen = None, None
                doc_mask = (
                    build_doc_mask(x, BOS_ID)
                    if args.pack_doc_mask and cu_seqlens is None
                    else None
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    warmup_loss = model(
                        x,
                        y,
                        doc_mask=doc_mask,
                        cu_seqlens=cu_seqlens,
                        max_seqlen=max_seqlen,
                    )
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
            if args.use_fa4 and _FA4_AVAILABLE and args.pack_doc_mask:
                cu_seqlens, max_seqlen = build_cu_seqlens(x, BOS_ID, pad_to=cu_seqlens_budget)
            else:
                cu_seqlens, max_seqlen = None, None
            doc_mask = (
                build_doc_mask(x, BOS_ID)
                if args.pack_doc_mask and cu_seqlens is None
                else None
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(
                    x,
                    y,
                    doc_mask=doc_mask,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )
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
            # Correct for special tokens (0 bytes) in targets, same as eval_val.
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
