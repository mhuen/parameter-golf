"""
Universal Transformer GPT — byte-level tokenizer variant.

Same model architecture as train_universal.py, but uses EfficientByteTokenizer
instead of SentencePiece. Each token represents exactly one UTF-8 byte, so
BPB = bits_per_token (no subword-to-byte accounting needed).

Tokenizer configs (via env vars):
    DISCARD_UNUSED_BYTES=1  (default) 206 used bytes, vocab=208
    DISCARD_UNUSED_BYTES=0           all 256 bytes, vocab=258
    FOLD=uppercase                   fold uppercase->lowercase, vocab=182

Data: expects byte260 shards (PureByteTokenizer format: bos=1, bytes=4..259 as uint16).
Tokens are remapped on-the-fly to EfficientByteTokenizer IDs at load time.
BOS tokens at document boundaries are preserved for PACK_DOC_MASK support.
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

    # Learnable category attention mask: off, bias, or lora
    catmask_mode = os.environ.get("CATMASK_MODE", "off")  # off | bias | lora
    catmask_lr = float(os.environ.get("CATMASK_LR", 0.04))
    catmask_rank = int(os.environ.get("CATMASK_RANK", 1))
    catmask_lora_v = bool(int(os.environ.get("CATMASK_LORA_V", "0")))

    # Structural boundary attention bias (word/sentence/paragraph)
    struct_bias_mode = os.environ.get("STRUCT_BIAS_MODE", "off")  # off | bias | lora
    struct_bias_lr = float(os.environ.get("STRUCT_BIAS_LR", 0.04))
    struct_bias_rank = int(os.environ.get("STRUCT_BIAS_RANK", 1))
    struct_bias_lora_v = bool(int(os.environ.get("STRUCT_BIAS_LORA_V", "0")))
    struct_bias_word_bins = int(os.environ.get("STRUCT_BIAS_WORD_BINS", 4))
    struct_bias_sent_bins = int(os.environ.get("STRUCT_BIAS_SENT_BINS", 4))

    # Distance attention bias (T5-style log-bucketed)
    dist_bias_mode = os.environ.get("DIST_BIAS_MODE", "off")  # off | bias | lora
    dist_bias_lr = float(os.environ.get("DIST_BIAS_LR", 0.04))
    dist_bias_rank = int(os.environ.get("DIST_BIAS_RANK", 1))
    dist_bias_lora_v = bool(int(os.environ.get("DIST_BIAS_LORA_V", "0")))
    dist_bias_buckets = int(os.environ.get("DIST_BIAS_BUCKETS", 32))
    dist_bias_max_distance = int(os.environ.get("DIST_BIAS_MAX_DISTANCE", 128))

    # Semantic RoPE: dimension pairs per boundary type (0=disabled)
    rope_word_pairs = int(os.environ.get("ROPE_WORD_PAIRS", 0))
    rope_sent_pairs = int(os.environ.get("ROPE_SENT_PAIRS", 0))
    rope_para_pairs = int(os.environ.get("ROPE_PARA_PAIRS", 0))
    # Frequency bases per type (tuned so lowest freq completes ~1 cycle over 2k-5k byte input tokens)
    rope_word_base = float(os.environ.get("ROPE_WORD_BASE", 1000.0))
    rope_sent_base = float(os.environ.get("ROPE_SENT_BASE", 50.0))
    rope_para_base = float(os.environ.get("ROPE_PARA_BASE", 10.0))

    # N-gram prior (additive logit bias from byte-level n-gram statistics)
    ngram_prior = bool(int(os.environ.get("NGRAM_PRIOR", "0")))
    ngram_order = int(os.environ.get("NGRAM_ORDER", 8))
    ngram_top_n = int(os.environ.get("NGRAM_TOP_N", 20_000_000))
    ngram_confidence_c = float(os.environ.get("NGRAM_CONFIDENCE_C", 3.0))
    ngram_scale_init = float(os.environ.get("NGRAM_SCALE_INIT", 1.0))
    ngram_max_data_bytes = int(os.environ.get("NGRAM_MAX_DATA_BYTES", 500_000_000))

    # Compressed linear layer mode: dense (default CastedLinear), kronecker, monarch
    linear_mode = os.environ.get("LINEAR_MODE", "dense")
    kronecker_terms = int(os.environ.get("KRONECKER_TERMS", 4))
    monarch_nblocks = int(os.environ.get("MONARCH_NBLOCKS", 0))  # 0 = auto (sqrt(n))

    # Byte tokenizer config
    discard_unused_bytes = bool(int(os.environ.get("DISCARD_UNUSED_BYTES", "1")))
    fold = os.environ.get("FOLD", "")  # comma-separated ByteCategory values


# -----------------------------
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
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,conv_scale,cat_attn_logits,cat_lora_down,cat_q_up,cat_k_up,cat_v_up,struct_bias_weights,struct_lora_down,struct_q_up,struct_k_up,struct_v_up,dist_bias_weights,dist_lora_down,dist_q_up,dist_k_up,dist_v_up",
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

# Byte260 shard format: PureByteTokenizer IDs (bos=1, bytes=4..259) stored as uint16.
# We use tok.remap_byte260_shard() + tok.filter_stream() to convert to token IDs.


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

    Uses tok.remap_byte260_shard() (byte260 format: bos=1, bytes=4..259),
    then tok.filter_stream() to apply the OtherTokenStrategy.
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
# N-GRAM PRIOR
# -----------------------------


def _encode_ngram_keys_np(tokens: np.ndarray, order: int, base: int) -> np.ndarray:
    """Encode full n-grams of length ``order`` as unique int64 keys.

    key[j] = tokens[j]*base^0 + tokens[j+1]*base^1 + ... + tokens[j+order-1]*base^(order-1)

    Returns array of length ``len(tokens) - order + 1``.
    """
    n = len(tokens) - order + 1
    keys = np.zeros(n, dtype=np.int64)
    for i in range(order):
        keys += tokens[i : i + n].astype(np.int64) * int(base**i)
    return keys


def build_ngram_tables(
    train_pattern: str,
    tok: "EfficientByteTokenizer",
    max_order: int = 8,
    top_n: int = 5_000_000,
    confidence_c: float = 3.0,
    max_data_bytes: int = 500_000_000,
    cache_dir: str | None = None,
    master_process: bool = True,
) -> dict[int, tuple[Tensor, Tensor]]:
    """Build byte-level n-gram lookup tables with chained backoff.

    Fully vectorized approach (following explore_bytes_ngram.ipynb):
      1. Encode full n-grams as int64 keys.
      2. ``np.unique`` to get all unique n-grams + counts in one shot.
      3. Extract context keys via integer modulo, next bytes via integer division.
      4. Select top-N contexts by total count, build conditional distributions.

    Backoff: for each context at order k, seen bytes use MLE (count/total).
    Unseen bytes are filled with the (k-1)-gram distribution for the shorter
    context (last k-2 tokens), chained down to unigram.  The combined
    distribution is renormalized so rows sum to 1.  This avoids the dilution
    problem of additive pseudo-count smoothing.

    Unigram level uses ``confidence_c`` floor for any truly unseen bytes.

    Returns dict mapping order → (sorted_keys [int64], log_probs [float16, (N, V)]).
    """
    # Check cache
    if cache_dir is not None:
        cache_path = (
            Path(cache_dir)
            / f"ngram_v2_o{max_order}_n{top_n}_c{confidence_c}_d{max_data_bytes}.pt"
        )
        if cache_path.exists():
            if master_process:
                print(f"ngram: loading cached tables from {cache_path}")
            return torch.load(cache_path, map_location="cpu")

    files = [Path(p) for p in sorted(glob.glob(train_pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {train_pattern}")

    vocab_size = tok.vocab_size
    base = vocab_size

    # Load training tokens
    if master_process:
        print(f"ngram: loading training tokens (max={max_data_bytes:,})...")
    token_chunks: list[np.ndarray] = []
    total = 0
    for f in files:
        shard = load_data_shard(f)
        toks = remap_shard_tokens(shard, tok).numpy().astype(np.int64)
        token_chunks.append(toks)
        total += len(toks)
        if max_data_bytes > 0 and total >= max_data_bytes:
            break
    all_tokens = np.concatenate(token_chunks)
    if max_data_bytes > 0 and len(all_tokens) > max_data_bytes:
        all_tokens = all_tokens[:max_data_bytes]
    del token_chunks
    if master_process:
        print(f"ngram: loaded {len(all_tokens):,} tokens")

    tables: dict[int, tuple[Tensor, Tensor]] = {}
    # Keep numpy versions of previous order for chained backoff lookups
    prev_np_keys: np.ndarray | None = None  # sorted context keys
    prev_np_logp: np.ndarray | None = None  # log-prob matrix (N, V)
    unigram_logp: np.ndarray | None = None  # (V,) ultimate fallback

    for order in range(1, max_order + 1):
        ctx_len = order - 1
        if len(all_tokens) < order:
            continue

        t_start = time.time()

        # ---- Unigram (order 1): just byte counts ----
        if ctx_len == 0:
            counts = np.bincount(all_tokens, minlength=vocab_size).astype(np.float64)
            smoothed = np.maximum(counts, confidence_c)
            logp = np.log(smoothed) - np.log(smoothed.sum())
            unigram_logp = logp
            tables[order] = (
                torch.zeros(1, dtype=torch.int64),
                torch.tensor(logp, dtype=torch.float16).unsqueeze(0),
            )
            # Store for chained backoff (unigram: single row, key=0)
            prev_np_keys = np.zeros(1, dtype=np.int64)
            prev_np_logp = logp.reshape(1, -1)
            if master_process:
                print(f"  order {order}: unigram ({time.time() - t_start:.1f}s)")
            continue

        # ---- Higher orders: fully vectorized ----
        if master_process:
            print(f"  order {order}: encoding {order}-grams...")

        # Step 1: encode full n-grams and np.unique (single vectorized pass)
        full_keys = _encode_ngram_keys_np(all_tokens, order, base)
        n_positions = len(full_keys)
        if master_process:
            print(f"  order {order}: np.unique on {n_positions:,} keys...")
        ngram_ids, ngram_counts = np.unique(full_keys, return_counts=True)
        del full_keys

        # Step 2: extract context and next_byte via integer arithmetic
        ctx_divisor = int(base**ctx_len)
        context_keys = ngram_ids % ctx_divisor
        next_bytes_arr = ngram_ids // ctx_divisor
        del ngram_ids

        # Step 3: aggregate per-context totals
        unique_contexts, ctx_inverse = np.unique(context_keys, return_inverse=True)
        ctx_totals = np.zeros(len(unique_contexts), dtype=np.int64)
        np.add.at(ctx_totals, ctx_inverse, ngram_counts)
        n_unique_ctx = len(unique_contexts)

        # Step 4: select top-N contexts by total count
        if n_unique_ctx > top_n:
            top_ctx_idx = np.argpartition(ctx_totals, -top_n)[-top_n:]
        else:
            top_ctx_idx = np.arange(n_unique_ctx)
        actual_n = len(top_ctx_idx)
        coverage = ctx_totals[top_ctx_idx].sum() / n_positions

        # Build sorted selected context keys
        selected_ctx = unique_contexts[top_ctx_idx]
        sort_order = np.argsort(selected_ctx)
        sorted_ctx = selected_ctx[sort_order]

        # Step 5: map unique n-grams → selected contexts, build count matrix
        sel_idx = np.searchsorted(sorted_ctx, context_keys)
        sel_idx = np.clip(sel_idx, 0, actual_n - 1)
        sel_found = sorted_ctx[sel_idx] == context_keys

        count_matrix = np.zeros((actual_n, vocab_size), dtype=np.int64)
        np.add.at(
            count_matrix,
            (sel_idx[sel_found], next_bytes_arr[sel_found].astype(np.int64)),
            ngram_counts[sel_found],
        )
        del context_keys, next_bytes_arr, ngram_counts, ctx_inverse
        del unique_contexts, ctx_totals, sel_idx, sel_found

        # Step 6: MLE for seen bytes + chained backoff for unseen bytes
        seen = count_matrix > 0
        row_totals = count_matrix.sum(axis=1, keepdims=True).astype(np.float64)
        row_totals = np.maximum(row_totals, 1.0)
        mle_probs = count_matrix.astype(np.float64) / row_totals
        del count_matrix

        # Build fallback: look up shorter context in (k-1) table
        # shorter_key = context_key // base  (drops oldest token from context)
        shorter_keys = sorted_ctx // base
        assert prev_np_keys is not None and prev_np_logp is not None
        fb_idx = np.searchsorted(prev_np_keys, shorter_keys)
        fb_idx = np.clip(fb_idx, 0, len(prev_np_keys) - 1)
        fb_found = prev_np_keys[fb_idx] == shorter_keys
        # Use (k-1) row where found, else unigram
        fallback_logp = np.where(
            fb_found[:, None],
            prev_np_logp[fb_idx],
            unigram_logp[None, :],
        )
        fallback_probs = np.exp(fallback_logp)
        del fallback_logp, fb_idx, fb_found, shorter_keys

        # Combine: MLE for seen, fallback for unseen, then renormalize
        combined = np.where(seen, mle_probs, fallback_probs)
        combined /= combined.sum(axis=1, keepdims=True)
        log_probs = np.log(np.maximum(combined, 1e-30))
        del seen, mle_probs, fallback_probs, combined, row_totals

        tables[order] = (
            torch.tensor(sorted_ctx, dtype=torch.int64),
            torch.tensor(log_probs, dtype=torch.float16),
        )

        # Store for next order's chained backoff (replace previous)
        prev_np_keys = sorted_ctx
        prev_np_logp = log_probs.astype(np.float32)

        size_mb = (sorted_ctx.nbytes + log_probs.size * 2) / 1e6
        if master_process:
            print(
                f"  order {order}: {actual_n:,} contexts, coverage={coverage:.1%}, "
                f"~{size_mb:.0f} MB ({time.time() - t_start:.1f}s)"
            )
        del log_probs, sorted_ctx

    del all_tokens, prev_np_keys, prev_np_logp

    # Cache to disk
    if cache_dir is not None and master_process:
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(tables, cache_path)
        total_mb = (
            sum(
                k.numel() * k.element_size() + v.numel() * v.element_size()
                for k, v in tables.values()
            )
            / 1e6
        )
        print(f"ngram: cached to {cache_path} ({total_mb:.1f} MB)")

    return tables


class NgramPrior(nn.Module):
    """Byte-level n-gram prior with stupid backoff for logit biasing.

    Stores sorted (context_key, log_prob_vector) tables for orders 1..max_order.
    At each position, encodes the byte context via collision-free mixed-radix
    int64 keys, looks up via ``searchsorted``, and backs off through decreasing
    orders.  Unigram (order 1) is the universal fallback.

    Table buffers are **non-persistent** (not saved in checkpoints — rebuild or
    load from cache at startup).  The only learnable parameter is ``scale``.
    """

    def __init__(
        self,
        tables: dict[int, tuple[Tensor, Tensor]],
        vocab_size: int,
        scale_init: float = 1.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.base = vocab_size
        orders = sorted(tables.keys())
        self.max_order = max(orders)
        self.scale = nn.Parameter(torch.tensor(scale_init, dtype=torch.float32))
        for order in orders:
            keys, logp = tables[order]
            self.register_buffer(f"keys_{order}", keys, persistent=False)
            self.register_buffer(f"logp_{order}", logp, persistent=False)
        # Ascending order: higher orders overwrite lower (highest priority last)
        self._orders_asc: list[int] = sorted(orders)

    def _encode_contexts(self, input_ids: Tensor, ctx_len: int) -> Tensor:
        """(B, S) token IDs → (B, S) int64 context keys.

        For position t the context is ``input_ids[:, t-ctx_len+1 : t+1]`` —
        the last ``ctx_len`` tokens up to and including position t.  Left-padded
        with a sentinel (= vocab_size) so early positions get keys that won't
        match any table entry and naturally back off.
        """
        B, S = input_ids.shape
        if ctx_len == 0:
            return torch.zeros(B, S, device=input_ids.device, dtype=torch.int64)
        padded = F.pad(input_ids.long(), (ctx_len, 0), value=self.base)
        windows = padded[:, 1:].unfold(1, ctx_len, 1)  # (B, S, ctx_len)
        powers = self.base ** torch.arange(
            ctx_len, device=input_ids.device, dtype=torch.int64
        )
        return (windows * powers).sum(dim=-1)

    def forward(self, input_ids: Tensor) -> Tensor:
        """(B, S) → (B, S, V) additive log-prob bias (float32).

        Starts from unigram, then overwrites with each higher order where found.
        """
        B, S = input_ids.shape
        V = self.vocab_size
        # Unigram fallback (always present)
        result = (
            getattr(self, "logp_1")[0]
            .to(dtype=torch.float32)
            .view(1, 1, V)
            .expand(B, S, V)
        )
        for order in self._orders_asc:
            if order <= 1:
                continue
            keys_tbl = getattr(self, f"keys_{order}")
            logp_tbl = getattr(self, f"logp_{order}")
            ctx_keys = self._encode_contexts(input_ids, order - 1).reshape(-1)
            idx = torch.searchsorted(keys_tbl, ctx_keys).clamp(
                max=keys_tbl.shape[0] - 1
            )
            found = keys_tbl[idx] == ctx_keys  # (B*S,)
            looked_up = logp_tbl[idx].to(dtype=torch.float32).reshape(B, S, V)
            result = torch.where(found.reshape(B, S, 1), looked_up, result)
        return result * self.scale


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


class KroneckerLinear(nn.Module):
    """W = sum_k A_k ⊗ B_k. Uses reshape trick: (A⊗B)vec(X) = vec(B X A^T)."""

    def __init__(self, in_features, out_features, bias=False, num_terms=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Factor dimensions: pick largest divisor near sqrt for balanced blocks.
        self.p_in, self.q_in = _balanced_factors(in_features)
        self.p_out, self.q_out = _balanced_factors(out_features)
        self.num_terms = num_terms
        # Each term: A_k is (p_out, p_in), B_k is (q_out, q_in)
        self.A = nn.Parameter(torch.randn(num_terms, self.p_out, self.p_in))
        self.B = nn.Parameter(torch.randn(num_terms, self.q_out, self.q_in))
        scale = (in_features * out_features) ** -0.5
        nn.init.normal_(self.A, std=scale)
        nn.init.normal_(self.B, std=scale)

    def forward(self, x):
        # x: (..., in_features) -> (..., out_features)
        leading = x.shape[:-1]
        x = x.reshape(-1, self.q_in, self.p_in)
        # Vectorized sum over K Kronecker terms: out[b,i,j] = Σ_k B[k,i,m] x[b,m,n] A[k,j,n]
        out = torch.einsum("kim,bmn,kjn->bij", self.B.to(x.dtype), x, self.A.to(x.dtype))
        return out.reshape(*leading, self.out_features)


class MonarchLinear(nn.Module):
    """Monarch: two block-diagonal matmuls with a reshape (permutation) between them.
    Handles rectangular in->out via asymmetric block structure."""

    def __init__(self, in_features, out_features, bias=False, nblocks=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Auto nblocks: sqrt of the smaller dimension, rounded to nearest divisor.
        if nblocks <= 0:
            nblocks = _nearest_divisor(min(in_features, out_features),
                                       int(math.sqrt(min(in_features, out_features))))
        self.nblocks = nblocks
        if in_features % nblocks != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by nblocks ({nblocks})"
            )
        self.blk_in = in_features // nblocks
        # Stage 1: nblocks blocks of (blk_in x blk_in)
        self.w1 = nn.Parameter(torch.randn(nblocks, self.blk_in, self.blk_in))
        # After monarch shuffle: (blk_in, nblocks)
        # Stage 2: blk_in blocks of (nblocks -> blk_out2)
        if out_features % self.blk_in != 0:
            raise ValueError(
                f"out_features ({out_features}) must be divisible by blk_in={self.blk_in} "
                f"(= in_features // nblocks)"
            )
        self.blk_out2 = out_features // self.blk_in
        self.w2 = nn.Parameter(torch.randn(self.blk_in, nblocks, self.blk_out2))
        scale = (in_features * out_features) ** -0.25
        nn.init.normal_(self.w1, std=scale)
        nn.init.normal_(self.w2, std=scale)

    def forward(self, x):
        # x: (..., in_features) -> (..., out_features)
        leading = x.shape[:-1]
        x = x.reshape(-1, self.nblocks, self.blk_in)
        # Stage 1: block-diagonal bmm (nblocks as batch dim)
        x = torch.bmm(x.permute(1, 0, 2), self.w1.to(x.dtype))  # (nblocks, batch, blk_in)
        # Monarch shuffle fused into permute: (nblocks, batch, blk_in) -> (blk_in, batch, nblocks)
        x = x.permute(2, 1, 0).contiguous()
        # Stage 2: block-diagonal bmm (blk_in as batch dim)
        x = torch.bmm(x, self.w2.to(x.dtype))  # (blk_in, batch, blk_out2)
        return x.permute(1, 0, 2).reshape(*leading, self.out_features)


def _balanced_factors(n):
    """Find two factors of n closest to sqrt(n)."""
    s = int(math.sqrt(n))
    while n % s != 0:
        s -= 1
    return s, n // s


def _nearest_divisor(n, target):
    """Find the divisor of n closest to target."""
    best, best_dist = 1, abs(1 - target)
    for d in range(1, int(math.sqrt(n)) + 1):
        if n % d == 0:
            for candidate in (d, n // d):
                dist = abs(candidate - target)
                if dist < best_dist:
                    best, best_dist = candidate, dist
    return best


def make_linear(in_features, out_features, bias=False, mode="dense", **kwargs):
    """Factory: create a CastedLinear, KroneckerLinear, or MonarchLinear."""
    if mode == "dense":
        return CastedLinear(in_features, out_features, bias=bias)
    elif mode == "kronecker":
        return KroneckerLinear(
            in_features, out_features, bias=bias,
            num_terms=kwargs.get("kronecker_terms", 4),
        )
    elif mode == "monarch":
        return MonarchLinear(
            in_features, out_features, bias=bias,
            nblocks=kwargs.get("monarch_nblocks", 0),
        )
    else:
        raise ValueError(f"Unknown linear_mode: {mode!r}")


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


class SemanticRotary(nn.Module):
    """RoPE driven by semantic boundary IDs (word/sentence/paragraph counts)."""

    def __init__(self, configs):
        """configs: list of (num_pairs, base) for each boundary type."""
        super().__init__()
        self.configs = configs
        inv_freqs = []
        for num_pairs, base in configs:
            dims = num_pairs * 2
            inv_freq = 1.0 / (
                base ** (torch.arange(0, dims, 2, dtype=torch.float32) / dims)
            )
            inv_freqs.append(inv_freq)
        self.register_buffer("inv_freq", torch.cat(inv_freqs), persistent=False)
        self._offsets = []
        pos = 0
        for num_pairs, _ in configs:
            self._offsets.append((pos, pos + num_pairs))
            pos += num_pairs
        self.total_half = pos  # total cos/sin width

    def forward(self, boundary_ids_list, dtype):
        """boundary_ids_list: list of (B, S) long tensors, one per boundary type.
        Returns: cos (B, 1, S, total_half), sin (B, 1, S, total_half)"""
        parts_cos, parts_sin = [], []
        for ids, (start, end) in zip(boundary_ids_list, self._offsets):
            inv_f = self.inv_freq[start:end]  # (num_pairs,)
            freqs = ids.unsqueeze(-1).float() * inv_f  # (B, S, num_pairs)
            parts_cos.append(freqs.cos())
            parts_sin.append(freqs.sin())
        cos = (
            torch.cat(parts_cos, dim=-1).unsqueeze(1).to(dtype=dtype)
        )  # (B, 1, S, total_half)
        sin = torch.cat(parts_sin, dim=-1).unsqueeze(1).to(dtype=dtype)
        return cos, sin


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
        sem_rope_configs=None,
        linear_mode="dense",
        linear_kwargs=None,
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
        _lkw = linear_kwargs or {}
        self.c_q = make_linear(dim, dim, bias=False, mode=linear_mode, **_lkw)
        self.c_k = make_linear(dim, kv_dim, bias=False, mode=linear_mode, **_lkw)
        self.c_v = make_linear(dim, kv_dim, bias=False, mode=linear_mode, **_lkw)
        self.proj = make_linear(dim, dim, bias=False, mode=linear_mode, **_lkw)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(
            torch.full((num_heads,), qk_gain_init, dtype=torch.float32)
        )
        # Semantic RoPE: dedicate last N head dims to boundary-count rotation
        self.sem_rope_dims = 0
        self.sem_rotary = None
        if sem_rope_configs:
            self.sem_rotary = SemanticRotary(sem_rope_configs)
            self.sem_rope_dims = self.sem_rotary.total_half * 2
        # Position RoPE covers remaining dims
        pos_rope_dim = self.head_dim - self.sem_rope_dims
        self.rotary = Rotary(
            pos_rope_dim, base=rope_base, rope_dim_fraction=rope_dim_fraction
        )

    def forward(
        self,
        x,
        doc_mask=None,
        attn_bias=None,
        q_delta=None,
        k_delta=None,
        v_delta=None,
        boundary_ids=None,
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
        # Position RoPE (covers first head_dim - sem_rope_dims dimensions)
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        # Semantic RoPE on last sem_rope_dims dimensions
        if self.sem_rotary is not None and boundary_ids is not None:
            sem_cos, sem_sin = self.sem_rotary(boundary_ids, q.dtype)
            sd = self.sem_rope_dims
            half = sd // 2
            q1, q2 = q[..., -sd:-half], q[..., -half:]
            k1, k2 = k[..., -sd:-half], k[..., -half:]
            q = torch.cat(
                [
                    q[..., :-sd],
                    q1 * sem_cos + q2 * sem_sin,
                    q1 * (-sem_sin) + q2 * sem_cos,
                ],
                dim=-1,
            )
            k = torch.cat(
                [
                    k[..., :-sd],
                    k1 * sem_cos + k2 * sem_sin,
                    k1 * (-sem_sin) + k2 * sem_cos,
                ],
                dim=-1,
            )
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        if doc_mask is not None or attn_bias is not None:
            # Build explicit causal + bias mask for non-flash path
            mask = torch.zeros(1, 1, seqlen, seqlen, device=x.device, dtype=q.dtype)
            # Causal: -inf for future positions
            causal = torch.triu(
                torch.full(
                    (seqlen, seqlen), float("-inf"), device=x.device, dtype=q.dtype
                ),
                diagonal=1,
            )
            mask = mask + causal
            if doc_mask is not None:
                # doc_mask is (B, 1, S, S) bool — convert disallowed to -inf
                mask = mask + torch.where(doc_mask, 0.0, float("-inf"))
            if attn_bias is not None:
                # attn_bias is (B, H, S, S) float — additive bias
                mask = mask + attn_bias
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
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
    def __init__(self, dim, mlp_mult, leaky_relu_negative_slope: float | None = 0.5,
                 linear_mode="dense", linear_kwargs=None):
        super().__init__()
        hidden = mlp_mult * dim
        _lkw = linear_kwargs or {}
        self.fc = make_linear(dim, hidden, bias=False, mode=linear_mode, **_lkw)
        self.proj = make_linear(hidden, dim, bias=False, mode=linear_mode, **_lkw)
        self.proj._zero_init = True
        self.leaky_relu_negative_slope = leaky_relu_negative_slope

    def forward(self, x):
        if self.leaky_relu_negative_slope is None:
            x = torch.relu(self.fc(x))
        else:
            x = F.leaky_relu(self.fc(x), negative_slope=self.leaky_relu_negative_slope)
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

    def precompute_bias(self, input_ids: Tensor, dtype) -> Tensor:
        """Precompute (B, 1, S, C) one-hot for reuse across layers."""
        cat_ids = self.token_to_cat[input_ids]
        cat_oh = F.one_hot(cat_ids, NUM_BYTE_CATEGORIES).to(dtype=dtype)
        return cat_oh.unsqueeze(1)  # (B, 1, S, C)

    def bias_forward(self, cat_oh_expanded: Tensor, cat_attn_logits: Tensor) -> Tensor:
        """Compute (B, H, S, S) bias from precomputed (B, 1, S, C) and per-layer (H, C, C)."""
        q_contrib = cat_oh_expanded @ cat_attn_logits.unsqueeze(0)  # (B, H, S, C)
        return q_contrib @ cat_oh_expanded.transpose(-1, -2)  # (B, H, S, S)


class StructuralBoundaryBias(nn.Module):
    """Detects word/sentence/paragraph boundaries for structural attention bias.

    - bias mode: produces (B, H, S, S) additive bias from same-word/sentence/paragraph features.
    - lora mode: provides structural category IDs (position-within-word × position-within-sentence).
    """

    def __init__(
        self, tok: EfficientByteTokenizer, word_bins: int = 4, sent_bins: int = 4
    ):
        super().__init__()
        V = tok.vocab_size
        # Build boundary LUTs from tokenizer
        is_separator = torch.tensor(
            [bool(tok.mask(ByteCategory.SEPARATOR)[tid]) for tid in range(V)],
            dtype=torch.long,
        )
        is_sentence_end = torch.zeros(V, dtype=torch.long)
        is_newline = torch.zeros(V, dtype=torch.long)
        for tid in range(V):
            info = tok.token_info(tid)
            if info is not None:
                if info.byte_value in (46, 33, 63):  # . ! ?
                    is_sentence_end[tid] = 1
                if info.byte_value == 0x0A:  # \n
                    is_newline[tid] = 1
        self.register_buffer("is_separator", is_separator)
        self.register_buffer("is_sentence_end", is_sentence_end)
        self.register_buffer("is_newline", is_newline)
        self.has_newline = bool(is_newline.any())
        self.num_features = 3 if self.has_newline else 2

        # LoRA: binning for position-within-word and position-within-sentence
        self.word_bins = word_bins
        self.sent_bins = sent_bins
        self.num_struct_cats = word_bins * sent_bins
        word_edges = torch.tensor([2, 4, 8], dtype=torch.long)[: word_bins - 1]
        sent_edges = torch.tensor([4, 16, 64], dtype=torch.long)[: sent_bins - 1]
        self.register_buffer("word_bin_edges", word_edges)
        self.register_buffer("sent_bin_edges", sent_edges)

    def get_boundary_ids(self, input_ids: Tensor):
        """(B, S) → (word_id, sentence_id, paragraph_id or None)."""
        word_id = self.is_separator[input_ids].cumsum(dim=1)
        sentence_id = self.is_sentence_end[input_ids].cumsum(dim=1)
        paragraph_id = (
            self.is_newline[input_ids].cumsum(dim=1) if self.has_newline else None
        )
        return word_id, sentence_id, paragraph_id

    def precompute_bias(self, input_ids: Tensor) -> Tensor:
        """Precompute (B, F, S, S) structural features for reuse across layers."""
        word_id, sentence_id, paragraph_id = self.get_boundary_ids(input_ids)
        same_word = (word_id[:, :, None] == word_id[:, None, :]).float()
        same_sentence = (sentence_id[:, :, None] == sentence_id[:, None, :]).float()
        features = [same_word, same_sentence]
        if paragraph_id is not None:
            same_paragraph = (
                paragraph_id[:, :, None] == paragraph_id[:, None, :]
            ).float()
            features.append(same_paragraph)
        return torch.stack(features, dim=1)  # (B, F, S, S)

    def bias_forward(
        self, struct_features: Tensor, struct_bias_weights: Tensor
    ) -> Tensor:
        """Compute (B, H, S, S) bias from precomputed features and per-layer (H, F) weights."""
        return torch.einsum("bfqk,hf->bhqk", struct_features, struct_bias_weights)

    def get_struct_cat_ids(self, input_ids: Tensor) -> Tensor:
        """(B, S) → (B, S) structural category indices for LoRA mode."""
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)
        # Position within current word
        is_sep = self.is_separator[input_ids].bool()
        last_sep_pos = (
            torch.where(is_sep, positions, torch.zeros_like(positions))
            .cummax(dim=1)
            .values
        )
        pos_in_word = positions - last_sep_pos
        # Position within current sentence
        is_sent = self.is_sentence_end[input_ids].bool()
        last_sent_pos = (
            torch.where(is_sent, positions, torch.zeros_like(positions))
            .cummax(dim=1)
            .values
        )
        pos_in_sent = positions - last_sent_pos
        # Bin
        word_bin = torch.bucketize(pos_in_word, self.word_bin_edges)
        sent_bin = torch.bucketize(pos_in_sent, self.sent_bin_edges)
        return word_bin * self.sent_bins + sent_bin


class DistanceBias(nn.Module):
    """T5-style log-bucketed relative distance bias for attention.

    - bias mode: produces (1, H, S, S) additive distance bias.
    - lora mode: provides position bin IDs for per-position LoRA.
    """

    def __init__(self, num_buckets: int = 32, max_distance: int = 128):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance

        # Precompute distance-to-bucket mapping
        distances = torch.arange(max_distance + 1, dtype=torch.long)
        max_exact = num_buckets // 2
        is_small = distances < max_exact
        # Log-spaced buckets for distances >= max_exact
        log_ratio = torch.log(distances.float().clamp(min=1) / max_exact) / math.log(
            max_distance / max_exact
        )
        val_if_large = (
            max_exact + (log_ratio * (num_buckets - max_exact)).long()
        ).clamp(min=max_exact, max=num_buckets - 1)
        bucket_ids = torch.where(is_small, distances, val_if_large)
        self.register_buffer("distance_to_bucket", bucket_ids)
        # For LoRA: same bucketing on absolute positions
        self.register_buffer("position_to_bin", bucket_ids)

    def precompute_bias(self, seq_len: int, device) -> Tensor:
        """Precompute (S, S) bucket IDs for reuse across layers."""
        positions = torch.arange(seq_len, device=device)
        rel_dist = (positions[:, None] - positions[None, :]).clamp(
            min=0, max=self.max_distance
        )
        return self.distance_to_bucket[rel_dist]  # (S, S)

    def bias_forward(self, bucket_ids: Tensor, dist_bias_weights: Tensor) -> Tensor:
        """Compute (1, H, S, S) bias from precomputed (S, S) bucket IDs and per-layer weights."""
        return dist_bias_weights[:, bucket_ids].unsqueeze(0)  # (1, H, S, S)

    def get_pos_bin_ids(self, seq_len: int, device) -> Tensor:
        """(S,) → (1, S) position bin IDs for LoRA mode."""
        positions = torch.arange(seq_len, device=device).clamp(max=self.max_distance)
        return self.position_to_bin[positions].unsqueeze(0)  # (1, S)


def _compute_lora_deltas(normed_x, one_hot, down, q_up, k_up, v_up):
    """Compute category-conditional LoRA deltas via one-hot matmul.

    Args:
        normed_x: (B, S, dim) normalized input
        one_hot: (B, S, C) one-hot category encoding
        down: (r, dim) shared down-projection
        q_up, k_up: (C, out_dim, r) per-category up-projections
        v_up: (C, kv_dim, r) or None
    Returns:
        (q_delta, k_delta, v_delta) — v_delta is None if v_up is None
    """
    low = normed_x @ down.to(dtype=normed_x.dtype).T  # (B, S, r)

    def _delta(up):
        C, out_dim, r = up.shape
        selected = one_hot @ up.to(dtype=normed_x.dtype).reshape(
            C, -1
        )  # (B, S, out_dim*r)
        if r == 1:
            return selected * low  # (B, S, out_dim) * (B, S, 1) broadcast
        return (selected.reshape(-1, out_dim, r) @ low.reshape(-1, r, 1)).reshape(
            one_hot.shape[0], one_hot.shape[1], out_dim
        )

    return _delta(q_up), _delta(k_up), (_delta(v_up) if v_up is not None else None)


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
        sem_rope_configs=None,
        linear_mode="dense",
        linear_kwargs=None,
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
            sem_rope_configs=sem_rope_configs,
            linear_mode=linear_mode,
            linear_kwargs=linear_kwargs,
        )
        self.mlp = MLP(dim, mlp_mult, linear_mode=linear_mode, linear_kwargs=linear_kwargs)
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
        struct_lora=None,
        dist_lora=None,
        boundary_ids=None,
    ):
        conv_mod = conv if conv is not None else self.conv
        if conv_mod is not None and conv_scale is not None:
            x = x + conv_scale.to(dtype=x.dtype)[None, None, :] * conv_mod(
                self.conv_norm(x)
            )
        n = self.attn_norm(x)
        # Compute LoRA deltas from all active sources
        q_delta, k_delta, v_delta = None, None, None
        for lora_tuple in (cat_lora, struct_lora, dist_lora):
            if lora_tuple is not None:
                oh, down, q_up, k_up, v_up = lora_tuple
                qd, kd, vd = _compute_lora_deltas(n, oh, down, q_up, k_up, v_up)
                q_delta = qd if q_delta is None else q_delta + qd
                k_delta = kd if k_delta is None else k_delta + kd
                if vd is not None:
                    v_delta = vd if v_delta is None else v_delta + vd
        attn_out = self.attn(
            n,
            doc_mask=doc_mask,
            attn_bias=attn_bias,
            q_delta=q_delta,
            k_delta=k_delta,
            v_delta=v_delta,
            boundary_ids=boundary_ids,
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
        struct_bias_mode="off",
        struct_bias_rank=1,
        struct_bias_lora_v=False,
        num_struct_features=3,
        num_struct_cats=16,
        dist_bias_mode="off",
        dist_bias_rank=1,
        dist_bias_lora_v=False,
        dist_bias_buckets=32,
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
        head_dim = dim // num_heads if num_heads > 0 else 0
        kv_dim = num_kv_heads * head_dim

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
        r = catmask_rank
        if catmask_mode == "lora":
            self.cat_lora_down = nn.Parameter(
                torch.randn(r, dim, dtype=torch.float32) * (1.0 / dim**0.5)
            )
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

        # Structural boundary: bias mode — per-head weight for same-word/sentence/paragraph
        self.struct_bias_weights = (
            nn.Parameter(
                torch.zeros(num_heads, num_struct_features, dtype=torch.float32)
            )
            if struct_bias_mode == "bias"
            else None
        )
        # Structural boundary: lora mode
        SC = num_struct_cats
        sr = struct_bias_rank
        if struct_bias_mode == "lora":
            self.struct_lora_down = nn.Parameter(
                torch.randn(sr, dim, dtype=torch.float32) * (1.0 / dim**0.5)
            )
            self.struct_q_up = nn.Parameter(
                torch.zeros(SC, dim, sr, dtype=torch.float32)
            )
            self.struct_k_up = nn.Parameter(
                torch.zeros(SC, kv_dim, sr, dtype=torch.float32)
            )
            self.struct_v_up = (
                nn.Parameter(torch.zeros(SC, kv_dim, sr, dtype=torch.float32))
                if struct_bias_lora_v
                else None
            )
        else:
            self.struct_lora_down = None
            self.struct_q_up = None
            self.struct_k_up = None
            self.struct_v_up = None

        # Distance: bias mode — per-head weight for each distance bucket
        self.dist_bias_weights = (
            nn.Parameter(torch.zeros(num_heads, dist_bias_buckets, dtype=torch.float32))
            if dist_bias_mode == "bias"
            else None
        )
        # Distance: lora mode
        DB = dist_bias_buckets
        dr = dist_bias_rank
        if dist_bias_mode == "lora":
            self.dist_lora_down = nn.Parameter(
                torch.randn(dr, dim, dtype=torch.float32) * (1.0 / dim**0.5)
            )
            self.dist_q_up = nn.Parameter(torch.zeros(DB, dim, dr, dtype=torch.float32))
            self.dist_k_up = nn.Parameter(
                torch.zeros(DB, kv_dim, dr, dtype=torch.float32)
            )
            self.dist_v_up = (
                nn.Parameter(torch.zeros(DB, kv_dim, dr, dtype=torch.float32))
                if dist_bias_lora_v
                else None
            )
        else:
            self.dist_lora_down = None
            self.dist_q_up = None
            self.dist_k_up = None
            self.dist_v_up = None


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

        # ---- Precompute n-gram marginalization matrices (static) ----
        # margin_i: (V, H_i) maps token probs → level-output probs via matmul.
        # margin_i[t, j] = 1.0 iff token t is active at level i and maps to output j.
        for i, head in enumerate(heads):
            H = head.out_features
            margin = torch.zeros(vocab_size, H)
            margin.scatter_(1, all_indices[i].unsqueeze(1), all_masks[i].unsqueeze(1))
            self.register_buffer(f"margin_{i}", margin)

    def forward(self, x, cat_prior=None, token_prior=None, ngram_logp=None):
        """Return (B, S, V) log-probabilities assembled from the hierarchy.

        Args:
            x: (B, S, D) hidden states.
            cat_prior: (B, S, num_categories) additive mask for level-0 logits.
            token_prior: (B, S, V) additive mask for final log-probs (e.g. UTF-8).
            ngram_logp: (B, S, V) token-level n-gram log-probs.  Marginalized into
                per-level conditional priors and added to each head's logits before
                log_softmax.
        """
        B, S, _ = x.shape
        log_p = torch.zeros(B, S, self.vocab_size, device=x.device, dtype=x.dtype)
        # Precompute n-gram probs once for all levels (float32 for log/exp precision)
        ngram_probs = ngram_logp.float().exp() if ngram_logp is not None else None
        for i, head in enumerate(self.heads):
            logits = head(x)
            if i == 0 and cat_prior is not None:
                logits = logits + cat_prior
            if self._is_leaf[i]:
                logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
            # Per-level n-gram prior: marginalize token probs → level conditional
            if ngram_probs is not None:
                margin = getattr(self, f"margin_{i}")  # (V, H_i)
                level_probs = ngram_probs @ margin  # (B, S, H_i)
                level_prior = level_probs.clamp(min=1e-30).log()
                level_prior = level_prior - level_prior.logsumexp(dim=-1, keepdim=True)
                logits = logits + level_prior.to(dtype=logits.dtype)
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
        struct_bias_mode="off",
        struct_bias_rank=1,
        struct_bias_lora_v=False,
        struct_bias_word_bins=4,
        struct_bias_sent_bins=4,
        dist_bias_mode="off",
        dist_bias_rank=1,
        dist_bias_lora_v=False,
        dist_bias_buckets=32,
        dist_bias_max_distance=128,
        sem_rope_configs=None,
        sem_rope_types=None,
        tok=None,
        ngram_tables=None,
        ngram_scale_init=1.0,
        linear_mode="dense",
        linear_kwargs=None,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.num_layers = num_layers
        self.catmask_mode = catmask_mode
        self.struct_bias_mode = struct_bias_mode
        self.dist_bias_mode = dist_bias_mode
        self.sem_rope_configs = sem_rope_configs or []
        self.sem_rope_types = sem_rope_types or []  # "word", "sent", "para"
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
                    sem_rope_configs=sem_rope_configs if sem_rope_configs else None,
                    linear_mode=linear_mode,
                    linear_kwargs=linear_kwargs,
                )
                for _ in range(num_blocks)
            ]
        )

        # Determine structural feature count and category count (needs struct_bias_mod)
        _num_struct_features = 3  # default: word + sentence + paragraph
        _num_struct_cats = struct_bias_word_bins * struct_bias_sent_bins
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
                    struct_bias_mode=struct_bias_mode,
                    struct_bias_rank=struct_bias_rank,
                    struct_bias_lora_v=struct_bias_lora_v,
                    num_struct_features=_num_struct_features,
                    num_struct_cats=_num_struct_cats,
                    dist_bias_mode=dist_bias_mode,
                    dist_bias_rank=dist_bias_rank,
                    dist_bias_lora_v=dist_bias_lora_v,
                    dist_bias_buckets=dist_bias_buckets,
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
        # Structural boundary bias (word/sentence/paragraph)
        # Also needed when sem_rope is active (for boundary IDs)
        _needs_struct_mod = struct_bias_mode != "off" or bool(self.sem_rope_configs)
        if _needs_struct_mod:
            assert tok is not None, "struct_bias/sem_rope requires tok"
            self.struct_bias_mod = StructuralBoundaryBias(
                tok, word_bins=struct_bias_word_bins, sent_bins=struct_bias_sent_bins
            )
            # Update layer scalars num_features if module detected fewer features
            if (
                struct_bias_mode != "off"
                and self.struct_bias_mod.num_features != _num_struct_features
            ):
                for ls in self.layer_scalars:
                    if ls.struct_bias_weights is not None:
                        nf = self.struct_bias_mod.num_features
                        ls.struct_bias_weights = nn.Parameter(
                            torch.zeros(
                                ls.struct_bias_weights.shape[0], nf, dtype=torch.float32
                            )
                        )
        else:
            self.struct_bias_mod = None
        # Distance bias
        if dist_bias_mode != "off":
            self.dist_bias_mod = DistanceBias(
                num_buckets=dist_bias_buckets, max_distance=dist_bias_max_distance
            )
        else:
            self.dist_bias_mod = None
        # N-gram prior
        if ngram_tables is not None:
            self.ngram_prior_mod = NgramPrior(
                ngram_tables, vocab_size, scale_init=ngram_scale_init
            )
        else:
            self.ngram_prior_mod = None
        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)) and getattr(
                module, "_zero_init", False
            ):
                nn.init.zeros_(module.weight)
            elif isinstance(module, (KroneckerLinear, MonarchLinear)) and getattr(
                module, "_zero_init", False
            ):
                for p in module.parameters():
                    nn.init.zeros_(p)

    def _get_conv_args(self, ls, layer_idx):
        if not self.conv_enabled:
            return None, None
        if ls.conv is not None:
            return ls.conv, ls.conv_scale
        return None, self.shared_conv_scales[layer_idx]

    def _get_all_attn_args(self, ls, precomputed):
        """Return (attn_bias, cat_lora, struct_lora, dist_lora, boundary_ids) using precomputed tensors."""
        (
            cat_oh,
            cat_bias_oh,
            struct_oh,
            struct_features,
            dist_oh,
            dist_bucket_ids,
            boundary_ids,
        ) = precomputed
        attn_bias = None
        cat_lora, struct_lora, dist_lora = None, None, None

        # Catmask
        if self.cat_attn_mod is not None:
            if self.catmask_mode == "bias" and ls.cat_attn_logits is not None:
                attn_bias = self.cat_attn_mod.bias_forward(
                    cat_bias_oh, ls.cat_attn_logits
                )
            elif self.catmask_mode == "lora" and ls.cat_lora_down is not None:
                cat_lora = (
                    cat_oh,
                    ls.cat_lora_down,
                    ls.cat_q_up,
                    ls.cat_k_up,
                    ls.cat_v_up,
                )

        # Structural boundary
        if self.struct_bias_mod is not None:
            if self.struct_bias_mode == "bias" and ls.struct_bias_weights is not None:
                sb = self.struct_bias_mod.bias_forward(
                    struct_features, ls.struct_bias_weights
                )
                attn_bias = sb if attn_bias is None else attn_bias + sb
            elif self.struct_bias_mode == "lora" and ls.struct_lora_down is not None:
                struct_lora = (
                    struct_oh,
                    ls.struct_lora_down,
                    ls.struct_q_up,
                    ls.struct_k_up,
                    ls.struct_v_up,
                )

        # Distance
        if self.dist_bias_mod is not None:
            if self.dist_bias_mode == "bias" and ls.dist_bias_weights is not None:
                db = self.dist_bias_mod.bias_forward(
                    dist_bucket_ids, ls.dist_bias_weights
                )
                attn_bias = db if attn_bias is None else attn_bias + db
            elif self.dist_bias_mode == "lora" and ls.dist_lora_down is not None:
                dist_lora = (
                    dist_oh,
                    ls.dist_lora_down,
                    ls.dist_q_up,
                    ls.dist_k_up,
                    ls.dist_v_up,
                )

        return attn_bias, cat_lora, struct_lora, dist_lora, boundary_ids

    def _precompute_attn_extras(self, input_ids, dtype):
        """Precompute all input-dependent tensors once for all layers."""
        # Catmask
        cat_oh, cat_bias_oh = None, None
        if self.cat_attn_mod is not None:
            if self.catmask_mode == "lora":
                cat_ids = self.cat_attn_mod.get_cat_ids(input_ids)
                cat_oh = F.one_hot(cat_ids, NUM_BYTE_CATEGORIES).to(dtype=dtype)
            elif self.catmask_mode == "bias":
                cat_bias_oh = self.cat_attn_mod.precompute_bias(input_ids, dtype)

        # Structural boundary
        struct_oh, struct_features = None, None
        if self.struct_bias_mod is not None:
            if self.struct_bias_mode == "lora":
                struct_cat_ids = self.struct_bias_mod.get_struct_cat_ids(input_ids)
                struct_oh = F.one_hot(
                    struct_cat_ids, self.struct_bias_mod.num_struct_cats
                ).to(dtype=dtype)
            elif self.struct_bias_mode == "bias":
                struct_features = self.struct_bias_mod.precompute_bias(input_ids)

        # Distance
        dist_oh, dist_bucket_ids = None, None
        if self.dist_bias_mod is not None:
            if self.dist_bias_mode == "lora":
                dist_bin_ids = self.dist_bias_mod.get_pos_bin_ids(
                    input_ids.shape[1], input_ids.device
                )
                dist_bin_ids = dist_bin_ids.expand(input_ids.shape[0], -1)
                dist_oh = F.one_hot(dist_bin_ids, self.dist_bias_mod.num_buckets).to(
                    dtype=dtype
                )
            elif self.dist_bias_mode == "bias":
                dist_bucket_ids = self.dist_bias_mod.precompute_bias(
                    input_ids.shape[1], input_ids.device
                )

        # Semantic RoPE boundary IDs
        boundary_ids = None
        if self.sem_rope_types and self.struct_bias_mod is not None:
            word_id, sentence_id, paragraph_id = self.struct_bias_mod.get_boundary_ids(
                input_ids
            )
            _id_map = {"word": word_id, "sent": sentence_id, "para": paragraph_id}
            boundary_ids = [
                _id_map[t] for t in self.sem_rope_types if _id_map[t] is not None
            ]

        return (
            cat_oh,
            cat_bias_oh,
            struct_oh,
            struct_features,
            dist_oh,
            dist_bucket_ids,
            boundary_ids,
        )

    def forward(self, input_ids, target_ids, doc_mask=None):
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []

        # Precompute all input-dependent attention extras once for all layers
        precomputed = self._precompute_attn_extras(input_ids, x.dtype)

        for i in range(self.num_encoder_layers):
            ls = self.layer_scalars[i]
            mix = ls.resid_mix.to(dtype=x.dtype)
            x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
            conv, conv_scale = self._get_conv_args(ls, i)
            attn_bias, cat_lora, struct_lora, dist_lora, boundary_ids = (
                self._get_all_attn_args(ls, precomputed)
            )
            x = self.shared_blocks[self.block_map[i]](
                x,
                ls.attn_scale,
                ls.mlp_scale,
                conv=conv,
                conv_scale=conv_scale,
                doc_mask=doc_mask,
                attn_bias=attn_bias,
                cat_lora=cat_lora,
                struct_lora=struct_lora,
                dist_lora=dist_lora,
                boundary_ids=boundary_ids,
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
            attn_bias, cat_lora, struct_lora, dist_lora, boundary_ids = (
                self._get_all_attn_args(ls, precomputed)
            )
            x = self.shared_blocks[self.block_map[layer_idx]](
                x,
                ls.attn_scale,
                ls.mlp_scale,
                conv=conv,
                conv_scale=conv_scale,
                doc_mask=doc_mask,
                attn_bias=attn_bias,
                cat_lora=cat_lora,
                struct_lora=struct_lora,
                dist_lora=dist_lora,
                boundary_ids=boundary_ids,
            )
        x = self.final_norm(x)
        # Compute UTF-8 prior masks (purely from input_ids, causal)
        cat_prior, token_prior = None, None
        if self.utf8_prior_mod is not None:
            cat_prior, token_prior = self.utf8_prior_mod(input_ids)
        # N-gram prior
        ngram_logp = None
        if self.ngram_prior_mod is not None:
            ngram_logp = self.ngram_prior_mod(input_ids)
        if self.structured_output_logits:
            # Hierarchical: n-gram marginalized per level inside the head.
            # token_prior here is UTF-8 hard masks only (no n-gram, avoids double-count).
            log_p = self.structured_head(
                x,
                cat_prior=cat_prior,
                token_prior=token_prior,
                ngram_logp=ngram_logp,
            )
            return F.nll_loss(
                log_p.float().reshape(-1, log_p.size(-1)),
                target_ids.reshape(-1),
                reduction="mean",
            )
        # Flat head: merge n-gram into token_prior at token level
        if ngram_logp is not None:
            if token_prior is not None:
                token_prior = token_prior + ngram_logp
            else:
                token_prior = ngram_logp
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

    _needs_explicit_mask = (
        args.pack_doc_mask
        or args.catmask_mode == "bias"
        or args.struct_bias_mode == "bias"
        or args.dist_bias_mode == "bias"
    )
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
        log0(
            "pack_doc_mask:enabled (byte260 shards preserve BOS at document boundaries)"
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
    if args.struct_bias_mode != "off":
        log0(
            f"struct_bias:mode={args.struct_bias_mode} rank={args.struct_bias_rank} lr={args.struct_bias_lr}"
        )
    if args.dist_bias_mode != "off":
        log0(
            f"dist_bias:mode={args.dist_bias_mode} buckets={args.dist_bias_buckets} lr={args.dist_bias_lr}"
        )

    # Build semantic RoPE configs
    sem_rope_configs = []
    sem_rope_types = []
    if args.rope_word_pairs > 0:
        sem_rope_configs.append((args.rope_word_pairs, args.rope_word_base))
        sem_rope_types.append("word")
    if args.rope_sent_pairs > 0:
        sem_rope_configs.append((args.rope_sent_pairs, args.rope_sent_base))
        sem_rope_types.append("sent")
    if args.rope_para_pairs > 0:
        sem_rope_configs.append((args.rope_para_pairs, args.rope_para_base))
        sem_rope_types.append("para")
    if sem_rope_configs:
        head_dim = args.model_dim // args.num_heads
        total_sem = sum(p * 2 for p, _ in sem_rope_configs)
        log0(
            f"sem_rope: word={args.rope_word_pairs}pairs(base={args.rope_word_base}) "
            f"sent={args.rope_sent_pairs}pairs(base={args.rope_sent_base}) "
            f"para={args.rope_para_pairs}pairs(base={args.rope_para_base}) "
            f"total_dims={total_sem}/{head_dim}"
        )

    # Build n-gram tables (before model creation)
    ngram_tables = None
    if args.ngram_prior:
        ngram_tables = build_ngram_tables(
            args.train_files,
            tok,
            max_order=args.ngram_order,
            top_n=args.ngram_top_n,
            confidence_c=args.ngram_confidence_c,
            max_data_bytes=args.ngram_max_data_bytes,
            cache_dir=str(Path(run_dir).parent) if master_process else None,
            master_process=master_process,
        )
        log0(
            f"ngram_prior:enabled order={args.ngram_order} top_n={args.ngram_top_n} "
            f"scale_init={args.ngram_scale_init}"
        )

    _needs_tok = (
        args.structured_output_logits
        or args.utf8_prior
        or args.catmask_mode != "off"
        or args.struct_bias_mode != "off"
        or bool(sem_rope_configs)
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
            struct_bias_mode=args.struct_bias_mode,
            struct_bias_rank=args.struct_bias_rank,
            struct_bias_lora_v=args.struct_bias_lora_v,
            struct_bias_word_bins=args.struct_bias_word_bins,
            struct_bias_sent_bins=args.struct_bias_sent_bins,
            dist_bias_mode=args.dist_bias_mode,
            dist_bias_rank=args.dist_bias_rank,
            dist_bias_lora_v=args.dist_bias_lora_v,
            dist_bias_buckets=args.dist_bias_buckets,
            dist_bias_max_distance=args.dist_bias_max_distance,
            sem_rope_configs=sem_rope_configs if sem_rope_configs else None,
            sem_rope_types=sem_rope_types if sem_rope_types else None,
            tok=tok if _needs_tok else None,
            ngram_tables=ngram_tables,
            ngram_scale_init=args.ngram_scale_init,
            linear_mode=args.linear_mode,
            linear_kwargs={
                "kronecker_terms": args.kronecker_terms,
                "monarch_nblocks": args.monarch_nblocks,
            },
        )
        .to(device)
        .bfloat16()
    )
    for module in base_model.modules():
        if isinstance(module, (CastedLinear, KroneckerLinear, MonarchLinear, nn.Conv1d)):
            module.float()
        if isinstance(module, (Rotary, SemanticRotary)):
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
    # Collect special param IDs for routing to separate optimizers
    _special_attr_groups = {
        "catmask": (
            "cat_attn_logits",
            "cat_lora_down",
            "cat_q_up",
            "cat_k_up",
            "cat_v_up",
        ),
        "struct_bias": (
            "struct_bias_weights",
            "struct_lora_down",
            "struct_q_up",
            "struct_k_up",
            "struct_v_up",
        ),
        "dist_bias": (
            "dist_bias_weights",
            "dist_lora_down",
            "dist_q_up",
            "dist_k_up",
            "dist_v_up",
        ),
    }
    _special_param_ids: dict[str, set[int]] = {k: set() for k in _special_attr_groups}
    for ls in base_model.layer_scalars:
        for group_name, attrs in _special_attr_groups.items():
            for attr in attrs:
                p = getattr(ls, attr, None)
                if p is not None:
                    _special_param_ids[group_name].add(id(p))
    all_special_ids = set().union(*_special_param_ids.values())
    catmask_params = []
    struct_bias_params = []
    dist_bias_params = []
    _group_lists = {
        "catmask": catmask_params,
        "struct_bias": struct_bias_params,
        "dist_bias": dist_bias_params,
    }
    for ls in base_model.layer_scalars:
        for p in ls.parameters():
            if id(p) in conv_weight_ids:
                conv_params.append(p)
            elif id(p) in all_special_ids:
                for group_name, ids in _special_param_ids.items():
                    if id(p) in ids:
                        _group_lists[group_name].append(p)
                        break
            else:
                scalar_params.append(p)
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    if base_model.shared_conv_scales is not None:
        for p in base_model.shared_conv_scales:
            scalar_params.append(p)
    if base_model.ngram_prior_mod is not None:
        scalar_params.append(base_model.ngram_prior_mod.scale)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    if matrix_params:
        optimizer_muon = Muon(
            matrix_params,
            lr=args.matrix_lr,
            momentum=args.muon_momentum,
            backend_steps=args.muon_backend_steps,
            gram_ns=args.muon_gram_ns,
        )
        for group in optimizer_muon.param_groups:
            group["base_lr"] = args.matrix_lr
    else:
        optimizer_muon = None
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers = [optimizer_tok, optimizer_scalar]
    if optimizer_muon is not None:
        optimizers.append(optimizer_muon)
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
    if struct_bias_params:
        optimizer_struct_bias = torch.optim.Adam(
            [
                {
                    "params": struct_bias_params,
                    "lr": args.struct_bias_lr,
                    "base_lr": args.struct_bias_lr,
                }
            ],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.append(optimizer_struct_bias)
    if dist_bias_params:
        optimizer_dist_bias = torch.optim.Adam(
            [
                {
                    "params": dist_bias_params,
                    "lr": args.dist_bias_lr,
                    "base_lr": args.dist_bias_lr,
                }
            ],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.append(optimizer_dist_bias)
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
    if args.linear_mode != "dense":
        _lm_extra = (
            f"kronecker_terms:{args.kronecker_terms}" if args.linear_mode == "kronecker"
            else f"monarch_nblocks:{args.monarch_nblocks}"
        )
        log0(f"linear_mode:{args.linear_mode} {_lm_extra}")
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
        if optimizer_muon is not None:
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
