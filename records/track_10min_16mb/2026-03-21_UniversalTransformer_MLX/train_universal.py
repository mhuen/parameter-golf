"""
Universal Transformer GPT — PyTorch implementation.

Key idea: transformer blocks (attention + MLP) are shared across groups of layers,
with only per-layer scalars (attn_scale, mlp_scale, resid_mix) being unique per
application. This is ALBERT-style weight sharing.

Default config: 27 layers with 9 shared blocks (3 layers per block), dim=512,
8 heads, 4 KV heads, relu^2 MLP (2x expansion). This matches the baseline
parameter budget (~17M params) while providing 3x more depth via weight sharing.

The BLOCK_PATTERN env var controls which layers share weights. Set to "" for a
single shared block across all layers (extreme sharing), or specify per-layer
block indices like "0,0,0,1,1,1,..." for grouped sharing.
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
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# Import from the project root — add it to sys.path if running from records dir.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import optim as _optim_mod
from optim import Muon, _GRAM_NS_LIB
from data import (
    load_shard_sp1024,
    load_validation_tokens_sp1024,
    TokenStream,
    DistributedTokenLoader,
)
from modules import (
    RMSNorm,
    CastedLinear,
    KroneckerLinear,
    MonarchLinear,
    make_linear,
    GatedCausalConv,
    MLP,
    Rotary,
    apply_rotary_emb,
    build_doc_mask,
    restore_low_dim_params_to_fp32,
)

# -----------------------------
# HYPERPARAMETERS
# -----------------------------


class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get(
        "TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model"
    )
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

    # Model — 27 layers with 9 shared blocks (3 layers per block), width 512, relu^2 MLP.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 27))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    # Block sharing pattern: comma-separated block indices per layer.
    # E.g. "0,0,0,1,1,1" means layers 0-2 share block 0, layers 3-5 share block 1.
    # Default "" means all layers share a single block (original behavior).
    block_pattern = os.environ.get(
        "BLOCK_PATTERN",
        "0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8",
        # "BLOCK_PATTERN",
        # "0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8",
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

    # Causal convolution before attention for local n-gram mixing.
    conv_kernel_size = int(os.environ.get("CONV_KERNEL_SIZE", 4))
    conv_groups = int(os.environ.get("CONV_GROUPS", 0))  # 0 = depthwise (groups=dim)
    conv_shared = bool(
        int(os.environ.get("CONV_SHARED", "0"))
    )  # share conv weights across layers
    conv_enabled = bool(int(os.environ.get("CONV_ENABLED", "1")))
    conv_lr = float(os.environ.get("CONV_LR", 0.01))

    # Partial RoPE: fraction of head dimensions to apply rotary embeddings to (0.0-1.0).
    # Remaining dimensions are position-independent "semantic" channels.
    rope_dim_fraction = float(os.environ.get("ROPE_DIM_FRACTION", 1.0))

    # Compressed linear layer mode: dense (default CastedLinear), kronecker, monarch
    # LINEAR_MODE sets the default; LINEAR_MODE_ATTN / LINEAR_MODE_MLP override per-component
    linear_mode = os.environ.get("LINEAR_MODE", "dense")
    linear_mode_attn = os.environ.get("LINEAR_MODE_ATTN", "")  # "" = use LINEAR_MODE
    linear_mode_mlp = os.environ.get("LINEAR_MODE_MLP", "")  # "" = use LINEAR_MODE
    kronecker_terms = int(os.environ.get("KRONECKER_TERMS", 4))
    monarch_nblocks = int(os.environ.get("MONARCH_NBLOCKS", 0))  # 0 = auto (sqrt(n))
    muon_optimize_factors = bool(
        int(os.environ.get("MUON_OPTIMIZE_FACTORS", "0"))
    )  # Kronecker/Monarch factors
    muon_optimize_lm_head = bool(int(os.environ.get("MUON_OPTIMIZE_LM_HEAD", "0")))
    muon_optimize_conv = bool(
        int(os.environ.get("MUON_OPTIMIZE_CONV", "0"))
    )  # non-depthwise only

    # Packing with document-level attention masking. When enabled, BOS tokens mark
    # document boundaries and tokens cannot attend across documents within a packed sequence.
    pack_doc_mask = bool(int(os.environ.get("PACK_DOC_MASK", "0")))


# -----------------------------
# TOKENIZER + EVAL + QUANTIZATION (same as baseline)
# -----------------------------


def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def eval_val(
    args,
    model,
    rank,
    world_size,
    device,
    grad_accum_steps,
    val_tokens,
    base_bytes_lut,
    has_leading_space_lut,
    is_boundary_token_lut,
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
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (
                has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
            ).to(dtype=torch.int16)
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
# TRANSFORMER MODULES (Universal Transformer)
# -----------------------------


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        num_kv_heads,
        rope_base,
        qk_gain_init,
        rope_dim_fraction=1.0,
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
        linear_mode="dense",
        linear_mode_attn="",
        linear_mode_mlp="",
        linear_kwargs=None,
    ):
        super().__init__()
        _attn_mode = linear_mode_attn or linear_mode
        _mlp_mode = linear_mode_mlp or linear_mode
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim,
            num_heads,
            num_kv_heads,
            rope_base,
            qk_gain_init,
            rope_dim_fraction,
            linear_mode=_attn_mode,
            linear_kwargs=linear_kwargs,
        )
        self.mlp = MLP(
            dim, mlp_mult, linear_mode=_mlp_mode, linear_kwargs=linear_kwargs
        )
        # Optional shared conv (when conv is shared across layers).
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
        # Conv before attention: local n-gram mixing enriches Q/K/V inputs.
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
        # Per-layer conv (when not shared) and its scale.
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
        linear_mode="dense",
        linear_mode_attn="",
        linear_mode_mlp="",
        linear_kwargs=None,
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

        # Block sharing: parse pattern to determine which block each layer uses.
        if block_pattern.strip():
            self.block_map = [int(x) for x in block_pattern.strip().split(",")]
            if len(self.block_map) != num_layers:
                raise ValueError(
                    f"BLOCK_PATTERN has {len(self.block_map)} entries but NUM_LAYERS={num_layers}"
                )
        else:
            self.block_map = [0] * num_layers  # all layers share a single block

        # Conv lives in SharedBlock when shared, in LayerScalars when per-layer.
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
                    linear_mode=linear_mode,
                    linear_mode_attn=linear_mode_attn,
                    linear_mode_mlp=linear_mode_mlp,
                    linear_kwargs=linear_kwargs,
                )
                for _ in range(num_blocks)
            ]
        )

        # Per-layer scalars (unique per application). Conv scale is always per-layer.
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
        # When conv is shared, we still need a per-layer conv_scale.
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
            elif isinstance(module, (KroneckerLinear, MonarchLinear)) and getattr(
                module, "_zero_init", False
            ):
                for p in module.parameters():
                    nn.init.zeros_(p)

    def _materialize_weights(self):
        """Materialize cached W for all KroneckerLinear/MonarchLinear in shared blocks."""
        for m in self.shared_blocks.modules():
            if isinstance(m, (KroneckerLinear, MonarchLinear)):
                m.materialize()

    def _clear_weight_caches(self):
        """Clear materialized W caches to free memory."""
        for m in self.shared_blocks.modules():
            if isinstance(m, (KroneckerLinear, MonarchLinear)):
                m.clear_cache()

    def _get_conv_args(self, ls, layer_idx):
        """Return (conv, conv_scale) for a given layer."""
        if not self.conv_enabled:
            return None, None
        if ls.conv is not None:
            # Per-layer conv lives in LayerScalars.
            return ls.conv, ls.conv_scale
        # Shared conv: scale is per-layer, conv is in SharedBlock.
        return None, self.shared_conv_scales[layer_idx]

    def forward(self, input_ids, target_ids, doc_mask=None):
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []

        self._materialize_weights()

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

        self._clear_weight_caches()

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


def main():
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    _optim_mod._zeropower_standard_ns5 = torch.compile(
        _optim_mod._zeropower_standard_ns5
    )
    if args.muon_gram_ns and not _GRAM_NS_LIB:
        _optim_mod._zeropower_gram_ns5 = torch.compile(_optim_mod._zeropower_gram_ns5)

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
    # mem_efficient and math backends needed when using custom attention masks (flash doesn't support them).
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

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(
            f"TOKENIZER_PATH must point to a SentencePiece .model file: {args.tokenizer_path}"
        )
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE mismatch: {args.vocab_size} vs {int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens_sp1024(
        args.val_files, args.train_seq_len, args.val_max_tokens
    )
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = (
        build_sentencepiece_luts(sp, args.vocab_size, device)
    )
    log0(
        f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}"
    )
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")

    base_model = (
        GPT(
            vocab_size=args.vocab_size,
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
            linear_mode=args.linear_mode,
            linear_mode_attn=args.linear_mode_attn,
            linear_mode_mlp=args.linear_mode_mlp,
            linear_kwargs={
                "kronecker_terms": args.kronecker_terms,
                "monarch_nblocks": args.monarch_nblocks,
            },
        )
        .to(device)
        .bfloat16()
    )
    for module in base_model.modules():
        if isinstance(
            module, (CastedLinear, KroneckerLinear, MonarchLinear, nn.Conv1d)
        ):
            module.float()
        if isinstance(module, Rotary):
            module.inv_freq.data = module.inv_freq.data.float()
    restore_low_dim_params_to_fp32(
        base_model, control_patterns=CONTROL_TENSOR_NAME_PATTERNS
    )
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model = (
        DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False)
        if distributed
        else compiled_model
    )

    # Optimizer: shared_blocks 2D (non-conv) -> Muon, everything else -> Adam
    # Optimizer: shared_blocks 2D (non-conv) -> Muon, everything else -> Adam
    # Collect conv weight IDs. When muon_optimize_conv is enabled, non-depthwise
    # conv weights go to Muon (reshaped 3D->2D); depthwise stays on Adam (each
    # filter is 1xK, so NS orthogonalization is meaningless).
    conv_weight_ids = set()
    _muon_conv_ids = set()
    for m in base_model.modules():
        if isinstance(m, nn.Conv1d):
            for p in m.parameters():
                if args.muon_optimize_conv and m.groups < m.out_channels:
                    _muon_conv_ids.add(id(p))
                else:
                    conv_weight_ids.add(id(p))
    # When muon_optimize_factors is enabled, route >=2D Kronecker/Monarch factor params
    # to Muon (which reshapes 3D->2D for NS). Otherwise only 2D params go to Muon.
    _extra_muon_ids = set(_muon_conv_ids)
    if args.muon_optimize_factors:
        for m in base_model.shared_blocks.modules():
            if isinstance(m, (KroneckerLinear, MonarchLinear)):
                for p in m.parameters():
                    _extra_muon_ids.add(id(p))
    shared_blocks_named = list(base_model.shared_blocks.named_parameters())
    matrix_params = [
        p
        for n, p in shared_blocks_named
        if (p.ndim == 2 or id(p) in _extra_muon_ids)
        and not any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)
        and id(p) not in conv_weight_ids
    ]
    _matrix_param_ids = {id(p) for p in matrix_params}
    scalar_params = [
        p
        for n, p in shared_blocks_named
        if id(p) not in _matrix_param_ids and id(p) not in conv_weight_ids
    ]
    conv_params = [p for n, p in shared_blocks_named if id(p) in conv_weight_ids]
    for ls in base_model.layer_scalars:
        for p in ls.parameters():
            if id(p) in _muon_conv_ids:
                matrix_params.append(p)
            elif id(p) in conv_weight_ids:
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
    if matrix_params:
        optimizer_muon = Muon(
            matrix_params,
            lr=args.matrix_lr,
            momentum=args.muon_momentum,
            backend_steps=args.muon_backend_steps,
            gram_ns=args.muon_gram_ns,
            reshape_3d=True,
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
    if base_model.lm_head is not None:
        if args.muon_optimize_lm_head:
            matrix_params.append(base_model.lm_head.weight)
        else:
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
    _attn_mode = args.linear_mode_attn or args.linear_mode
    _mlp_mode = args.linear_mode_mlp or args.linear_mode
    if _attn_mode != "dense" or _mlp_mode != "dense":
        _lm_extra = (
            f"kronecker_terms:{args.kronecker_terms}"
            if "kronecker" in (_attn_mode, _mlp_mode)
            else ""
        )
        if "monarch" in (_attn_mode, _mlp_mode):
            _lm_extra += f" monarch_nblocks:{args.monarch_nblocks}"
        log0(
            f"linear_mode attn:{_attn_mode} mlp:{_mlp_mode} {_lm_extra.strip()} muon_factors:{args.muon_optimize_factors}"
        )
    _muon_extras = []
    if args.muon_optimize_lm_head:
        _muon_extras.append("lm_head")
    if args.muon_optimize_conv:
        _muon_extras.append("conv")
    if _muon_extras:
        log0(f"muon_optimize: {' '.join(_muon_extras)}")
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

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

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
            args.train_files, rank, world_size, device
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
                has_leading_space_lut,
                is_boundary_token_lut,
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
        has_leading_space_lut,
        is_boundary_token_lut,
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
