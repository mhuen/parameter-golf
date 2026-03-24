"""
xLSTM mLSTM Language Model — PyTorch implementation.

Replaces transformer self-attention with mLSTM (matrix LSTM, Beck et al. 2024).
Uses the parallel form during training: materializes an [S, S] gating matrix that
combines forget/input gates with query-key similarities, then applies it to values.

Architecture: embedding → mLSTM blocks → final norm → tied LM head.
No encoder/decoder split, no skip connections, no x0 residual mixing.
"""
from __future__ import annotations

import copy
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

# -----------------------------
# HYPERPARAMETERS
# -----------------------------

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
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

    # Model — xLSTM architecture.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 13))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 4))
    qk_dim_factor = float(os.environ.get("QK_DIM_FACTOR", 0.5))
    v_dim_factor = float(os.environ.get("V_DIM_FACTOR", 1.0))
    ffn_mult = float(os.environ.get("FFN_MULT", 2.0))
    gate_soft_cap = float(os.environ.get("GATE_SOFT_CAP", 15.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))

    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    ttt_lora_rank = int(os.environ.get("TTT_LORA_RANK", 8))
    ttt_lora_lr = float(os.environ.get("TTT_LORA_LR", 0.01))
    ttt_chunk_size = int(os.environ.get("TTT_CHUNK_SIZE", 256))
    ttt_eval_seq_len = int(os.environ.get("TTT_EVAL_SEQ_LEN", 1024))
    ttt_batch_size = int(os.environ.get("TTT_BATCH_SIZE", 64))

    # Packing with document-level masking. When enabled, BOS tokens mark document
    # boundaries: the mLSTM gating matrix is masked to prevent cross-document attention,
    # and the cumulative forget gate is reset at each document boundary.
    pack_doc_mask = bool(int(os.environ.get("PACK_DOC_MASK", "0")))

# -----------------------------
# MUON OPTIMIZER
# -----------------------------

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "norm,igate_preact.bias,fgate_preact.bias",
    ).split(",") if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS", ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",") if pattern
)

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
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
    def __init__(self, params, lr, momentum, backend_steps, nesterov=True):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params: continue
            lr, momentum, backend_steps, nesterov = group["lr"], group["momentum"], group["backend_steps"], group["nesterov"]
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov: g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed: dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss

# -----------------------------
# INFRASTRUCTURE (tokenizer, eval, quantization, data loading)
# -----------------------------

def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id): continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1; continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True; piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )

def load_validation_tokens(pattern, seq_len, max_tokens=0):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files: raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    if max_tokens > 0: tokens = tokens[: max_tokens + 1]
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0: raise ValueError(f"Validation split too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]

def eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len: raise ValueError("VAL_BATCH_SIZE too small")
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
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            doc_mask = build_doc_mask(x, BOS_ID) if args.pack_doc_mask else None
            doc_reset = (x == BOS_ID) if args.pack_doc_mask else None
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y, doc_mask=doc_mask, doc_reset_mask=doc_reset).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1); tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
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

# Quantization
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0

def tensor_nbytes(t): return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name, t, passthrough_orig_dtypes):
    if any(p in name for p in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS): return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def quantize_float_tensor(t):
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1) if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32)
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

def quantize_state_dict_int8(state_dict):
    quantized, scales, dtypes, passthrough = {}, {}, {}, {}
    passthrough_orig_dtypes, qmeta = {}, {}
    stats = dict.fromkeys(("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"), 0)
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel()); stats["num_tensors"] += 1; stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1; passthrough[name] = t; stats["int8_payload_bytes"] += tensor_nbytes(t); continue
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes); passthrough[name] = kept; stats["int8_payload_bytes"] += tensor_nbytes(kept); continue
        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0: qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q; scales[name] = s; dtypes[name] = str(t.dtype).removeprefix("torch."); stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
    obj = {"__quant_format__": "int8_clean_per_row_v1", "quantized": quantized, "scales": scales, "dtypes": dtypes, "passthrough": passthrough}
    if qmeta: obj["qmeta"] = qmeta
    if passthrough_orig_dtypes: obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats

def dequantize_state_dict_int8(obj):
    out = {}
    qmeta = obj.get("qmeta", {}); passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name]); s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            out[name] = (q.float() * s.to(torch.float32).view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str): out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out

# Data loading
def load_data_shard(file):
    header_bytes = 256 * np.dtype("<i4").itemsize; token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1: raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    if file.stat().st_size != header_bytes + num_tokens * token_bytes: raise ValueError(f"Shard size mismatch for {file}")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens: raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

class TokenStream:
    def __init__(self, pattern):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files: raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0; self.tokens = load_data_shard(self.files[0]); self.pos = 0
    def _advance_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files); self.tokens = load_data_shard(self.files[self.file_idx]); self.pos = 0
    def take(self, n):
        chunks = []; remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0: self._advance_file(); continue
            k = min(remaining, avail); chunks.append(self.tokens[self.pos : self.pos + k]); self.pos += k; remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

class DistributedTokenLoader:
    def __init__(self, pattern, rank, world_size, device):
        self.rank, self.world_size, self.device = rank, world_size, device; self.stream = TokenStream(pattern)
    def next_batch(self, global_tokens, seq_len, grad_accum_steps):
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len); y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# -----------------------------
# xLSTM MODEL MODULES
# -----------------------------

def soft_cap(x: Tensor, cap: float) -> Tensor:
    return cap * torch.tanh(x / cap)


class CastedLinear(nn.Linear):
    """Linear layer that casts weights to input dtype at compute time."""
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)


class RMSNormWeighted(nn.Module):
    """RMS normalization with learnable weight."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        normed = F.rms_norm(x, (x.size(-1),), eps=self.eps)
        return normed * self.weight.to(dtype=normed.dtype)


class MultiHeadNorm(nn.Module):
    """RMS normalization applied per head, with learnable weight."""
    def __init__(self, num_heads: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.weight = nn.Parameter(torch.ones(num_heads * head_dim, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, NH, S, DH]
        B, NH, S, DH = x.shape
        x_fp = x.float()
        x_normed = x_fp * torch.rsqrt(x_fp.square().mean(dim=-1, keepdim=True) + self.eps)
        # Reshape to [B, S, NH*DH], apply weight, keep flat
        x_normed = x_normed.transpose(1, 2).reshape(B, S, NH * DH)
        x_normed = x_normed * self.weight.to(dtype=x_normed.dtype)
        return x_normed.to(dtype=x.dtype)


class PerHeadGate(nn.Module):
    """Per-head linear gate: each head has its own weight vector and scalar bias."""
    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_heads, head_dim))
        self.bias = nn.Parameter(torch.zeros(num_heads))

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, NH, S, D] -> [B, NH, S]
        w = self.weight.to(x.dtype)
        b = self.bias.to(x.dtype)
        return (x * w[None, :, None, :]).sum(dim=-1) + b[None, :, None]


def build_doc_mask(input_ids: Tensor, bos_id: int) -> Tensor:
    """Build a causal block-diagonal mask from BOS document boundaries.
    Tokens only attend to earlier tokens within the same document.
    Returns (B, 1, S, S) boolean mask."""
    bsz, seq_len = input_ids.shape
    doc_ids = (input_ids == bos_id).cumsum(dim=1)  # (B, S)
    same_doc = doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)  # (B, S, S)
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device))
    return (same_doc & causal).unsqueeze(1)  # (B, 1, S, S)


def segmented_cumsum(x: Tensor, reset_mask: Tensor) -> Tensor:
    """Cumulative sum along dim=-1 that restarts at positions where reset_mask is True.
    x: (B, NH, S), reset_mask: (B, S) boolean — True at document starts (BOS positions).
    Uses only static-shape ops so it works with torch.compile(fullgraph=True)."""
    rm = reset_mask.unsqueeze(1).expand_as(x)  # (B, NH, S)
    full_cumsum = torch.cumsum(x, dim=-1)
    # At reset position t, the prior accumulated sum is cumsum[t] - x[t].
    # We subtract this correction to restart the cumsum from x[t].
    correction_at_reset = full_cumsum - x
    # Use -inf at non-reset positions so cummax carries the latest correction forward.
    neg_inf = torch.tensor(-float("inf"), device=x.device, dtype=x.dtype)
    correction_sparse = torch.where(rm, correction_at_reset, neg_inf)
    carried_correction, _ = torch.cummax(correction_sparse, dim=-1)
    carried_correction = torch.where(carried_correction.isinf(), torch.zeros_like(carried_correction), carried_correction)
    return full_cumsum - carried_correction


class mLSTMLayer(nn.Module):
    """mLSTM layer matching the official xLSTM architecture (Beck et al. 2024).

    Up-proj -> split(x_mlstm, z) -> causal conv1d + SiLU on x_mlstm ->
    q,k from conv branch, v from raw -> parallel mLSTM cell ->
    multihead norm -> skip + SiLU(z) gating -> down-proj.
    """
    def __init__(self, dim: int, num_heads: int, qk_dim_factor: float, v_dim_factor: float,
                 gate_soft_cap: float, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.qk_dim = int(dim * qk_dim_factor)
        self.v_dim = int(dim * v_dim_factor)
        self.head_qk_dim = self.qk_dim // num_heads
        self.head_v_dim = self.v_dim // num_heads
        self.eps = eps
        self.scale = self.head_qk_dim ** -0.5

        # Up-projection: dim -> 2 * v_dim, split into x_mlstm and z
        self.up_proj = CastedLinear(dim, 2 * self.v_dim, bias=False)

        # Causal depthwise conv1d on x_mlstm branch (kernel=4, left-pad by 3)
        self.conv1d = nn.Conv1d(self.v_dim, self.v_dim, kernel_size=4,
                                padding=0, groups=self.v_dim, bias=True)

        # Q, K from conv-activated branch; V from raw x_mlstm
        self.c_q = CastedLinear(self.v_dim, self.qk_dim, bias=False)
        self.c_k = CastedLinear(self.v_dim, self.qk_dim, bias=False)
        self.c_v = CastedLinear(self.v_dim, self.v_dim, bias=False)

        # Per-head gates from concatenated [q, k, v] per head
        head_gate_dim = self.head_qk_dim * 2 + self.head_v_dim
        self.igate_preact = PerHeadGate(num_heads, head_gate_dim)
        self.fgate_preact = PerHeadGate(num_heads, head_gate_dim)

        # Learnable skip connection
        self.learnable_skip = nn.Parameter(torch.ones(1))

        # Multi-head norm and down-projection
        self.multihead_norm = MultiHeadNorm(num_heads, self.head_v_dim, eps=eps)
        self.out_proj = CastedLinear(self.v_dim, dim, bias=False)

    def forward(self, x: Tensor, q_delta=None, v_delta=None, doc_mask=None, doc_reset_mask=None) -> Tensor:
        B, S, D = x.shape
        NH = self.num_heads

        # Up-project and split into x_mlstm and z (output gate branch)
        up = self.up_proj(x)  # [B, S, 2 * v_dim]
        x_mlstm, z = up[..., :self.v_dim], up[..., self.v_dim:]  # each [B, S, v_dim]

        # Causal conv1d + SiLU on x_mlstm
        x_mlstm_t = x_mlstm.transpose(1, 2)  # [B, v_dim, S]
        x_conv = self.conv1d(F.pad(x_mlstm_t, (3, 0))).transpose(1, 2)  # [B, S, v_dim]
        if doc_reset_mask is not None:
            # Fix conv leakage across document boundaries. The conv (kernel=4) at
            # positions d, d+1, d+2 of a new document reads 1-3 tokens from the
            # previous doc. We re-run the conv on an input where the K-1 positions
            # before each BOS are zeroed, then blend only the affected positions.
            K = self.conv1d.kernel_size[0] - 1  # 3
            non_first_bos = doc_reset_mask.clone()
            non_first_bos[:, 0] = False  # first BOS is already correctly zero-padded
            # Zero pre-boundary positions: for BOS at t, zero t-1, t-2, t-3
            pre_boundary = torch.zeros(B, S, dtype=torch.bool, device=x.device)
            for k in range(1, K + 1):
                pre_boundary[:, :-k] |= non_first_bos[:, k:]
            x_mlstm_reset = x_mlstm_t.clone()
            x_mlstm_reset *= (~pre_boundary).unsqueeze(1)  # [B, v_dim, S]
            x_conv_reset = self.conv1d(F.pad(x_mlstm_reset, (3, 0))).transpose(1, 2)
            # Blend: use reset version for the first K positions of each non-first doc
            post_boundary = torch.zeros(B, S, dtype=torch.bool, device=x.device)
            for k in range(K):
                post_boundary[:, k:] |= non_first_bos[:, :S - k] if k > 0 else non_first_bos
            x_conv = torch.where(post_boundary.unsqueeze(-1), x_conv_reset, x_conv)
        x_conv_act = F.silu(x_conv)

        # Q, K from conv-activated; V from raw x_mlstm
        q = self.c_q(x_conv_act) + (q_delta if q_delta is not None else 0)
        q = q.reshape(B, S, NH, self.head_qk_dim).transpose(1, 2)  # [B, NH, S, d_qk]
        k = self.c_k(x_conv_act).reshape(B, S, NH, self.head_qk_dim).transpose(1, 2)
        v = self.c_v(x_mlstm) + (v_delta if v_delta is not None else 0)
        v = v.reshape(B, S, NH, self.head_v_dim).transpose(1, 2)  # [B, NH, S, d_v]

        # Per-head gates from concatenated [q, k, v]
        gate_input = torch.cat([q, k, v], dim=-1)  # [B, NH, S, head_gate_dim]
        i_pre = self.igate_preact(gate_input)  # [B, NH, S]
        f_pre = self.fgate_preact(gate_input)  # [B, NH, S]

        # Log-space gates: forget uses sigmoid, input uses exponential gating
        log_f = F.logsigmoid(f_pre)  # [B, NH, S]
        log_i = i_pre  # exponential input gate: log(exp(i)) = i

        # Cumulative log forget gate — segmented by document when masking is active
        if doc_reset_mask is not None:
            log_f_cumsum = segmented_cumsum(log_f, doc_reset_mask)  # [B, NH, S]
        else:
            log_f_cumsum = torch.cumsum(log_f, dim=-1)  # [B, NH, S]

        # Gating matrix: log_D[t,s] = cumsum_f[t] - cumsum_f[s] + log_i[s]
        log_D = (log_f_cumsum[:, :, :, None]
                 - log_f_cumsum[:, :, None, :]
                 + log_i[:, :, None, :])  # [B, NH, S, S]

        # Causal mask (also blocks cross-document attention when doc_mask is provided)
        if doc_mask is not None:
            # doc_mask is (B, 1, S, S) — squeeze to broadcast over NH
            log_D = torch.where(doc_mask, log_D, torch.tensor(-1e9, device=x.device))
        else:
            causal = torch.tril(torch.ones(S, S, dtype=torch.bool, device=x.device))
            log_D = torch.where(causal, log_D, torch.tensor(-1e9, device=x.device))

        # Stabilize: subtract row-wise max before exp
        max_log_D = log_D.max(dim=-1, keepdim=True).values  # [B, NH, S, 1]
        log_D = log_D - max_log_D
        log_D = log_D.clamp(max=80.0)
        D = torch.exp(log_D)  # [B, NH, S, S]

        # Gated attention
        qk = (q @ k.transpose(-2, -1)) * self.scale  # [B, NH, S, S]
        attn = D * qk

        # Weighted sum of values
        h = attn @ v  # [B, NH, S, d_v]

        # Normalizer: max(|sum of gated attention weights|, exp(-max_log_D)) + eps
        normalizer = torch.clamp(attn.sum(dim=-1, keepdim=True).abs(),
                                 min=torch.exp((-max_log_D).clamp(max=80.0)))
        h = h / (normalizer + self.eps)

        # Multi-head norm -> skip connection + SiLU(z) output gating -> project
        h_norm = self.multihead_norm(h)  # [B, S, v_dim]
        h_out = F.silu(z) * (h_norm + self.learnable_skip * x_conv_act)
        return self.out_proj(h_out)


class mLSTMBlock(nn.Module):
    """Pre-norm mLSTM with residual connection (no separate FFN — the mLSTM layer's
    internal up-proj/gate/down-proj provides channel mixing)."""
    def __init__(self, dim, num_heads, qk_dim_factor, v_dim_factor, gate_soft_cap):
        super().__init__()
        self.norm = RMSNormWeighted(dim)
        self.mlstm = mLSTMLayer(dim, num_heads, qk_dim_factor, v_dim_factor, gate_soft_cap)

    def forward(self, x: Tensor, q_delta_fn=None, v_delta_fn=None, doc_mask=None, doc_reset_mask=None) -> Tensor:
        n = self.norm(x)
        qd = q_delta_fn(n) if q_delta_fn is not None else None
        vd = v_delta_fn(n) if v_delta_fn is not None else None
        return x + self.mlstm(n, qd, vd, doc_mask=doc_mask, doc_reset_mask=doc_reset_mask)


class xLSTM(nn.Module):
    """xLSTM language model: embedding → mLSTM blocks → final norm → tied LM head."""
    def __init__(self, vocab_size, num_layers, dim, num_heads, qk_dim_factor, v_dim_factor,
                 gate_soft_cap, logit_softcap, tied_embed_init_std):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            mLSTMBlock(dim, num_heads, qk_dim_factor, v_dim_factor, gate_soft_cap)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNormWeighted(dim)
        self.tied_embed_init_std = tied_embed_init_std
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)
            if isinstance(module, mLSTMLayer):
                # Forget gate: weight=zeros (already), bias=linspace(3.0, 6.0)
                with torch.no_grad():
                    module.fgate_preact.bias.copy_(torch.linspace(3.0, 6.0, module.num_heads))
                # Input gate: weight=zeros (already), bias=normal(0, 0.1)
                nn.init.normal_(module.igate_preact.bias, mean=0.0, std=0.1)

    def forward(self, input_ids: Tensor, target_ids: Tensor, lora=None, doc_mask=None, doc_reset_mask=None) -> Tensor:
        x = self.tok_emb(input_ids)
        for i, block in enumerate(self.blocks):
            qd = lora.q_loras[i] if lora else None
            vd = lora.v_loras[i] if lora else None
            x = block(x, qd, vd, doc_mask=doc_mask, doc_reset_mask=doc_reset_mask)
        x = self.final_norm(x)
        logits = F.linear(x, self.tok_emb.weight)
        logits = logits + (lora.lm_head_lora(x) if lora else 0)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        if lora:
            bsz, sl, V = logits.shape
            return F.cross_entropy(logits.float().reshape(-1, V), target_ids.reshape(-1), reduction="none").reshape(bsz, sl)
        return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), target_ids.reshape(-1), reduction="mean")

def restore_low_dim_params_to_fp32(module):
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()

# -----------------------------
# TEST-TIME TRAINING (LoRA)
# -----------------------------

BOS_ID = 1

class BatchedLinearLoRA(nn.Module):
    def __init__(self, bsz, in_features, out_features, rank):
        super().__init__()
        self.in_features = in_features
        self.A = nn.Parameter(torch.empty(bsz, rank, in_features))
        self.B = nn.Parameter(torch.zeros(bsz, out_features, rank))
        self.reset()
    def forward(self, x): return (x @ self.A.transpose(1, 2)) @ self.B.transpose(1, 2)
    def reset(self):
        bound = 1.0 / math.sqrt(self.in_features)
        with torch.no_grad(): self.A.uniform_(-bound, bound); self.B.zero_()

class BatchedTTTLoRA(nn.Module):
    """LoRA on mLSTM q/v projections + LM head."""
    def __init__(self, bsz, model, rank):
        super().__init__()
        dim = model.tok_emb.embedding_dim
        vocab = model.tok_emb.num_embeddings
        self.lm_head_lora = BatchedLinearLoRA(bsz, dim, vocab, rank)
        self.q_loras = nn.ModuleList()
        self.v_loras = nn.ModuleList()
        for block in model.blocks:
            self.q_loras.append(BatchedLinearLoRA(bsz, dim, block.mlstm.c_q.weight.shape[0], rank))
            self.v_loras.append(BatchedLinearLoRA(bsz, dim, block.mlstm.c_v.weight.shape[0], rank))
    def reset(self):
        for m in self.modules():
            if isinstance(m, BatchedLinearLoRA): m.reset()

def _reset_ttt_optimizer(opt):
    for group in opt.param_groups:
        for p in group['params']:
            s = opt.state.get(p)
            if not s: continue
            s['exp_avg'].zero_(); s['exp_avg_sq'].zero_(); s['step'].fill_(0)

def _build_ttt_optimizer(lora, args):
    return torch.optim.Adam(lora.parameters(), lr=args.ttt_lora_lr, betas=(args.beta1, args.beta2), eps=1e-10)

def _find_docs(all_tokens, include_next_bos=True):
    bos_positions = (all_tokens == BOS_ID).nonzero(as_tuple=True)[0].numpy()
    docs = []
    for i in range(len(bos_positions)):
        start = int(bos_positions[i])
        end = int(bos_positions[i + 1]) if i + 1 < len(bos_positions) else all_tokens.numel()
        if include_next_bos and i + 1 < len(bos_positions): end += 1
        assert end - start >= 2
        docs.append((start, end - start))
    return docs

def _compute_chunk_window(ci, pred_len, num_chunks, chunk_size, eval_seq_len):
    chunk_start = ci * chunk_size
    chunk_end = pred_len if ci == num_chunks - 1 else (ci + 1) * chunk_size
    win_start = max(0, chunk_end - eval_seq_len)
    return win_start, chunk_end - win_start, chunk_start - win_start, chunk_end - chunk_start

def _accumulate_bpb(ptl, x, y, batch_i, chunk_offset, chunk_len, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, loss_sum, byte_sum, token_count):
    lbl = ptl[batch_i, chunk_offset:chunk_offset + chunk_len].to(torch.float64)
    prev = x[batch_i, chunk_offset:chunk_offset + chunk_len]; tgt = y[batch_i, chunk_offset:chunk_offset + chunk_len]
    tok_bytes = base_bytes_lut[tgt].to(torch.float64)
    tok_bytes += has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]
    loss_sum += lbl.sum(); byte_sum += tok_bytes.sum(); token_count += chunk_len

def eval_val_ttt_lora(args, base_model, rank, world_size, device, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
    files = sorted(glob.glob(args.val_files))
    all_tokens = torch.cat([load_data_shard(Path(f)) for f in files])
    docs = _find_docs(all_tokens)
    rank_docs = docs[(len(docs) * rank) // world_size : (len(docs) * (rank + 1)) // world_size]
    chunk_size, eval_seq_len, batch_size, lora_rank = args.ttt_chunk_size, args.ttt_eval_seq_len, args.ttt_batch_size, args.ttt_lora_rank
    rank_docs.sort(key=lambda d: (d[1] - 2) // chunk_size)
    base_model.eval()
    for p in base_model.parameters(): p.requires_grad_(False)
    lora = BatchedTTTLoRA(batch_size, base_model, lora_rank).to(device)
    opt = _build_ttt_optimizer(lora, args)
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    byte_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    for bi in range(0, len(rank_docs), batch_size):
        batch = rank_docs[bi:bi + batch_size]; bsz = len(batch)
        if bsz == batch_size:
            cur_lora, cur_opt = lora, opt; cur_lora.reset(); _reset_ttt_optimizer(cur_opt)
        else:
            cur_lora = BatchedTTTLoRA(bsz, base_model, lora_rank).to(device); cur_opt = _build_ttt_optimizer(cur_lora, args)
        pred_lens = [dl - 1 for _, dl in batch]
        num_chunks = [(pl + chunk_size - 1) // chunk_size for pl in pred_lens]
        max_nc = max(num_chunks)
        for ci in range(max_nc):
            cs = _compute_chunk_window(ci, (ci + 1) * chunk_size, ci + 1, chunk_size, eval_seq_len)
            context_size, chunk_offset = cs[1], cs[2]
            active = [ci < nc for nc in num_chunks]; needs_train = any(ci < nc - 1 for nc in num_chunks)
            x = torch.zeros(bsz, context_size, dtype=torch.int64, device=device)
            y = torch.zeros(bsz, context_size, dtype=torch.int64, device=device)
            doc_info = []
            for b in range(bsz):
                if not active[b]: doc_info.append((0, 0)); continue
                ds, dl = batch[b]
                ws, wl, co, cl = _compute_chunk_window(ci, pred_lens[b], num_chunks[b], chunk_size, eval_seq_len)
                chunk = all_tokens[ds + ws: ds + ws + wl + 1].to(dtype=torch.int64, device=device)
                x[b, :wl] = chunk[:-1]; y[b, :wl] = chunk[1:]; doc_info.append((co, cl))
            if needs_train:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16): ptl = base_model(x, y, lora=cur_lora)
            else:
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16): ptl = base_model(x, y, lora=cur_lora)
            with torch.no_grad():
                for b in range(bsz):
                    if not active[b]: continue
                    co, cl = doc_info[b]
                    _accumulate_bpb(ptl, x, y, b, co, cl, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, loss_sum, byte_sum, token_count)
            if needs_train:
                mask = torch.tensor([float(ci < num_chunks[b] - 1) for b in range(bsz)], device=device)
                per_doc = ptl[:, chunk_offset:chunk_offset + chunk_size].mean(dim=-1)
                cur_opt.zero_grad(); (per_doc * mask).sum().backward(); cur_opt.step()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM); dist.all_reduce(byte_sum, op=dist.ReduceOp.SUM); dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
    return float(loss_sum.item() / token_count.item()), float((loss_sum.item() / math.log(2.0)) / byte_sum.item())

# -----------------------------
# TRAINING
# -----------------------------

def main():
    global zeropower_via_newtonschulz5
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0")); world_size = int(os.environ.get("WORLD_SIZE", "1")); local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if "GRAD_ACCUM_STEPS" in os.environ:
        grad_accum_steps = int(os.environ["GRAD_ACCUM_STEPS"])
    else:
        if 8 % world_size != 0: raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
        grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank); torch.cuda.set_device(device)
    if distributed: dist.init_process_group(backend="nccl", device_id=device); dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)

    logfile = None
    if master_process: os.makedirs("logs", exist_ok=True); logfile = f"logs/{args.run_id}.txt"; print(logfile)
    def log0(msg, console=True):
        if not master_process: return
        if console: print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f: print(msg, file=f)

    log0(code, console=False); log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False); log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout, console=False)
    log0("=" * 100, console=False)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size: raise ValueError(f"VOCAB_SIZE mismatch")
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len, args.val_max_tokens)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")

    base_model = xLSTM(
        vocab_size=args.vocab_size, num_layers=args.num_layers, dim=args.model_dim,
        num_heads=args.num_heads, qk_dim_factor=args.qk_dim_factor, v_dim_factor=args.v_dim_factor,
        gate_soft_cap=args.gate_soft_cap, logit_softcap=args.logit_softcap,
        tied_embed_init_std=args.tied_embed_init_std,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear): module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # Optimizer split: 2D block params (excluding control) → Muon, rest → Adam
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [p for n, p in block_named_params if p.ndim == 2 and not any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    scalar_params = [p for n, p in block_named_params if p.ndim < 2 or any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    # final_norm params
    for p in base_model.final_norm.parameters(): scalar_params.append(p)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam([{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}], betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    optimizer_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps)
    for group in optimizer_muon.param_groups: group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam([{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}], betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params} architecture:xLSTM num_layers:{args.num_layers} dim:{args.model_dim} heads:{args.num_heads} pack_doc_mask:{args.pack_doc_mask}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps} seed:{args.seed}")

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    def zero_grad_all():
        for opt in optimizers: opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    def lr_mul(step, elapsed_ms):
        if args.warmdown_iters <= 0: return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1); warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    if args.warmup_steps > 0:
        initial_model_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed: model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                doc_mask = build_doc_mask(x, BOS_ID) if args.pack_doc_mask else None
                doc_reset = (x == BOS_ID) if args.pack_doc_mask else None
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16): warmup_loss = model(x, y, doc_mask=doc_mask, doc_reset_mask=doc_reset)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers: opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True): opt.load_state_dict(state)
        zero_grad_all()
        if distributed: model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    training_time_ms = 0.0; stop_after_step = None
    torch.cuda.synchronize(); t0 = time.perf_counter(); step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            torch.cuda.synchronize(); training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
            log0(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms")
            torch.cuda.synchronize(); t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed: model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            doc_mask = build_doc_mask(x, BOS_ID) if args.pack_doc_mask else None
            doc_reset = (x == BOS_ID) if args.pack_doc_mask else None
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(x, y, doc_mask=doc_mask, doc_reset_mask=doc_reset)
            train_loss += loss.detach(); (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        for group in optimizer_muon.param_groups: group["momentum"] = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for opt in optimizers:
            for group in opt.param_groups: group["lr"] = group["base_lr"] * scale
        if args.grad_clip_norm > 0: torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers: opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None):
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms")
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device); dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX); reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap: stop_after_step = step

    log0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB")

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        log0(f"Serialized model: {os.path.getsize('final_model.pt')} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO(); torch.save(quant_obj, quant_buf)
    quant_blob = zlib.compress(quant_buf.getvalue(), level=9)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f: f.write(quant_blob)
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(f"Serialized model int8+zlib: {os.path.getsize('final_model.int8.ptz')} bytes (payload_ratio:{ratio:.2f}x)")

    if distributed: dist.barrier()
    with open("final_model.int8.ptz", "rb") as f: quant_blob_disk = f.read()
    base_model.load_state_dict(dequantize_state_dict_int8(torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")), strict=True)
    torch.cuda.synchronize(); t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    torch.cuda.synchronize()
    log0(f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms")
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    torch._dynamo.reset(); torch.cuda.synchronize(); t_ttt = time.perf_counter()
    ttt_val_loss, ttt_val_bpb = eval_val_ttt_lora(args, base_model, rank, world_size, device, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    torch.cuda.synchronize()
    log0(f"final_int8_ttt_lora val_loss:{ttt_val_loss:.4f} val_bpb:{ttt_val_bpb:.4f} eval_time:{1000.0 * (time.perf_counter() - t_ttt):.0f}ms")

    if distributed: dist.destroy_process_group()

if __name__ == "__main__":
    main()
