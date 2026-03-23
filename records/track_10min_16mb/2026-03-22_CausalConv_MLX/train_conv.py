"""
Causal Convolutional LM — PyTorch implementation.

Replaces transformer attention blocks with causal depthwise-separable 1D convolutions
of increasing kernel sizes. Each block has a depthwise causal conv for spatial mixing +
pointwise relu^2 MLP for channel mixing, both writing to the residual stream.
Maintains encoder/decoder skip connections and x0 residual mixing from the baseline.
"""
from __future__ import annotations

import copy
import datetime
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
# KERNEL SIZES
# -----------------------------

KERNEL_SIZES = [
    int(k)
    for k in os.environ.get(
        "KERNEL_SIZES", "3,3,3,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5"
    ).split(",")
]

# -----------------------------
# HYPERPARAMETERS
# -----------------------------

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = f"{os.environ.get('RUN_ID', str(uuid.uuid4()))}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
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

    # Model.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 23))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    mlp_mult = int(os.environ.get("MLP_MULT", 1))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    conv_expand = int(os.environ.get("CONV_EXPAND", 1))

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

# -----------------------------
# MUON OPTIMIZER
# -----------------------------

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "conv_scale,mlp_scale,conv_scales,mlp_scales,resid_mix,resid_mixes,skip_weight,skip_weights",
    ).split(",") if p
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    p for p in os.environ.get("INT8_KEEP_FLOAT_FP32_NAME_PATTERNS", ",".join(CONTROL_TENSOR_NAME_PATTERNS)).split(",") if p
)

def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16(); X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed: X = X.T
    for _ in range(steps):
        A = X @ X.T; B = b * A + c * A @ A; X = a * X + B @ X
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
        world_size = dist.get_world_size() if distributed else 1; rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params: continue
            lr, momentum, backend_steps, nesterov = group["lr"], group["momentum"], group["backend_steps"], group["nesterov"]
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad; state = self.state[p]
                    if "momentum_buffer" not in state: state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]; buf.mul_(momentum).add_(g)
                    if nesterov: g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed: dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype); p.add_(g, alpha=-lr); curr += p.numel()
        return loss

# -----------------------------
# INFRASTRUCTURE
# -----------------------------

def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab_size = int(sp.vocab_size()); table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16); has_leading_space_np = np.zeros((table_size,), dtype=np.bool_); is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for tid in range(sp_vocab_size):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid): continue
        is_boundary_token_np[tid] = False
        if sp.is_byte(tid): base_bytes_np[tid] = 1; continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"): has_leading_space_np[tid] = True; piece = piece[1:]
        base_bytes_np[tid] = len(piece.encode("utf-8"))
    return (torch.tensor(base_bytes_np, dtype=torch.int16, device=device), torch.tensor(has_leading_space_np, dtype=torch.bool, device=device), torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device))

def load_validation_tokens(pattern, seq_len, max_tokens=0):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files: raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    if max_tokens > 0: tokens = tokens[: max_tokens + 1]
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[: usable + 1]

def eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size; seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64); val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bss in range(seq_start, seq_end, local_batch_seqs):
            bse = min(bss + local_batch_seqs, seq_end)
            local = val_tokens[bss * args.train_seq_len : bse * args.train_seq_len + 1].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len); y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16): batch_loss = model(x, y).detach()
            btc = float(y.numel()); val_loss_sum += batch_loss.to(torch.float64) * btc; val_token_count += btc
            prev_ids = x.reshape(-1); tgt_ids = y.reshape(-1)
            tb = base_bytes_lut[tgt_ids].to(torch.int16)
            tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.int16)
            val_byte_count += tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM); dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM); dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    vl = val_loss_sum / val_token_count; bpt = vl.item() / math.log(2.0); tpb = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(vl.item()), float(bpt * tpb)

INT8_KEEP_FLOAT_MAX_NUMEL = 65_536; INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16; INT8_PER_ROW_SCALE_DTYPE = torch.float16; INT8_CLIP_Q = 99.99984 / 100.0
def tensor_nbytes(t): return int(t.numel()) * int(t.element_size())
def keep_float_tensor(name, t, pod):
    if any(p in name for p in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS): return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}: pod[name] = str(t.dtype).removeprefix("torch."); return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t
def quantize_float_tensor(t):
    t32 = t.float()
    if t32.ndim == 2:
        ca = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1) if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32)
        cl = torch.maximum(torch.minimum(t32, ca[:, None]), -ca[:, None]); sc = (ca / 127.0).clamp_min(1.0 / 127.0)
        return torch.clamp(torch.round(cl / sc[:, None]), -127, 127).to(torch.int8).contiguous(), sc.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    ca = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    sc = torch.tensor(ca / 127.0 if ca > 0 else 1.0, dtype=torch.float32)
    return torch.clamp(torch.round(torch.clamp(t32, -ca, ca) / sc), -127, 127).to(torch.int8).contiguous(), sc
def quantize_state_dict_int8(sd):
    quantized, scales, dtypes, passthrough, pod, qmeta = {}, {}, {}, {}, {}, {}
    stats = dict.fromkeys(("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"), 0)
    for name, tensor in sd.items():
        t = tensor.detach().to("cpu").contiguous(); stats["param_count"] += int(t.numel()); stats["num_tensors"] += 1; stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if not t.is_floating_point(): stats["num_nonfloat_tensors"] += 1; passthrough[name] = t; stats["int8_payload_bytes"] += tensor_nbytes(t); continue
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL: kept = keep_float_tensor(name, t, pod); passthrough[name] = kept; stats["int8_payload_bytes"] += tensor_nbytes(kept); continue
        stats["num_float_tensors"] += 1; q, s = quantize_float_tensor(t)
        if s.ndim > 0: qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q; scales[name] = s; dtypes[name] = str(t.dtype).removeprefix("torch."); stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
    obj = {"__quant_format__": "int8_clean_per_row_v1", "quantized": quantized, "scales": scales, "dtypes": dtypes, "passthrough": passthrough}
    if qmeta: obj["qmeta"] = qmeta
    if pod: obj["passthrough_orig_dtypes"] = pod
    return obj, stats
def dequantize_state_dict_int8(obj):
    out = {}; qmeta = obj.get("qmeta", {}); pod = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dt = getattr(torch, obj["dtypes"][name]); s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            out[name] = (q.float() * s.to(torch.float32).view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dt).contiguous()
        else: out[name] = (q.float() * float(s.item())).to(dtype=dt).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous(); od = pod.get(name)
        if isinstance(od, str): out_t = out_t.to(dtype=getattr(torch, od)).contiguous()
        out[name] = out_t
    return out

def load_data_shard(file):
    hb = 256 * np.dtype("<i4").itemsize; tb = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1: raise ValueError(f"Unexpected shard header for {file}")
    nt = int(header[2])
    if file.stat().st_size != hb + nt * tb: raise ValueError(f"Shard size mismatch for {file}")
    tokens_np = np.fromfile(file, dtype="<u2", count=nt, offset=hb)
    if tokens_np.size != nt: raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

class TokenStream:
    def __init__(self, pattern):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files: raise FileNotFoundError(f"No files: {pattern}")
        self.file_idx = 0; self.tokens = load_data_shard(self.files[0]); self.pos = 0
    def _advance_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files); self.tokens = load_data_shard(self.files[self.file_idx]); self.pos = 0
    def take(self, n):
        chunks = []; rem = n
        while rem > 0:
            av = self.tokens.numel() - self.pos
            if av <= 0: self._advance_file(); continue
            k = min(rem, av); chunks.append(self.tokens[self.pos:self.pos + k]); self.pos += k; rem -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

class DistributedTokenLoader:
    def __init__(self, pattern, rank, world_size, device):
        self.rank, self.world_size, self.device = rank, world_size, device; self.stream = TokenStream(pattern)
    def next_batch(self, global_tokens, seq_len, grad_accum_steps):
        lt = global_tokens // (self.world_size * grad_accum_steps); prs = lt + 1
        chunk = self.stream.take(prs * self.world_size); start = self.rank * prs
        local = chunk[start:start + prs].to(dtype=torch.int64)
        return local[:-1].reshape(-1, seq_len).to(self.device, non_blocking=True), local[1:].reshape(-1, seq_len).to(self.device, non_blocking=True)

# -----------------------------
# CONV MODEL MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps=None): super().__init__(); self.eps = eps
    def forward(self, x): return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class CastedLinear(nn.Linear):
    def forward(self, x): return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)

def restore_low_dim_params_to_fp32(module):
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class CausalDepthwiseConv(nn.Module):
    """Causal depthwise 1D convolution — each channel gets its own kernel.
    Input: (B, T, D), Output: (B, T, D).
    Causality via left-padding by (kernel_size - 1).
    """
    def __init__(self, dim: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.dim = dim
        scale = 1.0 / math.sqrt(kernel_size)
        # PyTorch conv1d depthwise: weight shape (C_out, C_in/groups, K) = (dim, 1, K)
        self.weight = nn.Parameter(torch.randn(dim, 1, kernel_size) * scale)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, D) → (B, D, T) for conv1d
        xt = x.transpose(1, 2)
        if self.kernel_size > 1:
            xt = F.pad(xt, (self.kernel_size - 1, 0))
        y = F.conv1d(xt, self.weight.to(xt.dtype), groups=self.dim)
        return y.transpose(1, 2)  # (B, T, D)


class MLP(nn.Module):
    """Pointwise MLP with relu^2 activation."""
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = dim * mlp_mult
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class ConvBlock(nn.Module):
    """Conv block: RMSNorm → CausalConv → scale → residual, RMSNorm → MLP → scale → residual."""
    def __init__(self, dim: int, kernel_size: int, mlp_mult: int):
        super().__init__()
        self.conv_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.conv = CausalDepthwiseConv(dim, kernel_size)
        self.mlp = MLP(dim, mlp_mult)
        self.conv_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        conv_out = self.conv(self.conv_norm(x))
        x = x + self.conv_scale.to(dtype=x.dtype)[None, None, :] * conv_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class ConvLM(nn.Module):
    """Convolutional LM with encoder/decoder skip structure."""
    def __init__(self, vocab_size, num_layers, model_dim, mlp_mult, kernel_sizes,
                 logit_softcap, tied_embed_init_std):
        super().__init__()
        if logit_softcap <= 0.0: raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        if len(kernel_sizes) != num_layers: raise ValueError(f"Need {num_layers} kernel sizes, got {len(kernel_sizes)}")
        self.logit_softcap = logit_softcap
        self.tied_embed_init_std = tied_embed_init_std
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([ConvBlock(model_dim, kernel_sizes[i], mlp_mult) for i in range(num_layers)])
        self.final_norm = RMSNorm()
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor, lora=None) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []

        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        x = self.final_norm(x)
        logits = F.linear(x, self.tok_emb.weight)
        logits = logits + (lora.lm_head_lora(x) if lora else 0)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        if lora:
            bsz, sl, V = logits.shape
            return F.cross_entropy(logits.float().reshape(-1, V), target_ids.reshape(-1), reduction="none").reshape(bsz, sl)
        return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), target_ids.reshape(-1), reduction="mean")


# -----------------------------
# TEST-TIME TRAINING (LoRA) — LM head only (no attention q/v)
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
    """LM head LoRA only — conv blocks have no q/v projections."""
    def __init__(self, bsz, model, rank):
        super().__init__()
        dim = model.tok_emb.embedding_dim; vocab = model.tok_emb.num_embeddings
        self.lm_head_lora = BatchedLinearLoRA(bsz, dim, vocab, rank)
        # Dummy empty lists so the model forward doesn't need branching
        self.q_loras = nn.ModuleList()
        self.v_loras = nn.ModuleList()
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
        assert end - start >= 2; docs.append((start, end - start))
    return docs

def _compute_chunk_window(ci, pred_len, num_chunks, chunk_size, eval_seq_len):
    cs = ci * chunk_size; ce = pred_len if ci == num_chunks - 1 else (ci + 1) * chunk_size
    ws = max(0, ce - eval_seq_len); return ws, ce - ws, cs - ws, ce - cs

def _accumulate_bpb(ptl, x, y, bi, co, cl, bbl, hsl, ibt, ls, bs, tc):
    lbl = ptl[bi, co:co + cl].to(torch.float64); prev = x[bi, co:co + cl]; tgt = y[bi, co:co + cl]
    tb = bbl[tgt].to(torch.float64); tb += hsl[tgt] & ~ibt[prev]
    ls += lbl.sum(); bs += tb.sum(); tc += cl

def eval_val_ttt_lora(args, base_model, rank, world_size, device, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
    files = sorted(glob.glob(args.val_files)); all_tokens = torch.cat([load_data_shard(Path(f)) for f in files])
    docs = _find_docs(all_tokens)
    rank_docs = docs[(len(docs) * rank) // world_size : (len(docs) * (rank + 1)) // world_size]
    cs, esl, bs, lr = args.ttt_chunk_size, args.ttt_eval_seq_len, args.ttt_batch_size, args.ttt_lora_rank
    rank_docs.sort(key=lambda d: (d[1] - 2) // cs)
    base_model.eval()
    for p in base_model.parameters(): p.requires_grad_(False)
    lora = BatchedTTTLoRA(bs, base_model, lr).to(device); opt = _build_ttt_optimizer(lora, args)
    loss_sum = torch.zeros((), device=device, dtype=torch.float64); byte_sum = torch.zeros((), device=device, dtype=torch.float64); token_count = torch.zeros((), device=device, dtype=torch.float64)
    for bi in range(0, len(rank_docs), bs):
        batch = rank_docs[bi:bi + bs]; bsz = len(batch)
        if bsz == bs: cur_lora, cur_opt = lora, opt; cur_lora.reset(); _reset_ttt_optimizer(cur_opt)
        else: cur_lora = BatchedTTTLoRA(bsz, base_model, lr).to(device); cur_opt = _build_ttt_optimizer(cur_lora, args)
        pred_lens = [dl - 1 for _, dl in batch]; num_chunks = [(pl + cs - 1) // cs for pl in pred_lens]; max_nc = max(num_chunks)
        for ci in range(max_nc):
            cw = _compute_chunk_window(ci, (ci + 1) * cs, ci + 1, cs, esl); context_size, chunk_offset = cw[1], cw[2]
            active = [ci < nc for nc in num_chunks]; needs_train = any(ci < nc - 1 for nc in num_chunks)
            x = torch.zeros(bsz, context_size, dtype=torch.int64, device=device); y = torch.zeros(bsz, context_size, dtype=torch.int64, device=device); doc_info = []
            for b in range(bsz):
                if not active[b]: doc_info.append((0, 0)); continue
                ds, dl = batch[b]; ws, wl, co, cl = _compute_chunk_window(ci, pred_lens[b], num_chunks[b], cs, esl)
                chunk = all_tokens[ds + ws:ds + ws + wl + 1].to(dtype=torch.int64, device=device)
                x[b, :wl] = chunk[:-1]; y[b, :wl] = chunk[1:]; doc_info.append((co, cl))
            if needs_train:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16): ptl = base_model(x, y, lora=cur_lora)
            else:
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16): ptl = base_model(x, y, lora=cur_lora)
            with torch.no_grad():
                for b in range(bsz):
                    if not active[b]: continue
                    co, cl = doc_info[b]; _accumulate_bpb(ptl, x, y, b, co, cl, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, loss_sum, byte_sum, token_count)
            if needs_train:
                mask = torch.tensor([float(ci < num_chunks[b] - 1) for b in range(bsz)], device=device)
                per_doc = ptl[:, chunk_offset:chunk_offset + cs].mean(dim=-1)
                cur_opt.zero_grad(); (per_doc * mask).sum().backward(); cur_opt.step()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM); dist.all_reduce(byte_sum, op=dist.ReduceOp.SUM); dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
    return float(loss_sum.item() / token_count.item()), float((loss_sum.item() / math.log(2.0)) / byte_sum.item())

# -----------------------------
# TRAINING
# -----------------------------

def main():
    global zeropower_via_newtonschulz5
    code = Path(__file__).read_text(encoding="utf-8"); args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0")); world_size = int(os.environ.get("WORLD_SIZE", "1")); local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if 8 % world_size != 0: raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    grad_accum_steps = 8 // world_size; grad_scale = 1.0 / grad_accum_steps
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

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size: raise ValueError("VOCAB_SIZE mismatch")
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len, args.val_max_tokens)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")

    kernel_sizes = KERNEL_SIZES[:args.num_layers]
    if len(kernel_sizes) < args.num_layers:
        kernel_sizes += [kernel_sizes[-1]] * (args.num_layers - len(kernel_sizes))

    base_model = ConvLM(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        mlp_mult=args.mlp_mult, kernel_sizes=kernel_sizes,
        logit_softcap=args.logit_softcap, tied_embed_init_std=args.tied_embed_init_std,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear): module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    log0(f"architecture:convolutional kernel_sizes:{kernel_sizes}")

    # Optimizer: 2D CastedLinear weights → Muon, everything else (conv weight is 3D, scales, etc.) → Adam
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [p for n, p in block_named_params if p.ndim == 2 and not any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    scalar_params = [p for n, p in block_named_params if p.ndim != 2 or any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    if base_model.skip_weights.numel() > 0: scalar_params.append(base_model.skip_weights)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam([{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}], betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    optimizer_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps)
    for group in optimizer_muon.param_groups: group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam([{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}], betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params} num_layers:{args.num_layers} dim:{args.model_dim} mlp_mult:{args.mlp_mult}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps} seed:{args.seed}")

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    def zero_grad_all():
        for opt in optimizers: opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    def lr_mul(step, elapsed_ms):
        if args.warmdown_iters <= 0: return 1.0
        if max_wallclock_ms is None:
            ws = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if ws <= step < args.iterations else 1.0
        sms = elapsed_ms / max(step, 1); wms = args.warmdown_iters * sms; rms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return rms / max(wms, 1e-9) if rms <= wms else 1.0

    if args.warmup_steps > 0:
        initial_model_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed: model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16): wl = model(x, y)
                (wl * grad_scale).backward()
            for opt in optimizers: opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True): opt.load_state_dict(state)
        zero_grad_all()
        if distributed: model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    training_time_ms = 0.0; stop_after_step = None; torch.cuda.synchronize(); t0 = time.perf_counter(); step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            torch.cuda.synchronize(); training_time_ms += 1000.0 * (time.perf_counter() - t0)
            vl, vb = eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
            log0(f"step:{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vb:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms")
            torch.cuda.synchronize(); t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations: log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0); scale = lr_mul(step, elapsed_ms)
        zero_grad_all(); train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed: model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(x, y)
            train_loss += loss.detach(); (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        for group in optimizer_muon.param_groups: group["momentum"] = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for opt in optimizers:
            for group in opt.param_groups: group["lr"] = group["base_lr"] * scale
        if args.grad_clip_norm > 0: torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers: opt.step()
        zero_grad_all()

        step += 1; atm = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None):
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} train_time:{atm:.0f}ms step_avg:{atm / step:.2f}ms")
        reached_cap = max_wallclock_ms is not None and atm >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            rct = torch.tensor(int(reached_cap), device=device); dist.all_reduce(rct, op=dist.ReduceOp.MAX); reached_cap = bool(rct.item())
        if stop_after_step is None and reached_cap: stop_after_step = step

    log0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB")

    if master_process: torch.save(base_model.state_dict(), "final_model.pt"); log0(f"Serialized model: {os.path.getsize('final_model.pt')} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    qb = io.BytesIO(); torch.save(quant_obj, qb); quant_blob = zlib.compress(qb.getvalue(), level=9)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f: f.write(quant_blob)
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(f"Serialized model int8+zlib: {os.path.getsize('final_model.int8.ptz')} bytes (payload_ratio:{ratio:.2f}x)")

    if distributed: dist.barrier()
    with open("final_model.int8.ptz", "rb") as f: qbd = f.read()
    base_model.load_state_dict(dequantize_state_dict_int8(torch.load(io.BytesIO(zlib.decompress(qbd)), map_location="cpu")), strict=True)
    torch.cuda.synchronize(); tq = time.perf_counter()
    qvl, qvb = eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    torch.cuda.synchronize()
    log0(f"final_int8_zlib_roundtrip val_loss:{qvl:.4f} val_bpb:{qvb:.4f} eval_time:{1000.0 * (time.perf_counter() - tq):.0f}ms")
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{qvl:.8f} val_bpb:{qvb:.8f}")

    torch._dynamo.reset(); torch.cuda.synchronize(); tt = time.perf_counter()
    tvl, tvb = eval_val_ttt_lora(args, base_model, rank, world_size, device, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    torch.cuda.synchronize()
    log0(f"final_int8_ttt_lora val_loss:{tvl:.4f} val_bpb:{tvb:.4f} eval_time:{1000.0 * (time.perf_counter() - tt):.0f}ms")

    if distributed: dist.destroy_process_group()

if __name__ == "__main__":
    main()
