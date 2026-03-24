"""
xLSTM mLSTM Language Model — using the official xlstm + mlstm_kernels packages.

Uses xLSTMLarge from the NX-AI xlstm package with Triton chunkwise kernels
from mlstm_kernels. Architecture: embedding → mLSTM blocks → final norm → LM head.
No weight tying (package does not support it).

Training infrastructure (Muon, DDP, data loading, eval, int8 quantization) is
shared with the hand-rolled train_xlstm.py.
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

from xlstm.xlstm_large.model import xLSTMLarge, xLSTMLargeConfig

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

    # Model — xLSTM architecture (maps to xLSTMLargeConfig).
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 6))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 4))
    qk_dim_factor = float(os.environ.get("QK_DIM_FACTOR", 0.5))
    v_dim_factor = float(os.environ.get("V_DIM_FACTOR", 1.0))
    ffn_proj_factor = float(os.environ.get("FFN_PROJ_FACTOR", 2.6667))
    gate_soft_cap = float(os.environ.get("GATE_SOFT_CAP", 15.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    embed_init_std = float(os.environ.get("EMBED_INIT_STD", 0.005))

    # Kernel backend (native = pure PyTorch, no Triton; avoids LLVM bug).
    # Triton alternatives: "chunkwise--triton_xl_chunk", "native_sequence__triton", "triton"
    chunkwise_kernel = os.environ.get("CHUNKWISE_KERNEL", "chunkwise--native_autograd")
    sequence_kernel = os.environ.get("SEQUENCE_KERNEL", "native_sequence__native")
    step_kernel = os.environ.get("STEP_KERNEL", "native")
    chunk_size = int(os.environ.get("CHUNK_SIZE", 64))

    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
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

# Control tensor patterns — 1D / norm / gate biases get Adam, not Muon.
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

# -----------------------------
# MUON OPTIMIZER
# -----------------------------

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

def eval_val(args, model, forward_fn, rank, world_size, device, grad_accum_steps, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
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
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = forward_fn(x, y).detach()
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
# MODEL CONSTRUCTION & INIT
# -----------------------------

def build_model(args):
    """Build an xLSTMLarge model from hyperparameters and apply custom init."""
    config = xLSTMLargeConfig(
        embedding_dim=args.model_dim,
        num_heads=args.num_heads,
        num_blocks=args.num_layers,
        vocab_size=args.vocab_size,
        qk_dim_factor=args.qk_dim_factor,
        v_dim_factor=args.v_dim_factor,
        ffn_proj_factor=args.ffn_proj_factor,
        gate_soft_cap=args.gate_soft_cap,
        output_logit_soft_cap=args.logit_softcap,
        chunkwise_kernel=args.chunkwise_kernel,
        sequence_kernel=args.sequence_kernel,
        step_kernel=args.step_kernel,
        chunk_size=args.chunk_size,
        mode="train",
        return_last_states=False,
    )
    model = xLSTMLarge(config)
    _custom_init(model, args)
    return model


def _custom_init(model: xLSTMLarge, args):
    """Apply our custom weight init on top of the package defaults."""
    with torch.no_grad():
        # Embedding: small normal init
        nn.init.normal_(model.embedding.weight, mean=0.0, std=args.embed_init_std)

        # LM head: small normal init (no weight tying in xLSTMLarge)
        nn.init.normal_(model.lm_head.weight, mean=0.0, std=args.embed_init_std)

        # Per-block gate bias init
        for block in model.backbone.blocks:
            layer = block.mlstm_layer
            # Forget gate bias: linspace(3.0, 6.0, num_heads)
            if hasattr(layer, "fgate_preact") and layer.fgate_preact.bias is not None:
                layer.fgate_preact.bias.copy_(torch.linspace(3.0, 6.0, args.num_heads))
            # Input gate bias: normal(0, 0.1)
            if hasattr(layer, "igate_preact") and layer.igate_preact.bias is not None:
                nn.init.normal_(layer.igate_preact.bias, mean=0.0, std=0.1)
            # Fused mode
            if hasattr(layer, "ifgate_preact") and layer.ifgate_preact.bias is not None:
                nh = args.num_heads
                layer.ifgate_preact.bias[:nh].normal_(mean=0.0, std=0.1)
                layer.ifgate_preact.bias[nh:].copy_(torch.linspace(3.0, 6.0, nh))


def restore_low_dim_params_to_fp32(module):
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()

# -----------------------------
# TRAINING
# -----------------------------

def main():
    global zeropower_via_newtonschulz5
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    # zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

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

    base_model = build_model(args).to(device).bfloat16()
    restore_low_dim_params_to_fp32(base_model)

    if distributed:
        model = DDP(base_model, device_ids=[local_rank], broadcast_buffers=False)
    else:
        model = base_model

    def train_forward(x, y):
        logits = model(x)
        return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                               y.reshape(-1), reduction="mean")

    # Optimizer split: 2D backbone params (excluding control) → Muon, rest → Adam.
    # Separate embedding and lm_head since no weight tying.
    backbone_named_params = list(base_model.backbone.named_parameters())
    matrix_params = [p for n, p in backbone_named_params if p.ndim == 2 and not any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    scalar_params = [p for n, p in backbone_named_params if p.ndim < 2 or any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]

    optimizer_embed = torch.optim.Adam(
        [{"params": [base_model.embedding.weight], "lr": args.embed_lr, "base_lr": args.embed_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizer_head = torch.optim.Adam(
        [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizer_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps)
    for group in optimizer_muon.param_groups: group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizers = [optimizer_embed, optimizer_head, optimizer_muon, optimizer_scalar]

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params} architecture:xLSTMLarge(pkg) num_layers:{args.num_layers} dim:{args.model_dim} heads:{args.num_heads}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps} seed:{args.seed}")
    log0(f"kernel: chunkwise={args.chunkwise_kernel} chunk_size={args.chunk_size}")

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
        base_model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed: model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16): warmup_loss = train_forward(x, y)
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

    # Eval forward uses non-DDP path
    def eval_forward(x, y):
        logits = base_model(x)
        return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                               y.reshape(-1), reduction="mean")

    training_time_ms = 0.0; stop_after_step = None
    torch.cuda.synchronize(); t0 = time.perf_counter(); step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            torch.cuda.synchronize(); training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(args, base_model, eval_forward, rank, world_size, device, grad_accum_steps, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
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
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = train_forward(x, y)
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
    q_val_loss, q_val_bpb = eval_val(args, base_model, eval_forward, rank, world_size, device, grad_accum_steps, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    torch.cuda.synchronize()
    log0(f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms")
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed: dist.destroy_process_group()

if __name__ == "__main__":
    main()
