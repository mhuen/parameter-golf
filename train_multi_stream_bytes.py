"""
Multi-Stream GPT — byte-level training script.

Trains MultiStreamGPT using EfficientByteTokenizer (byte260 shards).
Each token represents exactly one UTF-8 byte, so BPB = bits_per_token.

Follows the same training pipeline as train_universal_bytes.py:
  - Byte260 data loading with token remapping
  - Muon optimizer for 2D matrix params, Adam for scalars/embeddings
  - INT8 per-row quantization + zlib compression
  - Roundtrip dequantization validation

New features:
  - Bigram prior initialization: computes bigram statistics from N training
    sequences and sets BigramPriorLayer weights before training starts.
  - Multi-stream parameter routing: dedicated optimizer groups for bigram
    prior, builder embeddings, conv weights, matrix params, and scalars.

Tokenizer configs (via env vars):
    DISCARD_UNUSED_BYTES=1  (default) 206 used bytes, vocab=208
    DISCARD_UNUSED_BYTES=0           all 256 bytes, vocab=258
    FOLD=uppercase                   fold uppercase->lowercase, vocab=182

Data: expects byte260 shards (PureByteTokenizer format: bos=1, bytes=4..259
as uint16).  Tokens are remapped on-the-fly to EfficientByteTokenizer IDs.
"""

from __future__ import annotations

import copy
from datetime import datetime
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

from efficient_byte_tokenizer import ByteCategory, EfficientByteTokenizer
from multi_stream_gpt import (
    MultiStreamGPT,
    build_multi_stream_gpt,
)
from multi_stream_attention import StreamMixingConfig, MixingMode, MixingSource
from multi_streams import StreamID, StreamType
from modules import (
    CastedLinear,
    KroneckerLinear,
    MonarchLinear,
    restore_low_dim_params_to_fp32,
)
import optim as _optim_mod
from optim import Muon, _GRAM_NS_LIB
from debug_nan import NaNWatchdog
from data import (
    load_validation_tokens_byte260,
    DistributedTokenLoaderByte260,
)


# fmt: off
base_train_data = {
    "step": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900, 6000, 6100, 6200, 6300, 6400, 6500, 6600, 6700, 6800, 6900, 7000, 7100, 7200, 7300, 7400, 7500, 7600, 7700, 7800, 7900, 8000, 8100, 8200, 8300, 8400, 8500, 8600, 8700, 8800, 8900, 9000, 9100, 9200, 9300, 9400, 9500, 9600, 9700, 9800, 9900, 10000, 10100, 10200, 10300, 10400, 10500, 10600, 10700, 10800, 10900, 11000, 11100, 11200, 11300, 11400, 11500, 11600, 11700, 11800, 11900, 12000, 12100, 12200, 12300, 12400, 12500, 12600, 12700, 12800, 12900, 13000, 13100, 13200, 13300, 13400, 13500, 13600, 13700, 13800, 13900, 14000, 14100, 14200, 14300, 14400, 14500, 14600, 14700, 14800, 14900, 15000, 15100, 15200, 15300, 15400, 15500, 15600, 15700, 15800, 15900, 16000, 16100, 16200, 16300, 16400, 16500, 16600, 16700, 16800, 16900, 17000, 17100, 17200, 17300, 17400, 17500, 17600, 17700, 17800, 17900, 18000, 18100, 18200, 18300, 18400, 18500, 18600, 18700, 18800, 18900, 19000, 19100, 19200, 19300, 19400, 19500, 19600, 19700, 19800, 19900, 20000, 20100, 20200, 20300, 20400, 20500, 20600, 20700, 20800, 20900, 21000, 21100, 21200],
    "train_loss": [None, 5.4939, 3.8885, 3.8816, 3.3941, 3.0916, 3.0635, 2.8276, 2.8205, 2.6998, 2.7295, 1.9779, 1.4358, 1.4482, 1.4201, 1.2940, 1.3671, 1.1707, 1.1844, 1.2673, 1.4021, 1.0566, 1.1859, 1.1906, 1.1656, 1.1182, 1.1402, 1.1932, 1.0538, 1.1404, 1.0785, 0.8387, 0.9971, 1.1331, 1.0612, 1.0681, 1.1861, 1.0920, 1.0951, 1.0147, 1.1491, 1.0533, 1.0923, 1.0792, 1.0205, 1.0153, 1.0788, 1.1128, 1.0348, 1.0777, 1.0596, 1.0766, 1.3711, 0.8753, 1.0684, 1.0045, 1.1038, 0.9525, 0.9963, 1.0321, 0.9826, 1.0353, 1.0608, 0.9295, 1.0609, 0.9936, 0.9700, 1.0752, 1.0202, 1.0544, 0.9569, 0.9738, 1.0131, 0.9630, 0.9357, 1.0172, 1.0039, 0.9707, 1.0685, 1.0458, 0.9770, 0.9878, 1.0330, 0.9923, 1.0483, 1.0449, 1.0447, 0.9795, 0.8630, 1.1581, 1.0746, 0.9496, 0.9617, 0.9863, 0.9528, 0.9916, 0.9921, 0.9071, 1.0163, 0.9636, 0.9220, 0.9645, 0.9471, 0.8821, 0.9251, 1.0248, 1.0801, 1.0262, 0.9723, 0.9749, 0.8496, 0.9512, 0.9697, 0.9300, 0.9642, 0.8776, 1.0129, 0.9367, 1.0754, 0.9700, 0.9598, 0.9300, 0.9603, 1.1075, 0.8745, 0.8983, 1.1720, 0.9573, 0.9599, 0.9640, 0.9507, 1.1050, 0.9383, 1.6304, 1.0377, 0.9591, 1.0307, 0.8085, 0.9735, 0.9505, 1.0219, 1.0369, 0.8886, 0.9526, 0.9793, 0.8963, 0.9509, 0.9703, 0.9206, 1.0299, 0.9687, 0.9010, 0.9931, 0.9539, 0.9550, 0.9487, 0.9503, 0.9286, 0.9354, 0.9452, 0.9960, 0.9832, 1.0121, 0.9637, 0.9416, 0.9591, 0.9799, 0.8695, 1.0096, 0.9505, 0.9293, 0.9178, 0.9070, 0.9359, 0.9372, 0.9101, 0.9554, 0.9143, 0.9583, 0.9984, 0.9668, 0.9401, 0.8945, 0.9249, 0.9888, 0.9788, 0.9743, 1.0252, 0.9752, 1.2327, 0.9582, 0.9347, 0.9273, 0.9053, 1.0021, 1.0065, 0.9490, 0.8724, 0.8961, 0.9628, 0.9457, 0.8843, 0.9869, 0.9854, 1.0156, 0.9449, 0.9001, 1.0154, 0.9639, 1.0241, 0.9744, 0.8930, 0.9029, 0.8978, 0.9160, 0.9556, 0.9781, 0.9347, 0.9124, 0.8267, 0.8717, 0.9450, 0.9111],
    "train_bpb": [None, 7.9286, 5.6125, 5.6021, 4.8987, 4.4616, 4.4212, 4.0808, 4.0711, 3.8964, 3.9398, 2.8540, 2.0721, 2.0900, 2.0489, 1.8674, 1.9723, 1.6897, 1.7095, 1.8292, 2.0234, 1.5249, 1.7116, 1.7179, 1.6821, 1.6137, 1.6455, 1.7218, 1.5207, 1.6458, 1.5567, 1.2100, 1.4386, 1.6353, 1.5315, 1.5415, 1.7112, 1.5761, 1.5806, 1.4644, 1.6587, 1.5203, 1.5759, 1.5573, 1.4726, 1.4653, 1.5570, 1.6061, 1.4934, 1.5553, 1.5293, 1.5537, 1.9787, 1.2628, 1.5419, 1.4495, 1.5930, 1.3747, 1.4380, 1.4896, 1.4182, 1.4943, 1.5310, 1.3412, 1.5311, 1.4338, 1.3996, 1.5519, 1.4725, 1.5221, 1.3809, 1.4051, 1.4622, 1.3898, 1.3504, 1.4682, 1.4489, 1.4010, 1.5422, 1.5091, 1.4102, 1.4254, 1.4908, 1.4317, 1.5131, 1.5082, 1.5075, 1.4135, 1.2454, 1.6715, 1.5507, 1.3705, 1.3875, 1.4233, 1.3748, 1.4311, 1.4319, 1.3088, 1.4663, 1.3906, 1.3303, 1.3920, 1.3668, 1.2731, 1.3351, 1.4786, 1.5587, 1.4810, 1.4034, 1.4070, 1.2260, 1.3728, 1.3994, 1.3422, 1.3916, 1.2664, 1.4619, 1.3520, 1.5520, 1.4000, 1.3851, 1.3422, 1.3860, 1.5984, 1.2619, 1.2962, 1.6913, 1.3816, 1.3853, 1.3913, 1.3721, 1.5946, 1.3544, 2.3524, 1.4974, 1.3841, 1.4872, 1.1668, 1.4051, 1.3717, 1.4748, 1.4963, 1.2824, 1.3748, 1.4132, 1.2933, 1.3720, 1.4003, 1.3287, 1.4863, 1.3979, 1.3001, 1.4333, 1.3767, 1.3782, 1.3692, 1.3717, 1.3404, 1.3501, 1.3642, 1.4371, 1.4189, 1.4602, 1.3909, 1.3589, 1.3843, 1.4142, 1.2550, 1.4571, 1.3714, 1.3407, 1.3241, 1.3091, 1.3505, 1.3524, 1.3133, 1.3789, 1.3196, 1.3831, 1.4406, 1.3953, 1.3568, 1.2907, 1.3348, 1.4270, 1.4127, 1.4062, 1.4794, 1.4076, 1.7784, 1.3828, 1.3489, 1.3383, 1.3066, 1.4465, 1.4525, 1.3695, 1.2588, 1.2932, 1.3894, 1.3649, 1.2760, 1.4245, 1.4222, 1.4654, 1.3636, 1.2990, 1.4656, 1.3911, 1.4781, 1.4064, 1.2888, 1.3030, 1.2957, 1.3220, 1.3791, 1.4114, 1.3491, 1.3169, 1.1930, 1.2579, 1.3639, 1.3150],
}
# fmt: on
_base_train_lookup = {
    s: (l, b)
    for s, l, b in zip(
        base_train_data["step"], base_train_data["train_loss"], base_train_data["train_bpb"]
    )
    if l is not None
}

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------


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

    # Model architecture
    num_layers = int(os.environ.get("NUM_LAYERS", 6))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    multi_head_dim = int(os.environ.get("MULTI_HEAD_DIM", 128))
    context_dim = int(os.environ.get("CONTEXT_DIM", 64))
    n_max = int(os.environ.get("N_MAX", 32))
    mlp_hidden_dim = int(os.environ.get("MLP_HIDDEN_DIM", 0))  # 0 = auto
    gated_mlp_output = bool(int(os.environ.get("GATED_MLP_OUTPUT", "1")))
    gated_attn_output = bool(int(os.environ.get("GATED_ATTN_OUTPUT", "1")))
    gated_conv = bool(int(os.environ.get("GATED_CONV", "1")))
    k_shift = bool(int(os.environ.get("K_SHIFT", "1")))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 0.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    _vs = os.environ.get("VALUE_SOFTCAP", "30.0")
    value_softcap: float | None = None if _vs.lower() in ("none", "0") else float(_vs)
    structured_output_logits = bool(
        int(os.environ.get("STRUCTURED_OUTPUT_LOGITS", "1"))
    )
    utf8_prior = bool(int(os.environ.get("UTF8_PRIOR", "1")))
    init_noise_std = float(os.environ.get("INIT_NOISE_STD", 0.01))

    # Pre-processing conv
    num_preconv_layers = int(os.environ.get("NUM_PRECONV_LAYERS", 2))
    preconv_kernel_size = os.environ.get("PRECONV_KERNEL_SIZE", "4")
    preconv_groups = int(os.environ.get("PRECONV_GROUPS", 6))
    preconv_channel_shuffle = bool(int(os.environ.get("PRECONV_CHANNEL_SHUFFLE", "1")))

    # Per-block conv
    block_conv_map = os.environ.get("BLOCK_CONV_MAP", "")
    block_conv_kernel_size = os.environ.get("BLOCK_CONV_KERNEL_SIZE", "4")
    block_conv_groups = os.environ.get("BLOCK_CONV_GROUPS", "1")

    # Arithmetic attention
    block_arith_map = os.environ.get("BLOCK_ARITH_MAP", "")
    block_arith_n_max = int(os.environ.get("BLOCK_ARITH_N_MAX", 16))

    # Stream mixing (USE_STREAM_MIXING=0 uses concat-based CausalMultiStreamAttention)
    use_stream_mixing = bool(int(os.environ.get("USE_STREAM_MIXING", "0")))
    mixing_mode = os.environ.get("MIXING_MODE", "glu")
    mixing_source = os.environ.get("MIXING_SOURCE", "dynamic_bottleneck")
    mixing_bottleneck_dim = int(os.environ.get("MIXING_BOTTLENECK_DIM", 32))

    # Residual mixing, skip connections, and alpha/beta bounding
    use_resid_mix = bool(int(os.environ.get("USE_RESID_MIX", "0")))
    use_unet_skip = bool(int(os.environ.get("USE_UNET_SKIP", "0")))
    bound_alpha = bool(int(os.environ.get("BOUND_ALPHA", "0")))
    bound_beta = bool(int(os.environ.get("BOUND_BETA", "0")))

    # Bigram prior
    include_bigram_prior = bool(int(os.environ.get("INCLUDE_BIGRAM_PRIOR", "1")))
    bigram_prior_lr = float(os.environ.get("BIGRAM_PRIOR_LR", 0.01))
    bigram_init_smoothing = float(os.environ.get("BIGRAM_INIT_SMOOTHING", 0.1))

    # Optimizer
    builder_lr = float(os.environ.get("BUILDER_LR", 0.01))
    conv_lr = float(os.environ.get("CONV_LR", 0.01))
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
    depth_lr_decay = float(os.environ.get("DEPTH_LR_DECAY", 0.0))
    muon_optimize_conv = bool(int(os.environ.get("MUON_OPTIMIZE_CONV", "0")))
    muon_optimize_factors = bool(int(os.environ.get("MUON_OPTIMIZE_FACTORS", "0")))

    # Compressed linear layer mode
    linear_mode = os.environ.get("LINEAR_MODE", "dense")
    kronecker_terms = int(os.environ.get("KRONECKER_TERMS", 4))
    monarch_nblocks = int(os.environ.get("MONARCH_NBLOCKS", 0))

    # Byte tokenizer config
    discard_unused_bytes = bool(int(os.environ.get("DISCARD_UNUSED_BYTES", "1")))
    fold = os.environ.get("FOLD", "")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_int_or_list(raw: str) -> int | list[int]:
    """Parse ``'4'`` -> ``4`` or ``'2,4,8'`` -> ``[2, 4, 8]``."""
    if "," in raw:
        return [int(x) for x in raw.split(",")]
    return int(raw)


def _parse_instance_map(raw: str) -> list[int | None] | None:
    """Parse ``'0,None,0,1'`` -> ``[0, None, 0, 1]``.  Empty string -> ``None``."""
    if not raw.strip():
        return None
    result: list[int | None] = []
    for s in raw.split(","):
        s = s.strip()
        if s.lower() == "none":
            result.append(None)
        else:
            result.append(int(s))
    return result


# ---------------------------------------------------------------------------
# Byte-level BPB eval
# ---------------------------------------------------------------------------


def build_byte_bpb_lut(tok: EfficientByteTokenizer, device: torch.device):
    """Build a simple LUT: each non-special token = 1 byte, special tokens = 0 bytes."""
    base_bytes = np.zeros(tok.vocab_size, dtype=np.int16)
    base_bytes[tok.n_special :] = 1
    return torch.tensor(base_bytes, dtype=torch.int16, device=device)


def eval_prior_bpb(
    args,
    base_model,
    rank,
    world_size,
    device,
    grad_accum_steps,
    val_tokens,
    base_bytes_lut,
):
    """Compute standalone BPB of the bigram + UTF8 prior on validation data.

    Reuses the model's own builder, utf8_prior, and bigram_prior layers
    (same code path as the first steps of MultiStreamGPT.forward), skipping
    blocks and softcap.  Reports uniform, UTF8-only, and bigram+UTF8 BPB
    so off-by-one index errors are easy to spot.
    """
    from multi_streams import StreamID, StreamType

    _LOGIT_SID = StreamID(StreamType.LOGIT)

    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size

    utf8_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    bigram_only_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    bigram_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    base_model.eval()
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

            # Same path as MultiStreamGPT.forward steps 1-3, no blocks/softcap.
            streams, _compressed, _views = base_model.builder(x, dtype=torch.bfloat16)
            logits = streams[_LOGIT_SID]  # zeros (B, S, V)

            # UTF8 prior hard mask (skip the capped early version — we only
            # care about the final prediction, not block stability).
            token_mask = None
            if base_model.utf8_prior is not None:
                _cat_mask, token_mask = base_model.utf8_prior(x)
                token_mask = token_mask.to(dtype=logits.dtype)

            # UTF8-only loss
            utf8_logits = logits + token_mask if token_mask is not None else logits
            utf8_loss = F.cross_entropy(
                utf8_logits.float().reshape(-1, utf8_logits.size(-1)),
                y.reshape(-1),
                reduction="mean",
            )

            # Bigram-only loss (no UTF8 mask)
            bigram_only_logits = logits
            if base_model.bigram_prior is not None:
                bigram_only_logits = bigram_only_logits + base_model.bigram_prior(x)
            bigram_only_loss = F.cross_entropy(
                bigram_only_logits.float().reshape(-1, bigram_only_logits.size(-1)),
                y.reshape(-1),
                reduction="mean",
            )

            # Bigram + UTF8 loss
            bigram_utf8_logits = utf8_logits
            if base_model.bigram_prior is not None:
                bigram_utf8_logits = bigram_utf8_logits + base_model.bigram_prior(x)
            bigram_loss = F.cross_entropy(
                bigram_utf8_logits.float().reshape(-1, bigram_utf8_logits.size(-1)),
                y.reshape(-1),
                reduction="mean",
            )

            n = float(y.numel())
            utf8_loss_sum += utf8_loss.to(torch.float64) * n
            bigram_only_loss_sum += bigram_only_loss.to(torch.float64) * n
            bigram_loss_sum += bigram_loss.to(torch.float64) * n
            token_count += n
            byte_count += base_bytes_lut[y.reshape(-1)].to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        for t in (
            utf8_loss_sum,
            bigram_only_loss_sum,
            bigram_loss_sum,
            token_count,
            byte_count,
        ):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    tpb = token_count.item() / byte_count.item()
    vocab_size = base_model.builder.vocab_size
    uniform_bpb = math.log2(vocab_size) * tpb
    utf8_bpb = utf8_loss_sum.item() / token_count.item() / math.log(2.0) * tpb
    bigram_only_bpb = (
        bigram_only_loss_sum.item() / token_count.item() / math.log(2.0) * tpb
    )
    bigram_utf8_bpb = bigram_loss_sum.item() / token_count.item() / math.log(2.0) * tpb

    base_model.train()
    return uniform_bpb, utf8_bpb, bigram_only_bpb, bigram_utf8_bpb


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


# ---------------------------------------------------------------------------
# Control tensor patterns & INT8 quantization
# ---------------------------------------------------------------------------

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "alpha,beta,q_gain,shift_logit,mix_logits,gates,bigram_logits,resid_mix,skip_weight",
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


DistributedTokenLoader = DistributedTokenLoaderByte260

# ---------------------------------------------------------------------------
# Training wrapper
# ---------------------------------------------------------------------------


class MultiStreamGPTForTraining(nn.Module):
    """Thin wrapper: ``forward(input_ids, target_ids)`` -> scalar loss.

    Matches the interface expected by ``torch.compile`` + DDP and the
    existing training loop pattern where ``model(x, y)`` returns loss.
    """

    def __init__(self, model: MultiStreamGPT):
        super().__init__()
        self.model = model

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        loss, _logits = self.model.compute_loss(input_ids, target_ids=target_ids)
        return loss


# ---------------------------------------------------------------------------
# Tokenizer builder
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    nan_watchdog = NaNWatchdog.from_env(log_fn=None)  # uses stderr by default
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
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

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

    # Load and remap validation tokens
    val_tokens = load_validation_tokens_byte260(
        args.val_files, args.train_seq_len, tok, args.val_max_tokens
    )

    base_bytes_lut = build_byte_bpb_lut(tok, device)
    log0(f"val_bpb:enabled tokenizer_kind=efficient_byte vocab_size={vocab_size}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")

    # --- Parse per-layer / instance-map configs ---
    block_conv_map = _parse_instance_map(args.block_conv_map)
    block_arith_map = _parse_instance_map(args.block_arith_map)
    preconv_kernel_size = _parse_int_or_list(args.preconv_kernel_size)
    block_conv_kernel_size = _parse_int_or_list(args.block_conv_kernel_size)
    block_conv_groups = _parse_int_or_list(args.block_conv_groups)

    # --- Build mixing config ---
    mixing_config: StreamMixingConfig | None = None
    if args.use_stream_mixing:
        mixing_config = StreamMixingConfig(
            qk_mode=MixingMode(args.mixing_mode),
            source=MixingSource(args.mixing_source),
            bottleneck_dim=args.mixing_bottleneck_dim,
        )

    # --- Build model (with bigram init + structural calibration) ---
    base_model = build_multi_stream_gpt(
        tok=tok,
        vocab_size=vocab_size,
        context_dim=args.context_dim,
        n_max=args.n_max,
        logit_softcap=args.logit_softcap,
        structured_output_logits=args.structured_output_logits,
        calibrate_structural_stream=True,
        compile_calibration=False,
        calibration_sequence_length=args.train_seq_len,
        include_bigram_prior=args.include_bigram_prior,
        train_pattern=args.train_files,
        bigram_init_smoothing=args.bigram_init_smoothing,
        device=device,
        multi_head_dim=args.multi_head_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        num_layers=args.num_layers,
        num_preconv_layers=args.num_preconv_layers,
        preconv_kernel_size=preconv_kernel_size,
        preconv_groups=args.preconv_groups,
        preconv_channel_shuffle=args.preconv_channel_shuffle,
        block_conv_map=block_conv_map,
        block_conv_kernel_size=block_conv_kernel_size,
        block_conv_groups=block_conv_groups,
        block_arith_map=block_arith_map,
        block_arith_n_max=args.block_arith_n_max,
        mixing_config=mixing_config,
        mlp_hidden_dim=args.mlp_hidden_dim if args.mlp_hidden_dim > 0 else None,
        gated_mlp_output=args.gated_mlp_output,
        gated_attn_output=args.gated_attn_output,
        gated_conv=args.gated_conv,
        qk_gain_init=args.qk_gain_init,
        k_shift=args.k_shift,
        value_softcap=args.value_softcap,
        include_utf8_prior=args.utf8_prior,
        linear_mode=args.linear_mode,
        linear_kwargs={
            "kronecker_terms": args.kronecker_terms,
            "monarch_nblocks": args.monarch_nblocks,
        },
        use_resid_mix=args.use_resid_mix,
        use_unet_skip=args.use_unet_skip,
        bound_alpha=args.bound_alpha,
        bound_beta=args.bound_beta,
        init_noise_std=args.init_noise_std,
    )
    base_model = base_model.bfloat16()

    # Restore CastedLinear / KroneckerLinear / MonarchLinear / Conv1d to float32 weights.
    for module in base_model.modules():
        if isinstance(
            module, (CastedLinear, KroneckerLinear, MonarchLinear, nn.Conv1d)
        ):
            module.float()
    restore_low_dim_params_to_fp32(
        base_model, control_patterns=CONTROL_TENSOR_NAME_PATTERNS
    )

    # --- Depth-based gradient scaling ---
    if args.depth_lr_decay > 0:
        for block_idx, block in enumerate(base_model.blocks):
            scale = 1.0 / (1.0 + block_idx * args.depth_lr_decay)
            for p in block.parameters():
                p._depth_lr_scale = scale

    # --- Sanity-check: prior-only BPB ---
    uniform_bpb, utf8_bpb, bigram_only_bpb, bigram_utf8_bpb = eval_prior_bpb(
        args,
        base_model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
    )
    log0(
        f"prior_bpb: uniform={uniform_bpb:.4f} utf8_only={utf8_bpb:.4f} "
        f"bigram_only={bigram_only_bpb:.4f} bigram+utf8={bigram_utf8_bpb:.4f}"
    )

    # --- Wrap for training ---
    train_wrapper = MultiStreamGPTForTraining(base_model)
    if nan_watchdog.should_disable_compile():
        log0("DEBUG_NAN: torch.compile DISABLED for full diagnostic mode")
        run_model = train_wrapper
    else:
        run_model = torch.compile(train_wrapper, dynamic=False, fullgraph=True)
    model = (
        DDP(run_model, device_ids=[local_rank], broadcast_buffers=False)
        if distributed
        else run_model
    )

    # --- Optimizer setup ---
    # Pre-scan: identify conv weight parameter IDs.
    conv_weight_ids: set[int] = set()
    muon_conv_ids: set[int] = set()
    for m in base_model.modules():
        if isinstance(m, nn.Conv1d):
            for p in m.parameters():
                if args.muon_optimize_conv and m.groups < m.out_channels:
                    muon_conv_ids.add(id(p))
                else:
                    conv_weight_ids.add(id(p))

    # If muon_optimize_factors, route Kronecker/Monarch factor params to Muon.
    extra_muon_ids: set[int] = set(muon_conv_ids)
    if args.muon_optimize_factors:
        for m in base_model.modules():
            if isinstance(m, (KroneckerLinear, MonarchLinear)):
                for p in m.parameters():
                    extra_muon_ids.add(id(p))

    # Classify all parameters (named_parameters deduplicates shared params).
    bigram_params: list[nn.Parameter] = []
    builder_params: list[nn.Parameter] = []
    muon_params: list[nn.Parameter] = []
    scalar_params: list[nn.Parameter] = []
    conv_params: list[nn.Parameter] = []

    for name, param in base_model.named_parameters():
        pid = id(param)
        if pid in muon_conv_ids:
            muon_params.append(param)
        elif pid in conv_weight_ids:
            conv_params.append(param)
        elif "bigram_prior" in name:
            bigram_params.append(param)
        elif "builder" in name:
            builder_params.append(param)
        elif (param.ndim >= 2 or pid in extra_muon_ids) and not any(
            pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS
        ):
            muon_params.append(param)
        else:
            scalar_params.append(param)

    # Build optimizers.
    optimizers: list[torch.optim.Optimizer] = []

    if bigram_params:
        optimizer_bigram = torch.optim.Adam(
            [
                {
                    "params": bigram_params,
                    "lr": args.bigram_prior_lr,
                    "base_lr": args.bigram_prior_lr,
                }
            ],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.append(optimizer_bigram)

    if builder_params:
        optimizer_builder = torch.optim.Adam(
            [
                {
                    "params": builder_params,
                    "lr": args.builder_lr,
                    "base_lr": args.builder_lr,
                }
            ],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.append(optimizer_builder)

    optimizer_muon = None
    if muon_params:
        optimizer_muon = Muon(
            muon_params,
            lr=args.matrix_lr,
            momentum=args.muon_momentum,
            backend_steps=args.muon_backend_steps,
            gram_ns=args.muon_gram_ns,
            reshape_3d=True,
        )
        for group in optimizer_muon.param_groups:
            group["base_lr"] = args.matrix_lr
        optimizers.append(optimizer_muon)

    if scalar_params:
        optimizer_scalar = torch.optim.Adam(
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
        optimizers.append(optimizer_scalar)

    if conv_params:
        optimizer_conv = torch.optim.Adam(
            [
                {
                    "params": conv_params,
                    "lr": args.conv_lr,
                    "base_lr": args.conv_lr,
                }
            ],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.append(optimizer_conv)

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
    log0(f"model_params:{n_params}")
    log0(
        f"optimizer_groups: bigram={len(bigram_params)} builder={len(builder_params)} "
        f"muon={len(muon_params)} scalar={len(scalar_params)} conv={len(conv_params)}"
    )
    if args.linear_mode != "dense":
        log0(
            f"linear_mode:{args.linear_mode} kronecker_terms:{args.kronecker_terms} "
            f"monarch_nblocks:{args.monarch_nblocks} muon_factors:{args.muon_optimize_factors}"
        )
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(
        f"mixing: mode={args.mixing_mode} source={args.mixing_source} "
        f"bottleneck_dim={args.mixing_bottleneck_dim}"
    )
    log0(
        f"arch: layers={args.num_layers} heads={args.num_heads} kv_heads={args.num_kv_heads} "
        f"multi_head_dim={args.multi_head_dim} context_dim={args.context_dim} "
        f"structured_output_logits={args.structured_output_logits} "
        f"utf8_prior={args.utf8_prior} "
        f"resid_mix={args.use_resid_mix} unet_skip={args.use_unet_skip} "
        f"bound_alpha={args.bound_alpha} bound_beta={args.bound_beta}"
    )
    log0(
        f"preconv: layers={args.num_preconv_layers} kernel={args.preconv_kernel_size} "
        f"groups={args.preconv_groups} shuffle={args.preconv_channel_shuffle}"
    )
    log0(
        f"bigram_prior:{args.include_bigram_prior} "
        f"smoothing={args.bigram_init_smoothing} lr={args.bigram_prior_lr}"
    )
    log0(
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr} "
        f"builder_lr:{args.builder_lr} conv_lr:{args.conv_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")

    # --- NaN watchdog hooks (level 2 only) ---
    if nan_watchdog.should_disable_compile():
        nan_watchdog.attach_hooks(base_model)
        torch.autograd.set_detect_anomaly(True)
        log0("DEBUG_NAN: anomaly detection ON, forward hooks attached")

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
        warmdown_start = max(args.iterations - args.warmdown_iters, 0)
        step_mul = (
            max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
            if warmdown_start <= step < args.iterations
            else 1.0
        )
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

    # --- Warmup ---
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

    # --- Training loop ---
    training_time_ms = 0.0
    stop_after_step = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0
    while True:
        last_step = step == args.iterations or (
            stop_after_step is not None and step >= stop_after_step
        )
        if last_step or (
            args.val_loss_every > 0 and step % args.val_loss_every == 0 and step > 0
        ):
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
            x, y = train_loader.next_batch(
                args.train_batch_tokens, args.train_seq_len, grad_accum_steps
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps
        nan_watchdog.check_loss(train_loss, step)

        frac = (
            min(step / args.muon_momentum_warmup_steps, 1.0)
            if args.muon_momentum_warmup_steps > 0
            else 1.0
        )
        warmed_momentum = (
            1 - frac
        ) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        if optimizer_muon is not None:
            for group in optimizer_muon.param_groups:
                group["momentum"] = warmed_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale
        nan_watchdog.check_params_and_grads(base_model, step)
        if nan_watchdog.triggered:
            log0(f"DEBUG_NAN: terminating at step {step}")
            break
        if args.depth_lr_decay > 0:
            for p in base_model.parameters():
                if p.grad is not None and hasattr(p, "_depth_lr_scale"):
                    p.grad.mul_(p._depth_lr_scale)
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
            cur_bpb = tl / math.log(2.0) * tpb
            base = _base_train_lookup.get(step)
            delta_loss = f" ({tl - base[0]:+.4f})" if base else ""
            delta_bpb = f" ({cur_bpb - base[1]:+.4f})" if base else ""
            log0(
                f"step:{step}/{args.iterations} train_loss:{tl:.4f}{delta_loss} "
                f"train_bpb:{cur_bpb:.4f}{delta_bpb} "
                f"train_time:{approx_training_time_ms:.0f}ms "
                f"step_avg:{approx_training_time_ms / step:.2f}ms"
            )
            nan_watchdog.log_step_summary(step, tl)
            nan_watchdog.log_alpha_beta_gates(base_model, step)
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

    # --- Save & quantize ---
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

    # --- Roundtrip validation ---
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
