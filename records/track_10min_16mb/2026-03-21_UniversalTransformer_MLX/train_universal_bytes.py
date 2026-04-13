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

from efficient_byte_tokenizer import ByteCategory, EfficientByteTokenizer  # noqa: E402

import optim as _optim_mod  # noqa: E402
from optim import Muon, _GRAM_NS_LIB  # noqa: E402
from data import (  # noqa: E402
    load_raw_shard,
    load_shard_byte260,
    load_validation_tokens_byte260,
    TokenStreamByte260,
    DistributedTokenLoaderByte260,
)
from modules import (  # noqa: E402
    RMSNorm,
    CastedLinear,
    KroneckerLinear,
    MonarchLinear,
    make_linear,
    GatedCausalConv,
    MLP,
    Rotary,
    SemanticRotary,
    apply_rotary_emb,
    build_doc_mask,
    restore_low_dim_params_to_fp32,
)
from byte_modules import (  # noqa: E402
    CategoryAttnBias,
    StructuralBoundaryBias,
    DistanceBias,
    UTF8Prior,
    StructuredOutputHead,
    NUM_BYTE_CATEGORIES,
)
from multi_stream_gpt import build_structural_stream, load_calibration_tokens  # noqa: E402
from multi_streams import CompositeStream  # noqa: E402


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
        base_train_data["step"],
        base_train_data["train_loss"],
        base_train_data["train_bpb"],
    )
    if l is not None
}

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

    # Structural stream: concatenate deterministic byte features with embedding
    structural_stream = bool(int(os.environ.get("STRUCTURAL_STREAM", "0")))
    structural_calibration_seqs = int(
        os.environ.get("STRUCTURAL_CALIBRATION_SEQS", 500)
    )

    # Byte tokenizer config
    discard_unused_bytes = bool(int(os.environ.get("DISCARD_UNUSED_BYTES", "1")))
    fold = os.environ.get("FOLD", "")  # comma-separated ByteCategory values


# -----------------------------
# BYTE-LEVEL BPB EVAL
# -----------------------------


def build_byte_bpb_lut(tok: EfficientByteTokenizer, device: torch.device):
    """Build a simple LUT: each non-special token = 1 byte, special tokens = 0 bytes."""
    base_bytes = np.zeros(tok.vocab_size, dtype=np.int16)
    base_bytes[tok.n_special :] = 1
    return torch.tensor(base_bytes, dtype=torch.int16, device=device)


# load_validation_byte260 imported from data.py


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
# Shard I/O: load_raw_shard imported from data.py
load_data_shard = load_raw_shard  # alias for backward compat


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


# TokenStreamByte260, DistributedTokenLoaderByte260 imported from data.py
TokenStream = TokenStreamByte260  # alias for local references
DistributedTokenLoader = DistributedTokenLoaderByte260  # alias for local references


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
        # Sigmoid gate: sigmoid(gate_logit) ∈ [0,1].  Initialised so that
        # sigmoid(gate_logit) ≈ scale_init (clamped to (ε, 1-ε) for invertibility).
        _si = max(min(scale_init, 1.0 - 1e-4), 1e-4)
        _gate_init = math.log(_si / (1.0 - _si))  # inverse sigmoid
        self.gate_logit = nn.Parameter(torch.tensor(_gate_init, dtype=torch.float32))
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
        return result * torch.sigmoid(self.gate_logit)


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
            sem_rope_configs=sem_rope_configs,
            linear_mode=_attn_mode,
            linear_kwargs=linear_kwargs,
        )
        self.mlp = MLP(
            dim, mlp_mult, linear_mode=_mlp_mode, linear_kwargs=linear_kwargs
        )
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
        linear_mode_attn="",
        linear_mode_mlp="",
        linear_kwargs=None,
        structural_stream=None,
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
        self.structural_stream = structural_stream
        embed_dim = model_dim
        if structural_stream is not None:
            embed_dim = model_dim - structural_stream.dim
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
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
                    linear_mode_attn=linear_mode_attn,
                    linear_mode_mlp=linear_mode_mlp,
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
        if self.structural_stream is not None:
            struct = self.structural_stream(input_ids, x.dtype)
            x = torch.cat([x, struct], dim=-1)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []

        # Materialize Kronecker/Monarch weights once for all shared-block reuses
        self._materialize_weights()

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
        # Free materialized weight caches after all layers are done
        self._clear_weight_caches()

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

    # Load and remap validation tokens (byte260 -> EfficientByteTokenizer IDs)
    val_tokens = load_validation_tokens_byte260(
        args.val_files, args.train_seq_len, tok, args.val_max_tokens
    )

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

    # Build structural stream (if enabled)
    structural_stream_mod = None
    if args.structural_stream:
        if args.tie_embeddings:
            args.tie_embeddings = False
            log0("structural_stream: forcing tie_embeddings=False")
        structural_stream_mod = build_structural_stream(tok)
        s_dim = structural_stream_mod.dim
        e_dim = args.model_dim - s_dim
        if e_dim <= 0:
            raise ValueError(
                f"model_dim ({args.model_dim}) must be > structural_dim ({s_dim}). "
                f"Increase MODEL_DIM to at least {s_dim + 1}."
            )
        if e_dim < 64:
            log0(
                f"WARNING: embed_dim={e_dim} is small (< 64). "
                f"Consider increasing MODEL_DIM."
            )
        log0(
            f"structural_stream: dim={s_dim} embed_dim={e_dim} "
            f"components={len(structural_stream_mod.components)}"
        )
        cal_tokens = load_calibration_tokens(
            train_pattern=args.train_files,
            tok=tok,
            seq_len=args.train_seq_len,
            n_sequences=args.structural_calibration_seqs,
        )
        structural_stream_mod.calibrate(cal_tokens)
        log0("structural_stream: calibration done")

    _needs_tok = (
        args.structured_output_logits
        or args.utf8_prior
        or args.catmask_mode != "off"
        or args.struct_bias_mode != "off"
        or bool(sem_rope_configs)
        or args.structural_stream
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
            linear_mode_attn=args.linear_mode_attn,
            linear_mode_mlp=args.linear_mode_mlp,
            linear_kwargs={
                "kronecker_terms": args.kronecker_terms,
                "monarch_nblocks": args.monarch_nblocks,
            },
            structural_stream=structural_stream_mod,
        )
        .to(device)
        .bfloat16()
    )
    for module in base_model.modules():
        if isinstance(
            module, (CastedLinear, KroneckerLinear, MonarchLinear, nn.Conv1d)
        ):
            module.float()
        if isinstance(module, (Rotary, SemanticRotary)):
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
            if id(p) in _muon_conv_ids:
                matrix_params.append(p)
            elif id(p) in conv_weight_ids:
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
        scalar_params.append(base_model.ngram_prior_mod.gate_logit)

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
    if base_model.structured_head is not None:
        structured_params = list(base_model.structured_head.parameters())
        if structured_params:
            if args.muon_optimize_lm_head:
                matrix_params.extend(structured_params)
            else:
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
        warmed_momentum = (
            1 - frac
        ) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        if optimizer_muon is not None:
            for group in optimizer_muon.param_groups:
                group["momentum"] = warmed_momentum
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
