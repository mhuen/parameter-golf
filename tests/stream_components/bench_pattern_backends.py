"""Benchmark core computational patterns across torch/numpy/C backends.

Compares 5 backends (torch GPU+compile, torch GPU eager, torch CPU, numpy, inline C)
on the 5 fundamental patterns used by the structural stream components.

Usage:
    # Run all patterns on all available backends:
    python tests/stream_components/bench_pattern_backends.py

    # Specific patterns and backends:
    python tests/stream_components/bench_pattern_backends.py \
        --patterns segmented_cumsum affine_scan \
        --backends torch_cpu c_inline numpy

    # Larger sizes:
    python tests/stream_components/bench_pattern_backends.py --seq-len 8192 --batch-size 8
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

VOCAB_SIZE = 208
HASH_MODULUS = 2147483647  # 2^31 - 1
HASH_BASE = 257
PROJ_PRIME = 997
TABLE_DIM = 4  # output dim for table lookup

ALL_BACKENDS = [
    "torch_gpu_compile",
    "torch_gpu_eager",
    "torch_cpu",
    "numpy",
    "c_inline",
]
ALL_PATTERNS = [
    "table_lookup",
    "segmented_cumsum",
    "run_length",
    "rolling_hash",
    "affine_scan",
]

# ──────────────────────────────────────────────────────────────────────────────
# Synthetic data generation
# ──────────────────────────────────────────────────────────────────────────────


def make_inputs(B: int, S: int, device: str = "cpu") -> dict:
    """Generate synthetic inputs for all patterns."""
    rng = torch.Generator().manual_seed(42)
    input_ids = torch.randint(0, VOCAB_SIZE, (B, S), generator=rng, dtype=torch.long)
    # Sprinkle BOS tokens (id=0) at ~5% of positions for segmented patterns
    bos_mask = torch.rand(B, S, generator=rng) < 0.05
    bos_mask[:, 0] = True
    input_ids[bos_mask] = 0

    # Table for lookup
    table = torch.randn(VOCAB_SIZE, TABLE_DIM)

    # For segmented_cumsum: values and reset mask
    # NOTE: _segmented_cumsum uses a cummax trick that requires non-negative inputs
    # (the cumsum must be monotonically non-decreasing). This matches production
    # usage where inputs are boolean/count values.
    values = torch.randint(0, 5, (B, S), generator=rng).float()
    reset_mask = input_ids == 0

    # For run_length: category ids
    cat_ids = (input_ids % 7).long()

    # For rolling hash: byte_vals in [0, 256], effective lookback
    byte_vals = (input_ids % 256 + 1).long()
    eff_lb = torch.arange(S).unsqueeze(0).expand(B, S).clone()
    # Reset at BOS positions
    positions = torch.arange(S).unsqueeze(0).expand(B, S)
    last_bos = torch.where(reset_mask, positions, torch.zeros_like(positions)).cummax(
        dim=1
    ).values
    eff_lb = positions - last_bos

    # Hash powers
    W = 8  # default window
    powers = torch.tensor(
        [(HASH_BASE**i) % HASH_MODULUS for i in range(W)], dtype=torch.long
    )

    # For affine_scan: a in [0, 1] (multiplicative), b (additive)
    a_scan = torch.where(
        reset_mask, torch.zeros(B, S), torch.ones(B, S)
    ).double()
    b_scan = torch.randn(B, S).double() * 0.1

    return {
        "input_ids": input_ids.to(device),
        "table": table.to(device),
        "values": values.to(device),
        "reset_mask": reset_mask.to(device),
        "cat_ids": cat_ids.to(device),
        "byte_vals": byte_vals.to(device),
        "eff_lb": eff_lb.to(device),
        "powers": powers.to(device),
        "a_scan": a_scan.to(device),
        "b_scan": b_scan.to(device),
        "B": B,
        "S": S,
        "W": W,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Torch implementations (extracted from byte_stream_components.py / number_detection.py)
# ──────────────────────────────────────────────────────────────────────────────


def torch_table_lookup(inputs: dict) -> torch.Tensor:
    return inputs["table"][inputs["input_ids"]]


def torch_segmented_cumsum(inputs: dict) -> torch.Tensor:
    x = inputs["values"]
    reset_mask = inputs["reset_mask"]
    was_long = x.dtype == torch.long
    xf = x.float() if was_long else x
    full = torch.cumsum(xf, dim=1)
    correction_at_reset = full - xf
    neg_inf = torch.tensor(-float("inf"), device=x.device, dtype=xf.dtype)
    sparse = torch.where(reset_mask, correction_at_reset, neg_inf)
    carried, _ = torch.cummax(sparse, dim=1)
    carried = torch.where(carried.isinf(), torch.zeros_like(carried), carried)
    result = full - carried
    return result.long() if was_long else result


def torch_run_length(inputs: dict) -> torch.Tensor:
    cat_ids = inputs["cat_ids"]
    B, S = cat_ids.shape
    positions = torch.arange(S, device=cat_ids.device).unsqueeze(0).expand(B, S)
    changed = torch.ones(B, S, device=cat_ids.device, dtype=torch.bool)
    changed[:, 1:] = cat_ids[:, 1:] != cat_ids[:, :-1]
    last_change = (
        torch.where(changed, positions, torch.zeros_like(positions))
        .cummax(dim=1)
        .values
    )
    return (positions - last_change).float().log1p()


def torch_rolling_hash(inputs: dict) -> torch.Tensor:
    byte_vals = inputs["byte_vals"]
    eff_lb = inputs["eff_lb"]
    powers = inputs["powers"]
    B, S = byte_vals.shape
    W = inputs["W"]
    P = HASH_MODULUS
    device = byte_vals.device

    positions = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
    offsets = torch.arange(W, device=device)
    gather_pos = (positions.unsqueeze(-1) - offsets).clamp(min=0)
    gathered = byte_vals.gather(1, gather_pos.reshape(B, -1)).reshape(B, S, W)
    mask = eff_lb.unsqueeze(-1) >= offsets
    gathered = gathered * mask
    hash_val = ((gathered * powers) % P).sum(dim=-1) % P
    projected = hash_val % PROJ_PRIME
    angle = projected.float() * (2 * math.pi / PROJ_PRIME)
    return torch.stack([angle.sin(), angle.cos()], dim=-1)


def torch_affine_scan(inputs: dict) -> torch.Tensor:
    a = inputs["a_scan"]
    b = inputs["b_scan"]
    S = a.shape[1]
    a_cur = a.clone()
    b_cur = b.clone()
    offset = 1
    while offset < S:
        a_prev = F.pad(a_cur[:, :-offset], (offset, 0), value=1.0)
        b_prev = F.pad(b_cur[:, :-offset], (offset, 0), value=0.0)
        b_cur = a_cur * b_prev + b_cur
        a_cur = a_cur * a_prev
        offset *= 2
    return b_cur


# ──────────────────────────────────────────────────────────────────────────────
# Numpy implementations
# ──────────────────────────────────────────────────────────────────────────────


def numpy_table_lookup(inputs: dict) -> np.ndarray:
    ids = inputs["input_ids_np"]
    table = inputs["table_np"]
    return table[ids]


def numpy_segmented_cumsum(inputs: dict) -> np.ndarray:
    x = inputs["values_np"].astype(np.float32)
    reset = inputs["reset_mask_np"]
    full = np.cumsum(x, axis=1)
    correction_at_reset = full - x
    sparse = np.where(reset, correction_at_reset, -np.inf)
    carried = np.maximum.accumulate(sparse, axis=1)
    carried = np.where(np.isinf(carried), 0.0, carried)
    return full - carried


def numpy_run_length(inputs: dict) -> np.ndarray:
    cat_ids = inputs["cat_ids_np"]
    B, S = cat_ids.shape
    positions = np.arange(S, dtype=np.int64)[None, :].repeat(B, axis=0)
    changed = np.ones((B, S), dtype=bool)
    changed[:, 1:] = cat_ids[:, 1:] != cat_ids[:, :-1]
    last_change_raw = np.where(changed, positions, 0)
    last_change = np.maximum.accumulate(last_change_raw, axis=1)
    return np.log1p((positions - last_change).astype(np.float32))


def numpy_rolling_hash(inputs: dict) -> np.ndarray:
    byte_vals = inputs["byte_vals_np"]
    eff_lb = inputs["eff_lb_np"]
    powers = inputs["powers_np"]
    B, S = byte_vals.shape
    W = inputs["W"]
    P = HASH_MODULUS

    positions = np.arange(S, dtype=np.int64)[None, :].repeat(B, axis=0)
    offsets = np.arange(W, dtype=np.int64)
    gather_pos = np.clip(positions[:, :, None] - offsets[None, None, :], 0, S - 1)
    # Gather: fancy index per batch
    b_idx = np.arange(B)[:, None, None]
    gathered = byte_vals[b_idx, gather_pos]
    mask = eff_lb[:, :, None] >= offsets[None, None, :]
    gathered = gathered * mask
    hash_val = (gathered * powers[None, None, :] % P).sum(axis=-1) % P
    projected = hash_val % PROJ_PRIME
    angle = projected.astype(np.float32) * np.float32(2 * math.pi / PROJ_PRIME)
    return np.stack([np.sin(angle), np.cos(angle)], axis=-1)


def numpy_affine_scan(inputs: dict) -> np.ndarray:
    a = inputs["a_scan_np"].copy()
    b = inputs["b_scan_np"].copy()
    B, S = a.shape
    # Sequential scan -- O(S) per batch element
    for bi in range(B):
        for t in range(1, S):
            b[bi, t] = a[bi, t] * b[bi, t - 1] + b[bi, t]
    return b


# ──────────────────────────────────────────────────────────────────────────────
# Inline C via ctypes — optimized for Xeon Platinum 8470 (Sapphire Rapids)
#
# Optimizations:
#  - OpenMP for batch/element parallelism (52 cores)
#  - Transposed (S, B) layout for scan patterns → auto-vectorization w/ AVX-512
#  - Mersenne prime fast modular reduction (bit ops instead of division)
#  - Precomputed sin/cos + log1p lookup tables (eliminate transcendentals)
#  - restrict pointers for alias analysis
# ──────────────────────────────────────────────────────────────────────────────

C_SOURCE = r"""
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── Mersenne prime 2^31-1: fast modular reduction ────────────── */
#define MERSENNE_P 0x7FFFFFFFLL

static inline int64_t mod_mersenne(int64_t x) {
    /* For 0 <= x < MERSENNE_P^2.  Two reductions handle carry. */
    x = (x & MERSENNE_P) + (x >> 31);
    return x >= MERSENNE_P ? x - MERSENNE_P : x;
}

/* ── table_lookup: OpenMP over elements ──────────────────────── */
void table_lookup_f32(
    const int64_t* restrict ids,
    const float*   restrict table,
    float*         restrict out,
    int B, int S, int D
) {
    int N = B * S;
    #pragma omp parallel for schedule(static) if(N > 4096)
    for (int i = 0; i < N; i++) {
        memcpy(out + i * D, table + ids[i] * D, (size_t)D * sizeof(float));
    }
}

/* ── segmented_cumsum: transposed (S, B) layout ──────────────── *
 * Sequential over S, vectorised + parallelised over B.           *
 * Each thread owns a contiguous B-slice — no barriers needed.    */
void segmented_cumsum_f32(
    const float*  restrict values,  /* (S, B) contiguous */
    const int8_t* restrict reset,   /* (S, B) contiguous */
    float*        restrict out,     /* (S, B) contiguous */
    int B, int S
) {
    #pragma omp parallel if(B > 32)
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num(), nth = omp_get_num_threads();
#else
        int tid = 0, nth = 1;
#endif
        int b0 = (int)((int64_t)tid * B / nth);
        int b1 = (int)((int64_t)(tid + 1) * B / nth);

        for (int b = b0; b < b1; b++) out[b] = values[b];

        for (int t = 1; t < S; t++) {
            const float*  vt = values + (int64_t)t * B;
            const int8_t* rt = reset  + (int64_t)t * B;
            const float*  pt = out    + (int64_t)(t - 1) * B;
            float*        ct = out    + (int64_t)t * B;
            /* Inner loop auto-vectorises with AVX-512 (16 floats/iter) */
            for (int b = b0; b < b1; b++) {
                ct[b] = rt[b] ? vt[b] : pt[b] + vt[b];
            }
        }
    }
}

/* ── run_length: transposed (S, B) + log1p LUT ──────────────── */
void run_length_i64(
    const int64_t* restrict cat_ids,  /* (S, B) contiguous */
    float*         restrict out,      /* (S, B) contiguous */
    int B, int S
) {
    /* Precompute log1p table — fits comfortably in L2 */
    float* lut = (float*)malloc((size_t)S * sizeof(float));
    for (int i = 0; i < S; i++) lut[i] = log1pf((float)i);

    #pragma omp parallel if(B > 32)
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num(), nth = omp_get_num_threads();
#else
        int tid = 0, nth = 1;
#endif
        int b0 = (int)((int64_t)tid * B / nth);
        int b1 = (int)((int64_t)(tid + 1) * B / nth);
        int chunk = b1 - b0;

        int*     run  = (int*)calloc((size_t)chunk, sizeof(int));
        int64_t* prev = (int64_t*)malloc((size_t)chunk * sizeof(int64_t));

        /* t = 0 */
        for (int b = 0; b < chunk; b++) {
            prev[b] = cat_ids[b0 + b];
            run[b]  = 1;
            out[b0 + b] = 0.0f;
        }

        for (int t = 1; t < S; t++) {
            const int64_t* ids_t = cat_ids + (int64_t)t * B + b0;
            float*         out_t = out     + (int64_t)t * B + b0;
            for (int b = 0; b < chunk; b++) {
                int64_t cur = ids_t[b];
                int same = (cur == prev[b]);
                run[b] = same * run[b];        /* 0 if changed */
                out_t[b] = lut[run[b]];
                run[b]++;
                prev[b] = cur;
            }
        }
        free(run);
        free(prev);
    }
    free(lut);
}

/* ── rolling_hash: OpenMP + Mersenne + sin/cos LUT ───────────── */
void rolling_hash_sincos(
    const int64_t* restrict byte_vals,
    const int64_t* restrict eff_lb,
    const int64_t* restrict powers,
    float*         restrict out_sin,
    float*         restrict out_cos,
    int B, int S, int W,
    int64_t P, int64_t proj_prime
) {
    /* Precompute sin/cos LUT — ~8 KB for proj_prime=997, fits in L1 */
    float* sin_lut = (float*)malloc((size_t)proj_prime * sizeof(float));
    float* cos_lut = (float*)malloc((size_t)proj_prime * sizeof(float));
    float angle_scale = (float)(2.0 * M_PI / (double)proj_prime);
    for (int64_t i = 0; i < proj_prime; i++) {
        float a = (float)i * angle_scale;
        sin_lut[i] = sinf(a);
        cos_lut[i] = cosf(a);
    }

    int64_t N = (int64_t)B * S;
    #pragma omp parallel for schedule(static) if(N > 4096)
    for (int64_t i = 0; i < N; i++) {
        int b = (int)(i / S);
        int t = (int)(i % S);
        int64_t idx = (int64_t)b * S + t;
        int64_t lb = eff_lb[idx];
        int64_t hash = 0;
        for (int k = 0; k < W && k <= lb; k++) {
            int src = t - k;
            if (src < 0) src = 0;
            int64_t bv = byte_vals[(int64_t)b * S + src];
            hash = mod_mersenne(hash + mod_mersenne(bv * powers[k]));
        }
        int64_t projected = hash % proj_prime;
        out_sin[idx] = sin_lut[projected];
        out_cos[idx] = cos_lut[projected];
    }

    free(sin_lut);
    free(cos_lut);
}

/* ── affine_scan: transposed (S, B) layout ───────────────────── */
void affine_scan_f64(
    const double* restrict a,       /* (S, B) contiguous */
    const double* restrict b_in,    /* (S, B) contiguous */
    double*       restrict out,     /* (S, B) contiguous */
    int B, int S
) {
    #pragma omp parallel if(B > 32)
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num(), nth = omp_get_num_threads();
#else
        int tid = 0, nth = 1;
#endif
        int b0 = (int)((int64_t)tid * B / nth);
        int b1 = (int)((int64_t)(tid + 1) * B / nth);

        for (int b = b0; b < b1; b++) out[b] = b_in[b];

        /* Sequential over S, auto-vectorised over b (AVX-512: 8 doubles) */
        for (int t = 1; t < S; t++) {
            const double* at = a     + (int64_t)t * B;
            const double* bt = b_in  + (int64_t)t * B;
            const double* pt = out   + (int64_t)(t - 1) * B;
            double*       ct = out   + (int64_t)t * B;
            for (int b = b0; b < b1; b++) {
                ct[b] = at[b] * pt[b] + bt[b];
            }
        }
    }
}
"""


def _compile_c_lib() -> ctypes.CDLL | None:
    """Compile inline C to a shared library. Returns None on failure."""
    try:
        tmpdir = tempfile.mkdtemp(prefix="bench_c_")
        src_path = os.path.join(tmpdir, "patterns.c")
        lib_path = os.path.join(tmpdir, "patterns.so")
        with open(src_path, "w") as f:
            f.write(C_SOURCE)

        # Try compilation with increasing fallback
        flag_sets = [
            ["-march=sapphirerapids", "-fopenmp"],
            ["-march=native", "-fopenmp"],
            ["-march=native"],
            [],
        ]
        for extra_flags in flag_sets:
            flags = ["gcc", "-O3", "-fPIC", "-shared", "-lm"] + extra_flags
            flags += ["-o", lib_path, src_path]
            result = subprocess.run(flags, capture_output=True, text=True)
            if result.returncode == 0:
                label = " ".join(extra_flags) if extra_flags else "(baseline)"
                print(f"  [c_inline] compiled with: {label}")
                return ctypes.cdll.LoadLibrary(lib_path)

        print(f"  [c_inline] gcc compilation failed: {result.stderr.strip()}")
        return None
    except FileNotFoundError:
        print("  [c_inline] gcc not found, skipping C backend")
        return None


_C_LIB: ctypes.CDLL | None = None
_C_LIB_LOADED = False


def _get_c_lib() -> ctypes.CDLL | None:
    global _C_LIB, _C_LIB_LOADED
    if not _C_LIB_LOADED:
        _C_LIB = _compile_c_lib()
        _C_LIB_LOADED = True
    return _C_LIB


def _np_ptr(arr: np.ndarray, ctype=None):
    """Get a ctypes pointer to a numpy array's data."""
    if ctype is None:
        dt = arr.dtype
        if dt == np.float32:
            ctype = ctypes.c_float
        elif dt == np.float64:
            ctype = ctypes.c_double
        elif dt == np.int64:
            ctype = ctypes.c_int64
        elif dt == np.int8:
            ctype = ctypes.c_int8
        else:
            raise TypeError(f"Unsupported dtype: {dt}")
    return arr.ctypes.data_as(ctypes.POINTER(ctype))


def c_table_lookup(inputs: dict) -> np.ndarray:
    lib = _get_c_lib()
    ids = inputs["input_ids_np"]
    table = inputs["table_np"]
    B, S = ids.shape
    D = table.shape[1]
    out = np.empty((B, S, D), dtype=np.float32)
    lib.table_lookup_f32(
        _np_ptr(ids), _np_ptr(table), _np_ptr(out),
        ctypes.c_int(B), ctypes.c_int(S), ctypes.c_int(D),
    )
    return out


def _transpose_to_SB(arr: np.ndarray) -> np.ndarray:
    """(B, S, ...) -> (S, B, ...) contiguous."""
    return np.ascontiguousarray(np.moveaxis(arr, 0, 1))


def _transpose_to_BS(arr: np.ndarray) -> np.ndarray:
    """(S, B, ...) -> (B, S, ...) contiguous."""
    return np.ascontiguousarray(np.moveaxis(arr, 0, 1))


def c_segmented_cumsum(inputs: dict) -> np.ndarray:
    lib = _get_c_lib()
    values = _transpose_to_SB(inputs["values_np"].astype(np.float32))
    reset = _transpose_to_SB(inputs["reset_mask_np"].astype(np.int8))
    S, B = values.shape
    out = np.empty((S, B), dtype=np.float32)
    lib.segmented_cumsum_f32(
        _np_ptr(values), _np_ptr(reset), _np_ptr(out),
        ctypes.c_int(B), ctypes.c_int(S),
    )
    return _transpose_to_BS(out)


def c_run_length(inputs: dict) -> np.ndarray:
    lib = _get_c_lib()
    cat_ids = _transpose_to_SB(np.ascontiguousarray(inputs["cat_ids_np"], dtype=np.int64))
    S, B = cat_ids.shape
    out = np.empty((S, B), dtype=np.float32)
    lib.run_length_i64(
        _np_ptr(cat_ids), _np_ptr(out),
        ctypes.c_int(B), ctypes.c_int(S),
    )
    return _transpose_to_BS(out)


def c_rolling_hash(inputs: dict) -> np.ndarray:
    lib = _get_c_lib()
    byte_vals = np.ascontiguousarray(inputs["byte_vals_np"], dtype=np.int64)
    eff_lb = np.ascontiguousarray(inputs["eff_lb_np"], dtype=np.int64)
    powers = np.ascontiguousarray(inputs["powers_np"], dtype=np.int64)
    B, S = byte_vals.shape
    W = inputs["W"]
    out_sin = np.empty((B, S), dtype=np.float32)
    out_cos = np.empty((B, S), dtype=np.float32)
    lib.rolling_hash_sincos(
        _np_ptr(byte_vals), _np_ptr(eff_lb), _np_ptr(powers),
        _np_ptr(out_sin), _np_ptr(out_cos),
        ctypes.c_int(B), ctypes.c_int(S), ctypes.c_int(W),
        ctypes.c_int64(HASH_MODULUS), ctypes.c_int64(PROJ_PRIME),
    )
    return np.stack([out_sin, out_cos], axis=-1)


def c_affine_scan(inputs: dict) -> np.ndarray:
    lib = _get_c_lib()
    a = _transpose_to_SB(np.ascontiguousarray(inputs["a_scan_np"], dtype=np.float64))
    b = _transpose_to_SB(np.ascontiguousarray(inputs["b_scan_np"], dtype=np.float64))
    S, B = a.shape
    out = np.empty((S, B), dtype=np.float64)
    lib.affine_scan_f64(
        _np_ptr(a), _np_ptr(b), _np_ptr(out),
        ctypes.c_int(B), ctypes.c_int(S),
    )
    return _transpose_to_BS(out)


# ──────────────────────────────────────────────────────────────────────────────
# Pattern registry
# ──────────────────────────────────────────────────────────────────────────────

# Each entry: pattern_name -> { backend_name -> callable }
# callable takes an inputs dict and returns torch.Tensor or np.ndarray


def _build_registry() -> dict[str, dict[str, tuple]]:
    return {
        "table_lookup": {
            "torch": torch_table_lookup,
            "numpy": numpy_table_lookup,
            "c_inline": c_table_lookup,
        },
        "segmented_cumsum": {
            "torch": torch_segmented_cumsum,
            "numpy": numpy_segmented_cumsum,
            "c_inline": c_segmented_cumsum,
        },
        "run_length": {
            "torch": torch_run_length,
            "numpy": numpy_run_length,
            "c_inline": c_run_length,
        },
        "rolling_hash": {
            "torch": torch_rolling_hash,
            "numpy": numpy_rolling_hash,
            "c_inline": c_rolling_hash,
        },
        "affine_scan": {
            "torch": torch_affine_scan,
            "numpy": numpy_affine_scan,
            "c_inline": c_affine_scan,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark harness
# ──────────────────────────────────────────────────────────────────────────────


def _check_close(
    ref: torch.Tensor, out: torch.Tensor | np.ndarray, label: str, atol: float = 1e-4
) -> bool:
    """Check that two outputs are close. Returns True if match."""
    if isinstance(out, np.ndarray):
        out = torch.from_numpy(out)
    ref = ref.cpu().float()
    out = out.cpu().float()
    if ref.shape != out.shape:
        print(f"    SHAPE MISMATCH {label}: {ref.shape} vs {out.shape}")
        return False
    max_diff = (ref - out).abs().max().item()
    if max_diff > atol:
        print(f"    VALUE MISMATCH {label}: max_diff={max_diff:.6e} (atol={atol})")
        return False
    return True


def _time_fn(fn, inputs: dict, repeats: int, use_cuda_sync: bool) -> list[float]:
    """Time a function over repeats, return list of times in seconds."""
    times = []
    for _ in range(repeats):
        if use_cuda_sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(inputs)
        if use_cuda_sync:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def _warmup_compiled(fn, inputs: dict, max_warmup: int = 15) -> None:
    """Warmup torch.compile until consecutive runs converge."""
    prev_t = None
    for i in range(max_warmup):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(inputs)
        torch.cuda.synchronize()
        cur_t = time.perf_counter() - t0
        if i >= 1 and prev_t is not None:
            ratio = max(cur_t, prev_t) / max(min(cur_t, prev_t), 1e-9)
            if ratio < 2.0:
                break
        prev_t = cur_t


def _measure_transfer(tensor_or_array, direction: str, repeats: int = 10) -> float:
    """Measure CPU<->GPU transfer time in seconds (mean)."""
    if direction == "to_gpu":
        if isinstance(tensor_or_array, np.ndarray):
            t = torch.from_numpy(np.ascontiguousarray(tensor_or_array))
        else:
            t = tensor_or_array.cpu().contiguous()
        times = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            t.to("cuda")
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        return sum(times) / len(times)
    return 0.0


def run_pattern_benchmark(
    pattern_name: str,
    backends: list[str],
    B: int,
    S: int,
    repeats: int,
    window: int = 8,
    has_cuda: bool = False,
) -> list[dict]:
    """Benchmark a single pattern across backends. Returns list of result dicts."""
    registry = _build_registry()
    if pattern_name not in registry:
        raise ValueError(f"Unknown pattern: {pattern_name}")
    pattern_fns = registry[pattern_name]

    results = []

    # Generate inputs on CPU first (single source of truth for correctness)
    cpu_inputs = make_inputs(B, S, device="cpu")
    cpu_inputs["W"] = window
    if pattern_name == "rolling_hash":
        cpu_inputs["powers"] = torch.tensor(
            [(HASH_BASE**i) % HASH_MODULUS for i in range(window)], dtype=torch.long
        )

    # Add numpy versions of all arrays
    np_inputs = dict(cpu_inputs)
    for key, val in cpu_inputs.items():
        if isinstance(val, torch.Tensor):
            np_inputs[key + "_np"] = val.numpy().copy()

    # GPU inputs: same data, moved to CUDA (fixes correctness mismatch)
    gpu_inputs = None
    if has_cuda:
        gpu_inputs = {}
        for key, val in cpu_inputs.items():
            if isinstance(val, torch.Tensor):
                gpu_inputs[key] = val.to("cuda")
            else:
                gpu_inputs[key] = val

    # Get reference output from torch CPU
    with torch.no_grad():
        ref_output = pattern_fns["torch"](cpu_inputs)

    for backend in backends:
        if backend.startswith("torch_gpu") and not has_cuda:
            continue
        if backend == "c_inline" and _get_c_lib() is None:
            continue

        entry: dict = {"backend": backend, "pattern": pattern_name}

        # Select function and inputs
        if backend == "torch_gpu_compile":
            fn = torch.compile(pattern_fns["torch"], fullgraph=True)
            inputs = gpu_inputs
            use_cuda = True
        elif backend == "torch_gpu_eager":
            fn = pattern_fns["torch"]
            inputs = gpu_inputs
            use_cuda = True
        elif backend == "torch_cpu":
            fn = pattern_fns["torch"]
            inputs = cpu_inputs
            use_cuda = False
        elif backend == "numpy":
            fn = pattern_fns["numpy"]
            inputs = np_inputs
            use_cuda = False
        elif backend == "c_inline":
            fn = pattern_fns["c_inline"]
            inputs = np_inputs
            use_cuda = False
        else:
            continue

        # Correctness check
        with torch.no_grad():
            test_out = fn(inputs)
        # Tolerances: rolling hash has sin/cos of large ints; segmented_cumsum
        # has float32 precision drift between global-cumsum-then-subtract (torch)
        # vs sequential accumulation (C/numpy).
        if pattern_name == "rolling_hash":
            atol = 1e-2
        elif pattern_name == "segmented_cumsum":
            atol = 1e-1  # float32 cumsum over thousands of values loses precision
        else:
            atol = 1e-4
        ok = _check_close(ref_output, test_out, f"{pattern_name}/{backend}", atol=atol)
        entry["correct"] = ok

        # Warmup
        if backend == "torch_gpu_compile":
            with torch.no_grad():
                _warmup_compiled(fn, inputs)
        else:
            warmup_n = 3
            with torch.no_grad():
                for _ in range(warmup_n):
                    fn(inputs)

        # Timed runs
        with torch.no_grad():
            times = _time_fn(fn, inputs, repeats, use_cuda)
        entry["mean_ms"] = sum(times) / len(times) * 1000
        entry["min_ms"] = min(times) * 1000

        # Transfer time (CPU backends: cost to move output to GPU)
        if not backend.startswith("torch_gpu") and has_cuda:
            with torch.no_grad():
                sample_out = fn(inputs)
            entry["transfer_ms"] = _measure_transfer(sample_out, "to_gpu") * 1000
        else:
            entry["transfer_ms"] = 0.0

        results.append(entry)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Output formatting
# ──────────────────────────────────────────────────────────────────────────────


def print_pattern_results(pattern: str, results: list[dict], B: int, S: int) -> None:
    if not results:
        return
    print(f"\nPattern: {pattern}  |  B={B}, S={S}")
    print("-" * 75)
    print(
        f"{'Backend':<22} {'compute_ms':>12} {'min_ms':>10} "
        f"{'transfer_ms':>13} {'total_ms':>10} {'correct':>8}"
    )
    print("-" * 75)
    for r in results:
        total = r["mean_ms"] + r["transfer_ms"]
        correct_str = "ok" if r["correct"] else "FAIL"
        print(
            f"{r['backend']:<22} {r['mean_ms']:>12.3f} {r['min_ms']:>10.3f} "
            f"{r['transfer_ms']:>13.3f} {total:>10.3f} {correct_str:>8}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark stream component patterns across backends."
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        nargs="+",
        default=[2048, 4096],
        help="Sequence lengths to test (default: 2048 4096).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=20,
        help="Timed repeats per measurement (default: 20).",
    )
    parser.add_argument(
        "--backends",
        type=str,
        nargs="+",
        default=None,
        help=f"Backends to test (default: all available). Options: {ALL_BACKENDS}",
    )
    parser.add_argument(
        "--patterns",
        type=str,
        nargs="+",
        default=None,
        help=f"Patterns to test (default: all). Options: {ALL_PATTERNS}",
    )
    parser.add_argument(
        "--window",
        type=int,
        nargs="+",
        default=[8],
        help="Window sizes for rolling_hash (default: 8).",
    )
    args = parser.parse_args()

    backends = args.backends or ALL_BACKENDS
    patterns = args.patterns or ALL_PATTERNS
    has_cuda = torch.cuda.is_available()

    if not has_cuda:
        backends = [b for b in backends if not b.startswith("torch_gpu")]
        print("CUDA not available -- skipping GPU backends\n")

    # Pre-compile C library if needed
    if "c_inline" in backends:
        lib = _get_c_lib()
        if lib is None:
            backends = [b for b in backends if b != "c_inline"]

    print(f"Backends: {backends}")
    print(f"Patterns: {patterns}")
    print(f"Batch size: {args.batch_size}")
    print(f"Seq lengths: {args.seq_len}")
    print(f"Repeats: {args.repeats}")

    for seq_len in args.seq_len:
        for pattern in patterns:
            windows = args.window if pattern == "rolling_hash" else [8]
            for window in windows:
                label = pattern
                if pattern == "rolling_hash" and len(args.window) > 1:
                    label = f"{pattern} (W={window})"
                results = run_pattern_benchmark(
                    pattern_name=pattern,
                    backends=backends,
                    B=args.batch_size,
                    S=seq_len,
                    repeats=args.repeats,
                    window=window,
                    has_cuda=has_cuda,
                )
                print_pattern_results(label, results, args.batch_size, seq_len)

    print()


if __name__ == "__main__":
    main()
