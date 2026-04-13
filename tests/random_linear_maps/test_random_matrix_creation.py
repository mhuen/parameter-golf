#!/usr/bin/env python3
"""
Random vector/matrix generation from seeds & seed-search compression experiment.

Benchmarks seeded generation on GPU, then evaluates how well brute-force seed
search can approximate vectors of varying sizes.  The idea: represent an entire
parameter vector with a single integer seed + a deterministic generation
function.  Extreme compression via hash-based PRNG.

Usage:
    python tests/random_linear_maps/test_random_matrix_creation.py --device cuda
    python tests/random_linear_maps/test_random_matrix_creation.py --benchmark-only
    python tests/random_linear_maps/test_random_matrix_creation.py --evaluate-only --n-seeds 1000000
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow many distinct n_elements specialisations without hitting dynamo's
# default recompile limit (8).  We test ~20 different sizes across the
# benchmark + evaluation sweep.
torch._dynamo.config.cache_size_limit = 64

# ---------------------------------------------------------------------------
# SplitMix64 constants (unsigned hex reinterpreted as signed int64)
#   _GOLDEN = 0x9e3779b97f4a7c15  (golden-ratio * 2^64, rounded)
#   _MIX1   = 0xbf58476d1ce4e5b9
#   _MIX2   = 0x94d049bb133111eb
# ---------------------------------------------------------------------------
_GOLDEN: int = -7046029254386353131
_MIX1: int = -4658895280553007687
_MIX2: int = -7723592293110705685


# ---------------------------------------------------------------------------
# Part 1: Seeded Vector Generator
# ---------------------------------------------------------------------------


class SeededVectorGenerator(nn.Module):
    """Deterministic random vector generation from integer seeds.

    Uses a SplitMix64-style hash on ``(seed, element_index)`` pairs.
    Fully vectorised over the batch dimension, no Python loops,
    ``torch.compile(fullgraph=True)`` compatible.

    The output is a flat ``[batch, n]`` tensor.  Callers can reshape to any
    matrix layout afterwards — the hash only depends on the total element
    count, not on any 2-D shape.
    """

    def forward(self, seeds: torch.Tensor, n: int) -> torch.Tensor:
        """
        Args:
            seeds: ``[batch]`` int64 tensor of seed values.
            n:     number of elements per vector.

        Returns:
            ``[batch, n]`` float32 tensor with values in [-1, 1).
        """
        idx = torch.arange(n, device=seeds.device, dtype=torch.int64)

        # Combine seed and element position  →  [batch, n]
        x = seeds.unsqueeze(1) * _GOLDEN + idx.unsqueeze(0)

        # SplitMix64 finaliser (three rounds of xor-shift-multiply)
        x = (x ^ (x >> 30)) * _MIX1
        x = (x ^ (x >> 27)) * _MIX2
        x = x ^ (x >> 31)

        # Extract lower 32 random bits → float in [-1, 1)
        # (avoids int64-sign issues; 32 bits > float32's 23-bit mantissa)
        return (x & 0xFFFFFFFF).float() * (1.0 / 2147483648.0) - 1.0


# ---------------------------------------------------------------------------
# Part 2: Benchmark
# ---------------------------------------------------------------------------


def _estimate_peak_bytes(batch: int, n_elements: int) -> int:
    """Conservative peak-memory estimate for one generation call.

    Two int64 intermediates + one float32 output ≈ 20 bytes per element.
    """
    return batch * n_elements * 20


def benchmark_generation(device: torch.device) -> None:
    """Benchmark seeded vector generation: compiled vs. eager."""
    gen = SeededVectorGenerator().to(device)
    gen_compiled = torch.compile(gen, fullgraph=True, dynamic=False)

    configs = [
        # (batch, n_elements, label)
        (1_000, 256, "1K x 256"),
        (10_000, 256, "10K x 256"),
        (100_000, 256, "100K x 256"),
        (1_000_000, 256, "1M x 256"),
        (10_000, 1024, "10K x 1024"),
        (100_000, 1024, "100K x 1024"),
        (1_000, 262_144, "1K x 262144"),
        (5_000, 262_144, "5K x 262144"),
    ]

    print("\n" + "=" * 94)
    print("BENCHMARK: Seeded Vector Generation")
    print("=" * 94)
    hdr = (
        f"{'Config':<20} {'Eager (ms)':>14} "
        f"{'Compiled (ms)':>14} {'Speedup':>9} {'Melem/s':>12}"
    )
    print(hdr)
    print("-" * 94)

    n_warmup = 5
    n_trials = 20

    for batch, n_elem, label in configs:
        total_elements = batch * n_elem

        # Skip configs that would exceed GPU memory
        if device.type == "cuda":
            free_mem = torch.cuda.mem_get_info(device)[0]
            if _estimate_peak_bytes(batch, n_elem) > free_mem * 0.8:
                print(f"{label:<20} {'SKIP (OOM)':>14}")
                continue

        seeds = torch.arange(batch, device=device, dtype=torch.int64)

        # ---- Eager ----
        for _ in range(n_warmup):
            gen(seeds, n_elem)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        t0 = time.perf_counter()
        for _ in range(n_trials):
            gen(seeds, n_elem)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_eager = (time.perf_counter() - t0) / n_trials * 1000

        # ---- Compiled ----
        for _ in range(n_warmup):
            gen_compiled(seeds, n_elem)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        t0 = time.perf_counter()
        for _ in range(n_trials):
            gen_compiled(seeds, n_elem)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_compiled = (time.perf_counter() - t0) / n_trials * 1000

        speedup = t_eager / max(t_compiled, 1e-12)
        melem = total_elements / (t_compiled / 1000) / 1e6

        print(
            f"{label:<20} {t_eager:>11.2f} ms "
            f"{t_compiled:>11.2f} ms {speedup:>8.2f}x {melem:>10.0f}"
        )

    print("=" * 94)


# ---------------------------------------------------------------------------
# Part 3: Seed search
# ---------------------------------------------------------------------------


def _auto_chunk_size(
    n_elements: int, device: torch.device, fraction: float = 0.5
) -> int:
    """Pick the largest batch that fits in *fraction* of free GPU memory."""
    if device.type == "cuda":
        free = torch.cuda.mem_get_info(device)[0]
    else:
        free = 4 * 1024**3  # 4 GiB default for CPU

    bytes_per_candidate = max(n_elements * 20, 1)  # see _estimate_peak_bytes
    chunk = int(free * fraction / bytes_per_candidate)
    return max(chunk, 1)


@torch.no_grad()
def find_best_seed(
    target: torch.Tensor,
    generator: SeededVectorGenerator,
    n_seeds: int,
    chunk_size: int | None = None,
    seed_offset: int = 0,
    verbose: bool = True,
) -> tuple[int, float, torch.Tensor]:
    """Brute-force search for the seed whose generated vector best matches *target*.

    Args:
        target:      Tensor of any shape (will be flattened internally).
        generator:   A :class:`SeededVectorGenerator` instance.
        n_seeds:     Total number of seeds to try.
        chunk_size:  Batch size per iteration; auto-computed from free memory when *None*.
        seed_offset: First seed value.
        verbose:     Print progress lines.

    Returns:
        ``(best_seed, best_mse, best_vector)``  where *best_vector* has the
        same shape as *target*.
    """
    device = target.device
    orig_shape = target.shape
    target_flat = target.reshape(1, -1)  # [1, n]
    n = target_flat.shape[1]

    if chunk_size is None:
        chunk_size = _auto_chunk_size(n, device)
    chunk_size = min(chunk_size, n_seeds)

    if verbose:
        print(
            f"  Searching {n_seeds:,} seeds in chunks of {chunk_size:,} ({n:,} params)"
        )

    # Compile a fresh module for this element count
    gen_c = torch.compile(
        SeededVectorGenerator().to(device),
        fullgraph=True,
        dynamic=False,
    )
    # Warmup compilation
    _ws = torch.zeros(min(chunk_size, 8), device=device, dtype=torch.int64)
    gen_c(_ws, n)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    best_mse = float("inf")
    best_seed = -1
    t_start = time.perf_counter()
    seeds_done = 0

    for start in range(0, n_seeds, chunk_size):
        end = min(start + chunk_size, n_seeds)

        seeds = torch.arange(
            seed_offset + start,
            seed_offset + end,
            device=device,
            dtype=torch.int64,
        )

        candidates = gen_c(seeds, n)  # [chunk, n]
        diff = candidates - target_flat  # [chunk, n]
        mse_vals = (diff * diff).mean(dim=1)  # [chunk]

        chunk_best_idx = mse_vals.argmin()
        chunk_best_mse = mse_vals[chunk_best_idx].item()

        if chunk_best_mse < best_mse:
            best_mse = chunk_best_mse
            best_seed = seed_offset + start + chunk_best_idx.item()

        seeds_done += end - start

        if verbose and (
            seeds_done % max(chunk_size * 10, 1) < (end - start) or end == n_seeds
        ):
            elapsed = time.perf_counter() - t_start
            rate = seeds_done / max(elapsed, 1e-9)
            eta = (n_seeds - seeds_done) / max(rate, 1)
            print(
                f"    {seeds_done:>12,}/{n_seeds:,} seeds | "
                f"best MSE={best_mse:.6f} | "
                f"{rate:,.0f} seeds/s | "
                f"ETA {eta:.1f}s"
            )

    # Regenerate the winning vector and reshape to match target
    best_seed_t = torch.tensor([best_seed], device=device, dtype=torch.int64)
    best_vector = generator(best_seed_t, n).squeeze(0).view(orig_shape)

    return best_seed, best_mse, best_vector


@torch.no_grad()
def find_best_seed_affine(
    target: torch.Tensor,
    generator: SeededVectorGenerator,
    n_seeds: int,
    chunk_size: int | None = None,
    seed_offset: int = 0,
    verbose: bool = True,
) -> tuple[int, float, float, float, torch.Tensor]:
    """Search for seed + optimal affine transform (scale, shift).

    Reconstruction: ``vector = gen(seed, n) * scale + shift``

    For each candidate seed the optimal ``(scale, shift)`` are solved in
    closed form via least-squares, so the brute-force only iterates over
    seeds.  The optimal residual MSE is computed without materialising the
    fitted tensor::

        MSE_opt = var(target) - cov(g, target)² / var(g)

    which keeps the inner loop to three cheap reductions per chunk.

    Args:
        target:      Tensor of any shape (will be flattened internally).
        generator:   A :class:`SeededVectorGenerator` instance.
        n_seeds:     Total number of seeds to try.
        chunk_size:  Batch size per iteration; auto-computed from free memory when *None*.
        seed_offset: First seed value.
        verbose:     Print progress lines.

    Returns:
        ``(best_seed, scale, shift, best_mse, best_vector)``
    """
    device = target.device
    orig_shape = target.shape
    target_flat = target.reshape(1, -1).float()  # [1, n]
    n = target_flat.shape[1]

    # Pre-compute target statistics (constant across all candidates)
    t_mean = target_flat.mean()
    var_t = target_flat.var(correction=0)

    if chunk_size is None:
        chunk_size = _auto_chunk_size(n, device)
    chunk_size = min(chunk_size, n_seeds)

    if verbose:
        print(
            f"  Searching {n_seeds:,} seeds (affine) in chunks of "
            f"{chunk_size:,} ({n:,} params)"
        )

    # Compile a fresh module for this element count
    gen_c = torch.compile(
        SeededVectorGenerator().to(device),
        fullgraph=True,
        dynamic=False,
    )
    _ws = torch.zeros(min(chunk_size, 8), device=device, dtype=torch.int64)
    gen_c(_ws, n)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    best_mse = float("inf")
    best_seed = -1
    t_start = time.perf_counter()
    seeds_done = 0

    for start in range(0, n_seeds, chunk_size):
        end = min(start + chunk_size, n_seeds)

        seeds = torch.arange(
            seed_offset + start,
            seed_offset + end,
            device=device,
            dtype=torch.int64,
        )

        g = gen_c(seeds, n)  # [chunk, n]

        # Sufficient statistics — three reductions, no extra [chunk, n] alloc
        g_mean = g.mean(dim=1)  # [chunk]
        gt_mean = (g * target_flat).mean(dim=1)  # [chunk]
        g_sq_mean = (g * g).mean(dim=1)  # [chunk]

        cov = gt_mean - g_mean * t_mean  # [chunk]
        var_g = (g_sq_mean - g_mean * g_mean).clamp(min=1e-20)  # [chunk]

        # Optimal residual MSE (closed-form, no fitted tensor needed)
        mse_vals = (var_t - cov * cov / var_g).clamp(min=0)  # [chunk]

        chunk_best_idx = mse_vals.argmin()
        chunk_best_mse = mse_vals[chunk_best_idx].item()

        if chunk_best_mse < best_mse:
            best_mse = chunk_best_mse
            best_seed = seed_offset + start + chunk_best_idx.item()

        seeds_done += end - start

        if verbose and (
            seeds_done % max(chunk_size * 10, 1) < (end - start) or end == n_seeds
        ):
            elapsed = time.perf_counter() - t_start
            rate = seeds_done / max(elapsed, 1e-9)
            eta = (n_seeds - seeds_done) / max(rate, 1)
            print(
                f"    {seeds_done:>12,}/{n_seeds:,} seeds | "
                f"best MSE={best_mse:.6f} | "
                f"{rate:,.0f} seeds/s | "
                f"ETA {eta:.1f}s"
            )

    # Recover scale & shift for the winning seed
    best_seed_t = torch.tensor([best_seed], device=device, dtype=torch.int64)
    g_best = generator(best_seed_t, n).squeeze(0).float()
    g_best_mean = g_best.mean()
    cov_best = ((g_best - g_best_mean) * (target_flat.squeeze(0) - t_mean)).mean()
    var_g_best = ((g_best - g_best_mean) ** 2).mean().clamp(min=1e-20)
    best_scale = (cov_best / var_g_best).item()
    best_shift = (t_mean - best_scale * g_best_mean).item()

    best_vector = (g_best * best_scale + best_shift).view(orig_shape)

    return best_seed, best_scale, best_shift, best_mse, best_vector


# ---------------------------------------------------------------------------
# Part 4: Evaluation across parameter counts
# ---------------------------------------------------------------------------

# _DEFAULT_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536, 262144]
_DEFAULT_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def _metrics(target: torch.Tensor, approx: torch.Tensor) -> tuple[float, float]:
    """Return (relative_l2_error, cosine_similarity) for two flat tensors."""
    t = target.float()
    a = approx.float()
    rel_l2 = (torch.norm(a - t) / torch.norm(t)).item()
    cos = (
        F.cosine_similarity(t.unsqueeze(0), a.unsqueeze(0)).item()
        if t.numel() > 1
        else float("nan")
    )
    return rel_l2, cos


def evaluate_compression(
    device: torch.device,
    n_seeds: int = 10_000_000,
    sizes: list[int] | None = None,
) -> list[dict]:
    """Evaluate seed-based compression: raw (1 param) vs affine (3 params)."""
    if sizes is None:
        sizes = list(_DEFAULT_SIZES)

    gen = SeededVectorGenerator().to(device)

    sep = "=" * 115
    print(f"\n{sep}")
    print("EVALUATION: Seed-Based Vector Compression")
    print(f"Device: {device} | Base seed budget: {n_seeds:,}")
    print(sep)
    print(
        f"{'Params':>8} {'Seeds':>12}  "
        f"{'--- Raw (1 int) ---':^28s}  "
        f"{'--- Affine (seed+scale+shift) ---':^34s}"
    )
    print(
        f"{'':>8} {'':>12}  "
        f"{'MSE':>10} {'Rel L2':>10} {'CosSim':>8}  "
        f"{'MSE':>10} {'Rel L2':>10} {'CosSim':>8} {'Time':>8}"
    )
    print("-" * 115)

    results: list[dict] = []
    torch.manual_seed(42)  # reproducible targets

    for n_params in sizes:
        # Scale search budget down for bigger vectors (memory + time)
        scaled_seeds = min(n_seeds, max(100_000, n_seeds // max(1, n_params // 64)))

        target = torch.randn(n_params, device=device)

        # --- Raw search (1 param: seed) ---
        t0 = time.perf_counter()
        raw_seed, raw_mse, raw_vec = find_best_seed(
            target,
            gen,
            scaled_seeds,
            verbose=False,
        )
        raw_time = time.perf_counter() - t0
        raw_rl2, raw_cos = _metrics(target, raw_vec)

        # --- Affine search (3 params: seed + scale + shift) ---
        t0 = time.perf_counter()
        aff_seed, aff_scale, aff_shift, aff_mse, aff_vec = find_best_seed_affine(
            target,
            gen,
            scaled_seeds,
            verbose=False,
        )
        aff_time = time.perf_counter() - t0
        aff_rl2, aff_cos = _metrics(target, aff_vec)

        total_time = raw_time + aff_time

        print(
            f"{n_params:>8,} {scaled_seeds:>12,}  "
            f"{raw_mse:>10.6f} {raw_rl2:>10.6f} {raw_cos:>8.4f}  "
            f"{aff_mse:>10.6f} {aff_rl2:>10.6f} {aff_cos:>8.4f} {total_time:>7.1f}s"
        )

        results.append(
            dict(
                n_params=n_params,
                n_seeds_searched=scaled_seeds,
                raw=dict(seed=raw_seed, mse=raw_mse, rel_l2=raw_rl2, cos_sim=raw_cos),
                affine=dict(
                    seed=aff_seed,
                    scale=aff_scale,
                    shift=aff_shift,
                    mse=aff_mse,
                    rel_l2=aff_rl2,
                    cos_sim=aff_cos,
                ),
                time_s=total_time,
            )
        )

        # Verify affine reproduction
        repro = (
            gen(
                torch.tensor([aff_seed], device=device, dtype=torch.int64),
                n_params,
            ).squeeze(0)
            * aff_scale
            + aff_shift
        )
        max_diff = (repro - aff_vec).abs().max().item()
        if max_diff > 1e-5:
            print(f"  WARNING: affine reproduction mismatch  max|diff| = {max_diff}")

    print(sep)
    return results


# ---------------------------------------------------------------------------
# Part 5: Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Seeded random-vector generation & compression experiment",
    )
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (default: cuda if available, else cpu)",
    )
    ap.add_argument("--benchmark-only", action="store_true")
    ap.add_argument("--evaluate-only", action="store_true")
    ap.add_argument(
        "--n-seeds",
        type=int,
        default=10_000_000,
        help="Base seed budget for evaluation (default 10 M)",
    )
    ap.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=None,
        help="Parameter counts to evaluate (default: 1..262144)",
    )
    args = ap.parse_args()

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(device)}")
        torch.set_float32_matmul_precision("high")

    # Quick determinism sanity check
    gen = SeededVectorGenerator()
    seeds = torch.tensor([0, 1, 42, 12345], dtype=torch.int64)
    m1 = gen(seeds, 16)
    m2 = gen(seeds, 16)
    print(f"Determinism check: max|diff| = {(m1 - m2).abs().max().item()}")

    do_bench = not args.evaluate_only
    do_eval = not args.benchmark_only

    if do_bench:
        benchmark_generation(device)

    if do_eval:
        evaluate_compression(device, n_seeds=args.n_seeds, sizes=args.sizes)


if __name__ == "__main__":
    main()
