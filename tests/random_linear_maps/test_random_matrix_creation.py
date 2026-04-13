#!/usr/bin/env python3
"""
Random matrix generation from seeds & seed-search compression experiment.

Benchmarks seeded matrix generation on GPU, then evaluates how well
brute-force seed search can approximate matrices of varying sizes.
The idea: represent an entire matrix with a single integer seed + a
deterministic generation function. Extreme compression via hash-based PRNG.

Usage:
    python tests/random_linear_maps/test_random_matrix_creation.py --device cuda
    python tests/random_linear_maps/test_random_matrix_creation.py --benchmark-only
    python tests/random_linear_maps/test_random_matrix_creation.py --evaluate-only --n-seeds 1000000
"""

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

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
# Part 1: Seeded Matrix Generator
# ---------------------------------------------------------------------------

class SeededMatrixGenerator(nn.Module):
    """Deterministic random matrix generation from integer seeds.

    Uses a SplitMix64-style hash on ``(seed, element_index)`` pairs.
    Fully vectorised over the batch dimension, no Python loops,
    ``torch.compile(fullgraph=True)`` compatible.
    """

    def forward(self, seeds: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        """
        Args:
            seeds: ``[batch]`` int64 tensor of seed values.
            rows:  number of rows in each output matrix.
            cols:  number of columns in each output matrix.

        Returns:
            ``[batch, rows, cols]`` float32 tensor with values in ~[-1, 1].
        """
        n = rows * cols
        idx = torch.arange(n, device=seeds.device, dtype=torch.int64)

        # Combine seed and element position  →  [batch, n]
        x = seeds.unsqueeze(1) * _GOLDEN + idx.unsqueeze(0)

        # SplitMix64 finaliser (three rounds of xor-shift-multiply)
        x = (x ^ (x >> 30)) * _MIX1
        x = (x ^ (x >> 27)) * _MIX2
        x = x ^ (x >> 31)

        # Extract lower 32 random bits → float in [-1, 1)
        # (avoids int64-sign issues; 32 bits > float32's 23-bit mantissa)
        result = (x & 0xFFFFFFFF).float() * (1.0 / 2147483648.0) - 1.0
        return result.view(seeds.shape[0], rows, cols)


# ---------------------------------------------------------------------------
# Part 2: Benchmark
# ---------------------------------------------------------------------------

def _estimate_peak_bytes(batch: int, n_elements: int) -> int:
    """Conservative peak-memory estimate for one generation call.

    Two int64 intermediates + one float32 output ≈ 20 bytes per element.
    """
    return batch * n_elements * 20


def benchmark_generation(device: torch.device) -> None:
    """Benchmark seeded matrix generation: compiled vs. uncompiled."""
    gen = SeededMatrixGenerator().to(device)

    configs = [
        # (batch, rows, cols, label)
        (1_000, 16, 16, "1K x 16x16"),
        (10_000, 16, 16, "10K x 16x16"),
        (100_000, 16, 16, "100K x 16x16"),
        (1_000_000, 16, 16, "1M x 16x16"),
        (10_000, 32, 32, "10K x 32x32"),
        (100_000, 32, 32, "100K x 32x32"),
        (1_000, 512, 512, "1K x 512x512"),
        (5_000, 512, 512, "5K x 512x512"),
    ]

    # Pre-compile one fresh module per distinct (rows, cols) shape so we never
    # exceed the dynamo recompile limit on a single module instance.
    distinct_shapes = sorted({(r, c) for _, r, c, _ in configs})
    compiled_by_shape: dict[tuple[int, int], nn.Module] = {}
    for rows, cols in distinct_shapes:
        m = SeededMatrixGenerator().to(device)
        compiled_by_shape[(rows, cols)] = torch.compile(
            m, fullgraph=True, dynamic=False,
        )

    print("\n" + "=" * 94)
    print("BENCHMARK: Seeded Matrix Generation")
    print("=" * 94)
    hdr = (
        f"{'Config':<20} {'Eager (ms)':>14} "
        f"{'Compiled (ms)':>14} {'Speedup':>9} {'Melem/s':>12}"
    )
    print(hdr)
    print("-" * 94)

    n_warmup = 5
    n_trials = 20

    for batch, rows, cols, label in configs:
        n_elements = rows * cols
        total_elements = batch * n_elements

        # Skip configs that would exceed GPU memory
        if device.type == "cuda":
            free_mem = torch.cuda.mem_get_info(device)[0]
            if _estimate_peak_bytes(batch, n_elements) > free_mem * 0.8:
                print(f"{label:<20} {'SKIP (OOM)':>14}")
                continue

        seeds = torch.arange(batch, device=device, dtype=torch.int64)
        gen_compiled = compiled_by_shape[(rows, cols)]

        # ---- Eager ----
        for _ in range(n_warmup):
            gen(seeds, rows, cols)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        t0 = time.perf_counter()
        for _ in range(n_trials):
            gen(seeds, rows, cols)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_eager = (time.perf_counter() - t0) / n_trials * 1000

        # ---- Compiled ----
        for _ in range(n_warmup):
            gen_compiled(seeds, rows, cols)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        t0 = time.perf_counter()
        for _ in range(n_trials):
            gen_compiled(seeds, rows, cols)
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

def _auto_chunk_size(n_elements: int, device: torch.device,
                     fraction: float = 0.5) -> int:
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
    generator: SeededMatrixGenerator,
    n_seeds: int,
    chunk_size: int | None = None,
    seed_offset: int = 0,
    verbose: bool = True,
) -> tuple[int, float, torch.Tensor]:
    """Brute-force search for the seed whose generated matrix best matches *target*.

    Args:
        target:      ``[rows, cols]`` or ``[n]`` tensor to approximate.
        generator:   A :class:`SeededMatrixGenerator` instance (will be compiled internally).
        n_seeds:     Total number of seeds to try (``seed_offset .. seed_offset + n_seeds - 1``).
        chunk_size:  Batch size per iteration; auto-computed from free memory when *None*.
        seed_offset: First seed value.
        verbose:     Print progress lines.

    Returns:
        ``(best_seed, best_mse, best_matrix)``
    """
    device = target.device

    # Normalise target to 2-D
    if target.ndim == 1:
        rows, cols = 1, target.shape[0]
        target_2d = target.unsqueeze(0)
    else:
        rows, cols = target.shape
        target_2d = target

    n_elements = rows * cols
    target_flat = target_2d.reshape(1, -1)  # [1, n]

    if chunk_size is None:
        chunk_size = _auto_chunk_size(n_elements, device)
    chunk_size = min(chunk_size, n_seeds)

    if verbose:
        print(
            f"  Searching {n_seeds:,} seeds in chunks of {chunk_size:,} "
            f"(matrix {rows}x{cols} = {n_elements:,} params)"
        )

    # Compile a *fresh* module so each (rows, cols) shape gets its own dynamo
    # cache and we never hit the recompile limit across successive calls.
    gen_c = torch.compile(
        SeededMatrixGenerator().to(device), fullgraph=True, dynamic=False,
    )
    # Warmup compilation
    _ws = torch.zeros(min(chunk_size, 8), device=device, dtype=torch.int64)
    gen_c(_ws, rows, cols)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    best_mse = float("inf")
    best_seed = -1
    t_start = time.perf_counter()
    seeds_done = 0

    for start in range(0, n_seeds, chunk_size):
        end = min(start + chunk_size, n_seeds)

        seeds = torch.arange(
            seed_offset + start, seed_offset + end,
            device=device, dtype=torch.int64,
        )

        candidates = gen_c(seeds, rows, cols)                # [chunk, r, c]
        diff = candidates.reshape(seeds.shape[0], -1) - target_flat  # [chunk, n]
        mse_vals = (diff * diff).mean(dim=1)                 # [chunk]

        chunk_best_idx = mse_vals.argmin()
        chunk_best_mse = mse_vals[chunk_best_idx].item()

        if chunk_best_mse < best_mse:
            best_mse = chunk_best_mse
            best_seed = seed_offset + start + chunk_best_idx.item()

        seeds_done += end - start

        if verbose and (
            seeds_done % max(chunk_size * 10, 1) < (end - start)
            or end == n_seeds
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

    # Regenerate the winning matrix so the caller can inspect it
    best_seed_t = torch.tensor([best_seed], device=device, dtype=torch.int64)
    best_matrix = generator(best_seed_t, rows, cols).squeeze(0)
    if target.ndim == 1:
        best_matrix = best_matrix.squeeze(0)

    return best_seed, best_mse, best_matrix


# ---------------------------------------------------------------------------
# Part 4: Evaluation across parameter counts
# ---------------------------------------------------------------------------

_DEFAULT_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536, 262144]


def _nice_shape(n: int) -> tuple[int, int]:
    """Pick a roughly-square (rows, cols) factorisation of *n*."""
    sqrt_n = int(math.isqrt(n))
    if sqrt_n * sqrt_n == n and sqrt_n > 1:
        return sqrt_n, sqrt_n
    for r in range(sqrt_n, 0, -1):
        if n % r == 0:
            return r, n // r
    return 1, n


def evaluate_compression(
    device: torch.device,
    n_seeds: int = 10_000_000,
    sizes: list[int] | None = None,
) -> list[dict]:
    """Evaluate seed-based compression quality for many parameter counts."""
    if sizes is None:
        sizes = list(_DEFAULT_SIZES)

    gen = SeededMatrixGenerator().to(device)

    sep = "=" * 114
    print(f"\n{sep}")
    print("EVALUATION: Seed-Based Matrix Compression")
    print(f"Device: {device} | Base seed budget: {n_seeds:,}")
    print(sep)
    print(
        f"{'Params':>8} {'Shape':>12} {'Seeds':>12} "
        f"{'Best MSE':>12} {'Rel L2 Err':>12} {'Cos Sim':>10} {'Time (s)':>10}"
    )
    print("-" * 114)

    results: list[dict] = []
    torch.manual_seed(42)  # reproducible targets

    for n_params in sizes:
        rows, cols = _nice_shape(n_params)
        shape_str = f"{rows}x{cols}" if rows > 1 else f"1x{cols}"

        # Scale search budget down for bigger matrices (memory + time)
        scaled_seeds = min(
            n_seeds, max(100_000, n_seeds // max(1, n_params // 64))
        )

        # Random target (normal distribution, as typical weight init)
        target = torch.randn(rows, cols, device=device)
        if rows == 1:
            target = target.squeeze(0)  # 1-D for vectors

        t0 = time.perf_counter()
        best_seed, best_mse, best_matrix = find_best_seed(
            target, gen, scaled_seeds, verbose=False,
        )
        elapsed = time.perf_counter() - t0

        # Metrics
        t_flat = target.reshape(-1).float()
        b_flat = best_matrix.reshape(-1).float()

        rel_l2 = (torch.norm(b_flat - t_flat) / torch.norm(t_flat)).item()
        cos_sim = F.cosine_similarity(
            t_flat.unsqueeze(0), b_flat.unsqueeze(0),
        ).item()

        print(
            f"{n_params:>8,} {shape_str:>12} {scaled_seeds:>12,} "
            f"{best_mse:>12.6f} {rel_l2:>12.6f} {cos_sim:>10.4f} {elapsed:>10.1f}"
        )

        results.append(dict(
            n_params=n_params,
            shape=(rows, cols),
            n_seeds_searched=scaled_seeds,
            best_seed=best_seed,
            best_mse=best_mse,
            rel_l2=rel_l2,
            cos_sim=cos_sim,
            time_s=elapsed,
        ))

        # Verify deterministic reproduction
        r, c = (1, cols) if target.ndim == 1 else (rows, cols)
        repro = gen(
            torch.tensor([best_seed], device=device, dtype=torch.int64), r, c,
        ).squeeze(0)
        if target.ndim == 1:
            repro = repro.squeeze(0)
        max_diff = (repro - best_matrix).abs().max().item()
        if max_diff > 0:
            print(f"  WARNING: reproduction mismatch  max|diff| = {max_diff}")

    print(sep)
    return results


# ---------------------------------------------------------------------------
# Part 5: Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Seeded random-matrix generation & compression experiment",
    )
    ap.add_argument(
        "--device", type=str, default=None,
        help="Device (default: cuda if available, else cpu)",
    )
    ap.add_argument("--benchmark-only", action="store_true")
    ap.add_argument("--evaluate-only", action="store_true")
    ap.add_argument(
        "--n-seeds", type=int, default=10_000_000,
        help="Base seed budget for evaluation (default 10 M)",
    )
    ap.add_argument(
        "--sizes", type=int, nargs="+", default=None,
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
    gen = SeededMatrixGenerator()
    seeds = torch.tensor([0, 1, 42, 12345], dtype=torch.int64)
    m1 = gen(seeds, 4, 4)
    m2 = gen(seeds, 4, 4)
    print(f"Determinism check: max|diff| = {(m1 - m2).abs().max().item()}")

    do_bench = not args.evaluate_only
    do_eval = not args.benchmark_only

    if do_bench:
        benchmark_generation(device)

    if do_eval:
        evaluate_compression(device, n_seeds=args.n_seeds, sizes=args.sizes)


if __name__ == "__main__":
    main()
