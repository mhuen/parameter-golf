"""Distribution analysis for all structural stream components.

Runs the default ``build_multi_stream_components`` setup across multiple
sequence lengths and reports per-component, per-dimension statistics:
min, max, mean, std, and quantiles (1%, 25%, 50%, 75%, 99%).

Usage:
    # Print distribution tables:
    python tests/stream_components/test_component_distributions.py

    # Only specific components:
    python tests/stream_components/test_component_distributions.py -c boundary hash

    # As pytest (asserts mean/std/range are within acceptable bounds):
    pytest tests/stream_components/test_component_distributions.py -v
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

# -- path setup (matches existing test conventions) --
_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, _PROJECT_ROOT)

from data import load_shard_byte260  # noqa: E402
from efficient_byte_tokenizer import EfficientByteTokenizer  # noqa: E402
from byte_stream_components import (  # noqa: E402
    BoundaryComponent,
    ByteCategoryComponent,
    ByteCategoryStatsComponent,
    ByteHashComponent,
    CaseComponent,
    ColumnPositionComponent,
    DigitComputeComponent,
    DigitSequenceComponent,
    HashBoundary,
    MultiByteStateComponent,
    PunctuationDepthComponent,
    RepeatedByteComponent,
    VowelConsonantComponent,
)
from multi_streams import DocBoundaryComponent, SinCosPositionComponent  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_DATA_DIR = Path(_PROJECT_ROOT) / "data" / "datasets" / "fineweb10B_byte260"

_DEFAULT_SEQ_LENS = [1024, 2048, 4096, 16_384, 32_768]

# Thresholds for pytest assertions
_MAX_ABS_MEAN = 5.0  # per-dim mean should stay moderate
_MAX_STD = 10.0  # per-dim std should stay moderate
_MAX_ABS_VALUE = 500.0  # no extreme outlier values

# ---------------------------------------------------------------------------
# Component registry (mirrors build_multi_stream_components)
# ---------------------------------------------------------------------------


def _build_component_registry(
    tok: EfficientByteTokenizer,
) -> list[tuple[str, torch.nn.Module]]:
    """Return (label, component) pairs matching the default model setup."""
    return [
        ("DocBoundary", DocBoundaryComponent(bos_id=tok.bos_id, num_freqs=2)),
        ("SinCosPosition", SinCosPositionComponent(num_freqs=10)),
        ("ByteCategory", ByteCategoryComponent(tok=tok, embed_dim=4)),
        ("MultiByteState", MultiByteStateComponent(tok=tok, id_freqs=3)),
        ("Case", CaseComponent(tok=tok)),
        ("VowelConsonant", VowelConsonantComponent(tok=tok)),
        ("ColumnPosition", ColumnPositionComponent(tok=tok, num_freqs=3)),
        ("RepeatedByte", RepeatedByteComponent()),
        ("PunctuationDepth", PunctuationDepthComponent(tok=tok)),
        ("ByteCategoryStats", ByteCategoryStatsComponent(tok=tok)),
        ("DigitSequence", DigitSequenceComponent(tok=tok, id_freqs=5)),
        ("DigitCompute", DigitComputeComponent(tok=tok)),
        (
            "ByteHash_word_w20",
            ByteHashComponent(
                tok, window=20, num_hashes=2,
                boundary=HashBoundary.WORD, track_hits=True,
            ),
        ),
        (
            "ByteHash_w2",
            ByteHashComponent(
                tok, window=2, num_hashes=2, boundary=None, track_hits=True,
            ),
        ),
        (
            "ByteHash_w3",
            ByteHashComponent(
                tok, window=3, num_hashes=2, boundary=None, track_hits=True,
            ),
        ),
        (
            "ByteHash_w5",
            ByteHashComponent(
                tok, window=5, num_hashes=2, boundary=None, track_hits=True,
            ),
        ),
        (
            "ByteHash_w8",
            ByteHashComponent(
                tok, window=8, num_hashes=2, boundary=None, track_hits=True,
            ),
        ),
        (
            "ByteHash_digit_w8",
            ByteHashComponent(
                tok, window=8, num_hashes=2,
                boundary=HashBoundary.DIGIT, track_hits=True,
            ),
        ),
        (
            "Boundary",
            BoundaryComponent(
                tok,
                word_pos_freqs=3, word_id_freqs=3,
                sent_pos_freqs=2, sent_id_freqs=2,
                para_pos_freqs=2, para_id_freqs=2,
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Data loading (reuses pattern from test_component_benchmark)
# ---------------------------------------------------------------------------


def _load_input_ids(
    seq_len: int,
    batch_size: int = 1,
    data_dir: Path | None = None,
) -> torch.Tensor:
    """Load real data and return a (batch_size, seq_len) input_ids tensor."""
    data_dir = data_dir or Path(os.environ.get("DATA_PATH", str(_DEFAULT_DATA_DIR)))
    shards = sorted(data_dir.glob("fineweb_val_*.bin"))
    if not shards:
        shards = sorted(data_dir.glob("fineweb_train_*.bin"))
    if not shards:
        raise FileNotFoundError(
            f"No .bin shards found in {data_dir}. Set DATA_PATH or pass --data-dir."
        )
    tok = EfficientByteTokenizer()
    total_needed = seq_len * batch_size
    chunks = []
    collected = 0
    for shard in shards:
        t = load_shard_byte260(shard, tok)
        t = torch.from_numpy(t) if not isinstance(t, torch.Tensor) else t
        chunks.append(t)
        collected += t.numel()
        if collected >= total_needed:
            break
    tokens = torch.cat(chunks).long()
    if tokens.numel() < total_needed:
        repeats = (total_needed + tokens.numel() - 1) // tokens.numel()
        tokens = tokens.repeat(repeats)
    tokens = tokens[:total_needed]
    return tokens.reshape(batch_size, seq_len)


# ---------------------------------------------------------------------------
# Per-dimension statistics
# ---------------------------------------------------------------------------

_QUANTILES = [0.01, 0.25, 0.50, 0.75, 0.99]


@torch.no_grad()
def compute_dim_stats(output: torch.Tensor) -> list[dict]:
    """Compute per-dimension statistics for a (B, S, D) tensor.

    Returns a list of dicts (one per dimension) with keys:
    dim_idx, mean, std, min, max, q01, q25, q50, q75, q99.
    """
    # Flatten batch and sequence: (B*S, D)
    flat = output.float().reshape(-1, output.shape[-1])
    D = flat.shape[-1]

    means = flat.mean(dim=0)
    stds = flat.std(dim=0)
    mins = flat.min(dim=0).values
    maxs = flat.max(dim=0).values
    quantiles = torch.quantile(flat, torch.tensor(_QUANTILES), dim=0)  # (5, D)

    stats = []
    for d in range(D):
        stats.append({
            "dim_idx": d,
            "mean": means[d].item(),
            "std": stds[d].item(),
            "min": mins[d].item(),
            "max": maxs[d].item(),
            "q01": quantiles[0, d].item(),
            "q25": quantiles[1, d].item(),
            "q50": quantiles[2, d].item(),
            "q75": quantiles[3, d].item(),
            "q99": quantiles[4, d].item(),
        })
    return stats


# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------


def run_distribution_analysis(
    seq_lens: list[int],
    batch_size: int = 1,
    data_dir: Path | None = None,
    components: list[str] | None = None,
) -> dict[int, list[dict]]:
    """Run all components at each sequence length.

    Returns:
        {seq_len: [{"label": str, "dim": int, "dim_stats": [...]}, ...]}
    """
    tok = EfficientByteTokenizer()
    registry = _build_component_registry(tok)

    if components:
        lc_filters = [f.lower() for f in components]
        registry = [
            (label, comp) for label, comp in registry
            if any(f in label.lower() for f in lc_filters)
        ]
        if not registry:
            raise ValueError(f"No components matched filters: {components}")

    results_by_seqlen: dict[int, list[dict]] = {}

    for seq_len in seq_lens:
        input_ids = _load_input_ids(seq_len, batch_size=batch_size, data_dir=data_dir)
        seq_results = []

        for label, component in registry:
            with torch.no_grad():
                output = component(input_ids, dtype=torch.float32)
            dim_stats = compute_dim_stats(output)
            seq_results.append({
                "label": label,
                "dim": component.dim,
                "shape": tuple(output.shape),
                "dim_stats": dim_stats,
            })

        results_by_seqlen[seq_len] = seq_results

    return results_by_seqlen


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------


def print_distribution_report(
    results_by_seqlen: dict[int, list[dict]],
    verbose: bool = False,
) -> None:
    """Print distribution tables.

    Default mode prints a per-component summary (aggregated across dims).
    Verbose mode additionally prints per-dimension breakdowns.
    """
    for seq_len, seq_results in sorted(results_by_seqlen.items()):
        print()
        print(f"{'=' * 90}")
        print(f"  Sequence length: {seq_len:,}")
        print(f"{'=' * 90}")

        # -- Per-component summary (aggregate across dims) --
        header = (
            f"{'Component':<25} {'Dim':>4}  "
            f"{'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}  "
            f"{'Q01':>8} {'Q50':>8} {'Q99':>8}"
        )
        print(header)
        print("-" * len(header))

        for r in seq_results:
            stats = r["dim_stats"]
            # Aggregate: mean-of-means, mean-of-stds, global min/max, mean quantiles
            agg_mean = sum(s["mean"] for s in stats) / len(stats)
            agg_std = sum(s["std"] for s in stats) / len(stats)
            agg_min = min(s["min"] for s in stats)
            agg_max = max(s["max"] for s in stats)
            agg_q01 = sum(s["q01"] for s in stats) / len(stats)
            agg_q50 = sum(s["q50"] for s in stats) / len(stats)
            agg_q99 = sum(s["q99"] for s in stats) / len(stats)
            print(
                f"{r['label']:<25} {r['dim']:>4}  "
                f"{agg_mean:>8.3f} {agg_std:>8.3f} {agg_min:>8.3f} {agg_max:>8.3f}  "
                f"{agg_q01:>8.3f} {agg_q50:>8.3f} {agg_q99:>8.3f}"
            )

        # -- Per-dimension breakdown (verbose) --
        if verbose:
            for r in seq_results:
                print()
                print(f"  {r['label']} ({r['dim']} dims):")
                dim_header = (
                    f"    {'Dim':>4}  "
                    f"{'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}  "
                    f"{'Q01':>8} {'Q25':>8} {'Q50':>8} {'Q75':>8} {'Q99':>8}"
                )
                print(dim_header)
                print(f"    {'-' * (len(dim_header) - 4)}")
                for s in r["dim_stats"]:
                    print(
                        f"    {s['dim_idx']:>4}  "
                        f"{s['mean']:>8.3f} {s['std']:>8.3f} "
                        f"{s['min']:>8.3f} {s['max']:>8.3f}  "
                        f"{s['q01']:>8.3f} {s['q25']:>8.3f} "
                        f"{s['q50']:>8.3f} {s['q75']:>8.3f} {s['q99']:>8.3f}"
                    )

    print()


# ---------------------------------------------------------------------------
# Violation checking
# ---------------------------------------------------------------------------


def find_violations(
    results_by_seqlen: dict[int, list[dict]],
    max_abs_mean: float = _MAX_ABS_MEAN,
    max_std: float = _MAX_STD,
    max_abs_value: float = _MAX_ABS_VALUE,
) -> list[str]:
    """Return a list of human-readable violation strings."""
    violations = []
    for seq_len, seq_results in sorted(results_by_seqlen.items()):
        for r in seq_results:
            label = r["label"]
            for s in r["dim_stats"]:
                d = s["dim_idx"]
                prefix = f"[seq_len={seq_len}] {label} dim={d}"

                if abs(s["mean"]) > max_abs_mean:
                    violations.append(
                        f"{prefix}: |mean|={abs(s['mean']):.4f} > {max_abs_mean}"
                    )
                if s["std"] > max_std:
                    violations.append(
                        f"{prefix}: std={s['std']:.4f} > {max_std}"
                    )
                abs_max = max(abs(s["min"]), abs(s["max"]))
                if abs_max > max_abs_value:
                    violations.append(
                        f"{prefix}: |max_value|={abs_max:.4f} > {max_abs_value}"
                    )
    return violations


# ---------------------------------------------------------------------------
# pytest integration
# ---------------------------------------------------------------------------

import pytest  # noqa: E402


@pytest.fixture(scope="module")
def distribution_results() -> dict[int, list[dict]]:
    """Run distribution analysis once for all pytest tests."""
    return run_distribution_analysis(
        seq_lens=_DEFAULT_SEQ_LENS,
        batch_size=1,
    )


class TestComponentDistributions:
    """Verify component output distributions stay within healthy ranges."""

    @torch.no_grad()
    def test_no_nans_or_infs(
        self, distribution_results: dict[int, list[dict]]
    ) -> None:
        """No dimension should have NaN or Inf in its statistics."""
        bad = []
        for seq_len, seq_results in distribution_results.items():
            for r in seq_results:
                for s in r["dim_stats"]:
                    vals = [s["mean"], s["std"], s["min"], s["max"]]
                    for v in vals:
                        if v != v or abs(v) == float("inf"):  # NaN or Inf
                            bad.append(
                                f"[seq_len={seq_len}] {r['label']} "
                                f"dim={s['dim_idx']}: NaN/Inf detected"
                            )
                            break
        if bad:
            pytest.fail("NaN/Inf found:\n  " + "\n  ".join(bad))

    @torch.no_grad()
    def test_mean_within_bounds(
        self, distribution_results: dict[int, list[dict]]
    ) -> None:
        """Per-dimension mean should not exceed threshold."""
        violations = []
        for seq_len, seq_results in distribution_results.items():
            for r in seq_results:
                for s in r["dim_stats"]:
                    if abs(s["mean"]) > _MAX_ABS_MEAN:
                        violations.append(
                            f"[seq_len={seq_len}] {r['label']} dim={s['dim_idx']}: "
                            f"|mean|={abs(s['mean']):.4f} > {_MAX_ABS_MEAN}"
                        )
        if violations:
            pytest.fail(
                f"{len(violations)} mean violation(s):\n  "
                + "\n  ".join(violations)
            )

    @torch.no_grad()
    def test_std_within_bounds(
        self, distribution_results: dict[int, list[dict]]
    ) -> None:
        """Per-dimension std should not exceed threshold."""
        violations = []
        for seq_len, seq_results in distribution_results.items():
            for r in seq_results:
                for s in r["dim_stats"]:
                    if s["std"] > _MAX_STD:
                        violations.append(
                            f"[seq_len={seq_len}] {r['label']} dim={s['dim_idx']}: "
                            f"std={s['std']:.4f} > {_MAX_STD}"
                        )
        if violations:
            pytest.fail(
                f"{len(violations)} std violation(s):\n  "
                + "\n  ".join(violations)
            )

    @torch.no_grad()
    def test_no_extreme_values(
        self, distribution_results: dict[int, list[dict]]
    ) -> None:
        """No value should exceed the absolute value threshold."""
        violations = []
        for seq_len, seq_results in distribution_results.items():
            for r in seq_results:
                for s in r["dim_stats"]:
                    abs_max = max(abs(s["min"]), abs(s["max"]))
                    if abs_max > _MAX_ABS_VALUE:
                        violations.append(
                            f"[seq_len={seq_len}] {r['label']} dim={s['dim_idx']}: "
                            f"|max_value|={abs_max:.4f} > {_MAX_ABS_VALUE}"
                        )
        if violations:
            pytest.fail(
                f"{len(violations)} extreme value violation(s):\n  "
                + "\n  ".join(violations)
            )

    @torch.no_grad()
    def test_distributions_stable_across_seq_lens(
        self, distribution_results: dict[int, list[dict]]
    ) -> None:
        """Mean and std should not drift dramatically across sequence lengths.

        Compares each seq_len to the shortest one. If mean shifts by more
        than 2.0 or std ratio exceeds 3x, flag it.
        """
        seq_lens_sorted = sorted(distribution_results.keys())
        if len(seq_lens_sorted) < 2:
            pytest.skip("Need at least 2 sequence lengths to compare stability")

        baseline_len = seq_lens_sorted[0]
        baseline = distribution_results[baseline_len]
        drift_issues = []

        for seq_len in seq_lens_sorted[1:]:
            current = distribution_results[seq_len]
            for r_base, r_cur in zip(baseline, current):
                for s_base, s_cur in zip(r_base["dim_stats"], r_cur["dim_stats"]):
                    d = s_base["dim_idx"]
                    label = r_base["label"]

                    mean_shift = abs(s_cur["mean"] - s_base["mean"])
                    if mean_shift > 2.0:
                        drift_issues.append(
                            f"{label} dim={d}: mean shifted by {mean_shift:.4f} "
                            f"between seq_len={baseline_len} and {seq_len}"
                        )

                    if s_base["std"] > 1e-6:
                        std_ratio = s_cur["std"] / s_base["std"]
                        if std_ratio > 3.0 or std_ratio < 1.0 / 3.0:
                            drift_issues.append(
                                f"{label} dim={d}: std ratio={std_ratio:.4f} "
                                f"between seq_len={baseline_len} and {seq_len}"
                            )

        if drift_issues:
            pytest.fail(
                f"{len(drift_issues)} stability issue(s):\n  "
                + "\n  ".join(drift_issues)
            )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distribution analysis for stream components."
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=_DEFAULT_SEQ_LENS,
        help=f"Sequence lengths to test (default: {_DEFAULT_SEQ_LENS}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to byte260 data directory (default: auto-detect).",
    )
    parser.add_argument(
        "-c", "--component",
        type=str,
        nargs="+",
        default=None,
        dest="components",
        help="Only run components whose name contains one of the given strings "
        "(case-insensitive substring match).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print per-dimension breakdown for each component.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run violation checks and exit with error code if any fail.",
    )
    parser.add_argument(
        "--max-abs-mean",
        type=float,
        default=_MAX_ABS_MEAN,
        help=f"Max allowed |mean| per dim (default: {_MAX_ABS_MEAN}).",
    )
    parser.add_argument(
        "--max-std",
        type=float,
        default=_MAX_STD,
        help=f"Max allowed std per dim (default: {_MAX_STD}).",
    )
    parser.add_argument(
        "--max-abs-value",
        type=float,
        default=_MAX_ABS_VALUE,
        help=f"Max allowed |value| (default: {_MAX_ABS_VALUE}).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else None
    results = run_distribution_analysis(
        seq_lens=args.seq_lens,
        batch_size=args.batch_size,
        data_dir=data_dir,
        components=args.components,
    )

    print_distribution_report(results, verbose=args.verbose)

    if args.check:
        violations = find_violations(
            results,
            max_abs_mean=args.max_abs_mean,
            max_std=args.max_std,
            max_abs_value=args.max_abs_value,
        )
        if violations:
            print(f"\n{len(violations)} VIOLATION(S):")
            for v in violations:
                print(f"  {v}")
            sys.exit(1)
        else:
            print("All checks passed.")


if __name__ == "__main__":
    main()
