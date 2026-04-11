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
from modules import StreamComponent  # noqa: E402
from multi_streams import DocBoundaryComponent, SinCosPositionComponent  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_DATA_DIR = Path(_PROJECT_ROOT) / "data" / "datasets" / "fineweb10B_byte260"

_DEFAULT_SEQ_LENS = [1024, 2048, 4096, 16_384, 32_768]
_CALIB_SEQ_LEN = 4096

# Thresholds for pytest assertions (raw / uncalibrated)
_MAX_ABS_MEAN = 5.0  # per-dim mean should stay moderate
_MAX_STD = 10.0  # per-dim std should stay moderate
_MAX_ABS_VALUE = 500.0  # no extreme outlier values

# Two-tiered thresholds after calibration:
# Standardizable dims (normalized to mean≈0, std≈1)
_CALIB_STD_MAX_ABS_MEAN = 0.15
_CALIB_STD_MIN_STD = 0.5
_CALIB_STD_MAX_STD = 2.0
# Non-standardizable dims (rotation/sincos, bounded to unit circle)
_CALIB_NONSTD_MAX_ABS_MEAN = 1.0
_CALIB_NONSTD_MAX_STD = 1.0

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
        ("ByteCategory", ByteCategoryComponent(tok=tok)),
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
                tok,
                window=20,
                num_hashes=2,
                boundary=HashBoundary.WORD,
                track_hits=True,
            ),
        ),
        (
            "ByteHash_w2",
            ByteHashComponent(
                tok,
                window=2,
                num_hashes=2,
                boundary=None,
                track_hits=True,
            ),
        ),
        (
            "ByteHash_w3",
            ByteHashComponent(
                tok,
                window=3,
                num_hashes=2,
                boundary=None,
                track_hits=True,
            ),
        ),
        (
            "ByteHash_w5",
            ByteHashComponent(
                tok,
                window=5,
                num_hashes=2,
                boundary=None,
                track_hits=True,
            ),
        ),
        (
            "ByteHash_w8",
            ByteHashComponent(
                tok,
                window=8,
                num_hashes=2,
                boundary=None,
                track_hits=True,
            ),
        ),
        (
            "ByteHash_digit_w8",
            ByteHashComponent(
                tok,
                window=8,
                num_hashes=2,
                boundary=HashBoundary.DIGIT,
                track_hits=True,
            ),
        ),
        (
            "Boundary",
            BoundaryComponent(
                tok,
                word_pos_freqs=3,
                word_id_freqs=3,
                sent_pos_freqs=2,
                sent_id_freqs=2,
                para_pos_freqs=2,
                para_id_freqs=2,
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
        raise ValueError(
            f"Not enough tokens in {data_dir} to fill batch_size={batch_size}, seq_len={seq_len} (got {tokens.numel()})"
        )
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
        stats.append(
            {
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
            }
        )
    return stats


# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------


def run_distribution_analysis(
    seq_lens: list[int],
    batch_size: int = 1,
    data_dir: Path | None = None,
    components: list[str] | None = None,
    calibrate: bool = False,
    calib_seq_len: int = 4096,
    calib_batches: int = 10,
    calib_batch_size: int = 16,
) -> dict[int, list[dict]]:
    """Run all components at each sequence length.

    Args:
        calibrate: if True, calibrate each StreamComponent before running
            the analysis.  After calibration, ``component(...)`` returns
            standardized output (mean≈0, std≈1 on standardizable dims).
        calib_seq_len: sequence length for calibration data.
        calib_batches: number of batches to use for calibration.
        calib_batch_size: batch size for calibration data.

    Returns:
        {seq_len: [{"label": str, "dim": int, "dim_stats": [...],
                     "standardizable_mask": tuple[bool, ...]}, ...]}
    """
    tok = EfficientByteTokenizer()
    registry = _build_component_registry(tok)

    if components:
        lc_filters = [f.lower() for f in components]
        registry = [
            (label, comp)
            for label, comp in registry
            if any(f in label.lower() for f in lc_filters)
        ]
        if not registry:
            raise ValueError(f"No components matched filters: {components}")

    # --- Optional calibration pass ---
    if calibrate:
        calib_ids = _load_input_ids(
            calib_seq_len,
            batch_size=calib_batches * calib_batch_size,
            data_dir=data_dir,
        )
        calib_id_batches = list(calib_ids.split(calib_batch_size))
        n_calibrated = 0
        for label, component in registry:
            if isinstance(component, StreamComponent):
                component.calibrate(calib_id_batches)
                n_std = sum(int(m) for m in component.standardizable_mask)
                if n_std > 0:
                    n_calibrated += 1
        print(
            f"calibration: {n_calibrated} components calibrated "
            f"({len(calib_id_batches)} batches × {calib_batch_size} × {calib_seq_len})"
        )
        del calib_ids, calib_id_batches

    results_by_seqlen: dict[int, list[dict]] = {}

    for seq_len in seq_lens:
        input_ids = _load_input_ids(seq_len, batch_size=batch_size, data_dir=data_dir)
        seq_results = []

        for label, component in registry:
            mask = ()
            if isinstance(component, StreamComponent):
                mask = component.standardizable_mask
            with torch.no_grad():
                output = component(input_ids, dtype=torch.float32)
            dim_stats = compute_dim_stats(output)
            seq_results.append(
                {
                    "label": label,
                    "dim": component.dim,
                    "shape": tuple(output.shape),
                    "dim_stats": dim_stats,
                    "standardizable_mask": mask,
                }
            )

        results_by_seqlen[seq_len] = seq_results

    return results_by_seqlen


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------


def _dim_violations(s: dict, is_std: bool, calibrated: bool) -> list[str]:
    """Return short violation tags for a single dim's stats."""
    tags = []
    mean, std = abs(s["mean"]), s["std"]
    abs_max = max(abs(s["min"]), abs(s["max"]))

    if calibrated and is_std:
        if mean > _CALIB_STD_MAX_ABS_MEAN:
            tags.append("mean")
        if std < _CALIB_STD_MIN_STD:
            tags.append("std<")
        elif std > _CALIB_STD_MAX_STD:
            tags.append("std>")
    elif calibrated:
        if mean > _CALIB_NONSTD_MAX_ABS_MEAN:
            tags.append("mean")
        if std > _CALIB_NONSTD_MAX_STD:
            tags.append("std>")
    else:
        if mean > _MAX_ABS_MEAN:
            tags.append("mean")
        if std > _MAX_STD:
            tags.append("std>")

    if abs_max > _MAX_ABS_VALUE:
        tags.append("max")
    return tags


def print_distribution_report(
    results_by_seqlen: dict[int, list[dict]],
    verbose: bool = False,
    calibrated: bool = False,
) -> None:
    """Print distribution tables.

    Default mode prints a per-component summary (aggregated across dims).
    Verbose mode additionally prints per-dimension breakdowns.
    Violations are marked with ``!`` and a short tag (mean/std</std>/max).
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
            mask = r.get("standardizable_mask", ())
            # Aggregate: mean-of-means, mean-of-stds, global min/max, mean quantiles
            agg_mean = sum(s["mean"] for s in stats) / len(stats)
            agg_std = sum(s["std"] for s in stats) / len(stats)
            agg_min = min(s["min"] for s in stats)
            agg_max = max(s["max"] for s in stats)
            agg_q01 = sum(s["q01"] for s in stats) / len(stats)
            agg_q50 = sum(s["q50"] for s in stats) / len(stats)
            agg_q99 = sum(s["q99"] for s in stats) / len(stats)
            # Count dims with any violation
            n_bad = sum(
                1
                for s in stats
                if _dim_violations(
                    s, s["dim_idx"] < len(mask) and mask[s["dim_idx"]], calibrated
                )
            )
            flag = f" [{n_bad}!]" if n_bad else ""
            print(
                f"{r['label']:<25} {r['dim']:>4}  "
                f"{agg_mean:>8.3f} {agg_std:>8.3f} {agg_min:>8.3f} {agg_max:>8.3f}  "
                f"{agg_q01:>8.3f} {agg_q50:>8.3f} {agg_q99:>8.3f}{flag}"
            )

        # -- Per-dimension breakdown (verbose) --
        if verbose:
            for r in seq_results:
                print()
                mask = r.get("standardizable_mask", ())
                n_std = sum(int(m) for m in mask)
                std_info = f", {n_std} standardizable" if mask else ""
                print(f"  {r['label']} ({r['dim']} dims{std_info}):")
                dim_header = (
                    f"    {'':>1} {'Dim':>4}  "
                    f"{'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}  "
                    f"{'Q01':>8} {'Q25':>8} {'Q50':>8} {'Q75':>8} {'Q99':>8}"
                )
                print(dim_header)
                print(f"    {'-' * (len(dim_header) - 4)}")
                for s in r["dim_stats"]:
                    d = s["dim_idx"]
                    is_std = d < len(mask) and mask[d]
                    marker = "*" if is_std else " "
                    violations = _dim_violations(s, is_std, calibrated)
                    suffix = f"  !{','.join(violations)}" if violations else ""
                    print(
                        f"    {marker} {d:>4}  "
                        f"{s['mean']:>8.3f} {s['std']:>8.3f} "
                        f"{s['min']:>8.3f} {s['max']:>8.3f}  "
                        f"{s['q01']:>8.3f} {s['q25']:>8.3f} "
                        f"{s['q50']:>8.3f} {s['q75']:>8.3f} {s['q99']:>8.3f}"
                        f"{suffix}"
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
                    violations.append(f"{prefix}: std={s['std']:.4f} > {max_std}")
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
from dataclasses import dataclass as _dataclass


@_dataclass
class _DistTestCase:
    """Bundle of results + mode for a single test run."""

    mode: str  # "raw" or "calibrated"
    results: dict[int, list[dict]]


@pytest.fixture(scope="module", params=["raw", "calibrated"])
def distribution_case(request) -> _DistTestCase:
    """Run distribution analysis — once raw, once with calibration.

    Calibrated mode evaluates only at the calibration seq_len (matching
    the training script which calibrates at ``args.train_seq_len``).
    """
    if request.param == "calibrated":
        return _DistTestCase(
            mode="calibrated",
            results=run_distribution_analysis(
                seq_lens=[_CALIB_SEQ_LEN],
                batch_size=128,
                calibrate=True,
                calib_seq_len=_CALIB_SEQ_LEN,
                calib_batches=10,
                calib_batch_size=16,
            ),
        )
    return _DistTestCase(
        mode="raw",
        results=run_distribution_analysis(
            seq_lens=_DEFAULT_SEQ_LENS,
            batch_size=1,
        ),
    )


class TestComponentDistributions:
    """Verify component output distributions stay within healthy ranges.

    Runs twice: once on raw (uncalibrated) output with loose bounds, once
    on calibrated output with tight bounds.  The calibrated run catches
    dims that were forgotten or mis-marked in ``standardizable_mask``.
    """

    @staticmethod
    def _is_standardizable(r: dict, dim_idx: int) -> bool:
        mask = r.get("standardizable_mask", ())
        return dim_idx < len(mask) and mask[dim_idx]

    @torch.no_grad()
    def test_no_nans_or_infs(self, distribution_case: _DistTestCase) -> None:
        """No dimension should have NaN or Inf in its statistics."""
        bad = []
        for seq_len, seq_results in distribution_case.results.items():
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
    def test_mean_within_bounds(self, distribution_case: _DistTestCase) -> None:
        """Per-dimension |mean| check.

        Raw mode: uniform loose threshold (_MAX_ABS_MEAN) at all seq_lens.
        Calibrated mode (two-tiered, only at calibration seq_len=4096):
          - standardizable dims: |mean| < _CALIB_STD_MAX_ABS_MEAN
          - non-standardizable:  |mean| <= _CALIB_NONSTD_MAX_ABS_MEAN

        Calibrated mode only has data at the calibration seq_len (the fixture
        ensures this), so bounds are checked where they're meaningful.
        """
        calibrated = distribution_case.mode == "calibrated"
        violations = []
        for seq_len, seq_results in distribution_case.results.items():
            for r in seq_results:
                for s in r["dim_stats"]:
                    d = s["dim_idx"]
                    if calibrated:
                        limit = (
                            _CALIB_STD_MAX_ABS_MEAN
                            if self._is_standardizable(r, d)
                            else _CALIB_NONSTD_MAX_ABS_MEAN
                        )
                    else:
                        limit = _MAX_ABS_MEAN
                    if abs(s["mean"]) > limit:
                        tag = "std" if self._is_standardizable(r, d) else "nonstd"
                        violations.append(
                            f"[seq_len={seq_len}] {r['label']} dim={d} ({tag}): "
                            f"|mean|={abs(s['mean']):.4f} > {limit}"
                        )
        if violations:
            pytest.fail(
                f"{len(violations)} mean violation(s):\n  " + "\n  ".join(violations)
            )

    @torch.no_grad()
    def test_std_within_bounds(self, distribution_case: _DistTestCase) -> None:
        """Per-dimension std check.

        Raw mode: uniform loose threshold (_MAX_STD) at all seq_lens.
        Calibrated mode (two-tiered, only at calibration seq_len=4096):
          - standardizable dims: std in [_CALIB_STD_MIN_STD, _CALIB_STD_MAX_STD]
          - non-standardizable:  std <= _CALIB_NONSTD_MAX_STD

        See test_mean_within_bounds for rationale.
        """
        calibrated = distribution_case.mode == "calibrated"
        violations = []
        for seq_len, seq_results in distribution_case.results.items():
            for r in seq_results:
                for s in r["dim_stats"]:
                    d = s["dim_idx"]
                    if calibrated:
                        if self._is_standardizable(r, d):
                            if s["std"] < _CALIB_STD_MIN_STD:
                                violations.append(
                                    f"[seq_len={seq_len}] {r['label']} dim={d} (std): "
                                    f"std={s['std']:.4f} < {_CALIB_STD_MIN_STD}"
                                )
                            elif s["std"] > _CALIB_STD_MAX_STD:
                                violations.append(
                                    f"[seq_len={seq_len}] {r['label']} dim={d} (std): "
                                    f"std={s['std']:.4f} > {_CALIB_STD_MAX_STD}"
                                )
                        else:
                            if s["std"] > _CALIB_NONSTD_MAX_STD:
                                violations.append(
                                    f"[seq_len={seq_len}] {r['label']} dim={d} (nonstd): "
                                    f"std={s['std']:.4f} > {_CALIB_NONSTD_MAX_STD}"
                                )
                    else:
                        if s["std"] > _MAX_STD:
                            violations.append(
                                f"[seq_len={seq_len}] {r['label']} dim={d}: "
                                f"std={s['std']:.4f} > {_MAX_STD}"
                            )
        if violations:
            pytest.fail(
                f"{len(violations)} std violation(s):\n  " + "\n  ".join(violations)
            )

    @torch.no_grad()
    def test_no_extreme_values(self, distribution_case: _DistTestCase) -> None:
        """No value should exceed the absolute value threshold."""
        violations = []
        for seq_len, seq_results in distribution_case.results.items():
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
        default=8,
        help="Batch size (default: 1).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to byte260 data directory (default: auto-detect).",
    )
    parser.add_argument(
        "-c",
        "--component",
        type=str,
        nargs="+",
        default=None,
        dest="components",
        help="Only run components whose name contains one of the given strings "
        "(case-insensitive substring match).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print per-dimension breakdown for each component.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Calibrate components before analysis (shows post-standardization "
        "distributions). Uses tighter default thresholds with --check.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run violation checks and exit with error code if any fail.",
    )
    parser.add_argument(
        "--max-abs-mean",
        type=float,
        default=None,
        help="Max allowed |mean| per dim (default depends on --calibrate).",
    )
    parser.add_argument(
        "--max-std",
        type=float,
        default=None,
        help="Max allowed std per dim (default depends on --calibrate).",
    )
    parser.add_argument(
        "--max-abs-value",
        type=float,
        default=None,
        help="Max allowed |value| (default depends on --calibrate).",
    )
    args = parser.parse_args()

    # Pick thresholds based on mode, allow explicit override.
    # CLI --check uses uniform thresholds (the two-tiered per-dim logic
    # is in the pytest tests).  For calibrated mode the CLI defaults to
    # the tighter standardizable-dim thresholds as a quick sanity check.
    if args.calibrate:
        max_abs_mean = (
            args.max_abs_mean
            if args.max_abs_mean is not None
            else _CALIB_STD_MAX_ABS_MEAN
        )
        max_std = args.max_std if args.max_std is not None else _CALIB_STD_MAX_STD
        max_abs_value = (
            args.max_abs_value if args.max_abs_value is not None else _MAX_ABS_VALUE
        )
    else:
        max_abs_mean = (
            args.max_abs_mean if args.max_abs_mean is not None else _MAX_ABS_MEAN
        )
        max_std = args.max_std if args.max_std is not None else _MAX_STD
        max_abs_value = (
            args.max_abs_value if args.max_abs_value is not None else _MAX_ABS_VALUE
        )

    data_dir = Path(args.data_dir) if args.data_dir else None
    results = run_distribution_analysis(
        seq_lens=args.seq_lens,
        batch_size=args.batch_size,
        data_dir=data_dir,
        components=args.components,
        calibrate=args.calibrate,
    )

    print_distribution_report(results, verbose=args.verbose, calibrated=args.calibrate)

    if args.check:
        violations = find_violations(
            results,
            max_abs_mean=max_abs_mean,
            max_std=max_std,
            max_abs_value=max_abs_value,
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
