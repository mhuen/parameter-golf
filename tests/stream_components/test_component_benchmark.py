"""Benchmark and regression test for all structural stream components.

Loads a real byte260 data shard, runs each component from
``build_multi_stream_components``, measures wall-clock time, and optionally
saves / verifies reference output tensors for regression detection.

Usage:
    # Run benchmark + regression check (default):
    python tests/stream_components/test_component_benchmark.py

    # Save reference vectors (run once to establish baseline):
    python tests/stream_components/test_component_benchmark.py --save

    # Customize sequence length or number of repeats:
    python tests/stream_components/test_component_benchmark.py --seq-len 2048 --repeats 5

    # Run on GPU with torch.compile:
    python tests/stream_components/test_component_benchmark.py --device cuda --compile

    # As pytest (regression checks only, skips if no saved refs):
    pytest tests/stream_components/test_component_benchmark.py -v
"""

from __future__ import annotations

import argparse
import os
import sys
import time
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

_REF_DIR = Path(__file__).parent / "reference_vectors"

_DEFAULT_DATA_DIR = Path(_PROJECT_ROOT) / "data" / "datasets" / "fineweb10B_byte260"

_DEFAULT_SEQ_LEN = 128
_DEFAULT_BATCH_SIZE = 32
_DEFAULT_REPEATS = 5


# ---------------------------------------------------------------------------
# Component registry: (label, constructor_callable)
# ---------------------------------------------------------------------------


def _build_component_registry(
    tok: EfficientByteTokenizer,
) -> list[tuple[str, torch.nn.Module]]:
    """Return (label, component) pairs mirroring build_multi_stream_components."""
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
# Helpers
# ---------------------------------------------------------------------------


def _load_input_ids(
    seq_len: int,
    batch_size: int = 1,
    data_dir: Path | None = None,
) -> torch.Tensor:
    """Load the first shard and return a (batch_size, seq_len) input_ids tensor."""
    data_dir = data_dir or Path(
        os.environ.get("DATA_PATH", str(_DEFAULT_DATA_DIR))
    )
    shards = sorted(data_dir.glob("fineweb_val_*.bin"))
    if not shards:
        shards = sorted(data_dir.glob("fineweb_train_*.bin"))
    if not shards:
        raise FileNotFoundError(
            f"No .bin shards found in {data_dir}. "
            "Set DATA_PATH or pass --data-dir."
        )
    tok = EfficientByteTokenizer()
    tokens = load_shard_byte260(shards[0], tok)
    tokens = torch.from_numpy(tokens) if not isinstance(tokens, torch.Tensor) else tokens
    total_needed = seq_len * batch_size
    tokens = tokens.long()[:total_needed]
    return tokens.reshape(batch_size, seq_len)


def _ref_path(label: str) -> Path:
    return _REF_DIR / f"{label}.pt"


def _save_reference(label: str, tensor: torch.Tensor) -> None:
    _REF_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.cpu(), _ref_path(label))


def _load_reference(label: str) -> torch.Tensor | None:
    p = _ref_path(label)
    if p.exists():
        return torch.load(p, map_location="cpu", weights_only=True)
    return None


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(
    seq_len: int = _DEFAULT_SEQ_LEN,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    repeats: int = _DEFAULT_REPEATS,
    save: bool = False,
    data_dir: Path | None = None,
    device: str = "cpu",
    compile: bool = False,
    compile_warmup: int = 10,
) -> list[dict]:
    """Run all components and return timing + regression results."""
    input_ids = _load_input_ids(seq_len, batch_size=batch_size, data_dir=data_dir)
    input_ids = input_ids.to(device)
    tok = EfficientByteTokenizer()
    registry = _build_component_registry(tok)
    dtype = torch.float32
    use_cuda_sync = device != "cpu" and torch.cuda.is_available()

    results: list[dict] = []

    for label, component in registry:
        component = component.to(device)
        if compile:
            component = torch.compile(component, fullgraph=True)

        # -- warm-up: run until compiled execution stabilises --
        # torch.compile traces lazily and may retrace on early calls.
        # We run iterations until the last two are within 2× of each other
        # (or until max_warmup is hit) so timed runs never include compilation.
        max_warmup = compile_warmup if compile else 1
        prev_t = None
        for wi in range(max_warmup):
            if use_cuda_sync:
                torch.cuda.synchronize()
            w0 = time.perf_counter()
            with torch.no_grad():
                component(input_ids, dtype=dtype)
            if use_cuda_sync:
                torch.cuda.synchronize()
            cur_t = time.perf_counter() - w0
            # After at least 2 iters, check if times have converged
            if compile and wi >= 1 and prev_t is not None:
                ratio = max(cur_t, prev_t) / max(min(cur_t, prev_t), 1e-9)
                if ratio < 2.0:
                    break
            prev_t = cur_t

        # -- timed runs --
        times = []
        output = None
        for _ in range(repeats):
            if use_cuda_sync:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                output = component(input_ids, dtype=dtype)
            if use_cuda_sync:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        assert output is not None
        mean_ms = sum(times) / len(times) * 1000
        min_ms = min(times) * 1000

        entry: dict = {
            "label": label,
            "dim": component.dim,
            "shape": tuple(output.shape),
            "mean_ms": mean_ms,
            "min_ms": min_ms,
        }

        # -- save / verify reference (first batch item only) --
        ref_output = output[0:1].cpu()
        if save:
            _save_reference(label, ref_output)
            entry["saved"] = True
        else:
            ref = _load_reference(label)
            if ref is not None:
                # Compare overlapping prefix so seq_len needn't match saved refs
                T = min(ref_output.shape[1], ref.shape[1])
                ref_output_cmp = ref_output[:, :T]
                ref_cmp = ref[:, :T]
                entry["ref_tokens"] = T
                entry["total_tokens"] = ref_output.shape[1]
                match = torch.equal(ref_output_cmp, ref_cmp)
                if not match:
                    max_diff = (ref_output_cmp - ref_cmp).abs().max().item()
                    entry["regression"] = True
                    entry["max_diff"] = max_diff
                else:
                    entry["regression"] = False
            else:
                entry["regression"] = None  # no reference available

        results.append(entry)

    return results


def print_results(
    results: list[dict],
    save: bool = False,
    batch_size: int = 1,
    seq_len: int = 0,
    device: str = "cpu",
    compile: bool = False,
) -> None:
    """Pretty-print benchmark results."""
    print()
    print(f"Input shape: ({batch_size}, {seq_len})  device: {device}  compile: {compile}")
    print()
    print(f"{'Component':<25} {'Dim':>5} {'Mean(ms)':>10} {'Min(ms)':>10} ", end="")
    if save:
        print("Saved")
    else:
        print("Regression")
    print("-" * 72)

    for r in results:
        print(
            f"{r['label']:<25} {r['dim']:>5} {r['mean_ms']:>10.2f} {r['min_ms']:>10.2f} ",
            end="",
        )
        if save:
            print("yes" if r.get("saved") else "")
        else:
            reg = r.get("regression")
            if reg is None:
                print("(no ref)")
            elif reg:
                print(f"FAIL (max_diff={r['max_diff']:.6e})")
            else:
                print("ok")

    # Warn if regression checked fewer tokens than the full sequence
    if not save:
        partial = [
            r for r in results
            if r.get("ref_tokens") is not None
            and r["ref_tokens"] < r["total_tokens"]
        ]
        if partial:
            ref_t = partial[0]["ref_tokens"]
            total_t = partial[0]["total_tokens"]
            print(
                f"  Note: references cover {ref_t} of {total_t} tokens. "
                f"Re-run with --save --seq-len {total_t} to verify the full sequence."
            )

    print()


# ---------------------------------------------------------------------------
# pytest integration
# ---------------------------------------------------------------------------

import pytest  # noqa: E402


@pytest.fixture(scope="module")
def benchmark_results() -> list[dict]:
    """Run the benchmark once for all pytest tests (small batch for CI speed)."""
    return run_benchmark(seq_len=_DEFAULT_SEQ_LEN, batch_size=1, repeats=1, save=False)


class TestComponentRegression:
    """Verify each component output matches saved reference vectors."""

    @torch.no_grad()
    def test_all_components_match_reference(self, benchmark_results: list[dict]) -> None:
        """Each component with a saved reference must match exactly."""
        any_checked = False
        failures = []
        for r in benchmark_results:
            reg = r.get("regression")
            if reg is None:
                continue  # no reference saved yet
            any_checked = True
            if reg:
                failures.append(
                    f"{r['label']}: max_diff={r.get('max_diff', '?')}"
                )

        if not any_checked:
            pytest.skip(
                "No reference vectors found. "
                "Run with --save to create them first."
            )

        assert not failures, (
            "Component regression(s) detected:\n  " + "\n  ".join(failures)
        )

    @torch.no_grad()
    def test_component_dims_match_output(self, benchmark_results: list[dict]) -> None:
        """Each component's .dim must match its actual output last dimension."""
        for r in benchmark_results:
            assert r["shape"][-1] == r["dim"], (
                f"{r['label']}: .dim={r['dim']} but output shape={r['shape']}"
            )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark and regression-test stream components."
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save current outputs as reference vectors.",
    )
    parser.add_argument(
        "--seq-len", type=int, default=_DEFAULT_SEQ_LEN,
        help=f"Sequence length to benchmark (default: {_DEFAULT_SEQ_LEN}).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=_DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {_DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--repeats", type=int, default=_DEFAULT_REPEATS,
        help=f"Number of timed repeats per component (default: {_DEFAULT_REPEATS}).",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to byte260 data directory (default: auto-detect).",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to run on, e.g. 'cpu', 'cuda', 'cuda:0' (default: cpu).",
    )
    parser.add_argument(
        "--compile", action="store_true",
        help="Wrap each component with torch.compile(fullgraph=True).",
    )
    parser.add_argument(
        "--compile-warmup", type=int, default=10,
        help="Max warm-up iterations when --compile is set (default: 10). "
        "Exits early once consecutive times converge.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else None
    results = run_benchmark(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        repeats=args.repeats,
        save=args.save,
        data_dir=data_dir,
        device=args.device,
        compile=args.compile,
        compile_warmup=args.compile_warmup,
    )
    print_results(
        results, save=args.save, batch_size=args.batch_size, seq_len=args.seq_len,
        device=args.device, compile=args.compile,
    )

    # Exit with error if any regressions
    if not args.save:
        regressions = [r for r in results if r.get("regression") is True]
        if regressions:
            sys.exit(1)


if __name__ == "__main__":
    main()
