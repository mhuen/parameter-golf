"""Test: multi-stream attention vs TinyGPT on periodic pattern completion.

Synthetic task:
    Given a repeating character pattern like "abcabcabc", predict the continuation.
    - Random base patterns of period 2-6 using lowercase letters
    - 3-6 full repeats shown as context
    - All positions after the first full period are supervised (every position
      is deterministically predictable once the pattern is established)

No word boundaries are used since the patterns are continuous character streams.
Multiple hash window sizes (3, 5, 8) cover different period lengths.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import random
import string

import torch
import torch.nn.functional as F
from torch import Tensor

from efficient_byte_tokenizer import EfficientByteTokenizer
from byte_stream_components import ByteHashComponent, HashBoundary
from multi_streams import StreamType, StreamID, StreamDef, StreamSource, SinCosPositionComponent
from test_harness import (
    TinyGPT,
    MultiStreamTestModel,
    MultiStreamGPTTestModel,
    count_params,
    parse_test_args,
    maybe_compile,
    train_model,
    evaluate_autoregressive,
    evaluate_packed,
    show_examples,
    verify_causality,
)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def _random_pattern(period: int) -> str:
    """Generate a random base pattern of the given length using lowercase letters."""
    return "".join(random.choices(string.ascii_lowercase, k=period))


def make_sample(
    min_period: int = 2,
    max_period: int = 6,
    min_repeats: int = 3,
    max_repeats: int = 6,
    fixed_period: int | None = None,
) -> tuple[str, str]:
    """Generate one pattern-completion sample.

    The prompt is cut at a random offset within the last period, so the model
    cannot just copy the last N characters — it must genuinely detect the period
    and figure out where it is in the cycle.

    Returns:
        (prompt_text, answer_str) where answer_str completes the current partial
        period plus one full additional period.
    """
    period = (
        fixed_period
        if fixed_period is not None
        else random.randint(min_period, max_period)
    )
    repeats = random.randint(min_repeats, max_repeats)
    base = _random_pattern(period)
    # Build enough repeats to have room for the prompt + answer
    full_seq = base * (repeats + 2)
    # Cut at a random offset within the last shown period (1 to period-1 chars into it)
    # This means the prompt does NOT end at a period boundary
    offset = random.randint(1, period - 1) if period > 1 else 0
    prompt_len = period * repeats + offset
    # Answer: rest of current partial period + one full period
    answer_len = (period - offset) + period
    prompt = full_seq[:prompt_len]
    answer = full_seq[prompt_len : prompt_len + answer_len]
    return prompt, answer


def make_batch(
    tok: EfficientByteTokenizer,
    batch_size: int,
    min_period: int = 2,
    max_period: int = 6,
    min_repeats: int = 3,
    max_repeats: int = 6,
    fixed_period: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Generate a padded training batch.

    For training, supervise ALL positions after the first `period` characters:
    targets[i] = input_ids[i+1] for i >= period-1, targets[i] = -100 otherwise.

    Returns:
        input_ids: (B, max_seq_len)
        targets:   (B, max_seq_len) with -100 for non-supervised positions
    """
    samples = []
    for _ in range(batch_size):
        period = (
            fixed_period
            if fixed_period is not None
            else random.randint(min_period, max_period)
        )
        repeats = random.randint(min_repeats, max_repeats)
        base = _random_pattern(period)
        full_seq = base * repeats
        samples.append((full_seq, period))

    encoded = [torch.from_numpy(tok.encode(seq)).long() for seq, _ in samples]
    max_len = max(e.numel() for e in encoded)

    input_ids = torch.full((batch_size, max_len), tok.pad_id, dtype=torch.long)
    targets = torch.full((batch_size, max_len), -100, dtype=torch.long)

    for i, (enc, (_, period)) in enumerate(zip(encoded, samples)):
        seq_len = enc.numel()
        input_ids[i, :seq_len] = enc
        # Supervise all positions after the first full period:
        # position j predicts token at j+1, valid for j >= period-1
        sup_start = period - 1
        sup_end = seq_len - 1
        if sup_start < sup_end:
            targets[i, sup_start:sup_end] = input_ids[i, sup_start + 1 : sup_end + 1]

    return input_ids, targets


# ---------------------------------------------------------------------------
# Convenience wrappers for the harness
# ---------------------------------------------------------------------------


def make_train_batch(
    tok: EfficientByteTokenizer, batch_size: int
) -> tuple[Tensor, Tensor]:
    """Training batch generator compatible with train_model's MakeBatchFn."""
    return make_batch(tok, batch_size)


def make_eval_sample(fixed_period: int | None = None):
    """Return a callable () -> (prompt, answer) for evaluate_autoregressive."""

    def _fn() -> tuple[str, str]:
        return make_sample(fixed_period=fixed_period)

    return _fn


# ---------------------------------------------------------------------------
# Stream definitions
# ---------------------------------------------------------------------------


def build_stream_defs(tok: EfficientByteTokenizer) -> list[StreamDef]:
    """Stream definitions for pattern completion.

    No word boundary since patterns are continuous character streams.
    Multiple window sizes (3, 5, 8) to cover different period lengths.
    """
    return [
        StreamDef(name=StreamID(StreamType.LOGIT), dim=tok.vocab_size),
        StreamDef(name=StreamID(StreamType.TOKENS), read_only=True, source=StreamSource.ONE_HOT),
        StreamDef(
            name=StreamID(StreamType.STRUCTURAL),
            read_only=True,
            source=StreamSource.COMPONENTS,
            components=[
                SinCosPositionComponent(num_freqs=32),  # 12d
                ByteHashComponent(
                    tok,
                    window=20,
                    num_hashes=2,
                    boundary=HashBoundary.WORD,
                    track_hits=True,
                ),
                ByteHashComponent(
                    tok, window=2, num_hashes=2, boundary=None, track_hits=True
                ),
                ByteHashComponent(
                    tok, window=3, num_hashes=2, boundary=None, track_hits=True
                ),
                ByteHashComponent(
                    tok, window=5, num_hashes=2, boundary=None, track_hits=True
                ),
                ByteHashComponent(
                    tok, window=8, num_hashes=2, boundary=None, track_hits=True
                ),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_test_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = EfficientByteTokenizer()
    print(f"Device: {device}, vocab_size: {tok.vocab_size}\n")

    stream_defs = build_stream_defs(tok)

    # --- Build models ---
    ms_model = MultiStreamTestModel(
        stream_defs=stream_defs,
        vocab_size=tok.vocab_size,
        num_heads=1,
        head_dim=16,
        num_layers=1,
        k_shift=True,
        # use_block=True,
        # mlp_hidden_dim=16,
    )
    gpt_model = TinyGPT(
        vocab_size=tok.vocab_size,
        dim=64,
        num_layers=1,
        num_heads=2,
    )

    msgpt_model = MultiStreamGPTTestModel(
        tok=tok,
        vocab_size=tok.vocab_size,
        make_batch_fn=make_batch,
        num_heads=1,
        num_kv_heads=1,
        num_layers=1,
        multi_head_dim=16,
        mlp_hidden_dim=16,
    )

    count_params(ms_model, "Multi-stream")
    count_params(gpt_model, "TinyGPT")
    count_params(msgpt_model, "MultiStreamGPT")

    train_kwargs = dict(
        tok=tok,
        steps=1000,
        batch_size=64,
        lr=3e-2,
        eval_every=200,
        device=device,
        pack_documents=args.pack,
    )

    if args.pack:
        print("*** Document packing enabled (2 docs/seq, BOS-separated) ***\n")

    results = {}

    for name, model in [
        ("Multi-stream attention", ms_model),
        ("TinyGPT", gpt_model),
        ("MultiStreamGPT", msgpt_model),
    ]:
        print("=" * 60)
        print(f"Training: {name}")
        print("=" * 60)
        model = maybe_compile(model, args.compile)

        if args.pack:

            def eval_fn(m, d):
                return evaluate_packed(
                    m,
                    make_train_batch,
                    tok,
                    n_samples=200,
                    device=d,
                )
        else:

            def eval_fn(m, d):
                return evaluate_autoregressive(
                    m,
                    make_eval_sample(),
                    tok,
                    n_samples=200,
                    device=d,
                )

        model = train_model(
            model,
            make_batch_fn=make_train_batch,
            eval_fn=eval_fn,
            **train_kwargs,
        )

        # --- Overall evaluation ---
        overall_acc = evaluate_autoregressive(
            model,
            make_eval_sample(),
            tok,
            n_samples=500,
            device=device,
        )
        print(f"\n  Single-doc accuracy: {overall_acc:.1%}")
        if args.pack:
            packed_acc = evaluate_packed(
                model,
                make_train_batch,
                tok,
                n_samples=500,
                device=device,
            )
            print(f"  Packed accuracy:     {packed_acc:.1%}")

        # --- Per-period evaluation ---
        print("\n  Per-period accuracy:")
        period_accs = {}
        for p in range(2, 7):
            acc = evaluate_autoregressive(
                model,
                make_eval_sample(fixed_period=p),
                tok,
                n_samples=200,
                device=device,
            )
            period_accs[p] = acc
            print(f"    period={p}: {acc:.1%}")
        results[name] = period_accs

        # --- Show examples ---
        print("\n  Examples:")
        show_examples(model, make_eval_sample(), tok, device=device, n=5)

        verify_causality(
            model, tok, device, make_sample_fn=make_eval_sample(), label=name
        )
        print()

    # --- Summary comparison ---
    print("=" * 60)
    print("Summary: per-period accuracy")
    print("=" * 60)
    header = f"  {'Period':>8s}"
    for name in results:
        header += f"  {name:>28s}"
    print(header)
    for p in range(2, 7):
        row = f"  {p:>8d}"
        for name in results:
            row += f"  {results[name][p]:>28.1%}"
        print(row)
