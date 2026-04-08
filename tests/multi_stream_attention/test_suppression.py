"""Test: suppression / no-repeat — find the missing item from a set.

Synthetic task variations:
    1. "Alphabet: a-p. Seen: h,c,m,a,f,p,b,k,d,j. First missing: " → "e"
    2. "Alphabet: a-p. Seen: h,c,m,a,f,p,b,k,d,j. Last missing: "  → "o"
    3. "Sequence: 5,2,8,1,6,3. Missing from 1-9: "                  → "4"

The model must:
1. Track which symbols have been "seen" (set membership / suppression)
2. Identify gaps relative to a reference set
3. Handle variable-length seen lists with distractors

Key anti-reward-hacking measures:
- Large alphabet (16 letters or 1-digit numbers) → too many combos to memorize
- Variable N (seen count) → no fixed positional pattern
- Distractors mixed in → must distinguish signal from noise
- Multiple question types (first/last missing) → can't hardcode one strategy
- Shuffled seen order → no positional shortcut
- Uniform answer distribution via target-first sampling

Compares multi-stream attention (with context stream) vs TinyGPT baseline.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import random
import torch
from torch import Tensor

from efficient_byte_tokenizer import EfficientByteTokenizer
from byte_stream_components import ByteHashComponent, HashBoundary
from multi_streams import StreamType, StreamID, StreamDef
from test_harness import (
    TinyGPT,
    MultiStreamTestModel,
    count_params,
    train_model,
    evaluate_autoregressive,
    show_examples,
    verify_causality,
)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

ALPHABET = "abcdefghijklmnop"  # 16 letters
MIN_N, MAX_N = 6, 13  # seen count range (leaves 3-10 missing)
QUERY_TYPES = ("first", "last")

# Distractor pools — items that look like alphabet items but aren't
_DISTRACTORS_ALPHA = list("qrstuvwxyz")  # letters outside our alphabet
_DISTRACTORS_NUM = [str(i) for i in range(10)]  # digits


def make_sample(
    fixed_n: int | None = None,
    fixed_query: str | None = None,
) -> tuple[str, str]:
    """Generate one suppression-task sample with uniform answer distribution.

    Picks target first (uniform over ALPHABET), then constructs seen set so
    target is the first (or last) missing letter.

    Args:
        fixed_n: fix number of seen letters (for per-N eval).
        fixed_query: fix query type ("first" or "last") for per-type eval.

    Returns: (prompt, answer)
    """
    query = fixed_query or random.choice(QUERY_TYPES)

    # Pick target uniformly
    target_idx = random.randint(0, len(ALPHABET) - 1)
    answer = ALPHABET[target_idx]

    if query == "first":
        # All letters before target must be seen (so target is first missing)
        required = set(ALPHABET[:target_idx])
        forbidden = set()  # target itself is excluded
    else:  # "last"
        # All letters after target must be seen (so target is last missing)
        required = set(ALPHABET[target_idx + 1 :])
        forbidden = set()

    # Available to optionally include (everything except target and required)
    optional = set(ALPHABET) - required - {answer}

    # Determine N
    min_n = max(MIN_N, len(required))
    max_n = min(MAX_N, len(required) + len(optional))
    if min_n > max_n:
        return make_sample(fixed_n=fixed_n, fixed_query=fixed_query)

    if fixed_n is not None:
        if fixed_n < min_n or fixed_n > max_n:
            return make_sample(fixed_n=fixed_n, fixed_query=fixed_query)
        n = fixed_n
    else:
        n = random.randint(min_n, max_n)

    # Build seen set
    n_extra = n - len(required)
    extra = random.sample(sorted(optional), min(n_extra, len(optional)))
    seen = list(required) + extra
    random.shuffle(seen)

    # Add 0-3 distractors mixed into the seen list
    n_distractors = random.randint(0, 3)
    if n_distractors > 0:
        pool = _DISTRACTORS_ALPHA + _DISTRACTORS_NUM
        distractors = random.sample(pool, min(n_distractors, len(pool)))
        # Insert distractors at random positions
        for d in distractors:
            pos = random.randint(0, len(seen))
            seen.insert(pos, d)

    seen_str = ",".join(seen)
    alpha_range = f"{ALPHABET[0]}-{ALPHABET[-1]}"

    if query == "first":
        prompt = f"Alphabet: {alpha_range}. Seen: {seen_str}. First missing: "
    else:
        prompt = f"Alphabet: {alpha_range}. Seen: {seen_str}. Last missing: "

    return prompt, answer


def make_batch(tok: EfficientByteTokenizer, batch_size: int) -> tuple[Tensor, Tensor]:
    """Build a padded batch with supervision only on the answer character."""
    all_ids = []
    all_targets = []

    for _ in range(batch_size):
        prompt, answer = make_sample()

        prompt_ids = list(tok.encode(prompt))
        answer_ids = list(tok.encode(answer))
        full_ids = prompt_ids + answer_ids

        # Shifted targets: logits[i] predicts token at i+1
        targets = [-100] * (len(prompt_ids) - 1) + answer_ids + [-100]

        all_ids.append(full_ids)
        all_targets.append(targets)

    max_len = max(len(ids) for ids in all_ids)
    pad_id = tok.pad_id

    padded_ids = []
    padded_targets = []
    for ids, tgts in zip(all_ids, all_targets):
        pad_len = max_len - len(ids)
        padded_ids.append(ids + [pad_id] * pad_len)
        padded_targets.append(tgts + [-100] * pad_len)

    return (
        torch.tensor(padded_ids, dtype=torch.long),
        torch.tensor(padded_targets, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Stream definitions
# ---------------------------------------------------------------------------


def make_stream_defs(tok: EfficientByteTokenizer) -> list[StreamDef]:
    """Stream defs including a writable context stream for accumulating state."""
    return [
        StreamDef(name=StreamID(StreamType.LOGIT), dim=tok.vocab_size),
        StreamDef(name=StreamID(StreamType.CONTEXT), dim=48, auto_zeros=True),
        StreamDef(name=StreamID(StreamType.TOKENS), read_only=True, auto_onehot=True),
        StreamDef(
            name=StreamID(StreamType.STRUCTURAL),
            read_only=True,
            components=[
                ByteHashComponent(
                    tok,
                    window=3,
                    num_hashes=2,
                    boundary=HashBoundary.WORD,
                    track_hits=True,
                ),
                ByteHashComponent(
                    tok,
                    window=6,
                    num_hashes=2,
                    boundary=None,
                    track_hits=True,
                ),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Per-N / per-query evaluation helpers
# ---------------------------------------------------------------------------


def evaluate_per_n(
    model: torch.nn.Module,
    tok: EfficientByteTokenizer,
    device: str,
    n_samples_per: int = 200,
) -> dict[int, float]:
    """Evaluate accuracy broken down by number of seen letters."""
    results = {}
    for n in range(MIN_N, MAX_N + 1):

        def make_sample_fixed(n=n):
            return make_sample(fixed_n=n)

        acc = evaluate_autoregressive(
            model, make_sample_fixed, tok, n_samples=n_samples_per, device=device
        )
        results[n] = acc
    return results


def evaluate_per_query(
    model: torch.nn.Module,
    tok: EfficientByteTokenizer,
    device: str,
    n_samples_per: int = 200,
) -> dict[str, float]:
    """Evaluate accuracy broken down by query type."""
    results = {}
    for q in QUERY_TYPES:

        def make_sample_fixed(q=q):
            return make_sample(fixed_query=q)

        acc = evaluate_autoregressive(
            model, make_sample_fixed, tok, n_samples=n_samples_per, device=device
        )
        results[q] = acc
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = EfficientByteTokenizer()
    print(f"Device: {device}, vocab_size: {tok.vocab_size}")
    print(f"Alphabet: {ALPHABET!r} ({len(ALPHABET)} letters)")
    print(f"N (seen letters): {MIN_N}-{MAX_N}")
    print(f"Query types: {QUERY_TYPES}")
    print(f"Distractors: 0-3 per sample\n")

    # --- Stream definitions ---
    stream_defs = make_stream_defs(tok)

    # --- Build models ---
    ms_model = MultiStreamTestModel(
        stream_defs=stream_defs,
        vocab_size=tok.vocab_size,
        num_heads=2,
        head_dim=16,
        num_layers=2,
        # use_block=True,
        # mlp_hidden_dim=32,
    )

    gpt_model = TinyGPT(
        vocab_size=tok.vocab_size,
        dim=128,
        num_layers=2,
        num_heads=4,
        mlp_mult=2,
    )

    print("=" * 60)
    print("Model sizes")
    print("=" * 60)
    ms_params = count_params(ms_model, "Multi-Stream")
    gpt_params = count_params(gpt_model, "TinyGPT")
    print(f"  Ratio: {gpt_params / ms_params:.2f}x\n")

    train_kwargs = dict(
        tok=tok, steps=2000, batch_size=64, lr=3e-2, eval_every=500, device=device
    )

    models = [
        ("Multi-Stream (2 heads, 2 blocks, context)", ms_model),
        ("TinyGPT (dim=128, 2 layers, 4 heads)", gpt_model),
    ]

    results = {}
    for name, model in models:
        print("=" * 60)
        print(name)
        print("=" * 60)

        def eval_fn(m, d):
            return evaluate_autoregressive(m, make_sample, tok, n_samples=200, device=d)

        train_model(model, make_batch, **train_kwargs, eval_fn=eval_fn)

        # Overall accuracy
        overall_acc = evaluate_autoregressive(
            model, make_sample, tok, n_samples=500, device=device
        )
        print(f"\n  Overall accuracy: {overall_acc:.1%}")

        # Per-query accuracy
        per_q = evaluate_per_query(model, tok, device, n_samples_per=200)
        print("  Per-query accuracy:")
        for q, acc in per_q.items():
            print(f"    {q}: {acc:.1%}")

        # Per-N accuracy (sample a few)
        per_n = evaluate_per_n(model, tok, device, n_samples_per=150)
        print("  Per-N accuracy:")
        for n_seen, acc in sorted(per_n.items()):
            print(f"    N={n_seen}: {acc:.1%}")

        results[name] = {"overall": overall_acc, "per_q": per_q, "per_n": per_n}

        print("\n  Examples:")
        show_examples(model, make_sample, tok, device, n=6)

        verify_causality(model, tok, device, make_sample_fn=make_sample, label=name)
        print()

    # --- Summary ---
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for metric in ["overall"] + list(QUERY_TYPES):
        row = f"  {metric:>10s}"
        for name in results:
            if metric == "overall":
                val = results[name]["overall"]
            else:
                val = results[name]["per_q"][metric]
            row += f"  {val:>30.1%}"
        print(row)

    print(f"\n  Random baseline: {1 / len(ALPHABET):.1%}")
    print(f"  Multi-stream params: {ms_params:,}")
    print(f"  TinyGPT params:      {gpt_params:,}")
