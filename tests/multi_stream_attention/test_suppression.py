"""Test: suppression / no-repeat task -- can models find the first missing letter?

Synthetic task:
    "Seen: c,a,e. Next unused: " -> "b"

A random subset of N letters (N=2..4) is drawn from ALPHABET="abcdef" (6 options),
shuffled, and displayed. The model must predict the first alphabetically missing
letter from the full alphabet.

This tests the model's ability to:
1. Track which symbols have been "seen" (suppression / set membership)
2. Identify the first gap in an ordered set

Compares:
1. Multi-stream attention with context stream (writable scratch for accumulating
   seen-state) + structural hash features.
2. TinyGPT baseline sized to approximately match parameter count.
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import random
import string
import torch
import torch.nn.functional as F
from torch import Tensor

from efficient_byte_tokenizer import EfficientByteTokenizer
from byte_modules import ByteHashComponent, HashBoundary
from multi_streams import Stream, StreamDef
from test_harness import (
    TinyGPT,
    MultiStreamTestModel,
    count_params,
    train_model,
    evaluate_autoregressive,
    show_examples,
)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

ALPHABET = "abcdef"  # 6 letters, simple version


def make_sample(fixed_n: int | None = None) -> tuple[str, str]:
    """Generate one suppression-task sample.

    Args:
        fixed_n: if set, use exactly this many seen letters (for per-N eval).
            Otherwise, pick N uniformly from [2, 4].

    Returns: (prompt, answer)
        prompt: e.g. "Seen: c,a,e. Next unused: "
        answer: e.g. "b"
    """
    n = fixed_n if fixed_n is not None else random.randint(2, 4)
    seen = random.sample(list(ALPHABET), n)
    # Shuffle to make order unpredictable (harder than sorted)
    random.shuffle(seen)

    # First alphabetically missing letter
    seen_set = set(seen)
    answer = ""
    for ch in ALPHABET:
        if ch not in seen_set:
            answer = ch
            break

    prompt = f"Seen: {','.join(seen)}. Next unused: "
    return prompt, answer


def make_batch(
    tok: EfficientByteTokenizer, batch_size: int
) -> tuple[Tensor, Tensor]:
    """Build a padded batch with supervision only on the answer character.

    Returns: (input_ids, targets) both of shape (B, max_len)
        targets is -100 everywhere except the answer position.
    """
    all_ids = []
    all_targets = []

    for _ in range(batch_size):
        prompt, answer = make_sample()

        prompt_ids = list(tok.encode(prompt))
        answer_ids = list(tok.encode(answer))
        full_ids = prompt_ids + answer_ids

        # Shifted targets: logits[i] predicts token at i+1
        # So targets[prompt_len-1] = answer_ids[0], etc.
        targets = [-100] * (len(prompt_ids) - 1) + answer_ids + [-100]

        all_ids.append(full_ids)
        all_targets.append(targets)

    # Pad to max length in batch
    max_len = max(len(ids) for ids in all_ids)
    pad_id = tok.pad_id

    padded_ids = []
    padded_targets = []
    for ids, tgts in zip(all_ids, all_targets):
        pad_len = max_len - len(ids)
        padded_ids.append(ids + [pad_id] * pad_len)
        padded_targets.append(tgts + [-100] * pad_len)

    input_ids = torch.tensor(padded_ids, dtype=torch.long)
    targets = torch.tensor(padded_targets, dtype=torch.long)

    return input_ids, targets


# ---------------------------------------------------------------------------
# Stream definitions
# ---------------------------------------------------------------------------

def make_stream_defs(tok: EfficientByteTokenizer) -> list[StreamDef]:
    """Stream defs including a writable context stream for accumulating state."""
    return [
        StreamDef(name=Stream.LOGIT, dim=tok.vocab_size),
        StreamDef(name=Stream.CONTEXT, dim=48, auto_zeros=True),  # writable scratch
        StreamDef(name=Stream.TOKENS, read_only=True, auto_onehot=True),
        StreamDef(
            name=Stream.STRUCTURAL,
            read_only=True,
            components=[
                ByteHashComponent(
                    tok, window=3, num_hashes=2, boundary=HashBoundary.WORD, track_hits=True
                ),
                ByteHashComponent(
                    tok, window=6, num_hashes=2, boundary=None, track_hits=True
                ),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Per-N evaluation helper
# ---------------------------------------------------------------------------

def evaluate_per_n(
    model: torch.nn.Module,
    tok: EfficientByteTokenizer,
    device: str,
    n_samples_per: int = 200,
) -> dict[int, float]:
    """Evaluate accuracy broken down by number of seen letters."""
    results = {}
    for n in range(2, 5):  # N = 2, 3, 4

        def make_sample_fixed(n=n):
            return make_sample(fixed_n=n)

        acc = evaluate_autoregressive(
            model, make_sample_fixed, tok, n_samples=n_samples_per, device=device
        )
        results[n] = acc
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = EfficientByteTokenizer()
    print(f"Device: {device}, vocab_size: {tok.vocab_size}")
    print(f"Alphabet: {ALPHABET!r} ({len(ALPHABET)} letters)")
    print(f"N (seen letters): 2-4\n")

    # --- Stream definitions ---
    stream_defs = make_stream_defs(tok)

    # --- Build models ---
    ms_model = MultiStreamTestModel(
        stream_defs=stream_defs,
        vocab_size=tok.vocab_size,
        num_heads=2,
        head_dim=32,
        num_layers=2,
        use_block=True,
        mlp_hidden_dim=128,
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
    ms_params = count_params(ms_model, "Multi-Stream (2 heads, head_dim=32, 2 layers, context)")
    gpt_params = count_params(gpt_model, "TinyGPT (dim=128, 2 layers, 4 heads, mlp_mult=2)")
    print(f"  Ratio: {gpt_params / ms_params:.2f}x\n")

    train_kwargs = dict(
        tok=tok, steps=2000, batch_size=64, lr=3e-2, eval_every=400, device=device
    )

    models = [
        ("Multi-Stream Attention (2 heads, 2 layers, context stream)", ms_model),
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

        # Per-N accuracy
        per_n = evaluate_per_n(model, tok, device, n_samples_per=200)
        print("  Per-N accuracy:")
        for n_seen, acc in sorted(per_n.items()):
            print(f"    N={n_seen} (seen letters): {acc:.1%}")

        results[name] = {"overall": overall_acc, "per_n": per_n}

        # Show examples
        print("\n  Examples:")
        show_examples(model, make_sample, tok, device, n=5)
        print()

    # --- Summary comparison ---
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    header = f"  {'N':>4s}"
    for name in results:
        short = name[:35]
        header += f"  {short:>35s}"
    print(header)

    for n_seen in range(2, 5):
        row = f"  {n_seen:>4d}"
        for name in results:
            acc = results[name]["per_n"][n_seen]
            row += f"  {acc:>35.1%}"
        print(row)

    row = f"  {'all':>4s}"
    for name in results:
        acc = results[name]["overall"]
        row += f"  {acc:>35.1%}"
    print(row)

    # -----------------------------------------------------------------------
    # Harder variant: 10-letter alphabet, N from 3 to 7
    # Uncomment to run.
    # -----------------------------------------------------------------------
    #
    # ALPHABET_HARD = "abcdefghij"  # 10 letters
    #
    # def make_sample_hard(fixed_n: int | None = None) -> tuple[str, str]:
    #     n = fixed_n if fixed_n is not None else random.randint(3, 7)
    #     seen = random.sample(list(ALPHABET_HARD), n)
    #     random.shuffle(seen)
    #     seen_set = set(seen)
    #     answer = ""
    #     for ch in ALPHABET_HARD:
    #         if ch not in seen_set:
    #             answer = ch
    #             break
    #     prompt = f"Seen: {','.join(seen)}. Next unused: "
    #     return prompt, answer
    #
    # def make_batch_hard(
    #     tok: EfficientByteTokenizer, batch_size: int
    # ) -> tuple[Tensor, Tensor]:
    #     all_ids = []
    #     all_targets = []
    #     for _ in range(batch_size):
    #         prompt, answer = make_sample_hard()
    #         prompt_ids = tok.encode(prompt)
    #         answer_ids = tok.encode(answer)
    #         full_ids = list(prompt_ids) + list(answer_ids)
    #         targets = [-100] * len(prompt_ids) + list(answer_ids)
    #         all_ids.append(full_ids)
    #         all_targets.append(targets)
    #     max_len = max(len(ids) for ids in all_ids)
    #     pad_id = tok.pad_id
    #     padded_ids = []
    #     padded_targets = []
    #     for ids, tgts in zip(all_ids, all_targets):
    #         pad_len = max_len - len(ids)
    #         padded_ids.append(ids + [pad_id] * pad_len)
    #         padded_targets.append(tgts + [-100] * pad_len)
    #     input_ids = torch.tensor(padded_ids, dtype=torch.long)
    #     targets = torch.tensor(padded_targets, dtype=torch.long)
    #     return input_ids, targets
    #
    # print("\n" + "=" * 60)
    # print("HARDER VARIANT: alphabet='abcdefghij' (10 letters), N=3..7")
    # print("=" * 60)
    #
    # stream_defs_hard = make_stream_defs(tok)
    #
    # ms_model_hard = MultiStreamTestModel(
    #     stream_defs=stream_defs_hard,
    #     vocab_size=tok.vocab_size,
    #     num_heads=2,
    #     head_dim=32,
    #     num_layers=2,
    #     use_block=True,
    #     mlp_hidden_dim=128,
    # )
    #
    # gpt_model_hard = TinyGPT(
    #     vocab_size=tok.vocab_size,
    #     dim=128,
    #     num_layers=2,
    #     num_heads=4,
    #     mlp_mult=2,
    # )
    #
    # for name, model in [
    #     ("Multi-Stream (hard)", ms_model_hard),
    #     ("TinyGPT (hard)", gpt_model_hard),
    # ]:
    #     print(f"\n--- {name} ---")
    #     count_params(model, name)
    #
    #     def eval_fn_hard(m, d):
    #         return evaluate_autoregressive(
    #             m, make_sample_hard, tok, n_samples=200, device=d
    #         )
    #
    #     train_model(
    #         model, make_batch_hard, tok=tok, steps=3000, batch_size=64,
    #         lr=3e-2, eval_every=500, device=device, eval_fn=eval_fn_hard,
    #     )
    #
    #     overall = evaluate_autoregressive(
    #         model, make_sample_hard, tok, n_samples=500, device=device
    #     )
    #     print(f"  Overall accuracy: {overall:.1%}")
    #
    #     print("  Per-N accuracy:")
    #     for n_seen in range(3, 8):
    #         def _fn(n=n_seen):
    #             return make_sample_hard(fixed_n=n)
    #         acc = evaluate_autoregressive(model, _fn, tok, n_samples=200, device=device)
    #         print(f"    N={n_seen}: {acc:.1%}")
    #
    #     print("  Examples:")
    #     show_examples(model, make_sample_hard, tok, device, n=5)
