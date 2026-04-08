"""Test: selective copy — filter characters from a mixed alphanumeric code.

Synthetic task:
    "Code: abc123def. Copy only the letters: " → "abcdef"
    "Code: abc123def. Copy only the digits: "  → "123"

Mixed alphanumeric codes (6-16 chars, lowercase letters + digits).
Two filter categories: "letters" or "digits".
Answer: only the characters matching the requested category, in order.

Compares multi-stream attention (with context scratch stream) vs TinyGPT.
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
from byte_stream_components import ByteHashComponent, HashBoundary, BoundaryComponent
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

CATEGORIES = ("letters", "digits")
CODE_CHARS = string.ascii_lowercase + string.digits


def make_sample(category: str | None = None) -> tuple[str, str]:
    """Generate one selective-copy sample.

    Returns (prompt_text, answer_str).
    """
    if category is None:
        category = random.choice(CATEGORIES)

    code_len = random.randint(6, 16)

    # Generate code ensuring at least 1 char of the target category
    while True:
        code = "".join(random.choices(CODE_CHARS, k=code_len))
        if category == "letters":
            answer = "".join(ch for ch in code if ch in string.ascii_lowercase)
        else:
            answer = "".join(ch for ch in code if ch in string.digits)
        if len(answer) >= 1:
            break

    prompt = f"Code: {code}. Copy only the {category}: "
    return prompt, answer


def _make_sample_letters() -> tuple[str, str]:
    return make_sample("letters")


def _make_sample_digits() -> tuple[str, str]:
    return make_sample("digits")


def make_batch(
    tok: EfficientByteTokenizer,
    batch_size: int,
) -> tuple[Tensor, Tensor]:
    """Generate a padded batch for training.

    Returns:
        input_ids: (B, max_seq_len) — full sequence (prompt + answer)
        targets:   (B, max_seq_len) — shifted targets (-100 for prompt positions)
    """
    samples = [make_sample() for _ in range(batch_size)]

    encoded_full = []
    prompt_lengths = []
    for prompt, answer in samples:
        full_text = prompt + answer
        enc = torch.from_numpy(tok.encode(full_text)).long()
        encoded_full.append(enc)
        prompt_len = len(tok.encode(prompt))
        prompt_lengths.append(prompt_len)

    max_len = max(e.numel() for e in encoded_full)

    input_ids = torch.full((batch_size, max_len), tok.pad_id, dtype=torch.long)
    targets = torch.full((batch_size, max_len), -100, dtype=torch.long)

    for i, (enc, prompt_len) in enumerate(zip(encoded_full, prompt_lengths)):
        seq_len = enc.numel()
        input_ids[i, :seq_len] = enc

        # Supervise only the answer portion (teacher-forced, shifted by 1)
        sup_start = prompt_len - 1
        sup_end = seq_len - 1
        targets[i, sup_start:sup_end] = input_ids[i, sup_start + 1 : sup_end + 1]

    return input_ids, targets


# ---------------------------------------------------------------------------
# Stream definitions (includes context stream for scratch memory)
# ---------------------------------------------------------------------------


def make_stream_defs(tok: EfficientByteTokenizer) -> list[StreamDef]:
    return [
        StreamDef(name=StreamID(StreamType.LOGIT), dim=tok.vocab_size),
        StreamDef(
            name=StreamID(StreamType.CONTEXT), dim=32, auto_zeros=True
        ),  # writable scratch
        StreamDef(name=StreamID(StreamType.TOKENS), read_only=True, auto_onehot=True),
        StreamDef(
            name=StreamID(StreamType.STRUCTURAL),
            read_only=True,
            components=[
                ByteHashComponent(
                    tok,
                    window=6,
                    num_hashes=2,
                    boundary=HashBoundary.WORD,
                    track_hits=True,
                ),
                ByteHashComponent(
                    tok,
                    window=3,
                    num_hashes=2,
                    boundary=None,
                    track_hits=True,
                ),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = EfficientByteTokenizer()
    print(f"Device: {device}, vocab_size: {tok.vocab_size}\n")

    stream_defs = make_stream_defs(tok)

    # --- Build models ---
    print("=" * 60)
    print("Multi-stream attention")
    print("=" * 60)
    ms_model = MultiStreamTestModel(
        stream_defs=stream_defs,
        vocab_size=tok.vocab_size,
        num_heads=2,
        head_dim=32,
        num_layers=2,
        # use_block=True,
        # mlp_hidden_dim=16,
    )
    ms_params = count_params(ms_model, "Multi-stream")

    print()
    print("=" * 60)
    print("TinyGPT baseline")
    print("=" * 60)
    # Size TinyGPT to roughly match multi-stream param count
    gpt_model = TinyGPT(
        vocab_size=tok.vocab_size,
        dim=128,
        num_layers=2,
        num_heads=4,
        mlp_mult=2,
    )
    gpt_params = count_params(gpt_model, "TinyGPT")
    print(f"\n  Param ratio (GPT / MS): {gpt_params / ms_params:.2f}x\n")

    # --- Training ---
    train_kwargs = dict(
        make_batch_fn=make_batch,
        tok=tok,
        steps=2000,
        batch_size=64,
        lr=3e-2,
        eval_every=400,
        device=device,
    )

    def ms_eval_fn(model, dev):
        return evaluate_autoregressive(
            model, make_sample, tok, n_samples=200, device=dev
        )

    def gpt_eval_fn(model, dev):
        return evaluate_autoregressive(
            model, make_sample, tok, n_samples=200, device=dev
        )

    print("=" * 60)
    print("Training Multi-stream")
    print("=" * 60)
    ms_model = train_model(ms_model, eval_fn=ms_eval_fn, **train_kwargs)

    print()
    print("=" * 60)
    print("Training TinyGPT")
    print("=" * 60)
    gpt_model = train_model(gpt_model, eval_fn=gpt_eval_fn, **train_kwargs)

    # --- Per-category evaluation ---
    print()
    print("=" * 60)
    print("Per-category evaluation")
    print("=" * 60)

    results = {}
    for name, model in [("Multi-stream", ms_model), ("TinyGPT", gpt_model)]:
        overall = evaluate_autoregressive(
            model, make_sample, tok, n_samples=400, device=device
        )
        letters_acc = evaluate_autoregressive(
            model, _make_sample_letters, tok, n_samples=200, device=device
        )
        digits_acc = evaluate_autoregressive(
            model, _make_sample_digits, tok, n_samples=200, device=device
        )
        results[name] = {
            "overall": overall,
            "letters": letters_acc,
            "digits": digits_acc,
        }
        print(
            f"  {name:15s}  overall={overall:.1%}  "
            f"letters={letters_acc:.1%}  digits={digits_acc:.1%}"
        )

    # --- Show examples per category ---
    for name, model in [("Multi-stream", ms_model), ("TinyGPT", gpt_model)]:
        print()
        print(f"  Examples — {name}")
        print(f"  {'─' * 50}")
        print("  [letters]")
        show_examples(model, _make_sample_letters, tok, device=device, n=5)
        print("  [digits]")
        show_examples(model, _make_sample_digits, tok, device=device, n=5)

    # --- Causality verification ---
    print()
    print("=" * 60)
    print("Causality verification")
    print("=" * 60)
    verify_causality(
        ms_model, tok, device, make_sample_fn=make_sample, label="Multi-stream"
    )
    verify_causality(
        gpt_model, tok, device, make_sample_fn=make_sample, label="TinyGPT"
    )

    # --- Summary comparison ---
    print()
    print("=" * 60)
    print("Summary comparison")
    print("=" * 60)
    header = f"  {'Category':>10s}"
    for name in results:
        header += f"  {name:>15s}"
    print(header)
    for cat in ["overall", "letters", "digits"]:
        row = f"  {cat:>10s}"
        for name in results:
            row += f"  {results[name][cat]:>15.1%}"
        print(row)
    print()
    print(f"  Multi-stream params: {ms_params:,}")
    print(f"  TinyGPT params:      {gpt_params:,}")
