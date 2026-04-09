"""Test: associative recall — can multi-stream attention retrieve a value by key?

Synthetic task:
    "A=3 B=7 C=1. What is B? " → "7"

Each sample contains 3-8 random key-value pairs (uppercase letter = single digit).
A random key is queried and the model must predict the corresponding digit.
Only the answer token(s) are supervised.

Compares:
1. Multi-stream attention with structural hash features (word-level + raw n-gram).
2. TinyGPT baseline sized to approximately match parameter count.
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
from multi_streams import StreamType, StreamID, StreamDef
from test_harness import (
    TinyGPT,
    MultiStreamTestModel,
    MultiStreamGPTTestModel,
    count_params,
    train_model,
    evaluate_autoregressive,
    evaluate_packed,
    show_examples,
    verify_causality,
)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def make_sample() -> tuple[str, str]:
    """Generate one associative-recall sample.

    Returns: (prompt, answer)
        prompt: e.g. "A=3 B=7 C=1. What is B? "
        answer: e.g. "7"
    """
    num_pairs = random.randint(3, 8)
    keys = random.sample(string.ascii_uppercase, num_pairs)
    values = [str(random.randint(0, 9)) for _ in range(num_pairs)]

    pairs_str = " ".join(f"{k}={v}" for k, v in zip(keys, values))

    query_idx = random.randint(0, num_pairs - 1)
    query_key = keys[query_idx]
    answer = values[query_idx]

    prompt = f"{pairs_str}. What is {query_key}? "
    return prompt, answer


def make_batch(tok: EfficientByteTokenizer, batch_size: int) -> tuple[Tensor, Tensor]:
    """Build a padded batch with supervision only on answer tokens.

    Returns: (input_ids, targets) both of shape (B, max_len)
        targets is -100 everywhere except the answer position(s).
    """
    all_ids = []
    all_targets = []

    for _ in range(batch_size):
        prompt, answer = make_sample()

        prompt_ids = list(tok.encode(prompt))
        answer_ids = list(tok.encode(answer))
        full_ids = prompt_ids + answer_ids

        # Shifted targets: logits[i] predicts token at position i+1
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
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pack",
        action="store_true",
        help="Pack 2 documents per sequence (each prefixed with BOS)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = EfficientByteTokenizer()

    # --- Stream definitions for multi-stream model ---
    stream_defs = [
        StreamDef(
            name=StreamID(StreamType.LOGIT), dim=tok.vocab_size
        ),  # caller provides one-hot
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
                    tok, window=3, num_hashes=2, boundary=None, track_hits=True
                ),
            ],
        ),
    ]

    # --- Build models ---
    ms_model = MultiStreamTestModel(
        stream_defs=stream_defs,
        vocab_size=tok.vocab_size,
        num_heads=1,
        head_dim=32,
        num_layers=1,
        use_block=False,
    )

    gpt_model = TinyGPT(
        vocab_size=tok.vocab_size,
        dim=64,
        num_layers=1,
        num_heads=2,
        mlp_mult=2,
    )

    msgpt_model = MultiStreamGPTTestModel(
        tok=tok,
        vocab_size=tok.vocab_size,
        num_heads=1,
        num_kv_heads=1,
        num_layers=1,
        multi_head_dim=32,
        mlp_hidden_dim=16,
        num_preconv_layers=1,
        # preconv_groups=1,
    )

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

    models = [
        ("Multi-Stream Attention (1 head, head_dim=32, 1 layer)", ms_model),
        ("TinyGPT (dim=64, 1 layer, 2 heads, mlp_mult=2)", gpt_model),
        ("MultiStreamGPT (1H, 1L, mhd=32)", msgpt_model),
    ]

    for name, model in models:
        print("=" * 60)
        print(name)
        print("=" * 60)
        count_params(model, name)

        if args.pack:

            def eval_fn(m, d):
                return evaluate_packed(m, make_batch, tok, n_samples=200, device=d)
        else:

            def eval_fn(m, d):
                return evaluate_autoregressive(
                    m, make_sample, tok, n_samples=200, device=d
                )

        train_model(model, make_batch, **train_kwargs, eval_fn=eval_fn)

        # Always report both single-doc and packed accuracy at the end
        single_acc = evaluate_autoregressive(
            model,
            make_sample,
            tok,
            n_samples=200,
            device=device,
        )
        print(f"\n  Single-doc accuracy: {single_acc:.1%}")
        if args.pack:
            packed_acc = evaluate_packed(
                model,
                make_batch,
                tok,
                n_samples=200,
                device=device,
            )
            print(f"  Packed accuracy:     {packed_acc:.1%}")

        print("\n  Examples:")
        show_examples(model, make_sample, tok, device, n=5)

        verify_causality(model, tok, device, make_sample_fn=make_sample, label=name)
        print()
