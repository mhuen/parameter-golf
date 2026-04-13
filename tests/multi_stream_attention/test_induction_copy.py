"""Test: can a single attention head with K-shift + rolling hash copy a code phrase?

Synthetic task:
    "The code is <random letters 4-20>. Please repeat the code: <first 3 letters>"
    → model must predict the remaining letters of the code.

Compares two architectures:
1. Multi-stream attention: one-hot logit stream + structural read-only stream
   (position, word hash, word boundary). The hash gives Q/K word-matching;
   K-shift lets V read the token *after* the matched position.
2. Standard attention baseline: same one-hot + structural features concatenated
   into a single flat vector. Same Q/K/V/O projections, no stream separation.

With rolling hash + K-shift, a single attention head should be able to perfectly
copy a previously seen sequence — the core induction head mechanism.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import random
import string
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from efficient_byte_tokenizer import EfficientByteTokenizer
from byte_stream_components import ByteHashComponent, HashBoundary
from modules import RMSNorm, LearnableShift
from multi_streams import (
    StreamType,
    StreamID,
    StreamDef,
    StreamSource,
    MultiStreamBuilder,
)
from test_harness import (
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


CATEGORIES = ["letters", "digits", "random"]


def _random_code(category: str, length: int) -> str:
    """Generate a random code string of the given category."""
    if category == "letters":
        return "".join(random.choices(string.ascii_lowercase, k=length))
    elif category == "digits":
        return "".join(random.choices(string.digits, k=length))
    elif category == "random":
        # printable ASCII excluding space and control chars (33-126)
        return "".join(chr(random.randint(33, 126)) for _ in range(length))
    raise ValueError(f"Unknown category: {category}")


def _make_sample_raw(
    min_len: int = 4,
    max_len: int = 20,
    category: str | None = None,
) -> tuple[str, str, int, str]:
    """Generate one copy-task sample (internal, returns full detail).

    Returns: (full_text, code, prefix_len, category)
    """
    if category is None:
        category = random.choice(CATEGORIES)
    code_len = random.randint(min_len, max_len)
    code = _random_code(category, code_len)
    prefix_len = 3
    return (
        f"The code is {code}. Please repeat the code: {code[:prefix_len]}",
        code,
        prefix_len,
        category,
    )


def make_sample(
    min_len: int = 4,
    max_len: int = 20,
    category: str | None = None,
) -> tuple[str, str]:
    """Generate one copy-task sample (harness-compatible 2-tuple).

    Returns: (prompt, answer)
    """
    full_text, code, prefix_len, _cat = _make_sample_raw(min_len, max_len, category)
    return full_text, code[prefix_len:]


def _make_batch_raw(
    tok: EfficientByteTokenizer,
    batch_size: int,
    min_len: int = 4,
    max_len: int = 20,
    category: str | None = None,
) -> tuple[Tensor, Tensor]:
    """Generate a padded batch.

    Returns:
        input_ids: (B, max_seq_len) — full sequence including prompt + completion
        targets: (B, max_seq_len) — shifted targets (-100 for non-prediction positions)
    """
    samples = [_make_sample_raw(min_len, max_len, category) for _ in range(batch_size)]
    texts = []
    for full_text, code, prefix_len, _cat in samples:
        target_text = full_text + code[prefix_len:]
        texts.append(target_text)

    encoded = [torch.from_numpy(tok.encode(t)).long() for t in texts]
    max_len_seq = max(e.numel() for e in encoded)

    input_ids = torch.full((batch_size, max_len_seq), tok.pad_id, dtype=torch.long)
    targets = torch.full((batch_size, max_len_seq), -100, dtype=torch.long)

    for i, (enc, (full_text, code, prefix_len, _cat)) in enumerate(
        zip(encoded, samples)
    ):
        seq_len = enc.numel()
        input_ids[i, :seq_len] = enc

        prompt_text = full_text
        prompt_len = len(tok.encode(prompt_text))
        completion_text = code[prefix_len:]
        completion_len = len(tok.encode(completion_text))

        sup_start = prompt_len - 1
        sup_end = prompt_len + completion_len - 1
        targets[i, sup_start:sup_end] = input_ids[i, sup_start + 1 : sup_end + 1]

    return input_ids, targets


def make_batch(tok: EfficientByteTokenizer, batch_size: int) -> tuple[Tensor, Tensor]:
    """Harness-compatible batch function (default length range)."""
    return _make_batch_raw(tok, batch_size)


# ---------------------------------------------------------------------------
# Stream definitions
# ---------------------------------------------------------------------------


def make_stream_defs(tok: EfficientByteTokenizer) -> list[StreamDef]:
    """Stream definitions for the induction copy task."""
    return [
        StreamDef(name=StreamID(StreamType.LOGIT), dim=tok.vocab_size),
        StreamDef(
            name=StreamID(StreamType.STRUCTURAL),
            read_only=True,
            source=StreamSource.COMPONENTS,
            components=[
                ByteHashComponent(
                    tok,
                    window=12,
                    num_hashes=2,
                    boundary=HashBoundary.WORD,
                    track_hits=True,
                ),
                ByteHashComponent(
                    tok,
                    window=12,
                    num_hashes=2,
                    boundary=HashBoundary.DIGIT,
                    track_hits=True,
                ),
                ByteHashComponent(
                    tok,
                    window=2,
                    num_hashes=2,
                    boundary=None,
                    track_hits=True,
                ),
                ByteHashComponent(
                    tok,
                    window=3,
                    num_hashes=2,
                    boundary=None,
                    track_hits=True,
                ),
                ByteHashComponent(
                    tok,
                    window=5,
                    num_hashes=2,
                    boundary=None,
                    track_hits=True,
                ),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Model B: Standard attention baseline (flat concatenated input)
# ---------------------------------------------------------------------------


class StandardAttentionCopyModel(nn.Module):
    """Standard single-head attention on concatenated one-hot + structural features.

    Same total input information as the multi-stream model, but no stream
    separation — Q/K/V project from the full concatenated vector, and the
    output projects back to vocab_size.
    """

    def __init__(
        self,
        stream_defs: list[StreamDef],
        tok: EfficientByteTokenizer,
        head_dim: int = 64,
    ):
        super().__init__()
        self.vocab_size = tok.vocab_size
        self.builder = MultiStreamBuilder(stream_defs, vocab_size=tok.vocab_size)

        structural_dim = next(
            s.dim
            for s in self.builder.config.streams
            if s.name == StreamID(StreamType.STRUCTURAL)
        )
        input_dim = tok.vocab_size + structural_dim
        self.W_q = nn.Linear(input_dim, head_dim, bias=False)
        self.W_k = nn.Linear(input_dim, head_dim, bias=False)
        self.W_v = nn.Linear(input_dim, head_dim, bias=False)
        self.q_norm = RMSNorm()
        self.k_norm = RMSNorm()
        self.k_shift_mod = LearnableShift(num_channels=1)
        self.W_o = nn.Linear(head_dim, tok.vocab_size, bias=False)

        # Gated residual (same as multi-stream version)
        self.alpha_pre = nn.Parameter(torch.tensor(-2.0))
        self.gate = nn.Linear(head_dim, tok.vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        B, S = input_ids.shape
        onehot = F.one_hot(input_ids, self.vocab_size).float()
        streams, _, _ = self.builder(input_ids, logit=onehot)
        structural = streams[StreamID(StreamType.STRUCTURAL)]
        x = torch.cat([onehot, structural], dim=-1)  # (B, S, input_dim)

        q = self.q_norm(self.W_q(x)).unsqueeze(1)  # (B, 1, S, D)
        k = self.k_norm(self.W_k(x)).unsqueeze(1)
        k = self.k_shift_mod(k)
        v = self.W_v(x).unsqueeze(1)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.squeeze(1)  # (B, S, D)

        alpha = torch.sigmoid(self.alpha_pre)
        update = self.W_o(attn_out) * torch.sigmoid(self.gate(attn_out))
        return onehot + alpha * update


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_test_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = EfficientByteTokenizer()
    print(f"Device: {device}, vocab_size: {tok.vocab_size}\n")

    stream_defs = make_stream_defs(tok)

    MIN_CODE_LEN = 4
    MAX_CODE_LEN = 16

    # Bind training length range into batch function
    make_train_batch = partial(
        _make_batch_raw, min_len=MIN_CODE_LEN, max_len=MAX_CODE_LEN
    )

    ms_model = MultiStreamTestModel(
        stream_defs=stream_defs,
        vocab_size=tok.vocab_size,
        num_heads=1,
        head_dim=16,
        num_layers=1,
        k_shift=True,
    )

    flat_model = StandardAttentionCopyModel(stream_defs, tok, head_dim=16)

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

    print("=" * 60)
    print("Model sizes")
    print("=" * 60)
    ms_params = count_params(ms_model, "Multi-stream attention")
    flat_params = count_params(flat_model, "Standard attention (flat)")
    count_params(msgpt_model, "MultiStreamGPT")
    print()

    train_kwargs = dict(
        tok=tok,
        steps=1000,
        batch_size=64,
        lr=3e-2,
        eval_every=200,
        device=device,
        pack_documents=args.pack,
    )
    eval_ranges = [(4, 8), (8, 12), (12, 16), (16, 20)]

    if args.pack:
        print("*** Document packing enabled (2 docs/seq, BOS-separated) ***\n")

    results = {}

    models = [
        ("Multi-stream attention", ms_model),
        ("Standard attention (flat)", flat_model),
        ("MultiStreamGPT", msgpt_model),
    ]

    for name, model in models:
        print("=" * 60)
        print(name)
        print("=" * 60)
        model = maybe_compile(model, args.compile)

        if args.pack:

            def eval_fn(m, d, _min=MIN_CODE_LEN, _max=MAX_CODE_LEN):
                return evaluate_packed(
                    m,
                    partial(_make_batch_raw, min_len=_min, max_len=_max),
                    tok,
                    n_samples=200,
                    device=d,
                )
        else:

            def eval_fn(m, d, _min=MIN_CODE_LEN, _max=MAX_CODE_LEN):
                return evaluate_autoregressive(
                    m,
                    partial(make_sample, min_len=_min, max_len=_max),
                    tok,
                    n_samples=200,
                    device=d,
                )

        train_model(model, make_train_batch, **train_kwargs, eval_fn=eval_fn)

        print("\n  Final evaluation (per category x length range):")
        accs = {}
        for cat in CATEGORIES:
            for lo, hi in eval_ranges:
                acc = evaluate_autoregressive(
                    model,
                    partial(make_sample, min_len=lo, max_len=hi, category=cat),
                    tok,
                    n_samples=200,
                    device=device,
                )
                accs[(cat, lo, hi)] = acc
            print(
                f"    {cat:8s}  "
                + "  ".join(
                    f"{lo}-{hi}: {accs[(cat, lo, hi)]:.0%}" for lo, hi in eval_ranges
                )
            )
        results[name] = accs

        if args.pack:
            packed_acc = evaluate_packed(
                model,
                make_train_batch,
                tok,
                n_samples=200,
                device=device,
            )
            print(f"\n  Packed accuracy: {packed_acc:.1%}")

        print("\n  Examples:")
        show_examples(
            model,
            partial(make_sample, min_len=6, max_len=14),
            tok,
            device,
            n=6,
        )

        verify_causality(model, tok, device, make_sample_fn=make_sample, label=name)
        print()

    # Summary comparison
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for cat in CATEGORIES:
        print(f"\n  [{cat}]")
        header = f"    {'Range':>8s}"
        for name in results:
            header += f"  {name:>28s}"
        print(header)
        for lo, hi in eval_ranges:
            row = f"    {lo:2d}-{hi:2d}   "
            for name in results:
                row += f"  {results[name][(cat, lo, hi)]:>28.1%}"
            print(row)
