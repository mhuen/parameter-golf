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

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import random
import string

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from efficient_byte_tokenizer import EfficientByteTokenizer
from byte_modules import ByteHashComponent, BoundaryComponent, HashBoundary
from modules import RMSNorm, LearnableShift
from multi_streams import (
    Stream,
    StreamDef,
    MultiStreamBuilder,
    SinCosPositionComponent,
)
from multi_stream_attention import CausualMultiStreamAttention


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


def make_sample(
    min_len: int = 4,
    max_len: int = 20,
    category: str | None = None,
) -> tuple[str, str, int, str]:
    """Generate one copy-task sample.

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


def make_batch(
    tok: EfficientByteTokenizer,
    batch_size: int,
    min_len: int = 4,
    max_len: int = 20,
    category: str | None = None,
) -> tuple[Tensor, Tensor, list[int]]:
    """Generate a padded batch.

    Args:
        category: if None, each sample picks a random category (mixed training).
            If set, all samples use that category (per-category evaluation).

    Returns:
        input_ids: (B, max_seq_len) — full sequence including prompt + completion
        targets: (B, max_seq_len) — shifted targets (-100 for non-prediction positions)
        code_lengths: list of code string lengths
    """
    samples = [make_sample(min_len, max_len, category) for _ in range(batch_size)]
    texts = []
    code_lengths = []
    for full_text, code, prefix_len, _cat in samples:
        target_text = full_text + code[prefix_len:]
        texts.append(target_text)
        code_lengths.append(len(code))

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

    return input_ids, targets, code_lengths


# ---------------------------------------------------------------------------
# Shared stream setup
# ---------------------------------------------------------------------------


def make_stream_builder(tok: EfficientByteTokenizer) -> MultiStreamBuilder:
    """Create the stream builder used by both model variants."""
    return MultiStreamBuilder(
        stream_defs=[
            StreamDef(name=Stream.LOGIT, dim=tok.vocab_size),
            StreamDef(name=Stream.STRUCTURAL, read_only=True, components=[
                # SinCosPositionComponent(num_freqs=4),  # 8d
                ByteHashComponent(
                    tok,
                    window=12,
                    num_hashes=2,
                    boundary=HashBoundary.WORD,
                    track_hits=True,
                ),  # 6d
                ByteHashComponent(
                    tok,
                    window=12,
                    num_hashes=2,
                    boundary=HashBoundary.DIGIT,
                    track_hits=True,
                ),  # 6d
                ByteHashComponent(
                    tok,
                    window=3,
                    num_hashes=2,
                    boundary=None,
                    track_hits=True,
                ),  # 6d
                ByteHashComponent(
                    tok,
                    window=8,
                    num_hashes=2,
                    boundary=None,
                    track_hits=True,
                ),  # 6d
                # BoundaryComponent(
                #     tok,
                #     word_pos_freqs=2,
                #     word_id_freqs=2,
                #     sent_pos_freqs=0,
                #     sent_id_freqs=0,
                #     para_pos_freqs=0,
                #     para_id_freqs=0,
                # ),  # 8d
            ]),
        ],
        vocab_size=tok.vocab_size,
    )


# ---------------------------------------------------------------------------
# Model A: Multi-stream attention (structured read-only streams)
# ---------------------------------------------------------------------------


class MultiStreamCopyModel(nn.Module):
    """Multi-stream attention: one-hot logit stream + structural read-only stream."""

    def __init__(self, tok: EfficientByteTokenizer, head_dim: int = 64):
        super().__init__()
        self.vocab_size = tok.vocab_size
        self.builder = make_stream_builder(tok)
        self.attn = CausualMultiStreamAttention(
            multi_head_dim=head_dim,
            num_heads=1,
            num_kv_heads=1,
            stream_config=self.builder.config,
            k_shift=True,
        )

    def forward(self, input_ids: Tensor) -> Tensor:
        logit_stream = F.one_hot(input_ids, self.vocab_size).float()
        streams = self.builder(input_ids, logit=logit_stream)
        out = self.attn(streams)
        return out[Stream.LOGIT]


# ---------------------------------------------------------------------------
# Model B: Standard attention baseline (flat concatenated input)
# ---------------------------------------------------------------------------


class StandardAttentionCopyModel(nn.Module):
    """Standard single-head attention on concatenated one-hot + structural features.

    Same total input information as the multi-stream model, but no stream
    separation — Q/K/V project from the full concatenated vector, and the
    output projects back to vocab_size.
    """

    def __init__(self, tok: EfficientByteTokenizer, head_dim: int = 64):
        super().__init__()
        self.vocab_size = tok.vocab_size
        self.builder = make_stream_builder(tok)

        structural_dim = next(
            s.dim for s in self.builder.config.streams if s.name == Stream.STRUCTURAL
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
        streams = self.builder(input_ids, logit=onehot)
        structural = streams[Stream.STRUCTURAL]
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
# Training and evaluation
# ---------------------------------------------------------------------------


def train(
    model: nn.Module,
    tok: EfficientByteTokenizer,
    steps: int = 2000,
    batch_size: int = 32,
    lr: float = 3e-3,
    min_code_len: int = 4,
    max_code_len: int = 16,
    eval_every: int = 200,
    device: str = "cpu",
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}")

    for step in range(1, steps + 1):
        model.train()
        input_ids, targets, _ = make_batch(tok, batch_size, min_code_len, max_code_len)
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % eval_every == 0 or step == 1:
            acc = evaluate(
                model,
                tok,
                n_samples=200,
                min_len=min_code_len,
                max_len=max_code_len,
                device=device,
            )
            # Report k_shift value if available
            ks_mod = getattr(model, "k_shift_mod", None)
            if ks_mod is None:
                ks_mod = getattr(getattr(model, "attn", None), "k_shift_mod", None)
            ks_str = ""
            if ks_mod is not None:
                ks_str = f"  k_shift={torch.sigmoid(ks_mod.shift_logit).item():.4f}"
            print(
                f"    step {step:5d}  loss={loss.item():.4f}  char_acc={acc:.1%}{ks_str}"
            )

    return model


@torch.no_grad()
def evaluate(
    model: nn.Module,
    tok: EfficientByteTokenizer,
    n_samples: int = 200,
    min_len: int = 4,
    max_len: int = 16,
    category: str | None = None,
    device: str = "cpu",
) -> float:
    """Character-level accuracy on the copy task (autoregressive)."""
    model.eval()
    total_chars = 0
    correct_chars = 0

    for _ in range(n_samples):
        full_text, code, prefix_len, _cat = make_sample(min_len, max_len, category)
        remaining = code[prefix_len:]
        n_to_predict = len(remaining)
        if n_to_predict == 0:
            continue

        ids = torch.from_numpy(tok.encode(full_text)).long().unsqueeze(0).to(device)

        predicted_bytes = []
        for _ in range(n_to_predict):
            logits = model(ids)
            next_id = logits[0, -1].argmax().item()
            predicted_bytes.append(next_id)
            ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)

        expected_ids = tok.encode(remaining)
        for j in range(min(len(predicted_bytes), len(expected_ids))):
            total_chars += 1
            if predicted_bytes[j] == expected_ids[j]:
                correct_chars += 1
        total_chars += max(0, len(expected_ids) - len(predicted_bytes))

    return correct_chars / max(total_chars, 1)


def show_examples(
    model: nn.Module,
    tok: EfficientByteTokenizer,
    device: str,
    n: int = 3,
    category: str | None = None,
):
    model.eval()
    for _ in range(n):
        full_text, code, prefix_len, cat = make_sample(6, 14, category)
        remaining = code[prefix_len:]
        ids = torch.from_numpy(tok.encode(full_text)).long().unsqueeze(0).to(device)

        predicted = []
        with torch.no_grad():
            for _ in range(len(remaining)):
                logits = model(ids)
                next_id = logits[0, -1].argmax().item()
                predicted.append(next_id)
                ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)

        pred_str = tok.decode_to_str(predicted)
        match = "OK" if pred_str == remaining else "FAIL"
        print(
            f"    {match:4s} [{cat:7s}] code={code!r:22s}  prefix={code[:prefix_len]!r}  "
            f"expected={remaining!r:18s}  predicted={pred_str!r}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = EfficientByteTokenizer()
    print(f"Device: {device}, vocab_size: {tok.vocab_size}\n")

    train_kwargs = dict(
        tok=tok,
        steps=2000,
        batch_size=64,
        lr=3e-2,
        min_code_len=4,
        max_code_len=16,
        eval_every=400,
        device=device,
    )
    eval_ranges = [(4, 8), (8, 12), (12, 16), (16, 20)]

    results = {}

    for name, ModelClass in [
        ("Multi-stream attention", MultiStreamCopyModel),
        ("Standard attention (flat)", StandardAttentionCopyModel),
    ]:
        print("=" * 60)
        print(f"{name}")
        print("=" * 60)
        model = ModelClass(tok, head_dim=32)
        model = train(model, **train_kwargs)

        print("\n  Final evaluation (per category x length range):")
        accs = {}
        for cat in CATEGORIES:
            for lo, hi in eval_ranges:
                acc = evaluate(
                    model,
                    tok,
                    n_samples=200,
                    min_len=lo,
                    max_len=hi,
                    category=cat,
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

        print("\n  Examples:")
        for cat in CATEGORIES:
            show_examples(model, tok, device, n=2, category=cat)
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
