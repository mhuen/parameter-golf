"""Shared test infrastructure for multi-stream attention head tests.

Provides:
- TinyGPT: small standard transformer baseline (embedding + RoPE attention + MLP)
- MultiStreamTestModel: wraps MultiStreamBuilder + attention/block layers
- Training and evaluation utilities for synthetic copy/recall tasks
"""

import sys
import os
import math
from typing import Callable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from efficient_byte_tokenizer import EfficientByteTokenizer
from modules import RMSNorm
from multi_streams import (
    StreamType,
    StreamID,
    StreamDef,
    MultiStreamBuilder,
    MultiStreamConfig,
    CompressionType,
    CompressedView,
)
from multi_stream_attention import (
    CausalMultiStreamAttention,
    CausalArithmeticMultiStreamAttention,
    MultiStreamBlock,
    StreamMixingConfig,
)


# ---------------------------------------------------------------------------
# TinyGPT baseline (self-contained, no train_gpt.py dependency)
# ---------------------------------------------------------------------------


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache: tuple[int, Tensor, Tensor] | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if (
            self._cache is None
            or self._cache[0] != seq_len
            or self._cache[1].device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cache = (seq_len, freqs.cos()[None, None], freqs.sin()[None, None])
        return self._cache[1].to(dtype), self._cache[2].to(dtype)


def _apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class TinyAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, rope_base: float = 10000.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.c_q = nn.Linear(dim, dim, bias=False)
        self.c_k = nn.Linear(dim, dim, bias=False)
        self.c_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        B, S, D = x.shape
        q = self.c_q(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(S, x.device, q.dtype)
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, S, D))


class TinyMLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int = 2):
        super().__init__()
        hidden = dim * mlp_mult
        self.fc = nn.Linear(dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(F.leaky_relu(self.fc(x), negative_slope=0.5).square())


class TinyBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_mult: int = 2):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = TinyAttention(dim, num_heads)
        self.mlp = TinyMLP(dim, mlp_mult)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class TinyGPT(nn.Module):
    """Small GPT baseline: embedding -> blocks -> norm -> linear head.

    Returns logits (B, S, vocab_size), not loss.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 64,
        num_layers: int = 1,
        num_heads: int = 2,
        mlp_mult: int = 2,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList(
            [TinyBlock(dim, num_heads, mlp_mult) for _ in range(num_layers)]
        )
        self.norm = RMSNorm()
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


# ---------------------------------------------------------------------------
# Multi-stream test model
# ---------------------------------------------------------------------------


class MultiStreamTestModel(nn.Module):
    """Wraps MultiStreamBuilder + N attention/block layers.

    Returns logits (B, S, vocab_size) from the logit stream.
    """

    def __init__(
        self,
        stream_defs: list[StreamDef],
        vocab_size: int,
        num_heads: int = 1,
        num_kv_heads: int | None = None,
        head_dim: int = 32,
        num_layers: int = 1,
        use_block: bool = False,
        mlp_hidden_dim: int | None = None,
        k_shift: bool = True,
        mixing_config: StreamMixingConfig | None = None,
        arith_attn: CausalArithmeticMultiStreamAttention | None = None,
        compressions: dict | None = None,
        compress_streams: list[StreamID] | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size

        self.builder = MultiStreamBuilder(
            stream_defs,
            vocab_size=vocab_size,
            compressions=compressions,
            compress_streams=compress_streams,
        )
        print("Stream definitions:")
        for stream in self.builder.config.streams:
            print(f"  {stream.name}: dim={stream.dim}")

        if num_kv_heads is None:
            num_kv_heads = num_heads

        shared_kwargs = dict(
            multi_head_dim=num_heads * head_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            stream_config=self.builder.config,
        )

        if use_block:
            self.layers = nn.ModuleList(
                [
                    MultiStreamBlock(
                        **shared_kwargs,
                        mixing_config=mixing_config,
                        mlp_hidden_dim=mlp_hidden_dim,
                        k_shift=k_shift,
                        arith_attn=arith_attn,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    CausalMultiStreamAttention(
                        **shared_kwargs,
                        k_shift=k_shift,
                    )
                    for _ in range(num_layers)
                ]
            )

    def forward(self, input_ids: Tensor, **provided_streams: Tensor) -> Tensor:
        logit_onehot = F.one_hot(input_ids, self.vocab_size).float()
        streams, compressed, views = self.builder(
            input_ids, logit=logit_onehot, **provided_streams
        )
        for layer in self.layers:
            if isinstance(layer, MultiStreamBlock) and compressed is not None:
                streams = layer(streams, compressed=compressed, views=views)
            else:
                streams = layer(streams)
        return streams[StreamID(StreamType.LOGIT)]


# ---------------------------------------------------------------------------
# Training and evaluation utilities
# ---------------------------------------------------------------------------


def count_params(model: nn.Module, label: str = "") -> int:
    n = sum(p.numel() for p in model.parameters())
    if label:
        print(f"  {label}: {n:,} params")
    else:
        print(f"  params: {n:,}")
    return n


MakeBatchFn = Callable[
    [EfficientByteTokenizer, int],  # (tok, batch_size) ->
    tuple[Tensor, Tensor],  # (input_ids, targets)
]


def _collect_k_shift_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Find all LearnableShift modules used as k_shift in the model."""
    results = []
    # Direct k_shift_mod on the model (e.g. StandardAttentionCopyModel)
    ks = getattr(model, "k_shift_mod", None)
    if ks is not None:
        results.append(("", ks))
        return results
    # Direct attn.k_shift_mod (e.g. MultiStreamCopyModel)
    attn = getattr(model, "attn", None)
    if attn is not None:
        ks = getattr(attn, "k_shift_mod", None)
        if ks is not None:
            results.append(("", ks))
            return results
    # Walk model.layers (MultiStreamTestModel with CausalMultiStreamAttention or MultiStreamBlock)
    layers = getattr(model, "layers", None)
    if layers is not None:
        for i, layer in enumerate(layers):
            ks = getattr(layer, "k_shift_mod", None)
            if ks is None:
                ks = getattr(getattr(layer, "attn", None), "k_shift_mod", None)
            if ks is not None:
                label = f"L{i}" if len(layers) > 1 else ""
                results.append((label, ks))
    return results


def _format_k_shifts(model: nn.Module) -> str:
    """Format k_shift values for all layers into a compact string."""
    mods = _collect_k_shift_modules(model)
    if not mods:
        return ""
    parts = []
    for label, ks in mods:
        vals = torch.sigmoid(ks.shift_logit)
        if vals.numel() == 1:
            s = f"{vals.item():.3f}"
        else:
            s = "[" + ",".join(f"{v:.3f}" for v in vals.tolist()) + "]"
        if label:
            parts.append(f"{label}={s}")
        else:
            parts.append(s)
    return "  k_shift=" + " ".join(parts)


def _format_alpha_summary(param: Tensor) -> str:
    """Summarize a per-dimension alpha/beta parameter as mean(min..max) after sigmoid."""
    vals = torch.sigmoid(param)
    return f"{vals.mean().item():.3f}({vals.min().item():.3f}..{vals.max().item():.3f})"


def _format_alphas(model: nn.Module) -> str:
    """Format gating alpha summaries for multi-stream layers.

    For MultiStreamBlock layers: reports attn_alpha and mlp_alpha per writable stream.
    For bare CausalMultiStreamAttention: reports alpha_pre_sigmoid per writable stream.
    """
    layers = getattr(model, "layers", None)
    if layers is None:
        return ""

    parts = []
    for i, layer in enumerate(layers):
        prefix = f"L{i}" if len(layers) > 1 else ""
        # MultiStreamBlock has attn_alpha / mlp_alpha
        attn_alpha = getattr(layer, "attn_alpha", None)
        if attn_alpha is not None:
            for name, param in attn_alpha.items():
                lbl = f"{prefix}attn_α.{name}" if prefix else f"attn_α.{name}"
                parts.append(f"{lbl}={_format_alpha_summary(param)}")
            mlp_alpha = getattr(layer, "mlp_alpha", None)
            if mlp_alpha is not None:
                for name, param in mlp_alpha.items():
                    lbl = f"{prefix}mlp_α.{name}" if prefix else f"mlp_α.{name}"
                    parts.append(f"{lbl}={_format_alpha_summary(param)}")
            arith_alpha = getattr(layer, "arith_alpha", None)
            if arith_alpha is not None:
                for name, param in arith_alpha.items():
                    lbl = f"{prefix}arith_α.{name}" if prefix else f"arith_α.{name}"
                    parts.append(f"{lbl}={_format_alpha_summary(param)}")
            continue
        # Bare CausalMultiStreamAttention has alpha_pre_sigmoid
        alpha_ps = getattr(layer, "alpha_pre_sigmoid", None)
        if alpha_ps is not None:
            for name, param in alpha_ps.items():
                lbl = f"{prefix}α.{name}" if prefix else f"α.{name}"
                parts.append(f"{lbl}={_format_alpha_summary(param)}")

    if not parts:
        return ""
    return "  " + " ".join(parts)


def _pack_batch(
    make_batch_fn: MakeBatchFn,
    tok: EfficientByteTokenizer,
    batch_size: int,
    docs_per_seq: int = 2,
) -> tuple[Tensor, Tensor]:
    """Pack multiple documents per sequence, each prefixed with BOS.

    Generates ``docs_per_seq`` independent batches, strips trailing padding
    from each row, and concatenates them as ``[BOS doc_1 BOS doc_2 ...]``.
    The result is re-padded to uniform length.

    Returns: (input_ids, targets) — same contract as any MakeBatchFn.
    """
    batches = [make_batch_fn(tok, batch_size) for _ in range(docs_per_seq)]

    packed_ids_list: list[Tensor] = []
    packed_tgt_list: list[Tensor] = []
    bos = torch.tensor([tok.bos_id], dtype=torch.long)
    ignore = torch.tensor([-100], dtype=torch.long)

    for i in range(batch_size):
        row_parts_ids: list[Tensor] = []
        row_parts_tgt: list[Tensor] = []
        for ids_batch, tgt_batch in batches:
            doc_ids = ids_batch[i]
            doc_len = int((doc_ids != tok.pad_id).sum().item())
            row_parts_ids.append(bos)
            row_parts_ids.append(doc_ids[:doc_len])
            row_parts_tgt.append(ignore)
            row_parts_tgt.append(tgt_batch[i, :doc_len])
        packed_ids_list.append(torch.cat(row_parts_ids))
        packed_tgt_list.append(torch.cat(row_parts_tgt))

    max_len = max(r.numel() for r in packed_ids_list)
    padded_ids = torch.full((batch_size, max_len), tok.pad_id, dtype=torch.long)
    padded_tgt = torch.full((batch_size, max_len), -100, dtype=torch.long)
    for i, (ri, rt) in enumerate(zip(packed_ids_list, packed_tgt_list)):
        padded_ids[i, : ri.numel()] = ri
        padded_tgt[i, : rt.numel()] = rt

    return padded_ids, padded_tgt


def train_model(
    model: nn.Module,
    make_batch_fn: MakeBatchFn,
    tok: EfficientByteTokenizer,
    steps: int = 2000,
    batch_size: int = 64,
    lr: float = 3e-2,
    eval_fn: Callable | None = None,
    eval_every: int = 400,
    device: str = "cpu",
    pack_documents: bool = False,
) -> nn.Module:
    """Generic training loop for synthetic tasks.

    Args:
        pack_documents: if True, pack 2 documents per sequence (each prefixed
            with BOS) so that cross-document causal attention is exercised.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    for step in range(1, steps + 1):
        model.train()
        if pack_documents:
            input_ids, targets = _pack_batch(make_batch_fn, tok, batch_size)
        else:
            input_ids, targets = make_batch_fn(tok, batch_size)
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
            acc = eval_fn(model, device) if eval_fn else 0.0

            ks_str = _format_k_shifts(model)
            alpha_str = _format_alphas(model)
            pack_str = " [packed]" if pack_documents else ""
            print(
                f"    step {step:5d}  loss={loss.item():.4f}  acc={acc:.1%}{ks_str}{alpha_str}{pack_str}"
            )

    return model


MakeSampleFn = Callable[
    [],  # () ->
    tuple[str, str],  # (prompt_text, expected_completion)
]


@torch.no_grad()
def evaluate_autoregressive(
    model: nn.Module,
    make_sample_fn: MakeSampleFn,
    tok: EfficientByteTokenizer,
    n_samples: int = 200,
    device: str = "cpu",
) -> float:
    """Character-level accuracy via autoregressive generation."""
    model.eval()
    total = 0
    correct = 0

    for _ in range(n_samples):
        prompt, expected = make_sample_fn()
        n_predict = len(tok.encode(expected))
        if n_predict == 0:
            continue

        ids = torch.from_numpy(tok.encode(prompt)).long().unsqueeze(0).to(device)

        predicted = []
        for _ in range(n_predict):
            logits = model(ids)
            next_id = logits[0, -1].argmax().item()
            predicted.append(next_id)
            ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)

        expected_ids = tok.encode(expected)
        for j in range(min(len(predicted), len(expected_ids))):
            total += 1
            if predicted[j] == expected_ids[j]:
                correct += 1
        total += max(0, len(expected_ids) - len(predicted))

    return correct / max(total, 1)


@torch.no_grad()
def evaluate_packed(
    model: nn.Module,
    make_batch_fn: MakeBatchFn,
    tok: EfficientByteTokenizer,
    n_samples: int = 200,
    device: str = "cpu",
    docs_per_seq: int = 2,
) -> float:
    """Teacher-forced accuracy on packed (multi-document) sequences.

    Packs ``docs_per_seq`` documents per row, runs a single forward pass,
    then scores each document independently: for every supervised position
    the argmax prediction must match the target.

    Returns: fraction of correctly predicted supervised tokens.
    """
    model.eval()
    total = 0
    correct = 0
    remaining = n_samples

    while remaining > 0:
        bsz = min(remaining, 64)
        input_ids, targets = _pack_batch(make_batch_fn, tok, bsz, docs_per_seq)
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        logits = model(input_ids)  # (B, S, V)
        preds = logits.argmax(dim=-1)  # (B, S)

        mask = targets != -100
        total += int(mask.sum().item())
        correct += int((preds[mask] == targets[mask]).sum().item())
        remaining -= bsz * docs_per_seq  # each row has docs_per_seq documents

    return correct / max(total, 1)


@torch.no_grad()
def verify_causality(
    model: nn.Module,
    tok: EfficientByteTokenizer,
    device: str = "cpu",
    n_samples: int = 5,
    make_sample_fn: MakeSampleFn | None = None,
    seq_len: int | None = None,
    atol: float = 1e-3,
    label: str = "",
):
    """Verify that the model is strictly causal by comparing single-pass vs sequential logits.

    For each sample, runs the model once on the full sequence to get logits at every
    position, then runs the model T separate times on tokens[:1], tokens[:2], ...,
    tokens[:T]. The logits at position t must be identical in both cases — any
    difference means information is leaking from future tokens.

    Args:
        model: the model to test (must accept input_ids and return logits).
        tok: tokenizer.
        device: device string.
        n_samples: number of random sequences to test.
        make_sample_fn: if provided, generates (prompt, answer) pairs for input.
        seq_len: if make_sample_fn is None, use random token sequences of this length.
        atol: absolute tolerance for logit comparison.
        label: optional label for print output.

    Raises:
        AssertionError if any logit mismatch is found.
    """
    model.eval()
    prefix = f"  [{label}] " if label else "  "

    if seq_len is None and make_sample_fn is None:
        seq_len = 30

    overall_max = 0.0
    for i in range(n_samples):
        # Build input sequence
        if make_sample_fn is not None:
            prompt, answer = make_sample_fn()
            full_text = prompt + answer
            ids = torch.from_numpy(tok.encode(full_text)).long().unsqueeze(0).to(device)
        else:
            ids = torch.randint(0, tok.vocab_size, (1, seq_len), device=device)

        T = ids.size(1)
        if T < 2:
            continue

        # Single-pass: full sequence
        full_logits = model(ids)  # (1, T, V)

        # Sequential: run on tokens[:t] for t = 1..T, collect logit at last position
        max_diff = 0.0
        worst_pos = -1
        for t in range(1, T + 1):
            partial_logits = model(ids[:, :t])  # (1, t, V)
            seq_logit = partial_logits[0, t - 1]  # logit at position t-1
            full_logit = full_logits[0, t - 1]

            diff = (seq_logit - full_logit).abs().max().item()
            if diff > max_diff:
                max_diff = diff
                worst_pos = t - 1

        if max_diff > atol:
            raise AssertionError(
                f"{prefix}CAUSALITY VIOLATION in sample {i}: "
                f"max logit diff = {max_diff:.2e} at position {worst_pos} "
                f"(tolerance = {atol:.0e}, seq_len = {T})"
            )
        overall_max = max(overall_max, max_diff)

    print(
        f"{prefix}Causality check PASSED ({n_samples} samples, "
        f"max_diff={overall_max:.2e}, atol={atol:.0e})"
    )


@torch.no_grad()
def show_examples(
    model: nn.Module,
    make_sample_fn: MakeSampleFn,
    tok: EfficientByteTokenizer,
    device: str = "cpu",
    n: int = 5,
):
    """Print example predictions."""
    model.eval()
    for _ in range(n):
        prompt, expected = make_sample_fn()
        n_predict = len(tok.encode(expected))
        ids = torch.from_numpy(tok.encode(prompt)).long().unsqueeze(0).to(device)

        predicted = []
        for _ in range(n_predict):
            logits = model(ids)
            next_id = logits[0, -1].argmax().item()
            predicted.append(next_id)
            ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)

        pred_str = tok.decode_to_str(predicted)
        match = "OK" if pred_str == expected else "FAIL"
        print(f"    {match:4s}  expected={expected!r:20s}  predicted={pred_str!r}")
