"""Test: variable-based arithmetic — stress test for CausalArithmeticMultiStreamAttention.

Defines N variables with random values, then asks equations referencing them:

  Level 1 (pair selection + op):
      "a=42 b=7 c=15 d=99 ; a + c = " → "57"

  Level 2 (two-step, chained ops):
      "a=12 b=7 c=3 ; a + b - c = "   → "16"

  Level 3 (nested, two-step):
      "a=5 b=3 c=2 ; c * (a + b) = "  → "16"

Level 1 tests whether attention can select the correct pair from many
distractors and pick the right operation.

Levels 2-3 require compositional reasoning (two binary ops). These are
expected to challenge single-layer models and may need multiple layers.
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import random
import string
import torch
from torch import Tensor

from efficient_byte_tokenizer import EfficientByteTokenizer
from byte_modules import (
    ByteHashComponent,
    HashBoundary,
    DigitSequenceComponent,
    NumberExtractor,
)
from multi_streams import (
    StreamType,
    StreamID,
    StreamDef,
    SinCosPositionComponent,
    MultiStreamBuilder,
    CompressionType,
)
from multi_stream_attention import CausalArithmeticMultiStreamAttention
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

VAR_NAMES = list(string.ascii_lowercase[:10])  # a-j
MAX_VAL = 99
OPS_L1 = ["+", "-", "*"]


def _make_defs(names: list[str], vals: dict[str, int]) -> str:
    return " ".join(f"{v}={vals[v]}" for v in names)


def make_sample_l1(
    n_vars_range: tuple[int, int] = (3, 6),
    ops: list[str] | None = None,
) -> tuple[str, str]:
    """Level 1: single binary op on two selected variables."""
    ops = ops or OPS_L1
    n = random.randint(*n_vars_range)
    names = random.sample(VAR_NAMES, n)
    vals = {v: random.randint(1, MAX_VAL) for v in names}

    a_name, b_name = random.sample(names, 2)
    op = random.choice(ops)
    a, b = vals[a_name], vals[b_name]

    if op == "+":
        result = a + b
    elif op == "-":
        result = a - b
    elif op == "*":
        result = a * b
    else:
        raise ValueError(op)

    defs = _make_defs(names, vals)
    prompt = f"{defs} ; {a_name} {op} {b_name} = "
    return prompt, str(result)


def make_sample_l2(n_vars_range: tuple[int, int] = (3, 6)) -> tuple[str, str]:
    """Level 2: two chained ops, e.g. a + b - c."""
    n = random.randint(max(3, n_vars_range[0]), n_vars_range[1])
    names = random.sample(VAR_NAMES, n)
    vals = {v: random.randint(1, MAX_VAL) for v in names}

    a, b, c = random.sample(names, 3)
    va, vb, vc = vals[a], vals[b], vals[c]

    templates = [
        (f"{a} + {b} - {c}", va + vb - vc),
        (f"{a} - {b} + {c}", va - vb + vc),
        (f"{a} * {b} + {c}", va * vb + vc),
        (f"{a} * {b} - {c}", va * vb - vc),
    ]
    expr, result = random.choice(templates)

    defs = _make_defs(names, vals)
    prompt = f"{defs} ; {expr} = "
    return prompt, str(result)


def make_sample_l3(n_vars_range: tuple[int, int] = (3, 6)) -> tuple[str, str]:
    """Level 3: nested ops, e.g. c * (a + b)."""
    n = random.randint(max(3, n_vars_range[0]), n_vars_range[1])
    names = random.sample(VAR_NAMES, n)
    vals = {v: random.randint(1, MAX_VAL) for v in names}

    a, b, c = random.sample(names, 3)
    va, vb, vc = vals[a], vals[b], vals[c]

    templates = [
        (f"{c} * ({a} + {b})", vc * (va + vb)),
        (f"({a} + {b}) * {c}", (va + vb) * vc),
        (f"({a} - {b}) * {c}", (va - vb) * vc),
        (f"{c} * ({a} - {b})", vc * (va - vb)),
    ]
    expr, result = random.choice(templates)

    defs = _make_defs(names, vals)
    prompt = f"{defs} ; {expr} = "
    return prompt, str(result)


def make_sample_mixed() -> tuple[str, str]:
    """Mixed levels: 60% L1, 30% L2, 10% L3."""
    r = random.random()
    if r < 0.6:
        return make_sample_l1()
    elif r < 0.9:
        return make_sample_l2()
    else:
        return make_sample_l3()


def make_batch_from(
    make_fn,
    tok: EfficientByteTokenizer,
    batch_size: int,
) -> tuple[Tensor, Tensor]:
    """Build a padded batch with supervision only on the answer characters."""
    all_ids = []
    all_targets = []

    for _ in range(batch_size):
        prompt, answer = make_fn()
        prompt_ids = list(tok.encode(prompt))
        answer_ids = list(tok.encode(answer))
        full_ids = prompt_ids + answer_ids
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


# Curried batch functions for train_model
def make_batch_l1(tok, bs):
    return make_batch_from(make_sample_l1, tok, bs)


def make_batch_mixed(tok, bs):
    return make_batch_from(make_sample_mixed, tok, bs)


# ---------------------------------------------------------------------------
# Stream definitions
# ---------------------------------------------------------------------------

STRUCTURAL_ID = StreamID(StreamType.STRUCTURAL)
TOKENS_ID = StreamID(StreamType.TOKENS)


def make_stream_defs(tok: EfficientByteTokenizer) -> list[StreamDef]:
    return [
        StreamDef(name=StreamID(StreamType.LOGIT), dim=tok.vocab_size),
        StreamDef(name=StreamID(StreamType.TOKENS), read_only=True, auto_onehot=True),
        StreamDef(
            name=STRUCTURAL_ID,
            read_only=True,
            components=[
                SinCosPositionComponent(num_freqs=32),
                DigitSequenceComponent(tok=tok, id_freqs=5),
                ByteHashComponent(
                    tok,
                    window=6,
                    num_hashes=2,
                    boundary=HashBoundary.WORD,
                    track_hits=True,
                ),
            ],
        ),
    ]


def build_arith_model(
    tok: EfficientByteTokenizer,
    num_layers: int = 1,
    num_heads: int = 2,
    head_dim: int = 16,
    mlp_hidden_dim: int = 32,
    n_max: int = 16,
    d_head: int = 16,
) -> MultiStreamTestModel:
    """Build a MultiStreamTestModel with CausalArithmeticMultiStreamAttention."""
    stream_defs = make_stream_defs(tok)
    compress_ids = [STRUCTURAL_ID, TOKENS_ID]
    extractor = NumberExtractor(tok, n_max=n_max)
    _cfg = MultiStreamBuilder(stream_defs, vocab_size=tok.vocab_size).config
    arith_attn = CausalArithmeticMultiStreamAttention(
        stream_config=_cfg,
        tok=tok,
        n_max=n_max,
        compressed_stream_ids=compress_ids,
        d_head=d_head,
    )
    return MultiStreamTestModel(
        stream_defs=stream_defs,
        vocab_size=tok.vocab_size,
        num_heads=num_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        use_block=True,
        mlp_hidden_dim=mlp_hidden_dim,
        arith_attn=arith_attn,
        compressions={CompressionType.NUMBER: extractor},
        compress_streams=compress_ids,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = EfficientByteTokenizer()
    print(f"Device: {device}, vocab_size: {tok.vocab_size}")
    print(f"Task: variable-based arithmetic (vals 1..{MAX_VAL})")
    print(f"  L1: single op, 3-6 vars    e.g. 'a=42 b=7 c=15 ; a + c = '")
    print(f"  L2: two chained ops         e.g. 'a=12 b=7 c=3 ; a + b - c = '")
    print(f"  L3: nested ops              e.g. 'a=5 b=3 c=2 ; c * (a + b) = '")
    print()

    # Show a few samples
    print("Sample inputs:")
    for label, fn in [
        ("L1", make_sample_l1),
        ("L2", make_sample_l2),
        ("L3", make_sample_l3),
    ]:
        for _ in range(2):
            p, a = fn()
            print(f"  {label}: {p!r} → {a!r}")
    print()

    # --- Build models ---
    models: list[tuple[str, torch.nn.Module]] = []

    # 1. TinyGPT baseline
    models.append(
        (
            "TinyGPT (dim=64, 2L)",
            TinyGPT(
                vocab_size=tok.vocab_size, dim=64, num_layers=2, num_heads=4, mlp_mult=2
            ),
        )
    )

    # 2. MS basic (no arithmetic features)
    models.append(
        (
            "MS basic",
            MultiStreamTestModel(
                stream_defs=make_stream_defs(tok),
                vocab_size=tok.vocab_size,
                num_heads=2,
                head_dim=16,
                num_layers=1,
                use_block=True,
                mlp_hidden_dim=32,
            ),
        )
    )

    # # 3. MS + ArithAttn (1 layer)
    # models.append(("MS + ArithAttn (1L)", build_arith_model(tok, num_layers=1)))

    # # 4. MS + ArithAttn (2 layers)
    # models.append(("MS + ArithAttn (2L)", build_arith_model(tok, num_layers=2)))

    # 5. MS + ArithAttn (3 layers) — for compositional tasks
    models.append(("MS + ArithAttn (3L)", build_arith_model(tok, num_layers=3)))

    # --- Print param counts ---
    print("=" * 70)
    print("Model sizes")
    print("=" * 70)
    for name, model in models:
        count_params(model, name)
    print()

    # --- Training (on mixed levels) ---
    train_kwargs = dict(
        tok=tok,
        steps=3000,
        batch_size=64,
        lr=3e-2,
        eval_every=500,
        device=device,
    )

    results: dict[str, dict[str, float]] = {}
    for name, model in reversed(models):
        print("=" * 70)
        print(name)
        print("=" * 70)

        def eval_fn(m, d):
            return evaluate_autoregressive(
                m, make_sample_l1, tok, n_samples=200, device=d
            )

        train_model(model, make_batch_mixed, **train_kwargs, eval_fn=eval_fn)

        # --- Per-level evaluation ---
        level_results = {}
        for level_name, level_fn in [
            ("L1", make_sample_l1),
            ("L2", make_sample_l2),
            ("L3", make_sample_l3),
        ]:
            acc = evaluate_autoregressive(
                model,
                level_fn,
                tok,
                n_samples=300,
                device=device,
            )
            level_results[level_name] = acc
            print(f"  {level_name} accuracy: {acc:.1%}")

        # Per-op evaluation on L1
        print("  L1 per-op:")
        for op in OPS_L1:

            def make_op(op=op):
                return make_sample_l1(ops=[op])

            op_acc = evaluate_autoregressive(
                model,
                make_op,
                tok,
                n_samples=200,
                device=device,
            )
            print(f"    {op} accuracy: {op_acc:.1%}")

        results[name] = level_results

        print("\n  L1 examples:")
        show_examples(model, make_sample_l1, tok, device, n=5)
        print("  L2 examples:")
        show_examples(model, make_sample_l2, tok, device, n=5)
        print("  L3 examples:")
        show_examples(model, make_sample_l3, tok, device, n=3)

        verify_causality(
            model,
            tok,
            device,
            make_sample_fn=make_sample_mixed,
            label=name,
            atol=5e-3,
        )
        print()

    # --- Summary ---
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    header = f"  {'Model':30s}  {'L1':>6s}  {'L2':>6s}  {'L3':>6s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, levels in results.items():
        l1 = levels.get("L1", 0)
        l2 = levels.get("L2", 0)
        l3 = levels.get("L3", 0)
        print(f"  {name:30s}  {l1:5.1%}  {l2:5.1%}  {l3:5.1%}")
