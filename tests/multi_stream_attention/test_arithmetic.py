"""Test: simple arithmetic — addition and subtraction of small numbers.

Synthetic task:
    "3 + 7 = " → "10"
    "45 - 12 = " → "33"
    "8 - 15 = " → "-7"

Tests whether the model can leverage:
1. DigitComputeComponent (hardcoded arithmetic features from recent digit sequences)
2. CausalArithmeticMultiStreamAttention (attention-based pair selection over compressed numbers)

Compares multi-stream models with/without compute features vs TinyGPT baseline.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import random
import torch
from torch import Tensor

from efficient_byte_tokenizer import EfficientByteTokenizer
from byte_modules import NumberExtractor, ByteLogitHierarchy
from byte_stream_components import (
    ByteHashComponent,
    DiscreteEncoding,
    HashBoundary,
    DigitComputeComponent,
    PairwiseOp,
    DigitSequenceComponent,
)
from modules import RotationCodebook
from number_detection import DIGIT_IDX_DOT, DIGIT_IDX_END, NEXT_DIGIT_VOCAB
from multi_streams import (
    StreamType,
    StreamID,
    StreamDef,
    SinCosPositionComponent,
    CompressionType,
)
from multi_stream_attention import CausalArithmeticMultiStreamAttention
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

# OPS = ["+", "-", "*", "/", "%", "<", ">", "=="]
OPS = ["+", "-", "*", "/"]
MAX_A, MAX_B = 999, 999


def make_sample(ops: list[str] | None = None) -> tuple[str, str]:
    """Generate one arithmetic sample. Returns (prompt, answer)."""
    op = random.choice(ops or OPS)
    a = random.randint(0, MAX_A)
    # Avoid division/mod by zero
    b = random.randint(1 if op in ("/", "%") else 0, MAX_B)
    if op == "+":
        result = a + b
    elif op == "-":
        result = a - b
    elif op == "*":
        result = a * b
    elif op == "/":
        result = str(a / b)
        if "." in result:
            dot_pos = result.index(".")
            # Keep at most 6 decimal digits by truncation (not rounding, to avoid "off by 1 in the last digit" issues)
            result = result[: dot_pos + 1 + 6]
    elif op == "%":
        result = a % b
    elif op == "<":
        result = str(a < b)  # "True" or "False"
    elif op == ">":
        result = str(a > b)
    elif op == "==":
        result = str(a == b)
    prompt = f"{a} {op} {b} = "
    answer = str(result)
    return prompt, answer


def make_batch(tok: EfficientByteTokenizer, batch_size: int) -> tuple[Tensor, Tensor]:
    """Build a padded batch with supervision only on the answer characters."""
    all_ids = []
    all_targets = []

    for _ in range(batch_size):
        prompt, answer = make_sample()
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


# ---------------------------------------------------------------------------
# Oracle model (diagnostic — reads DigitComputeComponent output directly)
# ---------------------------------------------------------------------------


class OracleDigitComputeModel(torch.nn.Module):
    """Zero-param model that decodes DigitComputeComponent output as logits.

    Runs the component on input_ids, detects the operator in the prompt,
    reads the next-digit encoding at each answer position, and emits hard
    logits for the corresponding byte tokens.

    Supports both one-hot and rotation encodings via ``digit_encoding``.

    The component's next-digit encoding uses the active digit run length
    to index into the result string. At the '=' sign (active_len=0), it
    outputs the first digit; at the first answer digit (active_len=1),
    it outputs the second; etc. The sign is handled separately.
    """

    def __init__(
        self,
        tok: EfficientByteTokenizer,
        ops: set[PairwiseOp] | None = None,
        digit_encoding: DiscreteEncoding = DiscreteEncoding.ONEHOT,
    ):
        super().__init__()
        if ops is None:
            ops = {
                PairwiseOp.ADD,
                PairwiseOp.MUL,
                PairwiseOp.SUB,
                PairwiseOp.DIV,
            }
        self.tok = tok
        self.vocab_size = tok.vocab_size
        self._digit_encoding = DiscreteEncoding(digit_encoding)
        self.comp = DigitComputeComponent(
            tok, k=2, ops=ops, digit_encoding=digit_encoding
        )
        self._arith_dim = self.comp._arith_dim
        self._digit_dim = self.comp._digit_dim

        self._rotation_codebook: RotationCodebook | None = None
        if self._digit_encoding == DiscreteEncoding.ROTATION:
            self._rotation_codebook = RotationCodebook(NEXT_DIGIT_VOCAB)

        # Map task operator → component op index
        _char_to_pw = {
            "+": PairwiseOp.ADD,
            "-": PairwiseOp.SUB,
            "*": PairwiseOp.MUL,
            "/": PairwiseOp.DIV,
            "%": PairwiseOp.MOD,
        }
        arith_ops = self.comp._arith_ops
        self._op_byte_to_idx: dict[int, int] = {}
        for char, pw_op in _char_to_pw.items():
            if pw_op in arith_ops:
                self._op_byte_to_idx[ord(char)] = arith_ops.index(pw_op)

        self._eq_byte = ord("=")
        self._minus_byte = ord("-")

        # Build token_id → byte value, and byte → token_id lookups
        token_byte = torch.full((tok.vocab_size,), -1, dtype=torch.long)
        self._byte_to_tid: dict[int, int] = {}
        for tid in range(tok.vocab_size):
            info = tok.token_info(tid)
            if info is not None:
                token_byte[tid] = info.byte_value
                self._byte_to_tid[info.byte_value] = tid
        self.register_buffer("token_byte", token_byte)

        # Map digit index → byte value
        # Indices 0-9 → ASCII '0'-'9', 10 → '.', 11 → END (no token)
        self._digit_idx_to_byte: dict[int, int] = {}
        for d in range(10):
            self._digit_idx_to_byte[d] = ord("0") + d
        self._digit_idx_to_byte[DIGIT_IDX_DOT] = ord(".")
        # DIGIT_IDX_END has no byte mapping (signals end of number)

        # Dummy parameter so optimizers don't choke on empty param list
        self.learnable_gate = torch.nn.Parameter(torch.ones(1))

    def _read_next_digit(self, feats: Tensor, op_idx: int) -> tuple[float, int]:
        """Read sign and next-digit index from component features at one position.

        Returns (sign_val, digit_idx) where digit_idx is 0-9, DOT, or END.
        """
        ad = self._arith_dim
        dd = self._digit_dim
        off = op_idx * ad
        sign_val = feats[off].item()
        encoded = feats[off + 1 : off + 1 + dd]
        if self._digit_encoding == DiscreteEncoding.ONEHOT:
            digit_idx = encoded.argmax().item()
        else:
            digit_idx = self._rotation_codebook.decode(encoded.unsqueeze(0)).item()
        return sign_val, digit_idx

    def forward(self, input_ids: Tensor) -> Tensor:
        B, S = input_ids.shape
        V = self.vocab_size
        feats = self.comp(input_ids, torch.float32)  # (B, S, dim)
        logits = torch.zeros(B, S, V, device=input_ids.device)

        tb = self.token_byte[input_ids]  # (B, S) byte values

        for b in range(B):
            op_bv = None
            eq_pos = None
            for t in range(S):
                bv = tb[b, t].item()
                if bv == self._eq_byte:
                    eq_pos = t
                    break
                if bv in self._op_byte_to_idx:
                    op_bv = bv

            if op_bv is None or eq_pos is None:
                continue

            op_idx = self._op_byte_to_idx[op_bv]

            # Read sign from the '= ' position (space after =, active_len=0)
            sign_pos = min(eq_pos + 1, S - 1)
            sign_val, _ = self._read_next_digit(feats[b, sign_pos], op_idx)

            # Emit sign token if negative
            pos = eq_pos + 1  # logits[pos] predicts token at pos+1
            if sign_val < -0.5:
                minus_tid = self._byte_to_tid.get(self._minus_byte)
                if minus_tid is not None and pos < S:
                    logits[b, pos, minus_tid] = 100.0
                    pos += 1

            # Emit digits by reading the next-digit one-hot at each position
            # The component auto-tracks active_len, so feats at each answer
            # position already contain the correct next-digit for that index.
            while pos < S:
                _, digit_idx = self._read_next_digit(feats[b, pos], op_idx)
                if digit_idx == DIGIT_IDX_END:
                    break
                byte_val = self._digit_idx_to_byte.get(digit_idx)
                if byte_val is None:
                    break
                tid = self._byte_to_tid.get(byte_val)
                if tid is not None:
                    logits[b, pos, tid] = 100.0
                pos += 1

        return logits * self.learnable_gate


# ---------------------------------------------------------------------------
# Stream definitions
# ---------------------------------------------------------------------------


def make_stream_defs_basic(tok: EfficientByteTokenizer) -> list[StreamDef]:
    """Basic multi-stream: logit + tokens + structural hash (no digit compute)."""
    return [
        StreamDef(name=StreamID(StreamType.LOGIT), dim=tok.vocab_size),
        StreamDef(name=StreamID(StreamType.TOKENS), read_only=True, auto_onehot=True),
        StreamDef(
            name=StreamID(StreamType.STRUCTURAL),
            read_only=True,
            components=[
                SinCosPositionComponent(num_freqs=32),  # 12d
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


def make_stream_defs_digit_compute(tok: EfficientByteTokenizer) -> list[StreamDef]:
    """Multi-stream with DigitComputeComponent in structural stream."""
    return [
        StreamDef(name=StreamID(StreamType.LOGIT), dim=tok.vocab_size),
        StreamDef(name=StreamID(StreamType.TOKENS), read_only=True, auto_onehot=True),
        StreamDef(
            name=StreamID(StreamType.STRUCTURAL),
            read_only=True,
            components=[
                SinCosPositionComponent(num_freqs=32),  # 12d
                DigitSequenceComponent(tok=tok, id_freqs=5),
                # ByteHashComponent(
                #     tok,
                #     window=6,
                #     num_hashes=2,
                #     boundary=HashBoundary.WORD,
                #     track_hits=True,
                # ),
                DigitComputeComponent(
                    tok,
                    k=2,
                    ops={
                        PairwiseOp.ADD,
                        PairwiseOp.SUB,
                        PairwiseOp.MUL,
                        PairwiseOp.DIV,
                    },
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
    print(f"Device: {device}, vocab_size: {tok.vocab_size}")
    print(f"Task: {MAX_A}-digit + {MAX_B}-digit addition/subtraction\n")

    # --- Build models ---
    models: list[tuple[str, torch.nn.Module]] = []
    logit_hierarchy = ByteLogitHierarchy(
        vocab_size=tok.vocab_size, tok=tok, logit_softcap=30.0
    )
    # logit_hierarchy = None  # comment in to disable logit hierarchy and use flat logits instead

    # 1. TinyGPT baseline
    gpt_model = TinyGPT(
        vocab_size=tok.vocab_size,
        dim=64,
        num_layers=2,
        num_heads=4,
        mlp_mult=2,
    )
    models.append(("TinyGPT (dim=128, 2L, 4H)", gpt_model))

    # 2. Multi-stream basic (no digit features)
    ms_basic = MultiStreamTestModel(
        stream_defs=make_stream_defs_basic(tok),
        vocab_size=tok.vocab_size,
        num_heads=2,
        head_dim=16,
        num_layers=1,
        use_block=True,
        mlp_hidden_dim=16,
        logit_hierarchy=logit_hierarchy,  # comment in to use logit hierarchy
    )
    models.append(("MS basic (no digit)", ms_basic))

    # 3. Multi-stream with DigitComputeComponent
    ms_digit = MultiStreamTestModel(
        stream_defs=make_stream_defs_digit_compute(tok),
        vocab_size=tok.vocab_size,
        num_heads=2,
        head_dim=16,
        num_layers=1,
        use_block=True,
        mlp_hidden_dim=16,
        logit_hierarchy=logit_hierarchy,  # comment in to use logit hierarchy
    )
    models.append(("MS + DigitCompute", ms_digit))

    # 4. Multi-stream with CausalArithmeticMultiStreamAttention
    STRUCTURAL_ID = StreamID(StreamType.STRUCTURAL)
    TOKENS_ID = StreamID(StreamType.TOKENS)
    arith_stream_defs = make_stream_defs_basic(tok)
    compress_ids = [STRUCTURAL_ID, TOKENS_ID]
    extractor = NumberExtractor(tok, n_max=16)
    # Build a temporary builder to get stream_config for arith_attn
    from multi_streams import MultiStreamBuilder

    _cfg = MultiStreamBuilder(arith_stream_defs, vocab_size=tok.vocab_size).config
    arith_attn = CausalArithmeticMultiStreamAttention(
        stream_config=_cfg,
        tok=tok,
        n_max=16,
        compressed_stream_ids=compress_ids,
        d_head=16,
    )
    ms_arith = MultiStreamTestModel(
        stream_defs=arith_stream_defs,
        vocab_size=tok.vocab_size,
        num_heads=2,
        head_dim=16,
        num_layers=2,
        use_block=True,
        mlp_hidden_dim=16,
        arith_attn=arith_attn,
        compressions={CompressionType.NUMBER: extractor},
        compress_streams=compress_ids,
        logit_hierarchy=logit_hierarchy,  # comment in to use logit hierarchy
    )
    models.append(("MS + ArithAttn", ms_arith))

    # 5. MultiStreamGPT (full model)
    msgpt_model = MultiStreamGPTTestModel(
        tok=tok,
        vocab_size=tok.vocab_size,
        num_heads=2,
        num_kv_heads=2,
        num_layers=1,
        multi_head_dim=32,
        mlp_hidden_dim=16,
    )
    models.append(("MultiStreamGPT (2H, 1L)", msgpt_model))

    # 6. Oracle — directly decodes DigitComputeComponent output (one-hot)
    oracle = OracleDigitComputeModel(
        tok,
        ops={PairwiseOp.ADD, PairwiseOp.SUB, PairwiseOp.MUL, PairwiseOp.DIV},
    )
    models.append(("Oracle (onehot)", oracle))

    # 7. Oracle — rotation encoding
    oracle_rot = OracleDigitComputeModel(
        tok,
        ops={PairwiseOp.ADD, PairwiseOp.SUB, PairwiseOp.MUL, PairwiseOp.DIV},
        digit_encoding=DiscreteEncoding.ROTATION,
    )
    models.append(("Oracle (rotation)", oracle_rot))

    # --- Print param counts ---
    print("=" * 60)
    print("Model sizes")
    print("=" * 60)
    for name, model in models:
        count_params(model, name)
    print()

    # --- Training ---
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
    for name, model in reversed(models):
        print("=" * 60)
        print(name)
        print("=" * 60)
        model = maybe_compile(model, args.compile and "Oracle" not in name)

        if args.pack:

            def eval_fn(m, d):
                return evaluate_packed(m, make_batch, tok, n_samples=200, device=d)
        else:

            def eval_fn(m, d):
                return evaluate_autoregressive(
                    m, make_sample, tok, n_samples=200, device=d
                )

        if "Oracle" in name:
            train_kwargs_copy = train_kwargs.copy()
            train_kwargs_copy["steps"] = 1
            model = train_model(model, make_batch, **train_kwargs_copy, eval_fn=eval_fn)
        else:
            train_model(model, make_batch, **train_kwargs, eval_fn=eval_fn)

        overall_acc = evaluate_autoregressive(
            model, make_sample, tok, n_samples=500, device=device
        )
        print(f"\n  Single-doc accuracy: {overall_acc:.1%}")
        if args.pack:
            packed_acc = evaluate_packed(
                model,
                make_batch,
                tok,
                n_samples=500,
                device=device,
            )
            print(f"  Packed accuracy:     {packed_acc:.1%}")
        results[name] = overall_acc

        # Per-op evaluation
        for op in OPS:

            def make_sample_op(op=op):
                return make_sample(ops=[op])

            op_acc = evaluate_autoregressive(
                model, make_sample_op, tok, n_samples=200, device=device
            )
            print(f"  {op} accuracy: {op_acc:.1%}")

        print("\n  Examples:")
        show_examples(model, make_sample, tok, device, n=8)

        verify_causality(
            model, tok, device, make_sample_fn=make_sample, label=name, atol=5e-3
        )
        print()

    # --- Summary ---
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for name, acc in results.items():
        print(f"  {name:30s}  {acc:.1%}")
