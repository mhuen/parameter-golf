import time

import torch
import torch.nn.functional as F

from torch import Tensor, nn

from multi_streams import StreamType, StreamID
from multi_stream_attention import (
    CastedLinear,
    CausualMultiStreamAttentionViaMixing,
    CausualMultiStreamAttention,
    MixingMode,
    MixingSource,
    MultiStreamConfig,
    RMSNorm,
    StreamConfig,
    StreamMixingConfig,
)

LOGIT = StreamID(StreamType.LOGIT)
CONTEXT = StreamID(StreamType.CONTEXT)
TOKENS = StreamID(StreamType.TOKENS)
STRUCTURAL = StreamID(StreamType.STRUCTURAL)


class VanillaAttention(nn.Module):
    """Plain attention on concatenated streams with single W_o.

    Like CausalSelfAttention in train_gpt.py but without RoPE/doc_mask.
    Concatenates all streams, does Q/K/V projection, SDPA, single output
    projection back to full concat dim, then splits back to streams.
    """

    def __init__(
        self,
        multi_head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        stream_config: MultiStreamConfig,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = multi_head_dim // num_heads
        self.stream_config = stream_config

        total_dim = sum(s.dim for s in stream_config.streams)
        self.total_dim = total_dim
        kv_dim = num_kv_heads * self.head_dim

        self.c_q = CastedLinear(total_dim, multi_head_dim, bias=False)
        self.c_k = CastedLinear(total_dim, kv_dim, bias=False)
        self.c_v = CastedLinear(total_dim, kv_dim, bias=False)
        self.proj = CastedLinear(multi_head_dim, total_dim, bias=False)
        self.q_norm = RMSNorm()
        self.k_norm = RMSNorm()

        # Precompute split sizes for output
        self._split_sizes = [s.dim for s in stream_config.streams]

    def forward(self, input_streams: dict[StreamID, Tensor]) -> dict[StreamID, Tensor]:
        bsz, seqlen, _ = next(iter(input_streams.values())).shape

        x = torch.cat(
            [input_streams[s.name] for s in self.stream_config.streams], dim=-1
        )

        q = self.c_q(x).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = (
            self.c_k(x)
            .view(bsz, seqlen, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.c_v(x)
            .view(bsz, seqlen, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        q = self.q_norm(q)
        k = self.k_norm(k)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).reshape(bsz, seqlen, self.num_heads * self.head_dim)
        out = x + self.proj(y)  # residual connection

        # Split back to streams
        splits = out.split(self._split_sizes, dim=-1)
        return {s.name: split for s, split in zip(self.stream_config.streams, splits)}


def generate_structural_gated_copy(
    bsz: int, seqlen: int, stream_config: MultiStreamConfig, device: str = "cpu"
) -> tuple[dict[StreamID, torch.Tensor], torch.Tensor]:
    """Task 1: Structural-gated token retrieval.

    Target: for each position i, the logit output should be a projection of
    the token value from the causally-nearest position j < i whose structural
    embedding is most similar (highest dot product) to position i.

    Requires: Q from structural, K from structural, V from tokens → logit.
    """
    dims = {s.name: s.dim for s in stream_config.streams}
    struct_dim = dims[STRUCTURAL]
    token_dim = dims[TOKENS]
    logit_dim = dims[LOGIT]

    structural = F.normalize(
        torch.randn(bsz, seqlen, struct_dim, device=device), dim=-1
    )
    tokens = torch.randn(bsz, seqlen, token_dim, device=device)
    context = torch.randn(bsz, seqlen, dims[CONTEXT], device=device) * 0.1
    logit_in = torch.randn(bsz, seqlen, logit_dim, device=device) * 0.1

    scores = torch.bmm(structural, structural.transpose(1, 2))
    causal_mask = torch.tril(torch.ones(seqlen, seqlen, device=device))
    scores = scores * causal_mask + (-1e9) * (1 - causal_mask)
    attn_weights = F.softmax(scores, dim=-1)

    retrieved = torch.bmm(attn_weights, tokens)
    target = retrieved[..., :logit_dim]

    inputs = {
        LOGIT: logit_in,
        CONTEXT: context,
        TOKENS: tokens,
        STRUCTURAL: structural,
    }
    return inputs, target


def generate_cross_stream_retrieval(
    bsz: int, seqlen: int, stream_config: MultiStreamConfig, device: str = "cpu"
) -> tuple[dict[StreamID, torch.Tensor], torch.Tensor]:
    """Task 2: Cross-stream content-addressed retrieval.

    The context stream contains "query keys". Target: for each position i,
    retrieve token values from position j <= i with most similar context,
    write to logit stream.

    Requires: Q from context, K from context, V from tokens → logit.
    """
    dims = {s.name: s.dim for s in stream_config.streams}
    ctx_dim = dims[CONTEXT]
    token_dim = dims[TOKENS]
    logit_dim = dims[LOGIT]

    context = F.normalize(torch.randn(bsz, seqlen, ctx_dim, device=device), dim=-1)
    tokens = torch.randn(bsz, seqlen, token_dim, device=device)
    structural = torch.randn(bsz, seqlen, dims[STRUCTURAL], device=device) * 0.1
    logit_in = torch.randn(bsz, seqlen, logit_dim, device=device) * 0.1

    scores = torch.bmm(context, context.transpose(1, 2))
    causal_mask = torch.tril(torch.ones(seqlen, seqlen, device=device))
    scores = scores * causal_mask + (-1e9) * (1 - causal_mask)
    attn_weights = F.softmax(scores, dim=-1)

    retrieved = torch.bmm(attn_weights, tokens)
    target = retrieved[..., :logit_dim]

    inputs = {
        LOGIT: logit_in,
        CONTEXT: context,
        TOKENS: tokens,
        STRUCTURAL: structural,
    }
    return inputs, target


def train_and_eval(
    model: torch.nn.Module,
    task_fn,
    stream_config: MultiStreamConfig,
    num_steps: int = 300,
    bsz: int = 16,
    seqlen: int = 32,
    lr: float = 3e-4,
    device: str = "cpu",
) -> tuple[list[float], float]:
    """Train model on a synthetic task, return (loss_curve, elapsed_seconds)."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    t0 = time.perf_counter()
    for step in range(num_steps):
        inputs, target = task_fn(bsz, seqlen, stream_config, device)
        output = model(inputs)
        loss = F.mse_loss(output[LOGIT], target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    elapsed = time.perf_counter() - t0

    return losses, elapsed


if __name__ == "__main__":
    stream_dim_logit = 32
    stream_dim_context = 48
    stream_dim_tokens = 48
    stream_dim_structural = 24
    stream_config = MultiStreamConfig(
        streams=[
            StreamConfig(LOGIT, stream_dim_logit),
            StreamConfig(CONTEXT, stream_dim_context),
            StreamConfig(TOKENS, stream_dim_tokens, read_only=True),
            StreamConfig(STRUCTURAL, stream_dim_structural, read_only=True),
        ]
    )

    num_heads = 4
    num_kv_heads = 2
    head_dim = 16
    multi_head_dim = num_heads * head_dim

    # --- Shape & backward tests ---
    print("=" * 70)
    print("Shape & backward tests")
    print("=" * 70)

    bsz, seqlen = 2, 16
    for qk_mode in MixingMode:
        for source in MixingSource:
            mixing_config = StreamMixingConfig(
                qk_mode=qk_mode, source=source, bottleneck_dim=16
            )
            model = CausualMultiStreamAttentionViaMixing(
                multi_head_dim=multi_head_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                stream_config=stream_config,
                mixing_config=mixing_config,
            )
            input_streams = {
                s.name: torch.randn(bsz, seqlen, s.dim) for s in stream_config.streams
            }
            output = model(input_streams)
            loss = sum(v.sum() for v in output.values() if v.requires_grad)
            loss.backward()
            param_count = sum(p.numel() for p in model.parameters())
            print(f"  {qk_mode:8s} + {source:20s}: OK  ({param_count:,} params)")

    model_std = CausualMultiStreamAttention(
        multi_head_dim=multi_head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        stream_config=stream_config,
    )
    input_streams = {
        s.name: torch.randn(bsz, seqlen, s.dim) for s in stream_config.streams
    }
    output = model_std(input_streams)
    loss = sum(v.sum() for v in output.values() if v.requires_grad)
    loss.backward()
    param_count = sum(p.numel() for p in model_std.parameters())
    print(f"  {'standard':8s} + {'(baseline)':20s}: OK  ({param_count:,} params)")

    # --- Training experiments ---
    tasks = {
        "structural_gated_copy": generate_structural_gated_copy,
        "cross_stream_retrieval": generate_cross_stream_retrieval,
    }
    model_configs = {
        "vanilla": lambda: VanillaAttention(
            multi_head_dim=multi_head_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            stream_config=stream_config,
        ),
        "standard": lambda: CausualMultiStreamAttention(
            multi_head_dim=multi_head_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            stream_config=stream_config,
        ),
        "ms_additive_static": lambda: CausualMultiStreamAttentionViaMixing(
            multi_head_dim=multi_head_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            stream_config=stream_config,
            mixing_config=StreamMixingConfig(
                qk_mode=MixingMode.ADDITIVE,
                source=MixingSource.STATIC,
                bottleneck_dim=16,
            ),
        ),
        "ms_glu_static": lambda: CausualMultiStreamAttentionViaMixing(
            multi_head_dim=multi_head_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            stream_config=stream_config,
            mixing_config=StreamMixingConfig(
                qk_mode=MixingMode.GLU,
                source=MixingSource.STATIC,
                bottleneck_dim=16,
            ),
        ),
        "ms_glu_dyn_bn": lambda: CausualMultiStreamAttentionViaMixing(
            multi_head_dim=multi_head_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            stream_config=stream_config,
            mixing_config=StreamMixingConfig(
                qk_mode=MixingMode.GLU,
                source=MixingSource.DYNAMIC_BOTTLENECK,
                bottleneck_dim=16,
            ),
        ),
    }

    num_steps = 3000
    for task_name, task_fn in tasks.items():
        print(f"\n{'=' * 70}")
        print(f"Task: {task_name} ({num_steps} steps)")
        print(f"{'=' * 70}")
        print(
            f"  {'Model':<25s} {'Params':>8s} {'Init Loss':>10s} "
            f"{'Final Loss':>10s} {'Ratio':>8s} {'Time (s)':>9s} {'step/s':>8s}"
        )
        print(f"  {'-' * 80}")

        for model_name, model_fn in model_configs.items():
            torch.manual_seed(42)
            model = model_fn()
            param_count = sum(p.numel() for p in model.parameters())
            losses, elapsed = train_and_eval(
                model,
                task_fn,
                stream_config,
                num_steps=num_steps,
                bsz=16,
                seqlen=32,
                lr=3e-4,
            )
            init_loss = sum(losses[:5]) / 5
            final_loss = sum(losses[-5:]) / 5
            ratio = final_loss / init_loss
            steps_per_sec = num_steps / elapsed
            print(
                f"  {model_name:<25s} {param_count:>8,} {init_loss:>10.4f} "
                f"{final_loss:>10.4f} {ratio:>7.1%} {elapsed:>9.2f} {steps_per_sec:>8.1f}"
            )
