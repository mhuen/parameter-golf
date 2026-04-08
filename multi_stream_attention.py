import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from enum import StrEnum
from dataclasses import dataclass

from modules import RMSNorm, CastedLinear, LearnableShift, make_linear
from multi_streams import (
    StreamType,
    StreamID,
    StreamConfig,
    MultiStreamConfig,
    CompressedView,
    CompressionType,
)


class MixingMode(StrEnum):
    ADDITIVE = "additive"
    GLU = "glu"


class MixingSource(StrEnum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    DYNAMIC_BOTTLENECK = "dynamic_bottleneck"


@dataclass
class StreamMixingConfig:
    qk_mode: MixingMode = MixingMode.GLU
    source: MixingSource = MixingSource.DYNAMIC_BOTTLENECK
    bottleneck_dim: int = 32


class CausalMultiStreamAttentionViaMixing(nn.Module):
    """Multi-stream self attention with learned cross-stream mixing.

    Each stream produces independent Q, K, V projections. Before attention,
    a learned mixing step combines per-stream projections so that each head
    can build its query from one stream combination and match keys from another.

    Mixing modes:
    - Additive: weighted sum of per-stream projections (soft OR patterns)
    - GLU: primary ⊙ σ(gate), where primary and gate are each weighted sums
      of per-stream projections (conjunctive AND patterns)

    Mixing source:
    - Static: learned parameter logits (nearly free, head specialization)
    - Dynamic: input-dependent mixing via linear projection
    - Dynamic bottleneck: input-dependent mixing via bottleneck MLP

    After mixing, standard causal SDPA is used — no custom masks needed.
    """

    def __init__(
        self,
        multi_head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        stream_config: MultiStreamConfig,
        mixing_config: StreamMixingConfig = StreamMixingConfig(),
        qk_gain_init: float = 0.0,
        k_shift: bool = True,
        linear_mode: str = "dense",
        linear_kwargs: dict | None = None,
    ):
        if multi_head_dim % num_heads != 0:
            raise ValueError("multi_head_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        super().__init__()
        self.multi_head_dim = multi_head_dim
        self.head_dim = multi_head_dim // num_heads
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        self.stream_config = stream_config
        self.mixing_config = mixing_config
        self.num_streams = len(stream_config.streams)
        self._stream_lookup: dict[StreamID, StreamConfig] = {
            s.name: s for s in stream_config.streams
        }
        _lkw = linear_kwargs or {}

        # --- Per-stream Q/K/V projections ---
        self.W_q = nn.ModuleDict(
            {
                stream.key: make_linear(
                    stream.dim,
                    self.num_heads * self.head_dim,
                    bias=False,
                    mode=linear_mode,
                    **_lkw,
                )
                for stream in self.stream_config.streams
            },
        )
        self.W_k = nn.ModuleDict(
            {
                stream.key: make_linear(
                    stream.dim,
                    self.num_kv_heads * self.head_dim,
                    bias=False,
                    mode=linear_mode,
                    **_lkw,
                )
                for stream in self.stream_config.streams
            }
        )
        self.W_v = nn.ModuleDict(
            {
                stream.key: make_linear(
                    stream.dim,
                    self.num_kv_heads * self.head_dim,
                    bias=False,
                    mode=linear_mode,
                    **_lkw,
                )
                for stream in self.stream_config.streams
            }
        )

        # --- Cross-stream mixing parameters ---
        S = self.num_streams
        H = self.num_heads
        H_kv = self.num_kv_heads
        is_glu = mixing_config.qk_mode == MixingMode.GLU

        # Number of mixing logits needed
        # GLU: Q needs H × 2S (primary + gate), K needs H_kv × 2S, V needs H_kv × S
        # Additive: Q needs H × S, K needs H_kv × S, V needs H_kv × S
        qk_mult = 2 if is_glu else 1
        self._q_mix_size = H * qk_mult * S
        self._k_mix_size = H_kv * qk_mult * S
        self._v_mix_size = H_kv * S
        total_mix_logits = self._q_mix_size + self._k_mix_size + self._v_mix_size

        if mixing_config.source == MixingSource.STATIC:
            self.mix_logits = nn.Parameter(torch.zeros(total_mix_logits))
        elif mixing_config.source == MixingSource.DYNAMIC:
            sum_stream_dims = sum(s.dim for s in stream_config.streams)
            self.mix_proj = CastedLinear(
                in_features=sum_stream_dims,
                out_features=total_mix_logits,
                bias=False,
            )
        elif mixing_config.source == MixingSource.DYNAMIC_BOTTLENECK:
            sum_stream_dims = sum(s.dim for s in stream_config.streams)
            self.mix_proj_down = CastedLinear(
                in_features=sum_stream_dims,
                out_features=mixing_config.bottleneck_dim,
                bias=False,
            )
            self.mix_proj_up = CastedLinear(
                in_features=mixing_config.bottleneck_dim,
                out_features=total_mix_logits,
                bias=False,
            )

        # --- QK normalization and gain after mixing ---
        self.q_norm = RMSNorm()
        self.k_norm = RMSNorm()
        self.q_gain = nn.Parameter(
            torch.full((num_heads,), qk_gain_init, dtype=torch.float32)
        )
        self.k_shift_mod = (
            LearnableShift(num_channels=num_kv_heads) if k_shift else None
        )

        # --- Output projection (gated write-back per writable stream) ---
        self.alpha_pre_sigmoid = nn.ParameterDict(
            {
                stream.key: nn.Parameter(torch.full((stream.dim,), -2.0))
                for stream in self.stream_config.streams
                if not stream.read_only
            }
        )
        self.W_o_value = nn.ModuleDict(
            {
                stream.key: make_linear(
                    self.num_heads * self.head_dim,
                    stream.dim,
                    bias=False,
                    mode=linear_mode,
                    **_lkw,
                )
                for stream in self.stream_config.streams
                if not stream.read_only
            }
        )
        self.W_o_gate = nn.ModuleDict(
            {
                stream.key: make_linear(
                    self.num_heads * self.head_dim,
                    stream.dim,
                    bias=False,
                    mode=linear_mode,
                    **_lkw,
                )
                for stream in self.stream_config.streams
                if not stream.read_only
            }
        )

    def _compute_mix_logits(self, input_streams: dict[StreamID, Tensor]) -> Tensor:
        """Compute raw mixing logits. Returns shape depends on source:
        - Static: [total_mix_logits]
        - Dynamic: [bsz, seqlen, total_mix_logits]
        """
        if self.mixing_config.source == MixingSource.STATIC:
            return self.mix_logits
        else:
            x_cat = torch.cat(
                [input_streams[s.name] for s in self.stream_config.streams],
                dim=-1,
            )  # [bsz, seqlen, sum_stream_dims]
            if self.mixing_config.source == MixingSource.DYNAMIC:
                return self.mix_proj(x_cat)
            else:  # DYNAMIC_BOTTLENECK
                return self.mix_proj_up(F.silu(self.mix_proj_down(x_cat)))

    def _apply_mixing(
        self,
        stacked: Tensor,
        mix_logits: Tensor,
        num_heads: int,
        is_qk: bool,
    ) -> Tensor:
        """Apply mixing weights to stacked per-stream projections.

        Args:
            stacked: [bsz, num_heads, seqlen, S, head_dim]
            mix_logits: raw logits for this component (q, k, or v)
                Static: [num_heads * mix_per_head * S]
                Dynamic: [bsz, seqlen, num_heads * mix_per_head * S]
            num_heads: H for Q, H_kv for K/V
            is_qk: whether to use GLU mode (if configured)

        Returns: [bsz, num_heads, seqlen, head_dim]
        """
        S = self.num_streams
        use_glu = is_qk and self.mixing_config.qk_mode == MixingMode.GLU

        if use_glu:
            # Reshape logits to [(...), num_heads, 2, S] then softmax over S
            if mix_logits.dim() == 1:
                # Static: [num_heads * 2 * S] -> [1, num_heads, 1, 2, S]
                w = mix_logits.view(num_heads, 2, S)
                w = F.softmax(w, dim=-1)
                w = w.unsqueeze(0).unsqueeze(2)  # [1, H, 1, 2, S]
            else:
                # Dynamic: [bsz, seqlen, num_heads * 2 * S] -> [bsz, H, seqlen, 2, S]
                bsz, seqlen, _ = mix_logits.shape
                w = mix_logits.view(bsz, seqlen, num_heads, 2, S)
                w = F.softmax(w, dim=-1)
                w = w.permute(0, 2, 1, 3, 4)  # [bsz, H, seqlen, 2, S]

            # stacked: [bsz, H, seqlen, S, head_dim]
            # w: [..., H, ..., 2, S]
            w_primary = w[..., 0, :].unsqueeze(-1)  # [..., H, ..., S, 1]
            w_gate = w[..., 1, :].unsqueeze(-1)  # [..., H, ..., S, 1]

            primary = (stacked * w_primary).sum(dim=-2)  # [bsz, H, seqlen, head_dim]
            gate = (stacked * w_gate).sum(dim=-2)  # [bsz, H, seqlen, head_dim]
            return primary * torch.sigmoid(gate)
        else:
            # Additive: softmax over S, weighted sum
            if mix_logits.dim() == 1:
                # Static: [num_heads * S] -> [1, H, 1, S, 1]
                w = F.softmax(mix_logits.view(num_heads, S), dim=-1)
                w = w.unsqueeze(0).unsqueeze(2).unsqueeze(-1)
            else:
                # Dynamic: [bsz, seqlen, num_heads * S] -> [bsz, H, seqlen, S, 1]
                bsz, seqlen, _ = mix_logits.shape
                w = mix_logits.view(bsz, seqlen, num_heads, S)
                w = F.softmax(w, dim=-1)
                w = w.permute(0, 2, 1, 3).unsqueeze(-1)  # [bsz, H, seqlen, S, 1]

            return (stacked * w).sum(dim=-2)  # [bsz, H, seqlen, head_dim]

    def forward(
        self,
        input_streams: dict[StreamID, Tensor],
        skip_residual: bool = True,
    ) -> dict[StreamID, Tensor]:
        bsz, seqlen, _ = next(iter(input_streams.values())).shape

        # Validate input streams
        for stream_name, x in input_streams.items():
            assert x.shape[0] == bsz and x.shape[1] == seqlen, (
                f"Stream {stream_name} shape {x.shape} doesn't match batch/seq dims"
            )
            assert stream_name in self._stream_lookup, (
                f"Input stream {stream_name} not in stream configuration"
            )
            assert x.shape[2] == self._stream_lookup[stream_name].dim, (
                f"Stream {stream_name} dim {x.shape[2]}, expected {self._stream_lookup[stream_name].dim}"
            )

        # Step 1: Compute per-stream Q/K/V and stack
        q_list, k_list, v_list = [], [], []
        for stream in self.stream_config.streams:
            x = input_streams[stream.name]

            q = self.W_q[stream.key](x)  # [bsz, seqlen, H * head_dim]
            k = self.W_k[stream.key](x)  # [bsz, seqlen, H_kv * head_dim]
            v = self.W_v[stream.key](x)  # [bsz, seqlen, H_kv * head_dim]

            q_list.append(
                q.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
            )
            k_list.append(
                k.view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
            )
            v_list.append(
                v.view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
            )

        # Stack along stream dim: [bsz, H, seqlen, S, head_dim]
        q_stack = torch.stack(q_list, dim=-2)
        k_stack = torch.stack(k_list, dim=-2)
        v_stack = torch.stack(v_list, dim=-2)

        # Step 2: Compute mixing logits and split for Q, K, V
        raw_logits = self._compute_mix_logits(input_streams)

        if raw_logits.dim() == 1:
            q_logits = raw_logits[: self._q_mix_size]
            k_logits = raw_logits[
                self._q_mix_size : self._q_mix_size + self._k_mix_size
            ]
            v_logits = raw_logits[self._q_mix_size + self._k_mix_size :]
        else:
            q_logits = raw_logits[..., : self._q_mix_size]
            k_logits = raw_logits[
                ..., self._q_mix_size : self._q_mix_size + self._k_mix_size
            ]
            v_logits = raw_logits[..., self._q_mix_size + self._k_mix_size :]

        # Step 3: Apply mixing
        q_mixed = self._apply_mixing(q_stack, q_logits, self.num_heads, is_qk=True)
        k_mixed = self._apply_mixing(k_stack, k_logits, self.num_kv_heads, is_qk=True)
        v_mixed = self._apply_mixing(v_stack, v_logits, self.num_kv_heads, is_qk=False)

        # Step 4: QK normalization + per-head gain + optional K shift
        q_mixed = self.q_norm(q_mixed)
        k_mixed = self.k_norm(k_mixed)
        if self.k_shift_mod is not None:
            k_mixed = self.k_shift_mod(k_mixed)
        q_mixed = q_mixed * self.q_gain[None, :, None, None].to(q_mixed.dtype)

        # Step 5: Standard causal SDPA
        attn_out = F.scaled_dot_product_attention(
            q_mixed,
            k_mixed,
            v_mixed,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )  # [bsz, H, seqlen, head_dim]

        # Step 6: Output projection — gated write-back per writable stream
        attn_flat = attn_out.transpose(1, 2).reshape(
            bsz, seqlen, self.num_heads * self.head_dim
        )  # [bsz, seqlen, H * head_dim]

        output_streams: dict[StreamID, Tensor] = {}
        for stream in self.stream_config.streams:
            if stream.read_only:
                output_streams[stream.name] = input_streams[stream.name]
                continue

            value = self.W_o_value[stream.key](attn_flat)
            gate = torch.sigmoid(self.W_o_gate[stream.key](attn_flat))
            update = value * gate

            if skip_residual:
                output_streams[stream.name] = update
            else:
                alpha = torch.sigmoid(self.alpha_pre_sigmoid[stream.key])
                output_streams[stream.name] = (1 - alpha) * input_streams[
                    stream.name
                ] + alpha * update

        return output_streams


class CausalMultiStreamAttention(nn.Module):
    """Standard attention baseline operating on multi-stream interface.

    Concatenates all streams into a single vector for Q/K/V projection
    (full cross-stream expressiveness), then writes back only to writable
    streams via gated output projections.
    """

    def __init__(
        self,
        multi_head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        stream_config: MultiStreamConfig,
        qk_gain_init: float = 0.0,
        k_shift: bool = True,
        linear_mode: str = "dense",
        linear_kwargs: dict | None = None,
    ):
        if multi_head_dim % num_heads != 0:
            raise ValueError("multi_head_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        super().__init__()
        self.multi_head_dim = multi_head_dim
        self.head_dim = multi_head_dim // num_heads
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.stream_config = stream_config
        self._stream_lookup: dict[StreamID, StreamConfig] = {
            s.name: s for s in stream_config.streams
        }
        _lkw = linear_kwargs or {}

        sum_stream_dims = sum(s.dim for s in stream_config.streams)

        self.W_q = make_linear(
            sum_stream_dims,
            num_heads * self.head_dim,
            bias=False,
            mode=linear_mode,
            **_lkw,
        )
        self.W_k = make_linear(
            sum_stream_dims,
            num_kv_heads * self.head_dim,
            bias=False,
            mode=linear_mode,
            **_lkw,
        )
        self.W_v = make_linear(
            sum_stream_dims,
            num_kv_heads * self.head_dim,
            bias=False,
            mode=linear_mode,
            **_lkw,
        )

        self.q_norm = RMSNorm()
        self.k_norm = RMSNorm()
        self.q_gain = nn.Parameter(
            torch.full((num_heads,), qk_gain_init, dtype=torch.float32)
        )
        self.k_shift_mod = (
            LearnableShift(num_channels=num_kv_heads) if k_shift else None
        )

        self.alpha_pre_sigmoid = nn.ParameterDict(
            {
                stream.key: nn.Parameter(torch.full((stream.dim,), -2.0))
                for stream in stream_config.streams
                if not stream.read_only
            }
        )
        self.W_o_value = nn.ModuleDict(
            {
                stream.key: make_linear(
                    num_heads * self.head_dim,
                    stream.dim,
                    bias=False,
                    mode=linear_mode,
                    **_lkw,
                )
                for stream in stream_config.streams
                if not stream.read_only
            }
        )
        self.W_o_gate = nn.ModuleDict(
            {
                stream.key: make_linear(
                    num_heads * self.head_dim,
                    stream.dim,
                    bias=False,
                    mode=linear_mode,
                    **_lkw,
                )
                for stream in stream_config.streams
                if not stream.read_only
            }
        )

    def forward(
        self,
        input_streams: dict[StreamID, Tensor],
        skip_residual: bool = True,
    ) -> dict[StreamID, Tensor]:
        bsz, seqlen, _ = next(iter(input_streams.values())).shape

        # Validate input streams
        for stream_name, x in input_streams.items():
            assert x.shape[0] == bsz and x.shape[1] == seqlen, (
                f"Stream {stream_name} shape {x.shape} doesn't match batch/seq dims"
            )
            assert stream_name in self._stream_lookup, (
                f"Input stream {stream_name} not in stream configuration"
            )
            assert x.shape[2] == self._stream_lookup[stream_name].dim, (
                f"Stream {stream_name} dim {x.shape[2]}, expected {self._stream_lookup[stream_name].dim}"
            )

        # Concat all streams
        x = torch.cat(
            [input_streams[s.name] for s in self.stream_config.streams], dim=-1
        )  # [bsz, seqlen, sum_stream_dims]

        # Q/K/V projections
        q = self.W_q(x).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = (
            self.W_k(x)
            .view(bsz, seqlen, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.W_v(x)
            .view(bsz, seqlen, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.k_shift_mod is not None:
            k = self.k_shift_mod(k)
        q = q * self.q_gain[None, :, None, None].to(q.dtype)

        # Standard causal SDPA
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )  # [bsz, H, seqlen, head_dim]

        attn_flat = attn_out.transpose(1, 2).reshape(
            bsz, seqlen, self.num_heads * self.head_dim
        )

        # Gated write-back per writable stream
        output_streams: dict[StreamID, Tensor] = {}
        for stream in self.stream_config.streams:
            if stream.read_only:
                output_streams[stream.name] = input_streams[stream.name]
                continue

            value = self.W_o_value[stream.key](attn_flat)
            gate = torch.sigmoid(self.W_o_gate[stream.key](attn_flat))
            update = value * gate

            if skip_residual:
                output_streams[stream.name] = update
            else:
                alpha = torch.sigmoid(self.alpha_pre_sigmoid[stream.key])
                output_streams[stream.name] = (1 - alpha) * input_streams[
                    stream.name
                ] + alpha * update

        return output_streams


class MultiStreamMLP(nn.Module):
    """MLP that reads all streams but only writes to writable streams.

    Shared up-projection from concatenated streams into a hidden latent,
    then per-stream output projections. Optionally gated output per stream.

    Activation: leaky_relu(x).square() (squared activation, self-gating).
    """

    def __init__(
        self,
        stream_config: MultiStreamConfig,
        hidden_dim: int,
        gated_output: bool = True,
        leaky_relu_slope: float = 0.5,
        linear_mode: str = "dense",
        linear_kwargs: dict | None = None,
    ):
        super().__init__()
        self.stream_config = stream_config
        self.gated_output = gated_output
        self.leaky_relu_slope = leaky_relu_slope
        _lkw = linear_kwargs or {}

        sum_stream_dims = sum(s.dim for s in stream_config.streams)
        self.fc_up = make_linear(
            sum_stream_dims,
            hidden_dim,
            bias=False,
            mode=linear_mode,
            **_lkw,
        )

        self.proj_value = nn.ModuleDict(
            {
                s.key: make_linear(
                    hidden_dim,
                    s.dim,
                    bias=False,
                    mode=linear_mode,
                    **_lkw,
                )
                for s in stream_config.streams
                if not s.read_only
            }
        )
        if gated_output:
            self.proj_gate = nn.ModuleDict(
                {
                    s.key: make_linear(
                        hidden_dim,
                        s.dim,
                        bias=False,
                        mode=linear_mode,
                        **_lkw,
                    )
                    for s in stream_config.streams
                    if not s.read_only
                }
            )

    def forward(self, input_streams: dict[StreamID, Tensor]) -> dict[StreamID, Tensor]:
        """Returns update deltas for writable streams only."""
        x = torch.cat(
            [input_streams[s.name] for s in self.stream_config.streams], dim=-1
        )
        h = F.leaky_relu(self.fc_up(x), negative_slope=self.leaky_relu_slope)
        h = h.square()

        output: dict[StreamID, Tensor] = {}
        for s in self.stream_config.streams:
            if s.read_only:
                continue
            if self.gated_output:
                value = self.proj_value[s.key](h)
                gate = torch.sigmoid(self.proj_gate[s.key](h))
                output[s.name] = value * gate
            else:
                output[s.name] = self.proj_value[s.key](h)
        return output


class CausalArithmeticMultiStreamAttention(nn.Module):
    """Attention-based arithmetic module operating on compressed number streams.

    Uses the same ``MultiStreamConfig`` interface as ``CausalMultiStreamAttention``:
    reads all streams, writes to writable streams (primarily logit).

    Architecture:
    1. Pre-computes arithmetic results for ALL pairs (i, j) of detected
       numbers and encodes them as next-digit one-hot sequences (non-diff).
    2. Builds pair keys from compressed structural features of both numbers.
    3. Attention from each position selects which pair's result to use.
    4. Separate op selector softmax picks the operation.
    5. Hardcoded scatter maps the selected digit → vocab_size logit delta.
    6. Gated write-back to writable streams.

    Gradients flow through pair attention weights, op selector, and gate.
    The pre-computed digit bank is a constant (non-differentiable).
    """

    NEXT_DIGIT_VOCAB = 12
    DIGIT_IDX_DOT = 10
    DIGIT_IDX_END = 11
    _MAX_RESULT_LEN = 20  # max digits in a result string

    def __init__(
        self,
        stream_config: MultiStreamConfig,
        tok,  # EfficientByteTokenizer
        n_max: int = 16,
        compressed_stream_ids: list[StreamID] | None = None,
        d_head: int = 16,
        n_ops: int = 5,  # add, sub, mul, div, mod
        eps: float = 1e-8,
    ):
        super().__init__()
        self.stream_config = stream_config
        self.d_head = d_head
        self.n_ops = n_ops
        self.n_max = n_max
        self.eps = eps

        # Pre-compute pair indices (i < j)
        pair_i, pair_j = [], []
        for i in range(n_max):
            for j in range(i + 1, n_max):
                pair_i.append(i)
                pair_j.append(j)
        self.register_buffer("pair_i", torch.tensor(pair_i, dtype=torch.long))
        self.register_buffer("pair_j", torch.tensor(pair_j, dtype=torch.long))
        self._n_pairs = len(pair_i)

        # Which streams are in the compressed view (for K projection)
        # Default: all streams. Read-only ones come pre-gathered from the
        # builder; writable ones are re-gathered each forward call so they
        # reflect the current (post-attention) state.
        if compressed_stream_ids is None:
            self._compressed_ids = [s.name for s in stream_config.streams]
        else:
            self._compressed_ids = list(compressed_stream_ids)
        self._compressed_id_set = set(self._compressed_ids)

        # Split into read-only (pre-gathered) and writable (re-gather in forward)
        self._readonly_compressed_ids = [
            s.name for s in stream_config.streams
            if s.name in self._compressed_id_set and s.read_only
        ]
        self._writable_compressed_ids = [
            s.name for s in stream_config.streams
            if s.name in self._compressed_id_set and not s.read_only
        ]

        sum_stream_dims = sum(s.dim for s in stream_config.streams)
        sum_compressed_dims = sum(
            s.dim for s in stream_config.streams if s.name in self._compressed_id_set
        )

        # Pair key: separate projections for number_a and number_b features
        self.W_ka = nn.Linear(sum_compressed_dims, d_head, bias=False)
        self.W_kb = nn.Linear(sum_compressed_dims, d_head, bias=False)
        # Query from full-length streams
        self.W_q = nn.Linear(sum_stream_dims, d_head, bias=True)
        # Operation selector from full-length streams
        self.W_op = nn.Linear(sum_stream_dims, n_ops, bias=True)
        # Per writable stream gate (init near 0 → sigmoid ≈ 0.12)
        self.gates = nn.ParameterDict(
            {
                s.key: nn.Parameter(torch.tensor(-2.0))
                for s in stream_config.streams
                if not s.read_only
            }
        )

        # Hardcoded scatter: {0-9, '.', END} → token IDs
        digit_to_tid = torch.zeros(self.NEXT_DIGIT_VOCAB, dtype=torch.long)
        for tid in range(tok.vocab_size):
            info = tok.token_info(tid)
            if info is None:
                continue
            bv = info.byte_value
            if 0x30 <= bv <= 0x39:
                digit_to_tid[bv - 0x30] = tid
            elif bv == 0x2E:
                digit_to_tid[self.DIGIT_IDX_DOT] = tid
        self.register_buffer("digit_to_tid", digit_to_tid)

        # '-' token for sign
        minus_tid = 0
        for tid in range(tok.vocab_size):
            info = tok.token_info(tid)
            if info is not None and info.byte_value == 0x2D:
                minus_tid = tid
                break
        self.register_buffer("minus_tid", torch.tensor(minus_tid, dtype=torch.long))

        # Logit stream info
        logit_cfg = next(
            s for s in stream_config.streams if s.name == StreamID(StreamType.LOGIT)
        )
        self._logit_dim = logit_cfg.dim
        self._logit_key = logit_cfg.key

    @staticmethod
    def _result_to_string(value: float) -> str:
        av = abs(value)
        if av == int(av) and av < 1e15:
            return str(int(av))
        s = f"{av}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s

    def _build_digit_bank(self, values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Pre-compute digit sequences and signs for all pairs × ops.

        Args:
            values: (B, N) number values.
            mask: (B, N) valid mask.

        Returns:
            digit_seqs: (B, P, n_ops, max_len) int tensor of digit class indices.
            signs: (B, P, n_ops) float tensor (+1, -1, or 0).
        """
        B = values.shape[0]
        P = self._n_pairs
        n_ops = self.n_ops
        max_len = self._MAX_RESULT_LEN
        eps = self.eps

        digit_seqs = torch.full(
            (B, P, n_ops, max_len),
            self.DIGIT_IDX_END,
            dtype=torch.long,
            device=values.device,
        )
        signs = torch.zeros(B, P, n_ops, device=values.device)

        vals_cpu = values.detach().cpu()
        mask_cpu = mask.detach().cpu()
        pi = self.pair_i.cpu()
        pj = self.pair_j.cpu()

        for b in range(B):
            for p in range(P):
                i, j = pi[p].item(), pj[p].item()
                if not mask_cpu[b, i] or not mask_cpu[b, j]:
                    continue
                a = vals_cpu[b, i].item()
                bv = vals_cpu[b, j].item()
                b_abs = abs(bv) + eps
                a_abs = abs(a) + eps

                op_results = [
                    a + bv,
                    a - bv,
                    a * bv,
                    a / b_abs,
                    math.fmod(a, b_abs),
                ]
                for o in range(n_ops):
                    val = op_results[o]
                    # Sign
                    if val > 0:
                        signs[b, p, o] = 1.0
                    elif val < 0:
                        signs[b, p, o] = -1.0

                    s = self._result_to_string(val)
                    for c in range(min(len(s), max_len)):
                        ch = s[c]
                        if ch == ".":
                            digit_seqs[b, p, o, c] = self.DIGIT_IDX_DOT
                        elif ch.isdigit():
                            digit_seqs[b, p, o, c] = int(ch)
                        # else stays END

        return digit_seqs, signs

    def forward(
        self,
        input_streams: dict[StreamID, Tensor],
        compressed_streams: dict[StreamID, Tensor],
        view: CompressedView,
    ) -> dict[StreamID, Tensor]:
        """Compute arithmetic attention and return update deltas.

        Args:
            input_streams: full-length streams (B, S, d).
            compressed_streams: compressed streams (B, N, d) at number positions.
            view: CompressedView with positions, mask, metadata (values, active_len).

        Returns:
            Update deltas for writable streams.
        """
        bsz = next(iter(input_streams.values())).shape[0]
        seqlen = next(iter(input_streams.values())).shape[1]
        device = next(iter(input_streams.values())).device
        dtype = next(iter(input_streams.values())).dtype

        positions = view.positions  # (B, N)
        num_mask = view.mask  # (B, N)
        num_values = view.metadata["values"]  # (B, N)
        active_len = view.metadata["active_len"]  # (B, S) long
        P = self._n_pairs
        n_ops = self.n_ops

        # --- 1. Pre-compute digit bank (non-differentiable) ---
        digit_seqs, signs = self._build_digit_bank(num_values, num_mask)
        # digit_seqs: (B, P, n_ops, max_len)
        # signs: (B, P, n_ops)

        # --- 2. Build pair keys from compressed features ---
        # Read-only streams use pre-gathered compressed_streams from the builder;
        # writable streams are re-gathered here from live input_streams so they
        # reflect updates from earlier sub-layers (e.g., attention residual).
        comp_parts = [compressed_streams[sid] for sid in self._readonly_compressed_ids]
        if self._writable_compressed_ids:
            N = positions.shape[1]
            idx = positions.clamp(0, seqlen - 1).unsqueeze(-1)  # (B, N, 1)
            mask_f = num_mask.unsqueeze(-1).to(dtype=dtype)
            for sid in self._writable_compressed_ids:
                full = input_streams[sid]  # (B, S, d)
                idx_exp = idx.expand(-1, -1, full.shape[-1])  # (B, N, d)
                comp_parts.append(full.gather(1, idx_exp) * mask_f)
        x_comp = torch.cat(comp_parts, dim=-1)  # (B, N, d_comp)
        feat_a = x_comp[:, self.pair_i]  # (B, P, d_comp)
        feat_b = x_comp[:, self.pair_j]  # (B, P, d_comp)
        K_pair = self.W_ka(feat_a) + self.W_kb(feat_b)  # (B, P, d_head)

        # --- 3. Query from full streams ---
        x_full = torch.cat(
            [input_streams[s.name] for s in self.stream_config.streams], dim=-1
        )  # (B, S, sum_dims)
        Q = self.W_q(x_full)  # (B, S, d_head)

        # --- 4. Pair attention with causal masking ---
        scale = 1.0 / math.sqrt(self.d_head)
        scores = torch.bmm(Q, K_pair.transpose(1, 2)) * scale  # (B, S, P)

        # Causal: both numbers in pair must be at positions <= t
        pos_a = positions[:, self.pair_i]  # (B, P)
        pos_b = positions[:, self.pair_j]  # (B, P)
        pair_pos_max = torch.maximum(pos_a, pos_b)  # (B, P)
        t_idx = torch.arange(seqlen, device=device).view(1, -1, 1)  # (1, S, 1)
        causal_ok = pair_pos_max.unsqueeze(1) <= t_idx  # (B, S, P)
        # Validity: both numbers must be valid
        pair_valid = num_mask[:, self.pair_i] & num_mask[:, self.pair_j]  # (B, P)
        attn_mask = causal_ok & pair_valid.unsqueeze(1)  # (B, S, P)

        scores = scores.masked_fill(~attn_mask, float("-inf"))
        w_pair = F.softmax(scores, dim=-1).nan_to_num(0.0)  # (B, S, P)

        # --- 5. Op selector ---
        op_weights = F.softmax(self.W_op(x_full), dim=-1)  # (B, S, n_ops)

        # --- 6. Look up digits at active_len ---
        # active_len: (B, S) → clamp and index into digit_seqs
        al = active_len.clamp(max=self._MAX_RESULT_LEN - 1)  # (B, S)
        # Gather: digit_seqs is (B, P, n_ops, max_len), index with al
        # Expand al to (B, 1, 1, S) then gather along last dim
        # Reshape for gather: (B, P, n_ops, S)
        al_exp = al.unsqueeze(1).unsqueeze(1).expand(bsz, P, n_ops, seqlen)
        digit_seqs_t = digit_seqs.gather(3, al_exp)  # (B, P, n_ops, S)
        digit_seqs_t = digit_seqs_t.permute(0, 3, 1, 2)  # (B, S, P, n_ops)

        # One-hot encode
        digit_oh = F.one_hot(digit_seqs_t, self.NEXT_DIGIT_VOCAB).to(
            dtype
        )  # (B, S, P, n_ops, 12)

        # --- 7. Weighted combination ---
        # w_pair: (B, S, P) × op_weights: (B, S, n_ops) → joint selection
        # combined = sum over P and n_ops of w_pair * op_weights * digit_oh
        combined = torch.einsum(
            "bsp, bso, bspod -> bsd",
            w_pair,
            op_weights,
            digit_oh,
        )  # (B, S, 12)

        # Sign: weighted sign for '-' token
        # signs: (B, P, n_ops) → expand to (B, S, P, n_ops) via pair and op weights
        weighted_sign = torch.einsum(
            "bsp, bso, bpo -> bs",
            w_pair,
            op_weights,
            signs.to(dtype),
        )  # (B, S)

        # --- 8. Scatter to logit stream ---
        logit_delta = torch.zeros(
            bsz, seqlen, self._logit_dim, device=device, dtype=dtype
        )
        tid_exp = self.digit_to_tid.view(1, 1, -1).expand(bsz, seqlen, -1)
        logit_delta.scatter_add_(2, tid_exp, combined)
        # Add sign to '-' token
        minus_idx = self.minus_tid.view(1, 1, 1).expand(bsz, seqlen, 1)
        logit_delta.scatter_add_(2, minus_idx, weighted_sign.unsqueeze(-1))

        # --- 9. Gated write-back ---
        output: dict[StreamID, Tensor] = {}
        for s in self.stream_config.streams:
            if s.read_only:
                continue
            gate = torch.sigmoid(self.gates[s.key])
            if s.key == self._logit_key:
                output[s.name] = gate * logit_delta
            else:
                output[s.name] = torch.zeros_like(input_streams[s.name])

        return output


class MultiStreamBlock(nn.Module):
    """Multi-stream transformer block: pre-norm → attention → residual → pre-norm → MLP → residual.

    Norms are applied to all streams; residual updates only to writable streams.
    Each sub-layer uses independent per-dimension α (update scale) and β (residual
    scale): output = σ(β)*x + σ(α)*update. This allows the model to learn additive
    (β≈1), interpolating (α+β≈1), or full-replacement (α≈1, β≈0) behavior per
    dimension per stream.

    Optional ``arith_attn``: if set, a CausalArithmeticMultiStreamAttention
    module is applied between attention and MLP using compressed number streams.
    """

    def __init__(
        self,
        multi_head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        stream_config: MultiStreamConfig,
        mixing_config: StreamMixingConfig | None = None,
        mlp_hidden_dim: int | None = None,
        gated_mlp_output: bool = True,
        leaky_relu_slope: float = 0.5,
        qk_gain_init: float = 0.0,
        linear_mode: str = "dense",
        linear_kwargs: dict | None = None,
        k_shift: bool = True,
        arith_attn: CausalArithmeticMultiStreamAttention | None = None,
    ):
        super().__init__()
        self.stream_config = stream_config

        # Per-stream norms (weight-free RMSNorm)
        self.attn_norms = nn.ModuleDict(
            {s.key: RMSNorm() for s in stream_config.streams}
        )
        self.mlp_norms = nn.ModuleDict(
            {s.key: RMSNorm() for s in stream_config.streams}
        )

        # Attention
        attn_kwargs = dict(
            multi_head_dim=multi_head_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            stream_config=stream_config,
            qk_gain_init=qk_gain_init,
            linear_mode=linear_mode,
            linear_kwargs=linear_kwargs,
            k_shift=k_shift,
        )
        if mixing_config is not None:
            self.attn = CausalMultiStreamAttentionViaMixing(
                **attn_kwargs, mixing_config=mixing_config
            )
        else:
            self.attn = CausalMultiStreamAttention(**attn_kwargs)

        # MLP
        writable_dim = sum(s.dim for s in stream_config.streams if not s.read_only)
        if mlp_hidden_dim is None:
            mlp_hidden_dim = 4 * writable_dim
        self.mlp = MultiStreamMLP(
            stream_config=stream_config,
            hidden_dim=mlp_hidden_dim,
            gated_output=gated_mlp_output,
            leaky_relu_slope=leaky_relu_slope,
            linear_mode=linear_mode,
            linear_kwargs=linear_kwargs,
        )

        # Per-stream, per-dimension independent α (update scale) and β (residual scale)
        # output = σ(β) * x + σ(α) * update
        # Init: α=-2 (σ≈0.12, small update), β=2 (σ≈0.88, mostly keep)
        self.attn_alpha = nn.ParameterDict(
            {
                s.key: nn.Parameter(torch.full((s.dim,), -2.0))
                for s in stream_config.streams
                if not s.read_only
            }
        )
        self.attn_beta = nn.ParameterDict(
            {
                s.key: nn.Parameter(torch.full((s.dim,), 2.0))
                for s in stream_config.streams
                if not s.read_only
            }
        )
        self.mlp_alpha = nn.ParameterDict(
            {
                s.key: nn.Parameter(torch.full((s.dim,), -2.0))
                for s in stream_config.streams
                if not s.read_only
            }
        )
        self.mlp_beta = nn.ParameterDict(
            {
                s.key: nn.Parameter(torch.full((s.dim,), 2.0))
                for s in stream_config.streams
                if not s.read_only
            }
        )

        # Optional arithmetic attention (between attention and MLP)
        self.arith_attn = arith_attn
        if arith_attn is not None:
            self.arith_alpha = nn.ParameterDict(
                {
                    s.key: nn.Parameter(torch.full((s.dim,), -2.0))
                    for s in stream_config.streams
                    if not s.read_only
                }
            )
            self.arith_beta = nn.ParameterDict(
                {
                    s.key: nn.Parameter(torch.full((s.dim,), 2.0))
                    for s in stream_config.streams
                    if not s.read_only
                }
            )

    def forward(
        self,
        input_streams: dict[StreamID, Tensor],
        compressed: dict[str, dict[StreamID, Tensor]] | None = None,
        views: dict[str, CompressedView] | None = None,
    ) -> dict[StreamID, Tensor]:
        # --- Attention sub-layer ---
        normed = {
            s.name: self.attn_norms[s.key](input_streams[s.name])
            for s in self.stream_config.streams
        }
        attn_out = self.attn(normed, skip_residual=True)

        x: dict[StreamID, Tensor] = {}
        for s in self.stream_config.streams:
            if s.read_only:
                x[s.name] = input_streams[s.name]
            else:
                alpha = torch.sigmoid(self.attn_alpha[s.key])
                beta = torch.sigmoid(self.attn_beta[s.key])
                x[s.name] = beta * input_streams[s.name] + alpha * attn_out[s.name]

        # --- Optional arithmetic attention (between attention and MLP) ---
        if self.arith_attn is not None and compressed is not None and views is not None:
            num_compressed = compressed[CompressionType.NUMBER]
            num_view = views[CompressionType.NUMBER]
            arith_out = self.arith_attn(x, num_compressed, num_view)
            for s in self.stream_config.streams:
                if not s.read_only and s.name in arith_out:
                    alpha = torch.sigmoid(self.arith_alpha[s.key])
                    beta = torch.sigmoid(self.arith_beta[s.key])
                    x[s.name] = beta * x[s.name] + alpha * arith_out[s.name]

        # --- MLP sub-layer ---
        normed = {
            s.name: self.mlp_norms[s.key](x[s.name]) for s in self.stream_config.streams
        }
        mlp_out = self.mlp(normed)

        output: dict[StreamID, Tensor] = {}
        for s in self.stream_config.streams:
            if s.read_only:
                output[s.name] = x[s.name]
            else:
                alpha = torch.sigmoid(self.mlp_alpha[s.key])
                beta = torch.sigmoid(self.mlp_beta[s.key])
                output[s.name] = beta * x[s.name] + alpha * mlp_out[s.name]

        return output


if __name__ == "__main__":
    # Quick shape & backward sanity check
    stream_config = MultiStreamConfig(
        streams=[
            StreamConfig(StreamID(StreamType.LOGIT), 32),
            StreamConfig(StreamID(StreamType.CONTEXT), 48),
            StreamConfig(StreamID(StreamType.TOKENS), 48, read_only=True),
            StreamConfig(StreamID(StreamType.STRUCTURAL), 24, read_only=True),
        ]
    )
    bsz, seqlen = 2, 16
    kwargs = dict(
        multi_head_dim=64, num_heads=4, num_kv_heads=2, stream_config=stream_config
    )

    for qk_mode in MixingMode:
        for source in MixingSource:
            cfg = StreamMixingConfig(qk_mode=qk_mode, source=source, bottleneck_dim=16)
            model = CausalMultiStreamAttentionViaMixing(**kwargs, mixing_config=cfg)
            inp = {
                s.name: torch.randn(bsz, seqlen, s.dim) for s in stream_config.streams
            }
            out = model(inp)
            sum(v.sum() for v in out.values() if v.requires_grad).backward()
            n = sum(p.numel() for p in model.parameters())
            print(f"  {qk_mode:8s} + {source:20s}: OK  ({n:,} params)")

    model = CausalMultiStreamAttention(**kwargs)
    inp = {s.name: torch.randn(bsz, seqlen, s.dim) for s in stream_config.streams}
    out = model(inp)
    sum(v.sum() for v in out.values() if v.requires_grad).backward()
    n = sum(p.numel() for p in model.parameters())
    print(f"  {'standard':8s} + {'(baseline)':20s}: OK  ({n:,} params)")

    # --- MLP tests ---
    print(f"\n{'=' * 70}")
    print("MultiStreamMLP tests")
    print("=" * 70)

    for gated in [True, False]:
        mlp = MultiStreamMLP(
            stream_config=stream_config, hidden_dim=320, gated_output=gated
        )
        inp = {s.name: torch.randn(bsz, seqlen, s.dim) for s in stream_config.streams}
        out = mlp(inp)
        # MLP returns only writable stream deltas
        assert set(out.keys()) == {
            s.name for s in stream_config.streams if not s.read_only
        }
        loss = sum(v.sum() for v in out.values())
        loss.backward()
        n = sum(p.numel() for p in mlp.parameters())
        print(f"  gated={str(gated):5s}: OK  ({n:,} params)")

    # --- Block tests ---
    print(f"\n{'=' * 70}")
    print("MultiStreamBlock tests")
    print("=" * 70)

    for label, mix_cfg in [
        ("standard", None),
        (
            "additive_static",
            StreamMixingConfig(MixingMode.ADDITIVE, MixingSource.STATIC, 16),
        ),
        (
            "glu_dyn_bn",
            StreamMixingConfig(MixingMode.GLU, MixingSource.DYNAMIC_BOTTLENECK, 16),
        ),
    ]:
        for gated_mlp in [True, False]:
            block = MultiStreamBlock(
                **kwargs,
                mixing_config=mix_cfg,
                mlp_hidden_dim=320,
                gated_mlp_output=gated_mlp,
            )
            inp = {
                s.name: torch.randn(bsz, seqlen, s.dim) for s in stream_config.streams
            }
            out = block(inp)
            # Block returns all streams
            assert set(out.keys()) == {s.name for s in stream_config.streams}
            # Read-only streams unchanged
            for s in stream_config.streams:
                if s.read_only:
                    assert torch.equal(out[s.name], inp[s.name])
            loss = sum(v.sum() for v in out.values() if v.requires_grad)
            loss.backward()
            n = sum(p.numel() for p in block.parameters())
            tag = f"{label}/gated_mlp={gated_mlp}"
            print(f"  {tag:40s}: OK  ({n:,} params)")

    # Test k_shift integration
    for label, mix_cfg in [
        ("standard", None),
        (
            "glu_dyn_bn",
            StreamMixingConfig(MixingMode.GLU, MixingSource.DYNAMIC_BOTTLENECK, 16),
        ),
    ]:
        block = MultiStreamBlock(
            **kwargs,
            mixing_config=mix_cfg,
            mlp_hidden_dim=320,
            k_shift=True,
        )
        inp = {s.name: torch.randn(bsz, seqlen, s.dim) for s in stream_config.streams}
        out = block(inp)
        loss = sum(v.sum() for v in out.values() if v.requires_grad)
        loss.backward()
        n = sum(p.numel() for p in block.parameters())
        tag = f"{label}/k_shift=True"
        print(f"  {tag:40s}: OK  ({n:,} params)")
