from calendar import c
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from enum import StrEnum
from dataclasses import dataclass

from byte_modules import ByteLogitHierarchy
from modules import RMSNorm, CastedLinear, GatedCausalConv, LearnableShift, make_linear
from multi_streams import (
    StreamType,
    StreamID,
    StreamConfig,
    MultiStreamConfig,
    CompressedView,
    CompressionType,
)


# ---------------------------------------------------------------------------
# Stream normalization helpers
# ---------------------------------------------------------------------------


def _build_stream_norms(streams: list[StreamConfig]) -> nn.ModuleDict:
    """Create per-stream norm instances for streams that have a norm_type."""
    return nn.ModuleDict({
        s.key: s.norm_type() for s in streams if s.norm_type is not None
    })


def _apply_stream_norms(
    norms: nn.ModuleDict,
    streams: list[StreamConfig],
    data: dict[StreamID, Tensor],
) -> dict[StreamID, Tensor]:
    """Return dict with norms applied to streams that have them."""
    return {
        s.name: norms[s.key](data[s.name]) if s.key in norms else data[s.name]
        for s in streams
    }


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
        skip_residual: bool = True,
        alpha_pre_sigmoid_init: float = -2.0,
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
        self.skip_residual = skip_residual

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
        if not skip_residual:
            self.alpha_pre_sigmoid = nn.ParameterDict(
                {
                    stream.key: nn.Parameter(
                        torch.full((stream.dim,), alpha_pre_sigmoid_init)
                    )
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
    ) -> dict[StreamID, Tensor]:
        bsz, seqlen, _ = next(iter(input_streams.values())).shape

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

            if self.skip_residual:
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
        skip_residual: bool = True,
        alpha_pre_sigmoid_init: float = -2.0,
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
        self.skip_residual = skip_residual
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

        if not skip_residual:
            self.alpha_pre_sigmoid = nn.ParameterDict(
                {
                    stream.key: nn.Parameter(
                        torch.full((stream.dim,), alpha_pre_sigmoid_init)
                    )
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
    ) -> dict[StreamID, Tensor]:
        bsz, seqlen, _ = next(iter(input_streams.values())).shape

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

            if self.skip_residual:
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
        logit_hierarchy: ByteLogitHierarchy | None = None,
        input_stream_ids: list[StreamID] | None = None,
        output_stream_ids: list[StreamID] | None = None,
    ):
        super().__init__()
        self.stream_config = stream_config
        self.gated_output = gated_output
        self.leaky_relu_slope = leaky_relu_slope
        self._logit_hierarchy = logit_hierarchy
        _lkw = linear_kwargs or {}

        # Determine which streams are concatenated as MLP input
        if input_stream_ids is not None:
            id_set = set(input_stream_ids)
            self._input_streams = [s for s in stream_config.streams if s.name in id_set]
        else:
            self._input_streams = list(stream_config.streams)

        # Determine which writable streams this MLP outputs to
        all_writable = [s for s in stream_config.streams if not s.read_only]
        if output_stream_ids is not None:
            out_set = set(output_stream_ids)
            self._output_streams = [s for s in all_writable if s.name in out_set]
        else:
            self._output_streams = all_writable

        # Identify the logit stream ID (if hierarchy is active)
        self._logit_sid: StreamID | None = None
        if logit_hierarchy is not None:
            for s in stream_config.streams:
                if s.name.type == StreamType.LOGIT and not s.read_only:
                    self._logit_sid = s.name
                    break

        sum_stream_dims = sum(s.dim for s in self._input_streams)
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
                    self._proj_dim(s),
                    bias=False,
                    mode=linear_mode,
                    **_lkw,
                )
                for s in self._output_streams
            }
        )
        if gated_output:
            self.proj_gate = nn.ModuleDict(
                {
                    s.key: make_linear(
                        hidden_dim,
                        self._proj_dim(s),
                        bias=False,
                        mode=linear_mode,
                        **_lkw,
                    )
                    for s in self._output_streams
                }
            )

    def _proj_dim(self, s: StreamConfig) -> int:
        """Output projection dim: total_slots for the logit stream when hierarchy is active."""
        if self._logit_hierarchy is not None and s.name == self._logit_sid:
            return self._logit_hierarchy.total_slots
        return s.dim

    def forward(self, input_streams: dict[StreamID, Tensor]) -> dict[StreamID, Tensor]:
        """Returns update deltas for output streams only."""
        x = torch.cat([input_streams[s.name] for s in self._input_streams], dim=-1)
        h = F.leaky_relu(self.fc_up(x), negative_slope=self.leaky_relu_slope)
        h = h.square()

        output: dict[StreamID, Tensor] = {}
        for s in self._output_streams:
            if self.gated_output:
                value = self.proj_value[s.key](h)
                gate = torch.sigmoid(self.proj_gate[s.key](h))
                value = value * gate
            else:
                value = self.proj_value[s.key](h)
            if self._logit_hierarchy is not None and s.name == self._logit_sid:
                chunks = value.split(self._logit_hierarchy.level_sizes, dim=-1)
                value = self._logit_hierarchy.assemble_logits(list(chunks))
            output[s.name] = value
        return output

    @property
    def output_streams(self) -> list[StreamConfig]:
        """Stream configs this MLP writes to."""
        return list(self._output_streams)


class MultiStreamCausalConv(nn.Module):
    """Causal conv that reads selected streams and directly outputs writable stream dims.

    Concatenates input streams, runs a ``GatedCausalConv`` whose output dimension
    equals the sum of writable stream dims, then slices the output to produce
    per-stream update deltas.  No separate projection layers — the conv itself
    maps from input space to output space.

    The caller (``MultiStreamCausalConvLayers`` or ``MultiStreamBlock``) owns
    the α/β residual write-back.
    """

    def __init__(
        self,
        stream_config: MultiStreamConfig,
        kernel_size: int = 4,
        conv_groups: int = 0,
        gated_conv: bool = True,
        logit_hierarchy: ByteLogitHierarchy | None = None,
        input_stream_ids: list[StreamID] | None = None,
        output_stream_ids: list[StreamID] | None = None,
        channel_shift: int = 0,
    ):
        super().__init__()
        self.stream_config = stream_config
        self._logit_hierarchy = logit_hierarchy

        # Determine which streams are concatenated as conv input
        if input_stream_ids is not None:
            id_set = set(input_stream_ids)
            self._input_streams = [s for s in stream_config.streams if s.name in id_set]
        else:
            self._input_streams = list(stream_config.streams)

        # Identify the logit stream ID (if hierarchy is active)
        self._logit_sid: StreamID | None = None
        if logit_hierarchy is not None:
            for s in stream_config.streams:
                if s.name.type == StreamType.LOGIT and not s.read_only:
                    self._logit_sid = s.name
                    break

        # Writable streams to output (filtered by output_stream_ids if given)
        all_writable = [s for s in stream_config.streams if not s.read_only]
        if output_stream_ids is not None:
            out_set = set(output_stream_ids)
            self._writable_streams = [s for s in all_writable if s.name in out_set]
        else:
            self._writable_streams = all_writable
        self._out_dims = [self._out_dim(s) for s in self._writable_streams]
        out_dim_total = sum(self._out_dims)

        # Pad output dim to next multiple of groups so grouped conv works
        input_dim = sum(s.dim for s in self._input_streams)
        effective_groups = input_dim if conv_groups <= 0 else conv_groups
        padded_out_dim = math.ceil(out_dim_total / effective_groups) * effective_groups
        self._out_pad = padded_out_dim - out_dim_total  # channels to discard

        self.conv = GatedCausalConv(
            dim=input_dim,
            kernel_size=kernel_size,
            groups=conv_groups,
            channel_shift=channel_shift,
            gated=gated_conv,
            out_dim=padded_out_dim if padded_out_dim != input_dim else None,
        )

    @property
    def output_streams(self) -> list[StreamConfig]:
        """Stream configs this conv writes to."""
        return self._writable_streams

    def _out_dim(self, s: StreamConfig) -> int:
        """Output dim per stream: total_slots for logit (hierarchy) or s.dim."""
        if self._logit_hierarchy is not None and s.name == self._logit_sid:
            return self._logit_hierarchy.total_slots
        return s.dim

    def forward(self, input_streams: dict[StreamID, Tensor]) -> dict[StreamID, Tensor]:
        """Returns update deltas for writable streams only."""
        x = torch.cat([input_streams[s.name] for s in self._input_streams], dim=-1)
        h = self.conv(x)

        # Discard padding channels, then slice into per-stream deltas
        if self._out_pad > 0:
            h = h[..., : -self._out_pad]
        chunks = h.split(self._out_dims, dim=-1)
        output: dict[StreamID, Tensor] = {}
        for s, chunk in zip(self._writable_streams, chunks):
            if self._logit_hierarchy is not None and s.name == self._logit_sid:
                level_chunks = chunk.split(self._logit_hierarchy.level_sizes, dim=-1)
                chunk = self._logit_hierarchy.assemble_logits(list(level_chunks))
            output[s.name] = chunk
        return output


class MultiStreamCausalConvLayers(nn.Module):
    """Stack of ``MultiStreamCausalConv`` layers with per-stream norms and residuals.

    Each layer: pre-norm all streams → conv → σ(β)*x + σ(α)*update for writable
    streams.  Read-only streams pass through unchanged.

    Intended as a pre-processing stage to fill initial information into writable
    streams before the attention blocks.
    """

    def __init__(
        self,
        stream_config: MultiStreamConfig,
        num_layers: int,
        kernel_size: int | list[int] = 4,
        conv_groups: int = 0,
        gated_conv: bool = True,
        logit_hierarchy: ByteLogitHierarchy | None = None,
        input_stream_ids: list[StreamID] | None = None,
        output_stream_ids: list[StreamID] | None = None,
        conv_channel_shuffle: bool = True,
        alpha_init: float = -3.0,
        beta_init: float = 5.0,
    ):
        super().__init__()
        self.stream_config = stream_config
        self.num_layers = num_layers

        if isinstance(kernel_size, int):
            kernel_sizes = [kernel_size] * num_layers
        else:
            if len(kernel_size) != num_layers:
                raise ValueError(
                    f"kernel_size list length ({len(kernel_size)}) must match "
                    f"num_layers ({num_layers})"
                )
            kernel_sizes = kernel_size

        # Compute per-layer channel shifts for cross-group mixing.
        # Only meaningful when groups partition channels into 2+ groups
        # (not depthwise and not full conv).
        shift_step = 0
        if conv_channel_shuffle and num_layers > 1:
            if input_stream_ids is not None:
                id_set = set(input_stream_ids)
                input_dim = sum(
                    s.dim for s in stream_config.streams if s.name in id_set
                )
            else:
                input_dim = sum(s.dim for s in stream_config.streams)
            effective_groups = input_dim if conv_groups <= 0 else conv_groups
            if 1 < effective_groups < input_dim:
                shift_step = input_dim // effective_groups

        self.norms = nn.ModuleList(
            [
                _build_stream_norms(list(stream_config.streams))
                for _ in range(num_layers)
            ]
        )
        # Determine which writable streams this conv stack outputs to
        all_writable = [s for s in stream_config.streams if not s.read_only]
        if output_stream_ids is not None:
            out_set = set(output_stream_ids)
            self._output_streams = [s for s in all_writable if s.name in out_set]
        else:
            self._output_streams = all_writable

        self.convs = nn.ModuleList(
            [
                MultiStreamCausalConv(
                    stream_config=stream_config,
                    kernel_size=ks,
                    conv_groups=conv_groups,
                    gated_conv=gated_conv,
                    logit_hierarchy=logit_hierarchy,
                    input_stream_ids=input_stream_ids,
                    output_stream_ids=output_stream_ids,
                    channel_shift=i * shift_step,
                )
                for i, ks in enumerate(kernel_sizes)
            ]
        )
        self.alphas = nn.ModuleList(
            [
                nn.ParameterDict(
                    {
                        s.key: nn.Parameter(torch.full((s.dim,), alpha_init))
                        for s in self._output_streams
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.betas = nn.ModuleList(
            [
                nn.ParameterDict(
                    {
                        s.key: nn.Parameter(torch.full((s.dim,), beta_init))
                        for s in self._output_streams
                    }
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, input_streams: dict[StreamID, Tensor]) -> dict[StreamID, Tensor]:
        x = dict(input_streams)
        for layer_idx in range(self.num_layers):
            norms = self.norms[layer_idx]
            normed = _apply_stream_norms(norms, self.stream_config.streams, x)
            conv_out = self.convs[layer_idx](normed)
            for s in self._output_streams:
                alpha = torch.sigmoid(self.alphas[layer_idx][s.key])
                beta = torch.sigmoid(self.betas[layer_idx][s.key])
                mixed = beta * normed[s.name] + alpha * conv_out[s.name]
                x[s.name] = norms[s.key](mixed) if s.key in norms else mixed
        return x


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
        gate_init: float = -2.0,
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
            s.name
            for s in stream_config.streams
            if s.name in self._compressed_id_set and s.read_only
        ]
        self._writable_compressed_ids = [
            s.name
            for s in stream_config.streams
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
                s.key: nn.Parameter(torch.tensor(gate_init))
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

    def _build_digit_bank(self, values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Pre-compute digit sequences and signs for all pairs × ops.

        Pure tensor implementation — no Python loops, ``.item()``, or string
        ops — compatible with ``torch.compile(fullgraph=True)``.

        Digit extraction uses fixed-point arithmetic in float64:
        integer digits via ``floor(val / 10^k) % 10``, fractional digits via
        ``floor(val * 10^k) % 10``, with trailing-zero stripping limited to
        float64 significant precision (~15 digits).

        Args:
            values: (B, N) number values.
            mask: (B, N) valid mask.

        Returns:
            digit_seqs: (B, P, n_ops, max_len) int tensor of digit class indices.
            signs: (B, P, n_ops) float tensor (+1, -1, or 0).
        """
        P = self._n_pairs
        n_ops = self.n_ops
        L = self._MAX_RESULT_LEN
        DOT = self.DIGIT_IDX_DOT
        END = self.DIGIT_IDX_END
        eps = self.eps
        device = values.device

        # -- 1. Pair values and validity --
        a = values[:, self.pair_i].double()  # (B, P)
        bv = values[:, self.pair_j].double()  # (B, P)
        pair_valid = mask[:, self.pair_i] & mask[:, self.pair_j]  # (B, P)
        b_abs = bv.abs() + eps

        # -- 2. All 5 arithmetic operations --
        results = torch.stack(
            [a + bv, a - bv, a * bv, a / b_abs, torch.fmod(a, b_abs)],
            dim=-1,
        )  # (B, P, n_ops)

        # -- 3. Signs (invalid pairs → 0) --
        signs = torch.sign(results) * pair_valid.unsqueeze(-1).double()

        # -- 4. Digit extraction (float64 throughout) --
        abs_v = results.abs().clamp(max=1e15)  # (B, P, n_ops)
        flat = abs_v.reshape(-1)  # (M,)
        M = flat.shape[0]

        int_part = flat.floor()  # (M,)
        frac_part = flat - int_part  # (M,)

        # Number of integer digits (min 1 for the leading "0" when val < 1)
        n_int = (int_part.clamp(min=1).log10().floor().long() + 1).clamp(
            min=1, max=L
        )  # (M,)

        # Integer digits: digit[p] = floor(int_part / 10^(n_int-1-p)) % 10
        pos = torch.arange(L, device=device)  # (L,)
        power = n_int.unsqueeze(1) - 1 - pos.unsqueeze(0)  # (M, L)
        is_int_pos = power >= 0  # (M, L)
        divisor = 10.0 ** power.clamp(min=0).double()  # (M, L)
        int_digits = (int_part.unsqueeze(1) / divisor).floor().long() % 10

        # Fractional digits: digit[k] = floor(frac * 10^(k+1)) % 10
        F = min(L - 2, 18)  # cap at 18 to avoid int64 overflow from 10^19
        frac_pows = 10.0 ** torch.arange(
            1, F + 1, device=device, dtype=torch.float64
        )  # (F,)
        frac_digits = (
            (frac_part.unsqueeze(1) * frac_pows.unsqueeze(0)).floor().long() % 10
        )  # (M, F)

        # Limit to float64 significant precision (~15 digits total) and
        # strip trailing zeros by finding the last nonzero within that range.
        max_sig = (15 - n_int).clamp(min=0, max=F)  # (M,)
        fi = torch.arange(F, device=device).unsqueeze(0)  # (1, F)
        sig_frac = torch.where(
            fi < max_sig.unsqueeze(1), frac_digits, torch.zeros_like(frac_digits)
        )
        nonzero = sig_frac != 0
        n_frac = nonzero.flip(1).cummax(1).values.flip(1).sum(1)  # (M,)
        has_frac = n_frac > 0  # (M,)

        # -- 5. Assemble: [int_d0 .. int_dN, DOT?, frac_d0 .. frac_dK, END ..] --
        seq = torch.full((M, L), END, dtype=torch.long, device=device)
        seq = torch.where(is_int_pos, int_digits, seq)

        # DOT at position n_int (only when fractional part exists)
        is_dot = (pos.unsqueeze(0) == n_int.unsqueeze(1)) & has_frac.unsqueeze(1)
        seq = torch.where(is_dot, DOT, seq)

        # Fractional digits at positions (n_int + 1) .. (n_int + n_frac)
        frac_start = (n_int + 1).unsqueeze(1)  # (M, 1)
        frac_idx = pos.unsqueeze(0) - frac_start  # (M, L)
        is_frac = (
            (frac_idx >= 0)
            & (frac_idx < n_frac.unsqueeze(1))
            & has_frac.unsqueeze(1)
        )
        frac_at_pos = frac_digits.gather(1, frac_idx.clamp(0, F - 1))
        seq = torch.where(is_frac, frac_at_pos, seq)

        # Invalid pairs → all END
        valid_flat = pair_valid.unsqueeze(-1).expand(-1, P, n_ops).reshape(M)
        seq = torch.where(valid_flat.unsqueeze(1), seq, END)

        return seq.reshape(-1, P, n_ops, L), signs.float()

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
        w_pair = F.softmax(scores, dim=-1)
        # When all pairs are masked for a position, softmax(-inf,...) = NaN.
        # Use masked_fill (not nan_to_num) so backward gets clean zero gradients.
        all_masked = ~attn_mask.any(dim=-1, keepdim=True)  # (B, S, 1)
        w_pair = w_pair.masked_fill(all_masked, 0.0)  # (B, S, P)

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

    Optional ``conv``: if set, a MultiStreamCausalConv is applied between
    attention and MLP (before arith_attn if present).

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
        conv: MultiStreamCausalConv | None = None,
        arith_attn: CausalArithmeticMultiStreamAttention | None = None,
        logit_hierarchy: ByteLogitHierarchy | None = None,
        mlp_input_stream_ids: list[StreamID] | None = None,
        mlp_output_stream_ids: list[StreamID] | None = None,
        attn_alpha_init: float = -1.0,
        attn_beta_init: float = 5.0,
        mlp_alpha_init: float = -3.0,
        mlp_beta_init: float = 5.0,
        conv_alpha_init: float = -3.0,
        conv_beta_init: float = 5.0,
        arith_alpha_init: float = -3.0,
        arith_beta_init: float = 5.0,
    ):
        super().__init__()
        self.stream_config = stream_config

        # Per-stream norms — type determined by stream_config.norm_type
        all_streams = list(stream_config.streams)
        self.attn_norms = _build_stream_norms(all_streams)
        self.mlp_norms = _build_stream_norms(all_streams)

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
            skip_residual=True,
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
            logit_hierarchy=logit_hierarchy,
            input_stream_ids=mlp_input_stream_ids,
            output_stream_ids=mlp_output_stream_ids,
        )
        mlp_out_streams = self.mlp.output_streams

        # Per-stream, per-dimension independent α (update scale) and β (residual scale)
        # output = σ(β) * x + σ(α) * update
        # Init: α=-3 (σ≈0.05, tiny update), β=5 (σ≈0.99, near-identity passthrough)
        self.attn_alpha = nn.ParameterDict(
            {
                s.key: nn.Parameter(torch.full((s.dim,), attn_alpha_init))
                for s in stream_config.streams
                if not s.read_only
            }
        )
        self.attn_beta = nn.ParameterDict(
            {
                s.key: nn.Parameter(torch.full((s.dim,), attn_beta_init))
                for s in stream_config.streams
                if not s.read_only
            }
        )
        self.mlp_alpha = nn.ParameterDict(
            {
                s.key: nn.Parameter(torch.full((s.dim,), mlp_alpha_init))
                for s in mlp_out_streams
            }
        )
        self.mlp_beta = nn.ParameterDict(
            {
                s.key: nn.Parameter(torch.full((s.dim,), mlp_beta_init))
                for s in mlp_out_streams
            }
        )

        # Optional causal conv (between attention and MLP, before arith_attn)
        self.conv = conv
        if conv is not None:
            self.conv_norms = _build_stream_norms(all_streams)
            conv_out_streams = conv.output_streams
            self.conv_alpha = nn.ParameterDict(
                {
                    s.key: nn.Parameter(torch.full((s.dim,), conv_alpha_init))
                    for s in conv_out_streams
                }
            )
            self.conv_beta = nn.ParameterDict(
                {
                    s.key: nn.Parameter(torch.full((s.dim,), conv_beta_init))
                    for s in conv_out_streams
                }
            )

        # Optional arithmetic attention (between attention and MLP)
        self.arith_attn = arith_attn
        if arith_attn is not None:
            self.arith_norms = _build_stream_norms(all_streams)
            self.arith_alpha = nn.ParameterDict(
                {
                    s.key: nn.Parameter(torch.full((s.dim,), arith_alpha_init))
                    for s in stream_config.streams
                    if not s.read_only
                }
            )
            self.arith_beta = nn.ParameterDict(
                {
                    s.key: nn.Parameter(torch.full((s.dim,), arith_beta_init))
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
        streams = list(self.stream_config.streams)

        # --- Attention sub-layer: pre-norm → attn → mix → re-norm ---
        normed = _apply_stream_norms(self.attn_norms, streams, input_streams)
        attn_out = self.attn(normed)

        x: dict[StreamID, Tensor] = {}
        for s in streams:
            if s.read_only:
                x[s.name] = input_streams[s.name]
            else:
                alpha = torch.sigmoid(self.attn_alpha[s.key])
                beta = torch.sigmoid(self.attn_beta[s.key])
                mixed = beta * normed[s.name] + alpha * attn_out[s.name]
                x[s.name] = (
                    self.attn_norms[s.key](mixed) if s.key in self.attn_norms
                    else mixed
                )

        # --- Optional causal conv: pre-norm → conv → mix → re-norm ---
        if self.conv is not None:
            normed = _apply_stream_norms(self.conv_norms, streams, x)
            conv_out = self.conv(normed)
            for s in streams:
                if not s.read_only and s.name in conv_out:
                    alpha = torch.sigmoid(self.conv_alpha[s.key])
                    beta = torch.sigmoid(self.conv_beta[s.key])
                    mixed = beta * normed[s.name] + alpha * conv_out[s.name]
                    x[s.name] = (
                        self.conv_norms[s.key](mixed) if s.key in self.conv_norms
                        else mixed
                    )

        # --- Optional arithmetic attention: pre-norm → arith → mix → re-norm ---
        if self.arith_attn is not None and compressed is not None and views is not None:
            normed = _apply_stream_norms(self.arith_norms, streams, x)
            num_compressed = compressed[CompressionType.NUMBER]
            num_view = views[CompressionType.NUMBER]
            arith_out = self.arith_attn(normed, num_compressed, num_view)
            for s in streams:
                if not s.read_only and s.name in arith_out:
                    alpha = torch.sigmoid(self.arith_alpha[s.key])
                    beta = torch.sigmoid(self.arith_beta[s.key])
                    mixed = beta * normed[s.name] + alpha * arith_out[s.name]
                    x[s.name] = (
                        self.arith_norms[s.key](mixed) if s.key in self.arith_norms
                        else mixed
                    )

        # --- MLP sub-layer: pre-norm → mlp → mix → re-norm ---
        normed = _apply_stream_norms(self.mlp_norms, streams, x)
        mlp_out = self.mlp(normed)

        output: dict[StreamID, Tensor] = {}
        for s in streams:
            if s.read_only:
                output[s.name] = x[s.name]
            elif s.name in mlp_out:
                alpha = torch.sigmoid(self.mlp_alpha[s.key])
                beta = torch.sigmoid(self.mlp_beta[s.key])
                mixed = beta * normed[s.name] + alpha * mlp_out[s.name]
                output[s.name] = (
                    self.mlp_norms[s.key](mixed) if s.key in self.mlp_norms
                    else mixed
                )
            else:
                output[s.name] = x[s.name]

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

    # --- Conv tests ---
    print(f"\n{'=' * 70}")
    print("MultiStreamCausalConv tests")
    print("=" * 70)

    for gated_conv in [True, False]:
        conv = MultiStreamCausalConv(
            stream_config=stream_config,
            kernel_size=4,
            gated_conv=gated_conv,
        )
        inp = {s.name: torch.randn(bsz, seqlen, s.dim) for s in stream_config.streams}
        out = conv(inp)
        assert set(out.keys()) == {
            s.name for s in stream_config.streams if not s.read_only
        }
        loss = sum(v.sum() for v in out.values())
        loss.backward()
        n = sum(p.numel() for p in conv.parameters())
        tag = f"gated_conv={gated_conv}"
        print(f"  {tag:40s}: OK  ({n:,} params)")

    # --- ConvLayers tests ---
    print(f"\n{'=' * 70}")
    print("MultiStreamCausalConvLayers tests")
    print("=" * 70)

    for num_layers, ks in [(1, 4), (3, 4), (3, [2, 4, 8])]:
        layers = MultiStreamCausalConvLayers(
            stream_config=stream_config,
            num_layers=num_layers,
            kernel_size=ks,
        )
        inp = {s.name: torch.randn(bsz, seqlen, s.dim) for s in stream_config.streams}
        out = layers(inp)
        assert set(out.keys()) == {s.name for s in stream_config.streams}
        for s in stream_config.streams:
            if s.read_only:
                assert torch.equal(out[s.name], inp[s.name])
        loss = sum(v.sum() for v in out.values() if v.requires_grad)
        loss.backward()
        n = sum(p.numel() for p in layers.parameters())
        tag = f"layers={num_layers}/ks={ks}"
        print(f"  {tag:40s}: OK  ({n:,} params)")

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

    # Test block with conv
    conv_mod = MultiStreamCausalConv(stream_config=stream_config, kernel_size=4)
    block = MultiStreamBlock(
        **kwargs,
        mlp_hidden_dim=320,
        conv=conv_mod,
    )
    inp = {s.name: torch.randn(bsz, seqlen, s.dim) for s in stream_config.streams}
    out = block(inp)
    assert set(out.keys()) == {s.name for s in stream_config.streams}
    for s in stream_config.streams:
        if s.read_only:
            assert torch.equal(out[s.name], inp[s.name])
    loss = sum(v.sum() for v in out.values() if v.requires_grad)
    loss.backward()
    n = sum(p.numel() for p in block.parameters())
    print(f"  {'standard/conv=True':40s}: OK  ({n:,} params)")

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
