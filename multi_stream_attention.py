import torch
from torch import Tensor, nn
import torch.nn.functional as F
from enum import StrEnum
from dataclasses import dataclass

from modules import RMSNorm, CastedLinear, LearnableShift, make_linear
from multi_streams import StreamType, StreamID, StreamConfig, MultiStreamConfig


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


class CausualMultiStreamAttentionViaMixing(nn.Module):
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


class CausualMultiStreamAttention(nn.Module):
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


class MultiStreamBlock(nn.Module):
    """Multi-stream transformer block: pre-norm → attention → residual → pre-norm → MLP → residual.

    Norms are applied to all streams; residual updates only to writable streams.
    Each sub-layer uses independent per-dimension α (update scale) and β (residual
    scale): output = σ(β)*x + σ(α)*update. This allows the model to learn additive
    (β≈1), interpolating (α+β≈1), or full-replacement (α≈1, β≈0) behavior per
    dimension per stream.
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
            self.attn = CausualMultiStreamAttentionViaMixing(
                **attn_kwargs, mixing_config=mixing_config
            )
        else:
            self.attn = CausualMultiStreamAttention(**attn_kwargs)

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

    def forward(self, input_streams: dict[StreamID, Tensor]) -> dict[StreamID, Tensor]:
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
            model = CausualMultiStreamAttentionViaMixing(**kwargs, mixing_config=cfg)
            inp = {
                s.name: torch.randn(bsz, seqlen, s.dim) for s in stream_config.streams
            }
            out = model(inp)
            sum(v.sum() for v in out.values() if v.requires_grad).backward()
            n = sum(p.numel() for p in model.parameters())
            print(f"  {qk_mode:8s} + {source:20s}: OK  ({n:,} params)")

    model = CausualMultiStreamAttention(**kwargs)
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
