"""Shared building blocks for multi-stream and related training scripts."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Core layers
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CenterLastDim(nn.Module):
    """Subtract per-position mean from the last dimension.

    softmax(x) == softmax(x - mean(x)), so this is lossless for
    cross-entropy but prevents mean drift in the logit stream.
    """

    def forward(self, x: Tensor) -> Tensor:
        return x - x.mean(dim=-1, keepdim=True)


class CastedLinear(nn.Linear):
    """Linear layer that casts weights to input dtype at forward time."""

    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def _balanced_factors(n: int) -> tuple[int, int]:
    """Find two factors of n closest to sqrt(n)."""
    s = int(math.sqrt(n))
    while n % s != 0:
        s -= 1
    return s, n // s


def _nearest_divisor(n: int, target: int) -> int:
    """Find the divisor of n closest to target."""
    best, best_dist = 1, abs(1 - target)
    for d in range(1, int(math.sqrt(n)) + 1):
        if n % d == 0:
            for candidate in (d, n // d):
                dist = abs(candidate - target)
                if dist < best_dist:
                    best, best_dist = candidate, dist
    return best


class KroneckerLinear(nn.Module):
    """W = sum_k A_k ⊗ B_k. Materializes W then uses F.linear for speed.

    Factor params (A, B) are 3D. When muon_optimize_factors is enabled, Muon reshapes
    them to 2D (stacking the K term slices) for NS orthogonalization in factor space.
    NOTE: if per-factor NS underperforms, consider NS on the full materialized W
    gradient instead (requires capturing ∂L/∂W via hooks outside torch.compile and
    a custom optimizer step to chain-rule the orthogonalized update back to factors).
    """

    def __init__(self, in_features, out_features, bias=False, num_terms=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.p_in, self.q_in = _balanced_factors(in_features)
        self.p_out, self.q_out = _balanced_factors(out_features)
        self.num_terms = num_terms
        self.A = nn.Parameter(torch.randn(num_terms, self.p_out, self.p_in))
        self.B = nn.Parameter(torch.randn(num_terms, self.q_out, self.q_in))
        scale = (in_features * out_features) ** -0.5
        nn.init.normal_(self.A, std=scale)
        nn.init.normal_(self.B, std=scale)
        self._W_cache = None

    def materialize(self):
        """Build W = Σ_k kron(A_k, B_k) and cache. Stays in autograd graph."""
        A = self.A.to(torch.bfloat16)
        B = self.B.to(torch.bfloat16)
        W = torch.einsum("kij,kmn->imjn", A, B)
        self._W_cache = W.reshape(self.out_features, self.in_features)

    def clear_cache(self):
        self._W_cache = None

    def forward(self, x):
        if self._W_cache is not None:
            return F.linear(x, self._W_cache)
        A = self.A.to(x.dtype)
        B = self.B.to(x.dtype)
        W = torch.einsum("kij,kmn->imjn", A, B).reshape(
            self.out_features, self.in_features
        )
        return F.linear(x, W)


class MonarchLinear(nn.Module):
    """Monarch: two block-diagonal matmuls with a shuffle between them.
    Materializes W then uses F.linear for speed.

    See KroneckerLinear docstring for notes on muon_optimize_factors and the
    alternative of NS on the full materialized W gradient.
    """

    def __init__(self, in_features, out_features, bias=False, nblocks=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if nblocks <= 0:
            nblocks = _nearest_divisor(
                min(in_features, out_features),
                int(math.sqrt(min(in_features, out_features))),
            )
        self.nblocks = nblocks
        if in_features % nblocks != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by nblocks ({nblocks})"
            )
        self.blk_in = in_features // nblocks
        self.w1 = nn.Parameter(torch.randn(nblocks, self.blk_in, self.blk_in))
        if out_features % self.blk_in != 0:
            raise ValueError(
                f"out_features ({out_features}) must be divisible by blk_in={self.blk_in} "
                f"(= in_features // nblocks)"
            )
        self.blk_out2 = out_features // self.blk_in
        self.w2 = nn.Parameter(torch.randn(self.blk_in, nblocks, self.blk_out2))
        scale = (in_features * out_features) ** -0.25
        nn.init.normal_(self.w1, std=scale)
        nn.init.normal_(self.w2, std=scale)
        self._W_cache = None

    def materialize(self):
        """Build full W from Monarch factors and cache. Stays in autograd graph."""
        w1 = self.w1.to(torch.bfloat16)
        w2 = self.w2.to(torch.bfloat16)
        W = torch.einsum("jno,njk->jonk", w2, w1)
        self._W_cache = W.reshape(self.out_features, self.in_features)

    def clear_cache(self):
        self._W_cache = None

    def forward(self, x):
        if self._W_cache is not None:
            return F.linear(x, self._W_cache)
        w1 = self.w1.to(x.dtype)
        w2 = self.w2.to(x.dtype)
        W = torch.einsum("jno,njk->jonk", w2, w1).reshape(
            self.out_features, self.in_features
        )
        return F.linear(x, W)


def make_linear(in_features, out_features, bias=False, mode="dense", **kwargs):
    """Factory: create a CastedLinear, KroneckerLinear, or MonarchLinear."""
    if mode == "dense":
        return CastedLinear(in_features, out_features, bias=bias)
    elif mode == "kronecker":
        return KroneckerLinear(
            in_features,
            out_features,
            bias=bias,
            num_terms=kwargs.get("kronecker_terms", 4),
        )
    elif mode == "monarch":
        return MonarchLinear(
            in_features,
            out_features,
            bias=bias,
            nblocks=kwargs.get("monarch_nblocks", 0),
        )
    else:
        raise ValueError(f"Unknown linear_mode: {mode!r}")


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------


class Rotary(nn.Module):
    """RoPE positional encoding with cached cos/sin tables.

    When rope_dim_fraction < 1.0, only a subset of head dimensions receive
    positional encoding; the remainder pass through unchanged (partial RoPE).
    """

    def __init__(self, dim: int, base: float = 10000.0, rope_dim_fraction: float = 1.0):
        super().__init__()
        # Only compute frequencies for the rotated subset of dimensions.
        rope_dims = max(2, 2 * (int(dim * rope_dim_fraction) // 2))  # ensure even
        inv_freq = 1.0 / (
            base ** (torch.arange(0, rope_dims, 2, dtype=torch.float32) / rope_dims)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    rope_dims = cos.size(-1) * 2  # cos covers half the rotated dims
    if rope_dims >= x.size(-1):
        # Full RoPE: apply to all dimensions.
        half = x.size(-1) // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
    # Partial RoPE: only rotate first rope_dims, pass the rest through.
    x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
    half = rope_dims // 2
    x1, x2 = x_rope[..., :half], x_rope[..., half:]
    x_rotated = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
    return torch.cat((x_rotated, x_pass), dim=-1)


class SemanticRotary(nn.Module):
    """RoPE driven by semantic boundary IDs (word/sentence/paragraph counts)."""

    def __init__(self, configs):
        """configs: list of (num_pairs, base) for each boundary type."""
        super().__init__()
        self.configs = configs
        inv_freqs = []
        for num_pairs, base in configs:
            dims = num_pairs * 2
            inv_freq = 1.0 / (
                base ** (torch.arange(0, dims, 2, dtype=torch.float32) / dims)
            )
            inv_freqs.append(inv_freq)
        self.register_buffer("inv_freq", torch.cat(inv_freqs), persistent=False)
        self._offsets = []
        pos = 0
        for num_pairs, _ in configs:
            self._offsets.append((pos, pos + num_pairs))
            pos += num_pairs
        self.total_half = pos  # total cos/sin width

    def forward(self, boundary_ids_list, dtype):
        """boundary_ids_list: list of (B, S) long tensors, one per boundary type.
        Returns: cos (B, 1, S, total_half), sin (B, 1, S, total_half)"""
        parts_cos, parts_sin = [], []
        for ids, (start, end) in zip(boundary_ids_list, self._offsets, strict=True):
            inv_f = self.inv_freq[start:end]  # (num_pairs,)
            freqs = ids.unsqueeze(-1).float() * inv_f  # (B, S, num_pairs)
            parts_cos.append(freqs.cos())
            parts_sin.append(freqs.sin())
        cos = (
            torch.cat(parts_cos, dim=-1).unsqueeze(1).to(dtype=dtype)
        )  # (B, 1, S, total_half)
        sin = torch.cat(parts_sin, dim=-1).unsqueeze(1).to(dtype=dtype)
        return cos, sin


# ---------------------------------------------------------------------------
# Stream components (composable read-only stream builders)
# ---------------------------------------------------------------------------


def sincos_encode(ids: Tensor, num_freqs: int, base: float = 10000.0) -> Tensor:
    """Encode integer IDs as sin/cos features.  (...,) → (..., num_freqs*2).

    All operations are element-wise on the input IDs — causality is determined
    by how the IDs themselves are computed (e.g. cumsum, cummax).
    """
    if num_freqs == 0:
        return ids.new_zeros(*ids.shape, 0)
    dim = num_freqs * 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=ids.device) / dim)
    )
    angles = ids.unsqueeze(-1).float() * inv_freq  # (..., num_freqs)
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class RotationCodebook(nn.Module):
    """2D unit-circle codebook for *n_states* discrete classes.

    Each class is evenly spaced around the unit circle at angle
    ``index * 2π / n_states``, giving a ``(cos θ, sin θ)`` vector.

    Dot-product geometry:
    - Same class  → dot = 1
    - Adjacent    → dot = cos(2π / n_states)
    - Opposite    → dot ≈ -1  (for even n_states, exactly -1)

    The codebook is stored as a registered buffer so it moves with
    ``.to(device)`` / ``.to(dtype)`` automatically.

    Args:
        n_states: number of discrete states (determines angular spacing).
    """

    def __init__(self, n_states: int):
        super().__init__()
        self.n_states = n_states
        angles = torch.arange(n_states, dtype=torch.float32) * (
            2.0 * torch.pi / n_states
        )
        # (n_states, 2) codebook of unit vectors
        codebook = torch.stack([angles.cos(), angles.sin()], dim=-1)
        self.register_buffer("codebook", codebook)

    def encode(self, indices: Tensor) -> Tensor:
        """Map integer indices to 2D unit vectors.

        Args:
            indices: ``(...,)`` integer tensor with values in ``[0, n_states)``.

        Returns:
            ``(..., 2)`` float tensor with unit-norm rows.
        """
        return self.codebook[indices]

    def decode(self, vectors: Tensor) -> Tensor:
        """Map 2D vectors back to the nearest class index (argmax dot product).

        Args:
            vectors: ``(..., 2)`` float tensor.

        Returns:
            ``(...,)`` long tensor of class indices.
        """
        # (..., 2) @ (2, n_states) → (..., n_states)
        dots = vectors @ self.codebook.t()
        return dots.argmax(dim=-1)


def binary_embedding(num_classes: int, dim: int) -> Tensor:
    """Deterministic binary embedding for a small number of classes.

    Encodes class indices ``0..num_classes-1`` as ``{-1, +1}`` vectors using
    binary digits, giving maximal separation (cosine similarity ≤ 0 between
    any two distinct classes) with no learnable parameters.

    Requires ``2 ** dim >= num_classes``.

    Returns:
        Float tensor of shape ``(num_classes, dim)`` with values in {-1, +1}.
    """
    if num_classes > 2**dim:
        raise ValueError(
            f"binary_embedding: dim={dim} can encode at most {2**dim} classes, "
            f"got num_classes={num_classes}"
        )
    rows = [
        [float((i >> b) & 1) * 2 - 1 for b in range(dim)] for i in range(num_classes)
    ]
    return torch.tensor(rows)


# Stream components (SinCosPositionComponent, DocBoundaryComponent, CompositeStream)
# have moved to multi_streams.py.  sincos_encode remains here as a shared utility.


# ---------------------------------------------------------------------------
# Learnable causal shift
# ---------------------------------------------------------------------------


class LearnableShift(nn.Module):
    """Learnable fractional causal shift along a sequence dimension.

    Interpolates between identity (d=0) and a full one-position shift (d=1):
        out[t] = (1 - d) * x[t] + d * x[t-1]

    Zero-padded at t=0 to maintain causality.

    Typical use: shift K before attention so that Q[t] matching K[j]
    effectively matches against position j-1 but reads V from position j
    (single-head induction).

    Args:
        num_channels: independent shift parameters (e.g., num_kv_heads).
            Broadcasts along the dim immediately before seq_dim.
        seq_dim: sequence dimension index (default -2, matching (B, H, S, D)).
        init: pre-sigmoid init value. None → random uniform in [-2, 2].
            Fixed float examples: -5.0 → d ≈ 0.007 (near identity),
            -1.0 → d ≈ 0.27 (moderate shift).
    """

    def __init__(
        self,
        num_channels: int = 1,
        seq_dim: int = -2,
        init: float | None = None,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.seq_dim = seq_dim
        self.shift_logit = nn.Parameter(torch.empty(num_channels, dtype=torch.float32))
        if init is not None:
            nn.init.constant_(self.shift_logit, init)
        else:
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.shift_logit, -2.0, 2.0)

    def forward(self, x: Tensor) -> Tensor:
        d = torch.sigmoid(self.shift_logit)  # (C,) in [0, 1]
        sd = self.seq_dim % x.ndim

        # x_prev: x shifted right by 1 along seq_dim, zero-padded at start
        zero_shape = list(x.shape)
        zero_shape[sd] = 1
        x_prev = torch.cat(
            [x.new_zeros(zero_shape), x.narrow(sd, 0, x.shape[sd] - 1)], dim=sd
        )

        # Broadcast d to match x: expand along channel dim (one before seq_dim)
        shape = [1] * x.ndim
        if self.num_channels > 1:
            cd = (sd - 1) % x.ndim
            shape[cd] = self.num_channels
        d = d.view(shape).to(x.dtype)

        return (1 - d) * x + d * x_prev


# ---------------------------------------------------------------------------
# Attention utilities
# ---------------------------------------------------------------------------


def build_doc_mask(input_ids: Tensor, bos_id: int) -> Tensor:
    """Build a causal block-diagonal attention mask from BOS document boundaries.
    Tokens only attend to earlier tokens within the same document."""
    bsz, seq_len = input_ids.shape
    doc_ids = (input_ids == bos_id).cumsum(dim=1)  # (B, S)
    same_doc = doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)  # (B, S, S)
    causal = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device)
    )
    return (same_doc & causal).unsqueeze(1)  # (B, 1, S, S)


# ---------------------------------------------------------------------------
# Feed-forward
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """Two-layer MLP with squared activation: leaky_relu(x).square()."""

    def __init__(
        self,
        dim: int,
        mlp_mult: int,
        leaky_relu_negative_slope: float | None = 0.5,
        linear_mode: str = "dense",
        linear_kwargs: dict | None = None,
    ):
        super().__init__()
        hidden = mlp_mult * dim
        _lkw = linear_kwargs or {}
        self.fc = make_linear(dim, hidden, bias=False, mode=linear_mode, **_lkw)
        self.proj = make_linear(hidden, dim, bias=False, mode=linear_mode, **_lkw)
        self.proj._zero_init = True
        self.leaky_relu_negative_slope = leaky_relu_negative_slope

    def forward(self, x: Tensor) -> Tensor:
        if self.leaky_relu_negative_slope is None:
            x = torch.relu(self.fc(x))
        else:
            x = F.leaky_relu(self.fc(x), negative_slope=self.leaky_relu_negative_slope)
        return self.proj(x.square())


class GatedCausalConv(nn.Module):
    """Causal conv1d for local n-gram mixing.

    Two modes:
    - gated (default): gate = σ(conv_gate(x)), value = SiLU(conv_value(x)),
      out = gate * value
    - ungated: out = leaky_relu(conv(x)).square()

    Uses causal (left) padding so output[t] only depends on input[t-k+1..t].

    Std-dev repair: conv output is divided by ``sqrt(fan_in)`` where
    ``fan_in = kernel_size * (dim // groups)``, normalizing the sum of
    weighted input terms to preserve unit variance.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 4,
        groups: int = 0,
        gated: bool = True,
        channel_shift: int = 0,
        out_dim: int | None = None,
        value_softcap: float | None = None,
    ):
        super().__init__()
        out_dim = dim if out_dim is None else out_dim
        groups = dim if groups <= 0 else groups

        if dim % groups != 0:
            raise ValueError(f"dim ({dim}) must be divisible by groups ({groups})")
        if out_dim % groups != 0:
            raise ValueError(
                f"out_dim ({out_dim}) must be divisible by groups ({groups})"
            )

        self.pad = kernel_size - 1
        self.gated = gated
        self.value_softcap = SoftcapLinear(value_softcap) if value_softcap is not None else None
        # Shifting only matters when groups partition channels into 2+ groups
        self.channel_shift = channel_shift if groups not in (1, dim) else 0

        # Std-dev repair: 1/sqrt(fan_in) to normalize conv output variance
        fan_in = kernel_size * (dim // groups)
        self._conv_std_repair = 1.0 / math.sqrt(fan_in)

        if gated:
            self.conv_gate = nn.Conv1d(
                dim, out_dim, kernel_size, groups=groups, bias=False
            )
            self.conv_value = nn.Conv1d(
                dim, out_dim, kernel_size, groups=groups, bias=False
            )
            self.conv_value._zero_init = True
        else:
            self.conv = nn.Conv1d(dim, out_dim, kernel_size, groups=groups, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        h = x.transpose(1, 2)
        h = F.pad(h, (self.pad, 0))
        if self.channel_shift:
            h = torch.roll(h, shifts=self.channel_shift, dims=1)
        if self.gated:
            gate = torch.sigmoid(self.conv_gate(h) * self._conv_std_repair)
            value = F.silu(self.conv_value(h) * self._conv_std_repair)
            if self.value_softcap is not None:
                value = self.value_softcap(value)
            out = gate * value
        else:
            out = F.leaky_relu(
                self.conv(h) * self._conv_std_repair, negative_slope=0.5
            ).square()
        # No output unroll — model learns output channel assignment regardless
        return out.transpose(1, 2)


# ---------------------------------------------------------------------------
# Logit capping
# ---------------------------------------------------------------------------


class SoftcapLinear(nn.Module):
    """Soft-cap with an exactly linear core and smooth rational tails.

    Exactly identity in ``[-knee, knee]`` (gradient = 1).  Outside, a rational
    function smoothly asymptotes to ``±cap``.  C1-continuous at the knee.

    Compared to ``cap * tanh(x / cap)`` which always compresses gradients,
    this preserves perfect unit gradients for normal-magnitude values and only
    compresses outliers.

    No learnable parameters.
    """

    def __init__(self, cap: float, linear_regime_fraction: float = 0.8):
        super().__init__()
        self.cap = cap
        self.knee = linear_regime_fraction * cap
        self.r = cap - self.knee  # remaining headroom above knee

    def forward(self, x: Tensor) -> Tensor:
        excess = (x.abs() - self.knee).clamp(min=0)
        compression = excess.square() / (self.r + excess)
        return x - x.sign() * compression



# ---------------------------------------------------------------------------
# Mixed-precision utilities
# ---------------------------------------------------------------------------

# Default control tensor patterns (scalars, gains, etc. that should stay fp32)
DEFAULT_CONTROL_PATTERNS = (
    "attn_scale",
    "mlp_scale",
    "resid_mix",
    "q_gain",
    "skip_weight",
    "alpha",
    "beta",
)


def restore_low_dim_params_to_fp32(
    module: nn.Module,
    control_patterns: tuple[str, ...] = DEFAULT_CONTROL_PATTERNS,
) -> None:
    """Keep small/control parameters in fp32 even when the model body runs in bf16."""
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (
                param.ndim < 2 or any(pattern in name for pattern in control_patterns)
            ) and param.dtype != torch.float32:
                param.data = param.data.float()
