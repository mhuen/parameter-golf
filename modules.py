"""Shared building blocks for multi-stream and related training scripts."""

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


class CastedLinear(nn.Linear):
    """Linear layer that casts weights to input dtype at forward time."""

    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------


class Rotary(nn.Module):
    """RoPE positional encoding with cached cos/sin tables."""

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
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
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


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

    def __init__(self, dim: int, mlp_mult: int, leaky_relu_negative_slope: float | None = 0.5):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
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
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 4,
        groups: int = 0,
        gated: bool = True,
    ):
        super().__init__()
        groups = dim if groups <= 0 else groups
        self.pad = kernel_size - 1
        self.gated = gated

        if gated:
            self.conv_gate = nn.Conv1d(dim, dim, kernel_size, groups=groups, bias=False)
            self.conv_value = nn.Conv1d(
                dim, dim, kernel_size, groups=groups, bias=False
            )
            self.conv_value._zero_init = True
        else:
            self.conv = nn.Conv1d(dim, dim, kernel_size, groups=groups, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        h = x.transpose(1, 2)
        h = F.pad(h, (self.pad, 0))
        if self.gated:
            gate = torch.sigmoid(self.conv_gate(h))
            value = F.silu(self.conv_value(h))
            return (gate * value).transpose(1, 2)
        else:
            return F.leaky_relu(self.conv(h), negative_slope=0.5).square().transpose(1, 2)


# ---------------------------------------------------------------------------
# Mixed-precision utilities
# ---------------------------------------------------------------------------

# Default control tensor patterns (scalars, gains, etc. that should stay fp32)
DEFAULT_CONTROL_PATTERNS = (
    "attn_scale", "mlp_scale", "resid_mix", "q_gain",
    "skip_weight", "alpha", "beta",
)


def restore_low_dim_params_to_fp32(
    module: nn.Module,
    control_patterns: tuple[str, ...] = DEFAULT_CONTROL_PATTERNS,
) -> None:
    """Keep small/control parameters in fp32 even when the model body runs in bf16."""
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (
                param.ndim < 2
                or any(pattern in name for pattern in control_patterns)
            ) and param.dtype != torch.float32:
                param.data = param.data.float()
