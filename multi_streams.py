"""Multi-stream configuration, composable stream components, and stream builder.

Defines the stream abstraction used by multi-stream attention:
- Stream names, StreamConfig, MultiStreamConfig (data structures)
- CompositeStream (concatenates components into a single read-only stream)
- SinCosPositionComponent, DocBoundaryComponent (tokenizer-agnostic components)
- CompressedView, CompressionType (stream compression framework)
- MultiStreamBuilder (constructs stream dict from input_ids)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from modules import sincos_encode


# ---------------------------------------------------------------------------
# Stream configuration
# ---------------------------------------------------------------------------


class StreamType(StrEnum):
    LOGIT = "logit"
    CONTEXT = "context"
    TOKENS = "tokens"
    STRUCTURAL = "structural"


@dataclass(frozen=True)
class StreamID:
    """Hashable stream identifier supporting multiple streams of the same type.

    For backward compatibility, ``str(StreamID(Stream.LOGIT))`` returns ``"logit"``,
    matching existing ``nn.ModuleDict`` / ``nn.ParameterDict`` keys.
    """

    type: StreamType
    name: str = ""

    def __str__(self) -> str:
        return f"{self.type.value}:{self.name}" if self.name else self.type.value

    def __repr__(self) -> str:
        if self.name:
            return f"StreamID({self.type!r}, {self.name!r})"
        return f"StreamID({self.type!r})"


@dataclass
class StreamConfig:
    name: StreamID
    dim: int
    read_only: bool = False

    @property
    def key(self) -> str:
        """String key for use in nn.ModuleDict/ParameterDict."""
        return str(self.name)


@dataclass
class MultiStreamConfig:
    streams: list[StreamConfig]


@dataclass
class StreamDef:
    """Definition for a single stream in MultiStreamBuilder.

    Exactly one of these must determine the dimension:
    - ``dim``: explicit dimension (for externally-provided or auto-zeros streams)
    - ``components``: list of nn.Module components (dim auto-computed)
    - ``auto_onehot``: True to auto-create one-hot from input_ids (requires vocab_size)
    """

    name: StreamID
    read_only: bool = False
    dim: int | None = None
    components: list[nn.Module] | None = None
    auto_zeros: bool = False
    auto_onehot: bool = False


# ---------------------------------------------------------------------------
# Stream compression framework
# ---------------------------------------------------------------------------


class CompressionType(StrEnum):
    NUMBER = "number"


@dataclass
class CompressedView:
    """Result of a compression function.

    A compressed view selects N positions from a sequence of length S,
    producing a shorter representation that downstream modules can
    attend to efficiently.

    Attributes:
        positions: (B, N) int — indices into the original sequence.
        mask: (B, N) bool — valid entries (N is padded to a fixed max).
        metadata: compression-specific data (e.g., numeric values for NUMBER).
    """

    positions: Tensor
    mask: Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


# Type alias for compression functions: input_ids → CompressedView
CompressionFn = Callable[[Tensor], CompressedView]


# ---------------------------------------------------------------------------
# Composable stream components
# ---------------------------------------------------------------------------


class SinCosPositionComponent(nn.Module):
    """Absolute position sin/cos encoding as a stream component.

    No learnable params. Equivalent to the positional information RoPE provides,
    but as explicit features the attention can read.
    """

    def __init__(self, num_freqs: int = 8, base: float = 10000.0):
        super().__init__()
        self.num_freqs = num_freqs
        self.base = base
        self._dim = num_freqs * 2

    @property
    def dim(self) -> int:
        return self._dim

    def forward(self, input_ids: Tensor, dtype: torch.dtype) -> Tensor:
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device)
        enc = sincos_encode(positions, self.num_freqs, self.base)  # (S, dim)
        return enc.unsqueeze(0).expand(B, -1, -1).to(dtype=dtype)


class DocBoundaryComponent(nn.Module):
    """Document boundary (BOS cumsum) encoded as sin/cos.

    Causal: uses cumsum over BOS markers (only depends on positions ≤ t).
    With 1-3 docs per sequence, 1 frequency pair (2 dims) suffices.
    """

    def __init__(self, bos_id: int, num_freqs: int = 1, base: float = 10000.0):
        super().__init__()
        self.bos_id = bos_id
        self.num_freqs = num_freqs
        self.base = base
        self._dim = num_freqs * 2

    @property
    def dim(self) -> int:
        return self._dim

    def forward(self, input_ids: Tensor, dtype: torch.dtype) -> Tensor:
        doc_ids = (input_ids == self.bos_id).cumsum(dim=1)  # (B, S), causal
        return sincos_encode(doc_ids, self.num_freqs, self.base).to(dtype=dtype)


class CompositeStream(nn.Module):
    """Concatenates stream components into a single read-only stream.

    Each component must implement:
      .dim -> int
      .forward(input_ids: Tensor, dtype: torch.dtype) -> Tensor  # (B, S, component_dim)
    """

    def __init__(self, components: list[nn.Module]):
        super().__init__()
        self.components = nn.ModuleList(components)
        self._dim = sum(c.dim for c in components)

    @property
    def dim(self) -> int:
        return self._dim

    def forward(self, input_ids: Tensor, dtype: torch.dtype) -> Tensor:
        parts = [c(input_ids, dtype) for c in self.components]
        return torch.cat(parts, dim=-1)


# ---------------------------------------------------------------------------
# Stream builder
# ---------------------------------------------------------------------------


class MultiStreamBuilder(nn.Module):
    """Constructs the full stream dict from input_ids and optional provided streams.

    Each stream is defined by a ``StreamDef`` specifying how it is constructed:
    - External input: caller provides the tensor in ``forward(**provided_streams)``
    - Components: built automatically from composable ``nn.Module`` components
    - Auto one-hot: one-hot encoding of ``input_ids``
    - Auto zeros: zero-initialized tensor

    Usage:
        builder = MultiStreamBuilder([
            StreamDef(Stream.LOGIT, dim=vocab_size),              # caller provides
            StreamDef(Stream.TOKENS, auto_onehot=True, read_only=True),
            StreamDef(Stream.STRUCTURAL, read_only=True, components=[hash_comp, ...]),
            StreamDef(Stream.CONTEXT, dim=32, auto_zeros=True),   # writable scratch
        ], vocab_size=vocab_size)

        attn = CausalMultiStreamAttention(..., stream_config=builder.config)
        streams, compressed, views = builder(input_ids, logit=one_hot_tensor)
        out = attn(streams)
    """

    def __init__(
        self,
        stream_defs: list[StreamDef],
        vocab_size: int | None = None,
        compressions: dict[CompressionType, CompressionFn] | None = None,
        compress_streams: list[StreamID] | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self._defs = list(stream_defs)
        self._compressions = compressions or {}
        self._compress_stream_ids = compress_streams or []

        # Build CompositeStreams for component-based defs
        composites: dict[str, CompositeStream] = {}
        stream_configs: list[StreamConfig] = []

        for sd in self._defs:
            dim = self._resolve_dim(sd)
            stream_configs.append(
                StreamConfig(sd.name, dim=dim, read_only=sd.read_only)
            )
            if sd.components is not None:
                composites[str(sd.name)] = CompositeStream(sd.components)

        self.composites = nn.ModuleDict(composites)
        self._config = MultiStreamConfig(streams=stream_configs)

    def _resolve_dim(self, sd: StreamDef) -> int:
        """Compute and validate the dimension for a StreamDef."""
        if sd.auto_onehot:
            if self.vocab_size is None:
                raise ValueError(
                    f"Stream {sd.name} has auto_onehot=True but vocab_size not provided"
                )
            if sd.dim is not None and sd.dim != self.vocab_size:
                raise ValueError(
                    f"Stream {sd.name}: dim={sd.dim} conflicts with vocab_size={self.vocab_size}"
                )
            return self.vocab_size
        if sd.components is not None:
            comp_dim = sum(c.dim for c in sd.components)
            if sd.dim is not None and sd.dim != comp_dim:
                raise ValueError(
                    f"Stream {sd.name}: dim={sd.dim} conflicts with components dim={comp_dim}"
                )
            return comp_dim
        if sd.dim is None:
            raise ValueError(
                f"Stream {sd.name}: must specify dim, components, or auto_onehot"
            )
        return sd.dim

    @property
    def config(self) -> MultiStreamConfig:
        return self._config

    def forward(
        self,
        input_ids: Tensor,
        dtype: torch.dtype = torch.float32,
        **provided_streams: dict[StreamID, Tensor],
    ) -> tuple[
        dict[StreamID, Tensor],
        dict[CompressionType, dict[StreamID, Tensor]] | None,
        dict[CompressionType, CompressedView] | None,
    ]:
        """Build stream dict and optional compressed views.

        Args:
            input_ids: (B, S) token IDs.
            dtype: dtype for auto-constructed streams.
            **provided_streams: tensors keyed by stream name string
                (e.g. ``logit=tensor``). Required for streams without
                auto_onehot, auto_zeros, or components.

        Returns:
            Tuple of:
            - streams: dict mapping StreamID to (B, S, d) tensors.
            - compressed: dict mapping CompressionType to
              dict[StreamID, (B, N, d) tensors], or None if no compressions.
            - views: dict mapping CompressionType to CompressedView, or None.
        """
        B, S = input_ids.shape
        streams: dict[StreamID, Tensor] = {}

        for sd in self._defs:
            key = str(sd.name)
            if sd.components is not None:
                streams[sd.name] = self.composites[key](input_ids, dtype)
            elif sd.auto_onehot:
                streams[sd.name] = F.one_hot(input_ids, self.vocab_size).to(dtype=dtype)
            elif sd.auto_zeros:
                dim = self._resolve_dim(sd)
                streams[sd.name] = torch.zeros(
                    B, S, dim, dtype=dtype, device=input_ids.device
                )
            else:
                # Try exact key first, then fall back to type value
                # (kwargs are Python identifiers, so "logit" not "logit:NAME")
                if key in provided_streams:
                    streams[sd.name] = provided_streams[key]
                elif sd.name.type.value in provided_streams:
                    streams[sd.name] = provided_streams[sd.name.type.value]
                else:
                    raise KeyError(
                        f"Stream '{key}' must be provided in forward() call "
                        f"(got keys: {list(provided_streams.keys())})"
                    )

        # --- Optional stream compression ---
        if not self._compressions:
            return streams, None, None

        compressed: dict[CompressionType, dict[StreamID, Tensor]] = {}
        views: dict[CompressionType, CompressedView] = {}

        for comp_type, comp_fn in self._compressions.items():
            view = comp_fn(input_ids)
            views[comp_type] = view
            compressed[comp_type] = {}
            for stream_id in self._compress_stream_ids:
                if stream_id not in streams:
                    continue
                full = streams[stream_id]  # (B, S, d)
                # Gather features at compressed positions
                idx = view.positions.clamp(0, S - 1).unsqueeze(-1)  # (B, N, 1)
                idx = idx.expand(-1, -1, full.shape[-1])  # (B, N, d)
                gathered = full.gather(1, idx)  # (B, N, d)
                compressed[comp_type][stream_id] = gathered * view.mask.unsqueeze(
                    -1
                ).to(dtype=gathered.dtype)

        return streams, compressed, views
