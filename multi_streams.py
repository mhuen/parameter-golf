"""Multi-stream configuration, composable stream components, and stream builder.

Defines the stream abstraction used by multi-stream attention:
- Stream names, StreamConfig, MultiStreamConfig (data structures)
- CompositeStream (concatenates components into a single read-only stream)
- SinCosPositionComponent, DocBoundaryComponent (tokenizer-agnostic components)
- MultiStreamBuilder (constructs stream dict from input_ids)
"""

from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import Tensor, nn

from modules import sincos_encode


# ---------------------------------------------------------------------------
# Stream configuration
# ---------------------------------------------------------------------------


class Stream(StrEnum):
    LOGIT = "logit"
    CONTEXT = "context"
    TOKENS = "tokens"
    STRUCTURAL = "structural"


@dataclass
class StreamConfig:
    name: Stream
    dim: int
    read_only: bool = False


@dataclass
class MultiStreamConfig:
    streams: list[StreamConfig]


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
    """Constructs the full stream dict from input_ids.

    Combines a writable stream (provided by the caller, e.g. one-hot or
    embedding) with read-only structural streams built from composable
    components.

    Usage:
        builder = MultiStreamBuilder(
            writable_dim=vocab_size,
            structural_components=[SinCosPositionComponent(), ByteHashComponent(tok), ...],
        )
        # builder.config is the MultiStreamConfig for attention layers
        attn = CausualMultiStreamAttention(..., stream_config=builder.config)

        # In forward:
        streams = builder(input_ids, writable_stream=one_hot, dtype=torch.float32)
        out = attn(streams)
    """

    def __init__(
        self,
        writable_dim: int,
        structural_components: list[nn.Module] | None = None,
        writable_name: Stream = Stream.LOGIT,
        structural_name: Stream = Stream.STRUCTURAL,
    ):
        super().__init__()
        self.writable_name = writable_name
        self.structural_name = structural_name
        self.writable_dim = writable_dim

        if structural_components:
            self.structural = CompositeStream(structural_components)
            self._config = MultiStreamConfig(
                streams=[
                    StreamConfig(writable_name, dim=writable_dim, read_only=False),
                    StreamConfig(
                        structural_name,
                        dim=self.structural.dim,
                        read_only=True,
                    ),
                ]
            )
        else:
            self.structural = None
            self._config = MultiStreamConfig(
                streams=[
                    StreamConfig(writable_name, dim=writable_dim, read_only=False),
                ]
            )

    @property
    def config(self) -> MultiStreamConfig:
        return self._config

    def forward(
        self,
        input_ids: Tensor,
        writable_stream: Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> dict[Stream, Tensor]:
        """Build stream dict.

        Args:
            input_ids: (B, S) token IDs, used by structural components.
            writable_stream: (B, S, writable_dim) pre-computed writable stream
                (e.g. one-hot, embedding output).
            dtype: dtype for structural stream computation.

        Returns:
            dict mapping Stream names to tensors.
        """
        streams: dict[Stream, Tensor] = {self.writable_name: writable_stream}
        if self.structural is not None:
            streams[self.structural_name] = self.structural(input_ids, dtype)
        return streams
