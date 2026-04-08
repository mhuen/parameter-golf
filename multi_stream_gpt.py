"""Multi-stream GPT: full model assembling all multi-stream components.

- ``MultiStreamComponents``: lightweight dataclass holding pre-built stream
  infrastructure (builder, hierarchy, UTF-8 prior).
- ``build_multi_stream_components()``: default factory wiring all byte-stream
  components into a ready-to-use ``MultiStreamComponents``.
- ``BigramPriorLayer``: trainable (V, V) bigram logit prior.
- ``MultiStreamGPT``: top-level model module combining priors, conv pre-processing,
  and attention blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from byte_modules import ByteLogitHierarchy, NumberExtractor, UTF8Prior
from efficient_byte_tokenizer import EfficientByteTokenizer
from byte_stream_components import (
    BoundaryComponent,
    ByteCategoryComponent,
    ByteCategoryStatsComponent,
    ByteHashComponent,
    CaseComponent,
    ColumnPositionComponent,
    DigitComputeComponent,
    DigitSequenceComponent,
    HashBoundary,
    MultiByteStateComponent,
    PunctuationDepthComponent,
    RepeatedByteComponent,
    VowelConsonantComponent,
)
from multi_stream_attention import (
    CausalArithmeticMultiStreamAttention,
    MultiStreamBlock,
    MultiStreamCausalConv,
    MultiStreamCausalConvLayers,
    StreamMixingConfig,
)
from modules import softcap_linear
from multi_streams import (
    CompressionType,
    DocBoundaryComponent,
    MultiStreamBuilder,
    MultiStreamConfig,
    SinCosPositionComponent,
    StreamDef,
    StreamID,
    StreamType,
)


# ---------------------------------------------------------------------------
# MultiStreamComponents dataclass
# ---------------------------------------------------------------------------


@dataclass
class MultiStreamComponents:
    """All pre-built components needed to construct a MultiStreamGPT."""

    builder: MultiStreamBuilder
    logit_hierarchy: ByteLogitHierarchy
    utf8_prior: UTF8Prior
    tok: EfficientByteTokenizer

    @property
    def stream_config(self) -> MultiStreamConfig:
        return self.builder.config

    @property
    def vocab_size(self) -> int:
        return self.builder.vocab_size


# ---------------------------------------------------------------------------
# Default stream component factory
# ---------------------------------------------------------------------------


def build_multi_stream_components(
    tok: EfficientByteTokenizer,
    vocab_size: int,
    context_dim: int = 64,
    n_max: int = 32,
    logit_softcap: float = 30.0,
) -> MultiStreamComponents:
    """Build the default multi-stream setup with all byte-stream components.

    Creates four streams:
    - **LOGIT** (writable, dim=vocab_size): starts as zeros (uniform prior).
    - **TOKENS** (read-only, dim=vocab_size): one-hot of input_ids.
    - **CONTEXT** (writable, dim=context_dim): zero-initialized scratch.
    - **STRUCTURAL** (read-only, dim~239): precomputed byte-level features.

    Also configures number-position compression via ``NumberExtractor``.

    Returns:
        A :class:`MultiStreamComponents` dataclass with builder, hierarchy,
        and UTF-8 prior ready to be passed to :class:`MultiStreamGPT`.
    """
    # -- STRUCTURAL stream components (~239 dims total) --
    structural_components: list[nn.Module] = [
        DocBoundaryComponent(bos_id=tok.bos_id, num_freqs=2),  # 4
        SinCosPositionComponent(num_freqs=10),  # 20
        ByteCategoryComponent(tok=tok, embed_dim=4),  # 4
        MultiByteStateComponent(tok=tok, id_freqs=3),  # 8
        CaseComponent(tok=tok),  # 2
        VowelConsonantComponent(tok=tok),  # 2
        ColumnPositionComponent(tok=tok, num_freqs=3),  # 6
        RepeatedByteComponent(),  # 2
        PunctuationDepthComponent(tok=tok),  # 2
        ByteCategoryStatsComponent(tok=tok),  # 6
        DigitSequenceComponent(tok=tok, id_freqs=5),  # 12
        DigitComputeComponent(tok=tok),  # 107
        ByteHashComponent(
            tok, window=20, num_hashes=2, boundary=HashBoundary.WORD, track_hits=True
        ),  # 6
        ByteHashComponent(
            tok, window=2, num_hashes=2, boundary=None, track_hits=True
        ),  # 6
        ByteHashComponent(
            tok, window=3, num_hashes=2, boundary=None, track_hits=True
        ),  # 6
        ByteHashComponent(
            tok, window=5, num_hashes=2, boundary=None, track_hits=True
        ),  # 6
        ByteHashComponent(
            tok, window=8, num_hashes=2, boundary=None, track_hits=True
        ),  # 6
        ByteHashComponent(
            tok, window=8, num_hashes=2, boundary=HashBoundary.DIGIT, track_hits=True
        ),  # 6
        BoundaryComponent(
            tok,
            word_pos_freqs=3,
            word_id_freqs=3,
            sent_pos_freqs=2,
            sent_id_freqs=2,
            para_pos_freqs=2,
            para_id_freqs=2,
        ),  # 28
    ]

    # -- Stream definitions --
    logit_id = StreamID(StreamType.LOGIT)
    tokens_id = StreamID(StreamType.TOKENS)
    context_id = StreamID(StreamType.CONTEXT)
    structural_id = StreamID(StreamType.STRUCTURAL)

    stream_defs = [
        StreamDef(logit_id, read_only=False, normalize=False, dim=vocab_size, auto_zeros=True),
        StreamDef(tokens_id, read_only=True, normalize=False, auto_onehot=True),
        StreamDef(context_id, read_only=False, normalize=True, dim=context_dim, auto_zeros=True),
        StreamDef(structural_id, read_only=True, normalize=False, components=structural_components),
    ]

    # -- Compression (numbers) --
    compressions = {CompressionType.NUMBER: NumberExtractor(tok, n_max=n_max)}
    compress_streams = [logit_id, tokens_id, context_id, structural_id]

    builder = MultiStreamBuilder(
        stream_defs=stream_defs,
        vocab_size=vocab_size,
        compressions=compressions,
        compress_streams=compress_streams,
    )

    logit_hierarchy = ByteLogitHierarchy(vocab_size, tok, logit_softcap)
    utf8_prior = UTF8Prior(tok)

    return MultiStreamComponents(
        builder=builder,
        logit_hierarchy=logit_hierarchy,
        utf8_prior=utf8_prior,
        tok=tok,
    )


# ---------------------------------------------------------------------------
# Bigram prior layer
# ---------------------------------------------------------------------------


class BigramPriorLayer(nn.Module):
    """Trainable bigram logit prior.

    Stores a ``(vocab_size, vocab_size)`` matrix where entry ``[a, b]`` is the
    logit adjustment for token *b* following token *a*.  Applied as an additive
    delta to the LOGIT stream.

    At position *t*, ``input_ids[t]`` is the current token and ``logits[t]``
    predicts the next token, so ``bigram_logits[input_ids[t]]`` gives
    ``P(next | current)`` as unnormalized logit deltas.
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        self.bigram_logits = nn.Parameter(torch.zeros(vocab_size, vocab_size))

    def forward(self, input_ids: Tensor) -> Tensor:
        """Return (B, S, vocab_size) bigram logit deltas."""
        return self.bigram_logits[input_ids]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LOGIT_SID = StreamID(StreamType.LOGIT)


def _broadcast(val: int | list[int], n: int, name: str) -> list[int]:
    """Broadcast a scalar to a list of length *n*, or validate an existing list."""
    if isinstance(val, int):
        return [val] * n
    if len(val) != n:
        raise ValueError(f"{name} list length ({len(val)}) must match num_layers ({n})")
    return list(val)


def _build_instance_map(
    mapping: list[int | None] | None,
    num_layers: int,
    name: str,
) -> tuple[list[int | None], int]:
    """Validate an instance-sharing map and return (map, num_unique_instances).

    Args:
        mapping: ``None`` (feature disabled) or a list of length *num_layers*
            where each entry is ``None`` (layer skips this feature) or an
            integer instance index.
        num_layers: expected list length.
        name: for error messages.

    Returns:
        (validated_map, num_unique) where *num_unique* is 0 when mapping is
        ``None`` (feature disabled entirely).
    """
    if mapping is None:
        return [None] * num_layers, 0
    if len(mapping) != num_layers:
        raise ValueError(
            f"{name} length ({len(mapping)}) must match num_layers ({num_layers})"
        )
    indices = [i for i in mapping if i is not None]
    if not indices:
        return list(mapping), 0
    num_unique = max(indices) + 1
    # Validate contiguous indices 0..num_unique-1
    if set(indices) != set(range(num_unique)):
        raise ValueError(
            f"{name} instance indices must be contiguous from 0; "
            f"got {sorted(set(indices))}"
        )
    return list(mapping), num_unique


# ---------------------------------------------------------------------------
# MultiStreamGPT
# ---------------------------------------------------------------------------


class MultiStreamGPT(nn.Module):
    """Top-level multi-stream byte-level language model.

    Combines:

    1. Stream construction (``MultiStreamBuilder``)
    2. UTF-8 prior (hard constraints on impossible tokens)
    3. Trainable bigram prior
    4. Pre-processing causal conv layers (``MultiStreamCausalConvLayers``)
    5. Transformer attention blocks (``MultiStreamBlock``)
    6. Logit softcap

    The LOGIT stream is initialized to zeros (uniform prior), refined through
    priors and layers, and returned as the final prediction logits.
    """

    def __init__(
        self,
        components: MultiStreamComponents,
        multi_head_dim: int | list[int],
        num_heads: int | list[int],
        num_kv_heads: int | list[int],
        num_layers: int,
        # Pre-processing conv (MultiStreamCausalConvLayers)
        num_preconv_layers: int = 2,
        preconv_kernel_size: int | list[int] = 4,
        preconv_groups: int = 1,
        # Per-block conv mapping: list[int|None], length=num_layers
        block_conv_map: list[int | None] | None = None,
        block_conv_kernel_size: int | list[int] = 4,
        block_conv_groups: int | list[int] = 1,
        # Arithmetic mapping: list[int|None], length=num_layers
        arith_map: list[int | None] | None = None,
        arith_n_max: int = 16,
        # Attention config
        mixing_config: StreamMixingConfig | None = None,
        mlp_hidden_dim: int | None = None,
        gated_mlp_output: bool = True,
        qk_gain_init: float = 0.0,
        k_shift: bool = True,
        # Output
        logit_softcap: float = 30.0,
        # Priors
        include_bigram_prior: bool = True,
        # Linear mode
        linear_mode: str = "dense",
        linear_kwargs: dict | None = None,
        # Init noise
        init_noise_std: float = 0.01,
    ):
        super().__init__()

        self.builder = components.builder
        self.utf8_prior = components.utf8_prior
        self.logit_softcap = logit_softcap
        self.num_layers = num_layers

        stream_config = components.stream_config
        vocab_size = components.vocab_size

        # -- Broadcast per-layer params --
        per_mhd = _broadcast(multi_head_dim, num_layers, "multi_head_dim")
        per_heads = _broadcast(num_heads, num_layers, "num_heads")
        per_kv_heads = _broadcast(num_kv_heads, num_layers, "num_kv_heads")

        # -- Bigram prior --
        self.bigram_prior: BigramPriorLayer | None = None
        if include_bigram_prior:
            self.bigram_prior = BigramPriorLayer(vocab_size)

        # -- Pre-processing conv layers --
        self.preconv_layers: MultiStreamCausalConvLayers | None = None
        if num_preconv_layers > 0:
            self.preconv_layers = MultiStreamCausalConvLayers(
                stream_config=stream_config,
                num_layers=num_preconv_layers,
                kernel_size=preconv_kernel_size,
                conv_groups=preconv_groups,
                logit_hierarchy=components.logit_hierarchy,
            )

        # -- Arithmetic attention instances --
        arith_map_validated, num_arith = _build_instance_map(
            arith_map, num_layers, "arith_map"
        )
        self._arith_map = arith_map_validated
        self.arith_attns = nn.ModuleList()
        if num_arith > 0:
            compressed_sids = [s.name for s in stream_config.streams]
            for _ in range(num_arith):
                self.arith_attns.append(
                    CausalArithmeticMultiStreamAttention(
                        stream_config=stream_config,
                        tok=components.tok,
                        n_max=arith_n_max,
                        compressed_stream_ids=compressed_sids,
                    )
                )

        # -- Per-block conv instances --
        conv_map_validated, num_block_conv = _build_instance_map(
            block_conv_map, num_layers, "block_conv_map"
        )
        self._block_conv_map = conv_map_validated
        bconv_ks = (
            _broadcast(block_conv_kernel_size, num_block_conv, "block_conv_kernel_size")
            if num_block_conv > 0
            else []
        )
        bconv_g = (
            _broadcast(block_conv_groups, num_block_conv, "block_conv_groups")
            if num_block_conv > 0
            else []
        )
        self.block_convs = nn.ModuleList()
        for j in range(num_block_conv):
            self.block_convs.append(
                MultiStreamCausalConv(
                    stream_config=stream_config,
                    kernel_size=bconv_ks[j],
                    conv_groups=bconv_g[j],
                    logit_hierarchy=components.logit_hierarchy,
                    linear_mode=linear_mode,
                    linear_kwargs=linear_kwargs,
                )
            )

        # -- Attention blocks --
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            conv_i = (
                self.block_convs[conv_map_validated[i]]
                if conv_map_validated[i] is not None
                else None
            )
            arith_i = (
                self.arith_attns[arith_map_validated[i]]
                if arith_map_validated[i] is not None
                else None
            )
            self.blocks.append(
                MultiStreamBlock(
                    multi_head_dim=per_mhd[i],
                    num_heads=per_heads[i],
                    num_kv_heads=per_kv_heads[i],
                    stream_config=stream_config,
                    mixing_config=mixing_config,
                    mlp_hidden_dim=mlp_hidden_dim,
                    gated_mlp_output=gated_mlp_output,
                    qk_gain_init=qk_gain_init,
                    linear_mode=linear_mode,
                    linear_kwargs=linear_kwargs,
                    k_shift=k_shift,
                    conv=conv_i,
                    arith_attn=arith_i,
                    logit_hierarchy=components.logit_hierarchy,
                )
            )

        # -- Init noise for symmetry breaking --
        if init_noise_std > 0:
            self._apply_init_noise(init_noise_std)

    # -- Initialization --

    def _apply_init_noise(self, std: float) -> None:
        """Add small Gaussian noise to alpha/beta/gate params for symmetry breaking."""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if any(k in name for k in ("alpha", "beta", "gate")):
                    param.add_(torch.randn_like(param) * std)

    # -- Forward --

    def forward(
        self,
        input_ids: Tensor,
        dtype: torch.dtype = torch.bfloat16,
    ) -> tuple[Tensor, dict[StreamID, Tensor]]:
        """Full forward pass.

        Args:
            input_ids: (B, S) token IDs.
            dtype: dtype for stream construction.

        Returns:
            Tuple of (logits, streams) where logits is (B, S, vocab_size)
            and streams is the full stream dict after all layers.
        """
        # 1. Build streams: LOGIT=zeros, TOKENS=onehot, CONTEXT=zeros,
        #    STRUCTURAL=components.  compressed/views from NumberExtractor.
        streams, compressed, views = self.builder(input_ids, dtype=dtype)

        # 2. UTF-8 prior: compute once, use twice (capped early + hard at end).
        _cat_mask, token_mask = self.utf8_prior(input_ids)
        token_mask = token_mask.to(dtype=streams[_LOGIT_SID].dtype)

        # Early: capped finite bias so RMSNorm in blocks stays stable.
        streams[_LOGIT_SID] = streams[_LOGIT_SID] + token_mask.clamp(
            min=-self.logit_softcap
        )

        # 3. Bigram prior: trainable logit adjustments.
        #    Applied before blocks so the LOGIT stream starts with a
        #    data-driven prior rather than flat zeros.
        if self.bigram_prior is not None:
            streams[_LOGIT_SID] = streams[_LOGIT_SID] + self.bigram_prior(input_ids)

        # 3. Pre-processing conv layers.
        if self.preconv_layers is not None:
            streams = self.preconv_layers(streams)

        # 4. Attention blocks.
        for block in self.blocks:
            streams = block(streams, compressed=compressed, views=views)

        # 5. Extract logits, apply softcap, then UTF-8 prior.
        logits = streams[_LOGIT_SID]
        if self.logit_softcap > 0:
            logits = softcap_linear(x=logits, cap=self.logit_softcap)

        # 6. UTF-8 prior: hard -inf mask (reuses token_mask from step 2).
        logits = logits + token_mask

        return logits, streams

    def compute_loss(
        self,
        input_ids: Tensor,
        target_ids: Tensor | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> tuple[Tensor, Tensor]:
        """Forward + cross-entropy loss.

        Args:
            input_ids: (B, S) token IDs.
            target_ids: (B, S) target IDs, pre-shifted by the data loader
                (i.e. ``target_ids[t]`` is the expected next token after
                ``input_ids[t]``).  If ``None``, targets are derived by
                shifting ``input_ids`` by 1 (losing 1 position).
            dtype: dtype for stream construction.

        Returns:
            (loss, logits) where loss is a scalar and logits is (B, S, V).
        """
        logits, _streams = self.forward(input_ids, dtype=dtype)
        if target_ids is not None:
            # Pre-shifted: all S positions contribute to loss.
            flat_logits = logits.float().reshape(-1, logits.size(-1))
            flat_targets = target_ids.reshape(-1)
        else:
            # Self-shift: predict input_ids[t+1] from logits[t].
            flat_logits = logits[:, :-1].float().contiguous().view(-1, logits.size(-1))
            flat_targets = input_ids[:, 1:].contiguous().view(-1)
        loss = F.cross_entropy(flat_logits, flat_targets, reduction="mean")
        return loss, logits
