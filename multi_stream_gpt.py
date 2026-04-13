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
import time
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import glob
from pathlib import Path
import numpy as np

from byte_modules import ByteLogitHierarchy, NumberExtractor, UTF8Prior
from data import load_shard_byte260
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
from modules import (
    SoftcapLinear,
    RMSNorm,
    CenterLastDim,
    KroneckerLinear,
    MonarchLinear,
)
from multi_streams import (
    CompositeStream,
    CompressionType,
    DocBoundaryComponent,
    MultiStreamBuilder,
    MultiStreamConfig,
    SinCosPositionComponent,
    StreamDef,
    StreamID,
    StreamSource,
    StreamType,
)


# ---------------------------------------------------------------------------
# MultiStreamComponents dataclass
# ---------------------------------------------------------------------------


@dataclass
class MultiStreamComponents:
    """All pre-built components needed to construct a MultiStreamGPT."""

    builder: MultiStreamBuilder
    logit_hierarchy: ByteLogitHierarchy | None
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
    structured_output_logits: bool = True,
) -> MultiStreamComponents:
    """Build the default multi-stream setup with all byte-stream components.

    Creates four streams:
    - **LOGIT** (writable, dim=vocab_size): starts as zeros (uniform prior).
    - **TOKENS** (read-only, dim=vocab_size): one-hot of input_ids.
    - **CONTEXT** (writable, dim=context_dim): zero-initialized scratch.
    - **STRUCTURAL** (read-only, dim~158): precomputed byte-level features.

    Also configures number-position compression via ``NumberExtractor``.

    Returns:
        A :class:`MultiStreamComponents` dataclass with builder, hierarchy,
        and UTF-8 prior ready to be passed to :class:`MultiStreamGPT`.
    """
    # -- STRUCTURAL stream components (~158 dims total) --
    structural_components: list[nn.Module] = [
        DocBoundaryComponent(bos_id=tok.bos_id, num_freqs=2),  # 4
        SinCosPositionComponent(num_freqs=10),  # 20
        ByteCategoryComponent(tok=tok),  # 2
        MultiByteStateComponent(tok=tok, id_freqs=3),  # 8
        CaseComponent(tok=tok),  # 2
        VowelConsonantComponent(tok=tok),  # 2
        ColumnPositionComponent(tok=tok, num_freqs=3),  # 6
        RepeatedByteComponent(),  # 2
        PunctuationDepthComponent(tok=tok),  # 3
        ByteCategoryStatsComponent(tok=tok),  # 6
        DigitSequenceComponent(tok=tok, id_freqs=5),  # 13
        DigitComputeComponent(tok=tok),  # 27 (ROTATION encoding: 1 pair * (8*3 + 3))
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
        StreamDef(
            logit_id,
            source=StreamSource.ZEROS,
            read_only=False,
            norm_type=CenterLastDim,
            # input_norm_types=[CenterLastDim, RMSNorm],
            input_norm_types=[RMSNorm],
            dim=vocab_size,
        ),
        # StreamDef(tokens_id, source=StreamSource.ONE_HOT, read_only=True),
        StreamDef(
            context_id,
            source=StreamSource.EMBEDDING,
            read_only=False,
            norm_type=RMSNorm,
            input_norm_types=[RMSNorm],
            dim=context_dim,
        ),
        StreamDef(
            structural_id,
            source=StreamSource.COMPONENTS,
            read_only=True,
            norm_type=RMSNorm,
            components=structural_components,
        ),
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

    logit_hierarchy = (
        ByteLogitHierarchy(vocab_size, tok, logit_softcap)
        if structured_output_logits
        else None
    )
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
# Calibration and bigram initialization
# ---------------------------------------------------------------------------


# TODO: pull out and simplify shared loading logic from load_calibration_tokens and compute_bigram_log_probs
def load_calibration_tokens(
    train_pattern: str,
    tok: EfficientByteTokenizer,
    seq_len: int = 512,
    n_sequences: int = 320,
) -> Tensor:
    """Load token sequences from training shards for model calibration.

    Reads as many shard files as needed to produce up to *n_sequences*
    sequences and returns a single ``(N, seq_len)`` tensor.
    """
    needed = n_sequences * seq_len
    files = [Path(p) for p in sorted(glob.glob(train_pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {train_pattern}")

    chunks: list[Tensor] = []
    total = 0
    for f in files:
        tokens = load_shard_byte260(f, tok)
        chunks.append(tokens)
        total += tokens.numel()
        if total >= needed:
            break

    all_tokens = torch.cat(chunks)
    n_seqs = min(all_tokens.numel() // seq_len, n_sequences)
    return all_tokens[: n_seqs * seq_len].reshape(-1, seq_len)


def compute_bigram_log_probs(
    train_pattern: str,
    tok: EfficientByteTokenizer,
    vocab_size: int,
    smoothing: float = 0.1,
    max_data_bytes: int = 1_000_000,
    calibration_tokens: Tensor | None = None,
) -> Tensor:
    """Compute bigram log-probabilities from training data.

    Counts consecutive-token bigrams, smooths, normalizes, and returns
    ``(V, V)`` float32 log-probabilities.

    If ``calibration_tokens`` is provided (``(N, seq_len)``), uses those
    directly instead of loading from shard files.
    """
    if calibration_tokens is not None:
        all_tokens = calibration_tokens.reshape(-1).numpy().astype(np.int64)
        if max_data_bytes > 0 and len(all_tokens) > max_data_bytes:
            all_tokens = all_tokens[:max_data_bytes]
        if max_data_bytes > 0 and len(all_tokens) < max_data_bytes:
            raise ValueError(
                f"Not enough tokens in provided batches: {len(all_tokens):,} < max_data_bytes ({max_data_bytes:,})"
            )
    else:
        files = [Path(p) for p in sorted(glob.glob(train_pattern))]
        if not files:
            raise FileNotFoundError(f"No files found for pattern: {train_pattern}")

        print(f"bigram_init: loading tokens (max={max_data_bytes:,})...")

        token_chunks: list[np.ndarray] = []
        total = 0
        for f in files:
            toks = load_shard_byte260(f, tok).numpy().astype(np.int64)
            token_chunks.append(toks)
            total += len(toks)
            if max_data_bytes > 0 and total >= max_data_bytes:
                break
        all_tokens = np.concatenate(token_chunks)
        if max_data_bytes > 0 and len(all_tokens) > max_data_bytes:
            all_tokens = all_tokens[:max_data_bytes]
        del token_chunks

    print(f"bigram_init: counting bigrams over {len(all_tokens):,} tokens...")

    # Count bigrams: (all_tokens[i], all_tokens[i+1]) for all consecutive pairs.
    prev_tokens = all_tokens[:-1]
    next_tokens = all_tokens[1:]
    counts = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    np.add.at(counts, (prev_tokens, next_tokens), 1)
    del all_tokens, prev_tokens, next_tokens

    # Smooth, normalize, log-transform.
    counts += smoothing
    counts /= counts.sum(axis=1, keepdims=True)
    log_probs = np.log(counts).astype(np.float32)

    print(
        f"bigram_init: done (min={log_probs.min():.3f} "
        f"max={log_probs.max():.3f} mean={log_probs.mean():.3f})"
    )

    return torch.from_numpy(log_probs)


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
        preconv_channel_shuffle: bool = True,
        # Per-block conv mapping: list[int|None], length=num_layers
        block_conv_map: list[int | None] | None = None,
        block_conv_kernel_size: int | list[int] = 4,
        block_conv_groups: int | list[int] = 1,
        # Arithmetic mapping: list[int|None], length=num_layers
        block_arith_map: list[int | None] | None = None,
        block_arith_n_max: int = 16,
        # Attention config
        mixing_config: StreamMixingConfig | None = None,
        mlp_hidden_dim: int | None = None,
        gated_mlp_output: bool = True,
        gated_attn_output: bool = True,
        gated_conv: bool = True,
        qk_gain_init: float = 0.0,
        k_shift: bool = True,
        # Output
        logit_softcap: float = 30.0,
        value_softcap: float | None = 30.0,
        logit_stream_normalization_factor: float = 1.0,
        # Priors
        include_bigram_prior: bool = True,
        include_utf8_prior: bool = True,
        # Linear mode
        linear_mode: str = "dense",
        linear_kwargs: dict | None = None,
        # Conv input filtering (None = exclude TOKENS)
        conv_input_streams: list[StreamID] | None = None,
        # Conv output filtering (None = exclude LOGIT, only write to context)
        conv_output_streams: list[StreamID] | None = None,
        # Residual mixing and skip connections
        use_resid_mix: bool = False,
        use_unet_skip: bool = False,
        # Alpha/beta bounding (passed to each MultiStreamBlock)
        bound_alpha: bool = True,
        bound_beta: bool = True,
        # Init noise
        init_noise_std: float = 0.01,
    ):
        super().__init__()

        self.builder = components.builder
        self.utf8_prior = components.utf8_prior if include_utf8_prior else None
        self.logit_softcap = SoftcapLinear(logit_softcap) if logit_softcap > 0 else None
        self.logit_stream_normalization_factor = logit_stream_normalization_factor
        self.num_layers = num_layers

        stream_config = components.stream_config
        self._stream_config = stream_config
        self._writable_stream_configs = [
            s for s in stream_config.streams if not s.read_only
        ]

        # One-time init norms (from norm_type) before the first block.
        # Blocks use input_norm_types as pre-norm; no post-norm on residual.
        self.init_norms = nn.ModuleDict(
            {
                s.key: s.norm_type()
                for s in stream_config.streams
                if s.norm_type is not None and not s.read_only
            }
        )
        vocab_size = components.vocab_size

        # -- Broadcast per-layer params --
        per_mhd = _broadcast(multi_head_dim, num_layers, "multi_head_dim")
        per_heads = _broadcast(num_heads, num_layers, "num_heads")
        per_kv_heads = _broadcast(num_kv_heads, num_layers, "num_kv_heads")

        # -- Conv input (+output) stream filtering (default: exclude TOKENS) --
        if conv_input_streams is None:
            conv_input_streams = [
                s.name
                for s in stream_config.streams
                if s.name.type != StreamType.TOKENS
            ]

        if conv_output_streams is None:
            conv_output_streams = [
                s.name
                for s in stream_config.streams
                if not s.read_only and s.name.type != StreamType.LOGIT
            ]

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
                gated_conv=gated_conv,
                logit_hierarchy=components.logit_hierarchy,
                input_stream_ids=conv_input_streams,
                output_stream_ids=conv_output_streams,
                conv_channel_shuffle=preconv_channel_shuffle,
                bound_alpha=bound_alpha,
                bound_beta=bound_beta,
                value_softcap=value_softcap,
            )

        # -- Arithmetic attention instances --
        block_arith_map_validated, num_arith = _build_instance_map(
            block_arith_map, num_layers, "arith_map"
        )
        self._block_arith_map = block_arith_map_validated
        self.arith_attns = nn.ModuleList()
        if num_arith > 0:
            compressed_sids = [s.name for s in stream_config.streams]
            for _ in range(num_arith):
                self.arith_attns.append(
                    CausalArithmeticMultiStreamAttention(
                        stream_config=stream_config,
                        tok=components.tok,
                        n_max=block_arith_n_max,
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
                    gated_conv=gated_conv,
                    logit_hierarchy=components.logit_hierarchy,
                    input_stream_ids=conv_input_streams,
                    value_softcap=value_softcap,
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
                self.arith_attns[block_arith_map_validated[i]]
                if block_arith_map_validated[i] is not None
                else None
            )
            is_last_block = i == num_layers - 1
            self.blocks.append(
                MultiStreamBlock(
                    multi_head_dim=per_mhd[i],
                    num_heads=per_heads[i],
                    num_kv_heads=per_kv_heads[i],
                    stream_config=stream_config,
                    mixing_config=mixing_config,
                    mlp_hidden_dim=mlp_hidden_dim,
                    gated_mlp_output=gated_mlp_output,
                    gated_attn_output=gated_attn_output,
                    qk_gain_init=qk_gain_init,
                    linear_mode=linear_mode,
                    linear_kwargs=linear_kwargs,
                    k_shift=k_shift,
                    conv=conv_i,
                    arith_attn=arith_i,
                    logit_hierarchy=components.logit_hierarchy,
                    mlp_output_stream_ids=[_LOGIT_SID] if is_last_block else None,
                    value_softcap=value_softcap,
                    bound_alpha=bound_alpha,
                    bound_beta=bound_beta,
                )
            )

        # -- Resid mix: per-layer blend of current state with initial x0 --
        self.resid_mix_params: nn.ModuleList | None = None
        if use_resid_mix:
            self.resid_mix_params = nn.ModuleList(
                [
                    nn.ParameterDict(
                        {
                            s.key: nn.Parameter(
                                torch.stack([torch.ones(s.dim), torch.zeros(s.dim)])
                            )
                            for s in self._writable_stream_configs
                        }
                    )
                    for _ in range(num_layers)
                ]
            )

        # -- U-net skip connections: encoder-decoder split with LIFO skips --
        self._num_encoder_layers = num_layers // 2 if use_unet_skip else 0
        num_skips = self._num_encoder_layers
        self.skip_weights: nn.ModuleList | None = None
        if use_unet_skip and num_skips > 0:
            self.skip_weights = nn.ModuleList(
                [
                    nn.ParameterDict(
                        {
                            s.key: nn.Parameter(torch.ones(s.dim))
                            for s in self._writable_stream_configs
                        }
                    )
                    for _ in range(num_skips)
                ]
            )

        # -- Zero-init output projections, then apply noise for symmetry breaking --
        self._init_weights()
        if init_noise_std > 0:
            self._apply_init_noise(init_noise_std)

    # -- Initialization --

    def _init_weights(self) -> None:
        """Zero-init output projections marked with _zero_init."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)) and getattr(
                module, "_zero_init", False
            ):
                nn.init.zeros_(module.weight)
            elif isinstance(module, (KroneckerLinear, MonarchLinear)) and getattr(
                module, "_zero_init", False
            ):
                for p in module.parameters():
                    nn.init.zeros_(p)

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

        # 2. Bigram prior: trainable logit adjustments.
        #    Applied before blocks so the LOGIT stream starts with a
        #    data-driven prior rather than flat zeros.
        if self.bigram_prior is not None:
            streams[_LOGIT_SID] = streams[_LOGIT_SID] + self.bigram_prior(input_ids)

        # 2b. Initial normalization to each stream if specified.
        for s in self._stream_config.streams:
            if s.key in self.init_norms:
                streams[s.name] = self.init_norms[s.key](streams[s.name])

        # 2c. Scale LOGIT stream down so all streams are at ~unit scale.
        #     Reversed in step 5 before softcap.
        if self.logit_stream_normalization_factor != 1:
            streams[_LOGIT_SID] = streams[_LOGIT_SID] * (
                1.0 / self.logit_stream_normalization_factor
            )

        # 3. Pre-processing conv layers.
        if self.preconv_layers is not None:
            streams = self.preconv_layers(streams)

        # 3b. Capture x0 for resid_mix (writable streams only, after preconv).
        x0: dict[StreamID, Tensor] | None = None
        if self.resid_mix_params is not None:
            x0 = {s.name: streams[s.name] for s in self._writable_stream_configs}

        # 4. Attention blocks (with optional skip connections and resid_mix).
        skip_stack: list[dict[StreamID, Tensor]] = []
        num_enc = self._num_encoder_layers

        for i, block in enumerate(self.blocks):
            # Skip: consume in decoder half (LIFO, before resid_mix and block)
            if self.skip_weights is not None and i >= num_enc and skip_stack:
                skip_idx = i - num_enc
                skip_data = skip_stack.pop()
                sw = self.skip_weights[skip_idx]
                for s in self._writable_stream_configs:
                    streams[s.name] = (
                        streams[s.name]
                        + sw[s.key].to(dtype=streams[s.name].dtype) * skip_data[s.name]
                    )

            # Resid_mix: blend current state with x0
            if self.resid_mix_params is not None:
                mix = self.resid_mix_params[i]
                for s in self._writable_stream_configs:
                    m = mix[s.key].to(dtype=streams[s.name].dtype)
                    streams[s.name] = m[0] * streams[s.name] + m[1] * x0[s.name]

            # Block
            streams = block(streams, compressed=compressed, views=views)

            # Skip: store in encoder half (after block)
            if self.skip_weights is not None and i < num_enc:
                skip_stack.append(
                    {s.name: streams[s.name] for s in self._writable_stream_configs}
                )

        # 5. Scale logits back up, apply softcap.
        if self.logit_stream_normalization_factor != 1:
            streams[_LOGIT_SID] *= self.logit_stream_normalization_factor
        if self.logit_softcap is not None:
            streams[_LOGIT_SID] = self.logit_softcap(streams[_LOGIT_SID])

        # 6. UTF-8 prior: hard -inf mask, inference only.
        #    Skipped during training to avoid injecting a bimodal
        #    distribution that destabilises CenterLastDim + the ×factor
        #    scale-up (see NaN analysis).
        if self.utf8_prior is not None and not self.training:
            _cat_mask, token_mask = self.utf8_prior(input_ids)
            streams[_LOGIT_SID] = streams[_LOGIT_SID] + token_mask.to(
                dtype=streams[_LOGIT_SID].dtype
            )

        return streams[_LOGIT_SID], streams

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


# ----------------------------
# MultiStreamGPT factory class
# ----------------------------


# TODO: do placment to device, bfloat16 and restore_low_dim_params_to_fp32 here as well?
def build_multi_stream_gpt(
    tok: EfficientByteTokenizer,
    vocab_size: int,
    context_dim: int = 64,
    n_max: int = 32,
    logit_softcap: float = 30.0,
    structured_output_logits: bool = True,
    calibrate_structural_stream: bool = True,
    calibration_sequence_length: int | None = None,
    calibration_n_sequences: int = 500,
    calibration_tokens: Tensor | None = None,
    compile_calibration: bool = False,
    include_bigram_prior: bool = True,
    train_pattern: str = "./data/datasets/fineweb10B_byte260/fineweb_train_*.bin",
    bigram_init_smoothing: float = 0.1,
    device: torch.device | str = "cpu",
    **model_kwargs,
) -> MultiStreamGPT:
    """Convenience factory to build a MultiStreamGPT with default components.

    Calibration data can be provided in two ways:
    - ``calibration_tokens``: pre-built ``(N, seq_len)`` token tensor
    - ``train_pattern`` + ``calibration_sequence_length``: load from shard files
    If ``calibration_tokens`` is given it takes priority.

    The model (and components) are moved to ``device`` before calibration
    so that compiled calibration runs on the target device.
    """
    components = build_multi_stream_components(
        tok=tok,
        vocab_size=vocab_size,
        context_dim=context_dim,
        n_max=n_max,
        logit_softcap=logit_softcap,
        structured_output_logits=structured_output_logits,
    )
    components.builder.to(device)

    # calibrate structural stream
    if calibrate_structural_stream:
        print("Calibrating structural stream with training data...")

        if calibration_tokens is None:
            if calibration_sequence_length is None:
                raise ValueError(
                    "calibration_sequence_length must be specified when "
                    "calibrate_structural_stream is True and calibration_tokens "
                    "is not provided"
                )
            calibration_tokens = load_calibration_tokens(
                train_pattern=train_pattern,
                tok=tok,
                seq_len=calibration_sequence_length,
                n_sequences=calibration_n_sequences,
            )

        structural_stream: CompositeStream = components.builder.composites[
            str(StreamID(StreamType.STRUCTURAL))
        ]  # type: ignore
        t0 = time.perf_counter()
        structural_stream.calibrate(calibration_tokens, compile=compile_calibration)
        print(f"structural_calibration: {time.perf_counter() - t0:.2f}s")
    else:
        print("Skipping structural stream calibration")

    model = MultiStreamGPT(
        components=components,
        include_bigram_prior=include_bigram_prior,
        logit_softcap=logit_softcap,
        **model_kwargs,
    ).to(device)

    # compute bigram init from training data and load into the model
    if include_bigram_prior:
        if model.bigram_prior is None:
            raise ValueError("bigram_prior should be included based on the flag")
        t0 = time.perf_counter()
        bigram_log_probs = compute_bigram_log_probs(
            train_pattern=train_pattern,
            tok=tok,
            vocab_size=model.bigram_prior.bigram_logits.shape[0],
            smoothing=bigram_init_smoothing,
            calibration_tokens=calibration_tokens,
        )
        with torch.no_grad():
            model.bigram_prior.bigram_logits.data.copy_(bigram_log_probs)
        bl = model.bigram_prior.bigram_logits.data
        print(
            f"bigram_prior: mean={bl.mean():.3f} std={bl.std():.3f} "
            f"min={bl.min():.3f} max={bl.max():.3f} "
            f"({time.perf_counter() - t0:.2f}s)"
        )
    else:
        print("bigram_prior: not included, skipping initialization")

    # Print per-component parameter breakdown
    print("MultiStreamGPT components:")
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        if n > 0:
            suffix = ""
            if isinstance(child, nn.ModuleList) and len(child) > 0:
                suffix = f" ({len(child)} modules)"
            print(f"  {name}: {n:,} params{suffix}")
    total = sum(p.numel() for p in model.parameters())
    print(f"  total: {total:,} params")

    return model
