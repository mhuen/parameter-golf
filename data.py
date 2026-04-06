"""Data loading and token streaming for shard-based training data.

Two token formats are supported:
- sp1024: SentencePiece tokenized shards (uint16), used directly.
- byte260: Byte-level shards (uint16), remapped via EfficientByteTokenizer.

Shard format: 256-int32 header (magic=20240520, version=1, num_tokens)
followed by num_tokens uint16 values.
"""

import glob
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Shard I/O
# ---------------------------------------------------------------------------


def load_raw_shard(file: Path) -> np.ndarray:
    """Read a binary shard file and return raw uint16 numpy array."""
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return tokens_np


def load_shard_sp1024(file: Path) -> Tensor:
    """Load a SentencePiece shard as a torch Tensor."""
    return torch.from_numpy(load_raw_shard(file).astype(np.uint16, copy=False))


def load_shard_byte260(file: Path, tok) -> Tensor:
    """Load a byte260 shard, remap to EfficientByteTokenizer IDs."""
    raw = load_raw_shard(file)
    remapped = tok.remap_byte260_shard(raw)
    remapped = tok.filter_stream(remapped)
    return torch.from_numpy(remapped)


# ---------------------------------------------------------------------------
# Token streaming
# ---------------------------------------------------------------------------


class TokenStream:
    """Reads shards sequentially and wraps around forever.

    Args:
        pattern: Glob pattern for shard files.
        load_fn: Callable (Path) -> Tensor that loads a single shard.
            Use functools.partial(load_shard_byte260, tok=tok) for byte data,
            or load_shard_sp1024 for sentencepiece data.
    """

    def __init__(self, pattern: str, load_fn: Callable[[Path], Tensor] = load_shard_sp1024):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.load_fn = load_fn
        self.file_idx = 0
        self.tokens = self.load_fn(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = self.load_fn(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    """Per-rank token loader. Consumes a contiguous chunk from the shared
    token stream, then slices out one disjoint span per rank."""

    def __init__(
        self,
        pattern: str,
        rank: int,
        world_size: int,
        device: torch.device,
        load_fn: Callable[[Path], Tensor] = load_shard_sp1024,
    ):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern, load_fn=load_fn)

    def next_batch(
        self, global_tokens: int, seq_len: int, grad_accum_steps: int
    ) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(
            self.device, non_blocking=True
        )


# ---------------------------------------------------------------------------
# Validation data loading
# ---------------------------------------------------------------------------


def load_validation_sp1024(pattern: str, seq_len: int, max_tokens: int = 0) -> Tensor:
    """Load and concatenate SentencePiece validation shards."""
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_shard_sp1024(f) for f in files]).contiguous()
    if max_tokens > 0:
        tokens = tokens[: max_tokens + 1]
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split too short for seq_len={seq_len}")
    return tokens[: usable + 1]


def load_validation_byte260(pattern: str, seq_len: int, tok, max_tokens: int = 0) -> Tensor:
    """Load and concatenate byte260 validation shards, remapped via tokenizer."""
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_shard_byte260(f, tok) for f in files]).contiguous()
    if max_tokens > 0:
        tokens = tokens[: max_tokens + 1]
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split too short for seq_len={seq_len}")
    return tokens[: usable + 1]
