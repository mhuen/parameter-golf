"""Efficient byte-level tokenizer that maps only used UTF-8 byte values to contiguous IDs.

Architecture:
    ByteCategory  — StrEnum of token property categories
    ByteInfo      — frozen dataclass describing one byte value's inherent properties
    ByteTable     — all 256 ByteInfo entries + array-level category masks (byte-indexed)
    OtherTokenStrategy — what to do with tokens not in the `keep` set
    EfficientByteTokenizer — configurable tokenizer: assigns contiguous IDs, supports
                             filtering by category, folding (e.g. uppercase -> lowercase)

Usage:
    from efficient_byte_tokenizer import EfficientByteTokenizer, ByteCategory, OtherTokenStrategy

    # Default: all 206 used bytes, vocab_size=208
    tok = EfficientByteTokenizer()

    # All 256 bytes (including unused), vocab_size=258
    tok = EfficientByteTokenizer(discard_unused_bytes=False)

    # Letters only, drop everything else
    tok = EfficientByteTokenizer(keep=ByteCategory.LETTER)

    # Lowercase letters only (fold uppercase first), non-letters become boundary tokens
    tok = EfficientByteTokenizer(
        keep=ByteCategory.LETTER,
        fold=ByteCategory.UPPERCASE,
        other=OtherTokenStrategy.BOUNDARY,
    )

    # Multiple keep categories
    tok = EfficientByteTokenizer(keep={ByteCategory.LETTER, ByteCategory.DIGIT})

    # Encode / filter
    ids = tok.encode("Hello, World! 123")
    ids = tok.filter_stream(ids)   # apply DROP/BOUNDARY collapsing
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

_DROP_MARKER = np.uint16(0xFFFF)


# ---------------------------------------------------------------------------
# ByteCategory
# ---------------------------------------------------------------------------


class ByteCategory(StrEnum):
    """Token property categories. Non-exclusive — combine masks with & and |.

    Hierarchy:
        SPECIAL: PAD, BOS, BOUNDARY
        ASCII: DIGIT, LETTER (UPPERCASE/LOWERCASE, VOWEL/CONSONANT),
               SEPARATOR, PUNCTUATION, SYMBOL
        MULTIBYTE: MB_LEADING (MB_LEAD_2, MB_LEAD_3, MB_LEAD_4), MB_CONTINUATION
    """

    # Special tokens (not byte-backed, assigned by the tokenizer)
    PAD = "pad"
    BOS = "bos"
    BOUNDARY = "boundary"
    SPECIAL = "special"

    # ASCII structural (byte 32-126)
    ASCII = "ascii"
    DIGIT = "digit"
    LETTER = "letter"
    UPPERCASE = "uppercase"
    LOWERCASE = "lowercase"
    VOWEL = "vowel"
    CONSONANT = "consonant"
    SEPARATOR = "separator"
    PUNCTUATION = "punctuation"
    SYMBOL = "symbol"

    # UTF-8 multi-byte structure (byte 128-240)
    MULTIBYTE = "multibyte"
    MB_LEADING = "mb_leading"
    MB_LEAD_2 = "mb_lead_2"
    MB_LEAD_3 = "mb_lead_3"
    MB_LEAD_4 = "mb_lead_4"
    MB_CONTINUATION = "mb_continuation"


class OtherTokenStrategy(StrEnum):
    """How to handle tokens not in the `keep` set."""

    DROP = "drop"  # remove from stream
    PAD = "pad"  # replace with pad_id (preserves positions)
    BOUNDARY = (
        "boundary"  # collapse consecutive non-kept tokens into one boundary token
    )


# ---------------------------------------------------------------------------
# ByteInfo
# ---------------------------------------------------------------------------

_VOWELS = frozenset(b"aeiouAEIOU")
_PUNCTUATION = frozenset(b".,;:!?'\"()-[]{}/\\")


def _classify_byte(b: int) -> frozenset[ByteCategory]:
    """Determine the inherent categories of a raw byte value (0-255)."""
    cats: set[ByteCategory] = set()

    if 32 <= b < 127:
        cats.add(ByteCategory.ASCII)
        if 48 <= b <= 57:
            cats.add(ByteCategory.DIGIT)
        elif (65 <= b <= 90) or (97 <= b <= 122):
            cats.add(ByteCategory.LETTER)
            cats.add(ByteCategory.UPPERCASE if b <= 90 else ByteCategory.LOWERCASE)
            cats.add(ByteCategory.VOWEL if b in _VOWELS else ByteCategory.CONSONANT)
        elif b == 32:
            cats.add(ByteCategory.SEPARATOR)
        elif b in _PUNCTUATION:
            cats.add(ByteCategory.PUNCTUATION)
        else:
            cats.add(ByteCategory.SYMBOL)

    elif 0x80 <= b <= 0xBF:
        cats.add(ByteCategory.MULTIBYTE)
        cats.add(ByteCategory.MB_CONTINUATION)
    elif 0xC2 <= b <= 0xDF:
        cats.add(ByteCategory.MULTIBYTE)
        cats.add(ByteCategory.MB_LEADING)
        cats.add(ByteCategory.MB_LEAD_2)
    elif 0xE0 <= b <= 0xEF:
        cats.add(ByteCategory.MULTIBYTE)
        cats.add(ByteCategory.MB_LEADING)
        cats.add(ByteCategory.MB_LEAD_3)
    elif 0xF0 <= b <= 0xF4:
        cats.add(ByteCategory.MULTIBYTE)
        cats.add(ByteCategory.MB_LEADING)
        cats.add(ByteCategory.MB_LEAD_4)

    return frozenset(cats)


@dataclass(frozen=True)
class ByteInfo:
    """Inherent properties of a single byte value. Independent of any token ID mapping."""

    byte_value: int
    categories: frozenset[ByteCategory]

    @property
    def char(self) -> str | None:
        """Printable ASCII character, or None."""
        if 32 <= self.byte_value < 127:
            return chr(self.byte_value)
        return None

    @property
    def lowercase_byte(self) -> int | None:
        """Lowercase counterpart byte value, or None if not an uppercase letter."""
        if 65 <= self.byte_value <= 90:
            return self.byte_value + 32
        return None

    def has(self, cat: ByteCategory) -> bool:
        return cat in self.categories


# ---------------------------------------------------------------------------
# ByteTable
# ---------------------------------------------------------------------------


class ByteTable:
    """Properties of all 256 byte values with array-level access.

    Indexed by raw byte value (0-255). This is the single source of truth
    for byte properties — the tokenizer reindexes these into token-space.
    """

    def __init__(self) -> None:
        self.entries: tuple[ByteInfo, ...] = tuple(
            ByteInfo(byte_value=b, categories=_classify_byte(b)) for b in range(256)
        )
        # Precompute byte-indexed masks: shape (256,) bool
        self._masks: dict[ByteCategory, np.ndarray] = {}
        for cat in ByteCategory:
            if cat in (
                ByteCategory.PAD,
                ByteCategory.BOS,
                ByteCategory.BOUNDARY,
                ByteCategory.SPECIAL,
            ):
                self._masks[cat] = np.zeros(256, dtype=bool)
            else:
                m = np.array([cat in e.categories for e in self.entries], dtype=bool)
                m.flags.writeable = False
                self._masks[cat] = m

    def __getitem__(self, byte_value: int) -> ByteInfo:
        return self.entries[byte_value]

    def mask(self, category: ByteCategory | str) -> np.ndarray:
        """Byte-indexed boolean mask of shape (256,)."""
        return self._masks[ByteCategory(category)]


# Module-level singleton
BYTE_TABLE = ByteTable()


# ---------------------------------------------------------------------------
# EfficientByteTokenizer
# ---------------------------------------------------------------------------

# Byte values absent from the FineWeb 10B UTF-8 dataset
_UNUSED_BYTES = frozenset(range(0, 32)) | {127, 192, 193} | frozenset(range(241, 256))
_ALL_USED_BYTES = tuple(b for b in range(256) if b not in _UNUSED_BYTES)


class EfficientByteTokenizer:
    """Configurable byte-level tokenizer.

    Default (no args): all 206 used bytes -> vocab_size=208.
    With discard_unused_bytes=False: all 256 bytes -> vocab_size=258.
    With keep/fold/other: filtered/transformed token space.

    Token layout:
        [pad=0] [bos=1] ([boundary=2]) [byte_0] [byte_1] ...
        boundary token only present when other=BOUNDARY.
    """

    def __init__(
        self,
        *,
        discard_unused_bytes: bool = True,
        keep: ByteCategory | set[ByteCategory] | frozenset[ByteCategory] | None = None,
        fold: ByteCategory | set[ByteCategory] | frozenset[ByteCategory] | None = None,
        other: OtherTokenStrategy = OtherTokenStrategy.DROP,
    ) -> None:
        # Normalize inputs
        if isinstance(keep, ByteCategory):
            keep = frozenset({keep})
        elif keep is not None:
            keep = frozenset(keep)
        if isinstance(fold, ByteCategory):
            fold = frozenset({fold})
        elif fold is not None:
            fold = frozenset(fold)
        else:
            fold = frozenset()

        self.discard_unused_bytes = discard_unused_bytes
        self.keep_categories = keep
        self.fold_categories = fold
        self.other = OtherTokenStrategy(other)

        # --- Step 1: Build byte-level fold map (256 -> 256) ---
        fold_map = np.arange(256, dtype=np.uint8)
        if ByteCategory.UPPERCASE in fold:
            for entry in BYTE_TABLE.entries:
                if entry.lowercase_byte is not None:
                    fold_map[entry.byte_value] = entry.lowercase_byte
        self._fold_map = fold_map

        # --- Step 2: After folding, which bytes get their own token ID? ---
        base_bytes = _ALL_USED_BYTES if discard_unused_bytes else tuple(range(256))
        # Fold then deduplicate to get effective byte set
        folded_used = sorted(set(int(fold_map[b]) for b in base_bytes))

        # Filter by keep categories (if specified)
        if keep is not None:
            kept_bytes = tuple(
                b
                for b in folded_used
                if any(cat in BYTE_TABLE[b].categories for cat in keep)
            )
        else:
            kept_bytes = tuple(folded_used)

        # --- Step 3: Assign special tokens ---
        self.pad_id: int = 0
        self.bos_id: int = 1
        if self.other == OtherTokenStrategy.BOUNDARY:
            self.boundary_id: int | None = 2
            self.n_special: int = 3
        else:
            self.boundary_id = None
            self.n_special = 2

        self.used_bytes: tuple[int, ...] = kept_bytes
        self.vocab_size: int = self.n_special + len(kept_bytes)

        # --- Step 4: Build lookup tables ---

        # kept byte value -> token ID
        _kept_to_id: dict[int, int] = {}
        self._id_to_byte = np.zeros(self.vocab_size, dtype=np.uint8)
        for i, b in enumerate(kept_bytes):
            tid = self.n_special + i
            _kept_to_id[b] = tid
            self._id_to_byte[tid] = b

        # What non-kept bytes map to
        if self.other == OtherTokenStrategy.PAD:
            _non_kept_id = np.uint16(self.pad_id)
        elif self.other == OtherTokenStrategy.BOUNDARY:
            assert self.boundary_id is not None
            _non_kept_id = np.uint16(self.boundary_id)
        else:
            _non_kept_id = _DROP_MARKER

        # byte_to_id: raw byte -> token ID (incorporates fold + keep)
        self._byte_to_id = np.full(256, _non_kept_id, dtype=np.uint16)
        for b in base_bytes:
            folded = int(fold_map[b])
            if folded in _kept_to_id:
                self._byte_to_id[b] = _kept_to_id[folded]

        # --- Step 5: Lowercase token mapping ---
        self._to_lower = np.arange(self.vocab_size, dtype=np.uint16)
        for entry in BYTE_TABLE.entries:
            if entry.lowercase_byte is not None:
                uid = int(self._byte_to_id[entry.byte_value])
                lid = int(self._byte_to_id[entry.lowercase_byte])
                if (
                    uid != _DROP_MARKER
                    and lid != _DROP_MARKER
                    and uid >= self.n_special
                ):
                    self._to_lower[uid] = lid

        # --- Step 6: Token-space category masks ---
        self._masks: dict[ByteCategory, np.ndarray] = {}
        for cat in ByteCategory:
            m = np.zeros(self.vocab_size, dtype=bool)
            if cat == ByteCategory.PAD:
                m[self.pad_id] = True
            elif cat == ByteCategory.BOS:
                m[self.bos_id] = True
            elif cat == ByteCategory.BOUNDARY:
                if self.boundary_id is not None:
                    m[self.boundary_id] = True
            elif cat == ByteCategory.SPECIAL:
                m[: self.n_special] = True
            else:
                byte_mask = BYTE_TABLE.mask(cat)
                for ti in range(self.n_special, self.vocab_size):
                    m[ti] = byte_mask[self._id_to_byte[ti]]
            m.flags.writeable = False
            self._masks[cat] = m

    # --- Category access ---

    def mask(self, category: ByteCategory | str) -> np.ndarray:
        """Boolean mask of shape (vocab_size,) for a token category.

        Combinable: tok.mask("uppercase") & tok.mask("vowel")
        """
        return self._masks[ByteCategory(category)]

    def ids(self, category: ByteCategory | str) -> np.ndarray:
        """Array of token IDs belonging to a category."""
        return np.where(self.mask(category))[0]

    def token_info(self, token_id: int) -> ByteInfo | None:
        """Get the ByteInfo for a token ID, or None for special tokens."""
        if token_id < self.n_special:
            return None
        return BYTE_TABLE[int(self._id_to_byte[token_id])]

    # --- Encode / decode ---

    def encode(self, text: str) -> np.ndarray:
        """Encode a UTF-8 string to token IDs (uint16).

        Non-kept tokens are mapped according to the `other` strategy:
        DROP -> 0xFFFF markers (use filter_stream to remove),
        PAD -> pad_id, BOUNDARY -> boundary_id.
        """
        raw = np.frombuffer(text.encode("utf-8", errors="replace"), dtype=np.uint8)
        return self._byte_to_id[raw]

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.encode(t) for t in texts]

    def encode_with_bos(self, text: str) -> np.ndarray:
        """Encode with a leading BOS token."""
        encoded = self.encode(text)
        return np.concatenate([[self.bos_id], encoded]).astype(np.uint16)

    def decode(self, ids: np.ndarray | list[int]) -> bytes:
        """Decode token IDs back to raw bytes. Special tokens are skipped."""
        ids = np.asarray(ids)
        m = (ids >= self.n_special) & (ids != _DROP_MARKER)
        return self._id_to_byte[ids[m]].tobytes()

    def decode_to_str(self, ids: np.ndarray | list[int]) -> str:
        return self.decode(ids).decode("utf-8", errors="replace")

    # --- Stream filtering ---

    def filter_stream(self, ids: np.ndarray) -> np.ndarray:
        """Apply other-token strategy post-processing to a token stream.

        - DROP: removes 0xFFFF markers
        - BOUNDARY: collapses consecutive boundary tokens into one
        - PAD: no-op (pad tokens already in place)
        """
        ids = np.asarray(ids, dtype=np.uint16)
        if self.other == OtherTokenStrategy.DROP:
            return ids[ids != _DROP_MARKER]
        if self.other == OtherTokenStrategy.BOUNDARY and len(ids) > 0:
            bid = self.boundary_id
            is_dup = np.zeros(len(ids), dtype=bool)
            is_dup[1:] = (ids[1:] == bid) & (ids[:-1] == bid)
            return ids[~is_dup]
        return ids

    # --- Remapping ---

    def map_token_ids_to_lower(self, ids: np.ndarray) -> np.ndarray:
        """Map token IDs to their lowercase equivalents (A-Z -> a-z)."""
        return self._to_lower[np.asarray(ids)]

    def remap_byte_array(self, raw_bytes: np.ndarray) -> np.ndarray:
        """Convert a uint8 array of raw byte values (0-255) to token IDs.

        Incorporates fold and keep. Non-kept bytes handled per `other` strategy.
        """
        raw_bytes = np.asarray(raw_bytes, dtype=np.uint8)
        return self._byte_to_id[raw_bytes]

    # --- Description ---

    def describe(self) -> str:
        """Human-readable summary of this tokenizer configuration."""
        parts = [f"EfficientByteTokenizer(vocab_size={self.vocab_size}"]
        if not self.discard_unused_bytes:
            parts.append("discard_unused_bytes=False")
        if self.keep_categories is not None:
            parts.append(
                f"keep={{{','.join(c.value for c in sorted(self.keep_categories))}}}"
            )
        if self.fold_categories:
            parts.append(
                f"fold={{{','.join(c.value for c in sorted(self.fold_categories))}}}"
            )
        if self.keep_categories is not None:
            parts.append(f"other={self.other.value}")
        parts[-1] += ")"
        return ", ".join(parts)


if __name__ == "__main__":
    # --- Default config ---
    tok = EfficientByteTokenizer()
    print(tok.describe())
    print(f"  Used bytes: {len(tok.used_bytes)} / 256")

    text = "Hello, world! UTF-8: äöü 你好 🌍"
    ids = tok.encode(text)
    decoded = tok.decode_to_str(ids)
    print(f"  Round-trip: {text == decoded}")

    print("\n  Category counts:")
    for cat in ByteCategory:
        count = tok.mask(cat).sum()
        if count > 0:
            print(f"    {cat.value:>16s}: {count:3d}")

    # --- All 256 bytes ---
    print()
    tok_all = EfficientByteTokenizer(discard_unused_bytes=False)
    print(tok_all.describe())
    print(f"  Used bytes: {len(tok_all.used_bytes)} / 256")

    # --- Letters only, drop others ---
    print()
    tok_letters = EfficientByteTokenizer(keep=ByteCategory.LETTER)
    print(tok_letters.describe())
    ids = tok_letters.encode("Hello, World! 123")
    filtered = tok_letters.filter_stream(ids)
    print(f"  Raw IDs:      {ids}")
    print(f"  Filtered:     {filtered}")
    print(f"  Decoded:      {tok_letters.decode_to_str(filtered)!r}")

    # --- Lowercase letters, boundary for non-letters ---
    print()
    tok_lower = EfficientByteTokenizer(
        keep=ByteCategory.LETTER,
        fold=ByteCategory.UPPERCASE,
        other=OtherTokenStrategy.BOUNDARY,
    )
    print(tok_lower.describe())
    ids = tok_lower.encode("Hello, World! 123")
    filtered = tok_lower.filter_stream(ids)
    print(f"  Raw IDs:      {ids}")
    print(f"  Filtered:     {filtered}")
    print(f"  Decoded:      {tok_lower.decode_to_str(filtered)!r}")
    print(f"  Has uppercase: {tok_lower.mask('uppercase').any()}")

    # --- Multi-byte only ---
    print()
    tok_mb = EfficientByteTokenizer(
        keep=ByteCategory.MULTIBYTE, other=OtherTokenStrategy.PAD
    )
    print(tok_mb.describe())
    ids = tok_mb.encode("Hello 你好 🌍!")
    print(f"  IDs:     {ids}")
    print(f"  Decoded: {tok_mb.decode_to_str(ids)!r}")
