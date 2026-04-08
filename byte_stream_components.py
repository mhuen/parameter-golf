from __future__ import annotations

import math
from enum import StrEnum

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from efficient_byte_tokenizer import ByteCategory, EfficientByteTokenizer

from modules import sincos_encode
from byte_modules import (
    BYTE_CATEGORY_DEFS,
    NUM_BYTE_CATEGORIES,
)


class HashBoundary(StrEnum):
    """Boundary mode for ByteHashComponent rolling hash."""

    WORD = "word"  # reset at separators
    DIGIT = "digit"  # digit runs only
    CODEPOINT = "codepoint"  # multibyte sequences only


# ---------------------------------------------------------------------------
# Stream components (byte-tokenizer-specific, composable with CompositeStream)
# ---------------------------------------------------------------------------


class ByteCategoryComponent(nn.Module):
    """Byte category (8 classes) as learned embedding stream component.

    Categories: BOS, PAD, DIGIT, LETTER, SEPARATOR, PUNCTUATION, SYMBOL, MULTIBYTE.
    """

    def __init__(self, tok: EfficientByteTokenizer, embed_dim: int = 4):
        super().__init__()
        V = tok.vocab_size
        token_to_cat = torch.zeros(V, dtype=torch.long)
        for ci, cat in enumerate(BYTE_CATEGORY_DEFS):
            mask = tok.mask(cat)
            for tid in range(V):
                if mask[tid]:
                    token_to_cat[tid] = ci
        self.register_buffer("token_to_cat", token_to_cat)
        self.embed = nn.Embedding(NUM_BYTE_CATEGORIES, embed_dim)
        self._dim = embed_dim

    @property
    def dim(self) -> int:
        return self._dim

    def forward(self, input_ids: Tensor, dtype: torch.dtype) -> Tensor:
        cat_ids = self.token_to_cat[input_ids]  # (B, S), per-token lookup
        return self.embed(cat_ids).to(dtype=dtype)


class BoundaryComponent(nn.Module):
    """Word/sentence/paragraph boundary features as sin/cos stream component.

    Encodes both position-within-segment (distance from last boundary, via
    causal cummax) and cumulative boundary IDs (via causal cumsum) for each
    boundary type.

    All operations are causal: cumsum and cummax only depend on positions ≤ t.
    """

    def __init__(
        self,
        tok: EfficientByteTokenizer,
        word_pos_freqs: int = 3,
        word_id_freqs: int = 2,
        sent_pos_freqs: int = 3,
        sent_id_freqs: int = 2,
        para_pos_freqs: int = 2,
        para_id_freqs: int = 1,
        base: float = 10000.0,
    ):
        super().__init__()
        self.base = base
        V = tok.vocab_size

        is_separator = torch.tensor(
            [bool(tok.mask(ByteCategory.SEPARATOR)[tid]) for tid in range(V)],
            dtype=torch.long,
        )
        is_sentence_end = torch.zeros(V, dtype=torch.long)
        is_newline = torch.zeros(V, dtype=torch.long)
        for tid in range(V):
            info = tok.token_info(tid)
            if info is not None:
                if info.byte_value in (46, 33, 63):  # . ! ?
                    is_sentence_end[tid] = 1
                if info.byte_value == 0x0A:  # \n
                    is_newline[tid] = 1
        self.register_buffer("is_separator", is_separator)
        self.register_buffer("is_sentence_end", is_sentence_end)
        self.register_buffer("is_newline", is_newline)
        self.has_newline = bool(is_newline.any())

        self.word_pos_freqs = word_pos_freqs
        self.word_id_freqs = word_id_freqs
        self.sent_pos_freqs = sent_pos_freqs
        self.sent_id_freqs = sent_id_freqs
        self.para_pos_freqs = para_pos_freqs if self.has_newline else 0
        self.para_id_freqs = para_id_freqs if self.has_newline else 0

        self._dim = 2 * (
            word_pos_freqs
            + word_id_freqs
            + sent_pos_freqs
            + sent_id_freqs
            + self.para_pos_freqs
            + self.para_id_freqs
        )

    @property
    def dim(self) -> int:
        return self._dim

    def _pos_within(self, is_boundary: Tensor, input_ids: Tensor) -> Tensor:
        """Compute causal position within current segment.

        Uses cummax to track the last boundary position — only depends on
        positions ≤ t.  If no boundary seen yet, distance is from position 0.
        """
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)
        is_b = is_boundary[input_ids].bool()
        last_b_pos = (
            torch.where(is_b, positions, torch.zeros_like(positions))
            .cummax(dim=1)
            .values
        )
        return positions - last_b_pos

    def _encode_boundary(
        self,
        is_marker: Tensor,
        input_ids: Tensor,
        id_freqs: int,
        pos_freqs: int,
    ) -> list[Tensor]:
        """Encode one boundary type as sin/cos features (causal)."""
        parts = []
        if id_freqs > 0:
            boundary_id = is_marker[input_ids].cumsum(dim=1)  # causal
            parts.append(sincos_encode(boundary_id, id_freqs, self.base))
        if pos_freqs > 0:
            pos_within = self._pos_within(is_marker, input_ids)  # causal
            parts.append(sincos_encode(pos_within, pos_freqs, self.base))
        return parts

    def forward(self, input_ids: Tensor, dtype: torch.dtype) -> Tensor:
        parts: list[Tensor] = []
        parts.extend(
            self._encode_boundary(
                self.is_separator, input_ids, self.word_id_freqs, self.word_pos_freqs
            )
        )
        parts.extend(
            self._encode_boundary(
                self.is_sentence_end,
                input_ids,
                self.sent_id_freqs,
                self.sent_pos_freqs,
            )
        )
        if self.has_newline:
            parts.extend(
                self._encode_boundary(
                    self.is_newline,
                    input_ids,
                    self.para_id_freqs,
                    self.para_pos_freqs,
                )
            )
        return torch.cat(parts, dim=-1).to(dtype=dtype)


class MultiByteStateComponent(nn.Module):
    """UTF-8 multibyte state as raw features + codepoint ID sin/cos (no learnable params).

    Raw features (2 dims):
      Dim 0 (in_sequence): -1 = not in multibyte, +1 = in multibyte sequence.
      Dim 1 (remaining):   -1 = not in multibyte, 0-3 = continuation bytes still needed.

    Codepoint ID (id_freqs * 2 dims):
      Cumsum over lead bytes, encoded as sin/cos. Lets attention match
      "same codepoint" (like word_id matches "same word"). Non-multibyte
      positions get the ID of the last completed/current codepoint.

    Examples (with id_freqs=1):
        ASCII          → [-1, -1, sin(id), cos(id)]
        LEAD_3         → [+1, +2, sin(id), cos(id)]   id increments here
        CONT (2 rem)   → [+1, +1, sin(id), cos(id)]   same id as lead
        CONT (0 rem)   → [+1,  0, sin(id), cos(id)]   same id
        ASCII          → [-1, -1, sin(id), cos(id)]   same id until next lead

    Causal: uses bounded lookback of 3 positions (no sequential scan).
    """

    # Byte type constants (matches UTF8Prior)
    BT_BOS = 0
    BT_PAD = 1
    BT_ASCII = 2
    BT_LEAD_2 = 3
    BT_LEAD_3 = 4
    BT_LEAD_4 = 5
    BT_CONT = 6

    def __init__(
        self, tok: EfficientByteTokenizer, id_freqs: int = 1, base: float = 10000.0
    ):
        super().__init__()
        self.id_freqs = id_freqs
        self.base = base
        V = tok.vocab_size
        token_byte_type = torch.zeros(V, dtype=torch.long)
        for tid in range(V):
            info = tok.token_info(tid)
            if info is None:
                token_byte_type[tid] = self.BT_BOS if tid == tok.bos_id else self.BT_PAD
            elif info.has(ByteCategory.MB_CONTINUATION):
                token_byte_type[tid] = self.BT_CONT
            elif info.has(ByteCategory.MB_LEAD_2):
                token_byte_type[tid] = self.BT_LEAD_2
            elif info.has(ByteCategory.MB_LEAD_3):
                token_byte_type[tid] = self.BT_LEAD_3
            elif info.has(ByteCategory.MB_LEAD_4):
                token_byte_type[tid] = self.BT_LEAD_4
            else:
                token_byte_type[tid] = self.BT_ASCII
        self.register_buffer("token_byte_type", token_byte_type)

        # lead type → expected continuation count
        lead_expected = torch.zeros(7, dtype=torch.long)
        lead_expected[self.BT_LEAD_2] = 1
        lead_expected[self.BT_LEAD_3] = 2
        lead_expected[self.BT_LEAD_4] = 3
        self.register_buffer("lead_expected", lead_expected)

    @property
    def dim(self) -> int:
        return 2 + self.id_freqs * 2

    def forward(self, input_ids: Tensor, dtype: torch.dtype) -> Tensor:
        B, S = input_ids.shape
        device = input_ids.device

        byte_type = self.token_byte_type[input_ids]  # (B, S)

        # Bounded lookback (max 3) to count consecutive continuations
        is_cont = byte_type == self.BT_CONT
        c1 = is_cont
        c2 = torch.zeros(B, S, device=device, dtype=torch.bool)
        c2[:, 1:] = is_cont[:, 1:] & is_cont[:, :-1]
        c3 = torch.zeros(B, S, device=device, dtype=torch.bool)
        c3[:, 2:] = is_cont[:, 2:] & is_cont[:, 1:-1] & is_cont[:, :-2]
        cont_count = c1.long() + c2.long() + c3.long()  # (B, S), 0-3

        # Find lead byte via lookback
        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        lead_pos = (positions - cont_count).clamp(min=0)
        lead_type = byte_type.gather(1, lead_pos)

        # Remaining continuations expected
        non_cont_remaining = self.lead_expected[byte_type]
        cont_remaining = self.lead_expected[lead_type] - cont_count
        remaining = torch.where(is_cont, cont_remaining, non_cont_remaining)

        # In multibyte sequence? (lead or valid continuation)
        is_multibyte = (
            (byte_type == self.BT_LEAD_2)
            | (byte_type == self.BT_LEAD_3)
            | (byte_type == self.BT_LEAD_4)
            | is_cont
        )

        # Raw state features
        in_seq = torch.where(is_multibyte, 1.0, -1.0)
        remaining_f = torch.where(is_multibyte, remaining.float(), -1.0)
        parts = [torch.stack([in_seq, remaining_f], dim=-1)]

        # Codepoint ID: cumsum over lead bytes (causal), zeroed for non-multibyte
        if self.id_freqs > 0:
            is_lead = (
                (byte_type == self.BT_LEAD_2)
                | (byte_type == self.BT_LEAD_3)
                | (byte_type == self.BT_LEAD_4)
            )
            codepoint_id = is_lead.long().cumsum(dim=1)  # causal
            parts.append(sincos_encode(codepoint_id, self.id_freqs, self.base))

        return torch.cat(parts, dim=-1).to(dtype=dtype)


def _run_length(cat_ids: Tensor) -> Tensor:
    """Causal run length: log1p-compressed distance from last category change.

    Position 0 always has run_length=0 (start of first run).
    Within a run of identical values: log1p(0)=0, log1p(1)≈0.69, log1p(5)≈1.79, ...
    """
    B, S = cat_ids.shape
    positions = torch.arange(S, device=cat_ids.device).unsqueeze(0).expand(B, S)
    changed = torch.ones(B, S, device=cat_ids.device, dtype=torch.bool)
    changed[:, 1:] = cat_ids[:, 1:] != cat_ids[:, :-1]
    last_change = (
        torch.where(changed, positions, torch.zeros_like(positions))
        .cummax(dim=1)
        .values
    )
    return (positions - last_change).float().log1p()


class CaseComponent(nn.Module):
    """Letter case state + consecutive same-case run length. 2 dims, 0 params.

    Dim 0 (case_state): +1=uppercase, -1=lowercase, 0=non-letter.
    Dim 1 (run_length): consecutive positions with same case state (0-indexed).

    Causal: only depends on current and prior positions.
    """

    def __init__(self, tok: EfficientByteTokenizer):
        super().__init__()
        V = tok.vocab_size
        # 0=non-letter, 1=upper, 2=lower
        token_case = torch.zeros(V, dtype=torch.long)
        upper_mask = tok.mask(ByteCategory.UPPERCASE)
        lower_mask = tok.mask(ByteCategory.LOWERCASE)
        for tid in range(V):
            if upper_mask[tid]:
                token_case[tid] = 1
            elif lower_mask[tid]:
                token_case[tid] = 2
        self.register_buffer("token_case", token_case)

    @property
    def dim(self) -> int:
        return 2

    def forward(self, input_ids: Tensor, dtype: torch.dtype) -> Tensor:
        case_type = self.token_case[input_ids]  # (B, S), 0/1/2
        state = torch.where(case_type == 1, 1.0, torch.where(case_type == 2, -1.0, 0.0))
        run_len = _run_length(case_type)
        return torch.stack([state, run_len], dim=-1).to(dtype=dtype)


class VowelConsonantComponent(nn.Module):
    """Vowel/consonant state + consecutive run length. 2 dims, 0 params.

    Dim 0 (vc_state): +1=vowel, -1=consonant, 0=non-letter.
    Dim 1 (run_length): consecutive positions with same vc state (0-indexed).

    Causal: only depends on current and prior positions.
    """

    def __init__(self, tok: EfficientByteTokenizer):
        super().__init__()
        V = tok.vocab_size
        # 0=non-letter, 1=vowel, 2=consonant
        token_vc = torch.zeros(V, dtype=torch.long)
        vowel_mask = tok.mask(ByteCategory.VOWEL)
        consonant_mask = tok.mask(ByteCategory.CONSONANT)
        for tid in range(V):
            if vowel_mask[tid]:
                token_vc[tid] = 1
            elif consonant_mask[tid]:
                token_vc[tid] = 2
        self.register_buffer("token_vc", token_vc)

    @property
    def dim(self) -> int:
        return 2

    def forward(self, input_ids: Tensor, dtype: torch.dtype) -> Tensor:
        vc_type = self.token_vc[input_ids]  # (B, S), 0/1/2
        state = torch.where(vc_type == 1, 1.0, torch.where(vc_type == 2, -1.0, 0.0))
        run_len = _run_length(vc_type)
        return torch.stack([state, run_len], dim=-1).to(dtype=dtype)


class ColumnPositionComponent(nn.Module):
    """Position since last newline (column number), sin/cos encoded. 0 params.

    Useful for indentation-sensitive content (code, lists, tables, markdown).
    Causal: uses cummax to track last newline position.
    """

    def __init__(
        self, tok: EfficientByteTokenizer, num_freqs: int = 3, base: float = 10000.0
    ):
        super().__init__()
        self.num_freqs = num_freqs
        self.base = base
        V = tok.vocab_size
        is_newline = torch.zeros(V, dtype=torch.long)
        for tid in range(V):
            info = tok.token_info(tid)
            if info is not None and info.byte_value == 0x0A:
                is_newline[tid] = 1
        self.register_buffer("is_newline", is_newline)

    @property
    def dim(self) -> int:
        return self.num_freqs * 2

    def forward(self, input_ids: Tensor, dtype: torch.dtype) -> Tensor:
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)
        is_nl = self.is_newline[input_ids].bool()
        last_nl = (
            torch.where(is_nl, positions, torch.zeros_like(positions))
            .cummax(dim=1)
            .values
        )
        col_pos = positions - last_nl  # (B, S), causal
        return sincos_encode(col_pos, self.num_freqs, self.base).to(dtype=dtype)


class DigitSequenceComponent(nn.Module):
    """Digit run state + number ID. 0 params.

    Dim 0 (in_digit): +1=digit, -1=non-digit.
    Dim 1 (pos_in_number): log1p-compressed position within digit run, -1 for non-digits.
    Dims 2+ (number_id): sin/cos encoded cumsum over digit-run starts.
      Non-digit positions inherit the last number's ID (like codepoint_id).

    Causal: rising-edge detection + cumsum + cummax.
    """

    def __init__(
        self, tok: EfficientByteTokenizer, id_freqs: int = 1, base: float = 10000.0
    ):
        super().__init__()
        self.id_freqs = id_freqs
        self.base = base
        V = tok.vocab_size
        is_digit = torch.zeros(V, dtype=torch.long)
        digit_mask = tok.mask(ByteCategory.DIGIT)
        for tid in range(V):
            if digit_mask[tid]:
                is_digit[tid] = 1
        self.register_buffer("is_digit", is_digit)

    @property
    def dim(self) -> int:
        return 2 + self.id_freqs * 2

    def forward(self, input_ids: Tensor, dtype: torch.dtype) -> Tensor:
        B, S = input_ids.shape
        is_dig = self.is_digit[input_ids].bool()  # (B, S)

        # In-digit state
        in_digit = torch.where(is_dig, 1.0, -1.0)

        # Position within digit run: use _run_length on binary digit/non-digit
        digit_cat = is_dig.long()
        run_len = _run_length(digit_cat)  # log1p-compressed
        pos_in_num = torch.where(is_dig, run_len, -1.0)

        parts = [torch.stack([in_digit, pos_in_num], dim=-1)]

        # Number ID: cumsum over digit-run starts (rising edge)
        if self.id_freqs > 0:
            run_start = torch.zeros_like(is_dig)
            run_start[:, 0] = is_dig[:, 0]
            run_start[:, 1:] = is_dig[:, 1:] & ~is_dig[:, :-1]
            number_id = run_start.long().cumsum(dim=1)  # causal
            parts.append(sincos_encode(number_id, self.id_freqs, self.base))

        return torch.cat(parts, dim=-1).to(dtype=dtype)


class RepeatedByteComponent(nn.Module):
    """Repeated byte detection + log1p-compressed run length. 2 dims, 0 params.

    Dim 0 (is_repeat): +1=same as previous byte, -1=different (or position 0).
    Dim 1 (run_length): log1p-compressed consecutive same-byte count.

    Causal: compares adjacent positions only.
    """

    @property
    def dim(self) -> int:
        return 2

    def forward(self, input_ids: Tensor, dtype: torch.dtype) -> Tensor:
        run_len = _run_length(input_ids)  # log1p-compressed
        is_repeat = torch.where(run_len > 0, 1.0, -1.0)
        return torch.stack([is_repeat, run_len], dim=-1).to(dtype=dtype)


class PunctuationDepthComponent(nn.Module):
    """Bracket nesting depth + quote parity. 2 dims, 0 params.

    Dim 0 (bracket_depth): cumsum(opens) - cumsum(closes) for ( [ { vs ) ] }.
    Dim 1 (quote_state): +1=inside double quotes (odd cumsum), -1=outside (even).

    Bracket depth can go negative in noisy web text (unbalanced closes) —
    this is left as-is since it's also a useful signal.

    Causal: cumsum only depends on positions ≤ t.
    """

    def __init__(self, tok: EfficientByteTokenizer):
        super().__init__()
        V = tok.vocab_size
        is_open = torch.zeros(V, dtype=torch.long)
        is_close = torch.zeros(V, dtype=torch.long)
        is_quote = torch.zeros(V, dtype=torch.long)
        for tid in range(V):
            info = tok.token_info(tid)
            if info is not None:
                bv = info.byte_value
                if bv in (0x28, 0x5B, 0x7B):  # ( [ {
                    is_open[tid] = 1
                elif bv in (0x29, 0x5D, 0x7D):  # ) ] }
                    is_close[tid] = 1
                elif bv == 0x22:  # "
                    is_quote[tid] = 1
        self.register_buffer("is_open", is_open)
        self.register_buffer("is_close", is_close)
        self.register_buffer("is_quote", is_quote)

    @property
    def dim(self) -> int:
        return 2

    def forward(self, input_ids: Tensor, dtype: torch.dtype) -> Tensor:
        opens = self.is_open[input_ids].cumsum(dim=1)  # causal
        closes = self.is_close[input_ids].cumsum(dim=1)  # causal
        bracket_depth = (opens - closes).float()

        quote_count = self.is_quote[input_ids].cumsum(dim=1)  # causal
        quote_state = torch.where(quote_count % 2 == 1, 1.0, -1.0)

        return torch.stack([bracket_depth, quote_state], dim=-1).to(dtype=dtype)


_STATS_CATEGORIES = [
    ByteCategory.DIGIT,
    ByteCategory.LETTER,
    ByteCategory.SEPARATOR,
    ByteCategory.PUNCTUATION,
    ByteCategory.SYMBOL,
    ByteCategory.MULTIBYTE,
]


class ByteCategoryStatsComponent(nn.Module):
    """Running fraction of each byte category seen so far. 6 dims, 0 params.

    At position t, output[c] = count(category c in positions 0..t) / (t + 1).
    Gives a "document type" signal: prose has high letter fraction, code has
    high symbol/punctuation fraction, numeric data has high digit fraction.

    Categories: DIGIT, LETTER, SEPARATOR, PUNCTUATION, SYMBOL, MULTIBYTE.
    (BOS/PAD excluded — at most 1 token per sequence, fraction ≈ 0.)

    Causal via cumsum.
    """

    NUM_STATS_CATS = len(_STATS_CATEGORIES)

    def __init__(self, tok: EfficientByteTokenizer):
        super().__init__()
        V = tok.vocab_size
        token_to_cat_oh = torch.zeros(V, self.NUM_STATS_CATS)
        for ci, cat in enumerate(_STATS_CATEGORIES):
            mask = tok.mask(cat)
            for tid in range(V):
                if mask[tid]:
                    token_to_cat_oh[tid, ci] = 1.0
        self.register_buffer("token_to_cat_oh", token_to_cat_oh)

    @property
    def dim(self) -> int:
        return self.NUM_STATS_CATS

    def forward(self, input_ids: Tensor, dtype: torch.dtype) -> Tensor:
        cat_oh = self.token_to_cat_oh[input_ids]  # (B, S, 6)
        cumcounts = cat_oh.cumsum(dim=1)  # causal
        positions = torch.arange(
            1, input_ids.shape[1] + 1, device=input_ids.device, dtype=torch.float32
        )
        fractions = cumcounts / positions[None, :, None]
        return fractions.to(dtype=dtype)


class ByteHashComponent(nn.Module):
    """Rolling polynomial hash of byte sequences. 0 learnable params.

    Computes independent hash functions over bytes within the current segment
    (word, digit run, multibyte codepoint) or over the last `window` bytes.
    Each hash is projected to a small range via modular arithmetic, then
    sin/cos encoded for numerically stable attention matching.

    Identical byte sequences → identical features. Collision probability
    with num_hashes=2 is ≈ 1/(997×991) ≈ 0.0001%.

    Args:
        tok: byte tokenizer for byte value lookup.
        window: max lookback window (12 covers most English words).
        num_hashes: number of independent hash functions (output = 2 * num_hashes dims).
        boundary: HashBoundary.WORD (reset at separators),
                  HashBoundary.DIGIT (digit runs only),
                  HashBoundary.CODEPOINT (multibyte sequences only),
                  or None (last N bytes).

    Causal: only looks at positions ≤ t.
    """

    HASH_BASES = [257, 263, 269, 271, 277]
    MODULUS = 2147483647  # 2^31 - 1
    PROJ_PRIMES = [997, 991, 983, 977, 971]  # one per hash function

    def __init__(
        self,
        tok: EfficientByteTokenizer,
        window: int = 12,
        num_hashes: int = 2,
        boundary: HashBoundary | None = HashBoundary.WORD,
        track_hits: bool = True,
        hit_bucket_size: int = 251,
    ):
        super().__init__()
        self.window = window
        self.num_hashes = num_hashes
        self.boundary = boundary
        self.track_hits = track_hits
        self.hit_bucket_size = hit_bucket_size
        V = tok.vocab_size

        # Byte value lookup (+1 so real byte 0x00 ≠ masked-out zeros)
        byte_value = torch.zeros(V, dtype=torch.long)
        for tid in range(V):
            info = tok.token_info(tid)
            if info is not None:
                byte_value[tid] = info.byte_value + 1
        self.register_buffer("byte_value", byte_value)

        # Precompute base^k mod P for each hash function
        P = self.MODULUS
        for h in range(num_hashes):
            base = self.HASH_BASES[h]
            powers = torch.tensor(
                [(base**i) % P for i in range(window)], dtype=torch.long
            )
            self.register_buffer(f"powers_{h}", powers)

        # Boundary detection buffers
        if boundary == HashBoundary.WORD:
            is_sep = torch.zeros(V, dtype=torch.bool)
            sep_mask = tok.mask(ByteCategory.SEPARATOR)
            for tid in range(V):
                if sep_mask[tid]:
                    is_sep[tid] = True
            self.register_buffer("is_separator", is_sep)
        elif boundary == HashBoundary.DIGIT:
            is_dig = torch.zeros(V, dtype=torch.bool)
            d_mask = tok.mask(ByteCategory.DIGIT)
            for tid in range(V):
                if d_mask[tid]:
                    is_dig[tid] = True
            self.register_buffer("is_digit_buf", is_dig)
        elif boundary == HashBoundary.CODEPOINT:
            is_mb = torch.zeros(V, dtype=torch.bool)
            is_lead = torch.zeros(V, dtype=torch.bool)
            for tid in range(V):
                info = tok.token_info(tid)
                if info is not None:
                    if (
                        info.has(ByteCategory.MB_CONTINUATION)
                        or info.has(ByteCategory.MB_LEAD_2)
                        or info.has(ByteCategory.MB_LEAD_3)
                        or info.has(ByteCategory.MB_LEAD_4)
                    ):
                        is_mb[tid] = True
                    if (
                        info.has(ByteCategory.MB_LEAD_2)
                        or info.has(ByteCategory.MB_LEAD_3)
                        or info.has(ByteCategory.MB_LEAD_4)
                    ):
                        is_lead[tid] = True
            self.register_buffer("is_multibyte", is_mb)
            self.register_buffer("is_lead_byte", is_lead)

    @property
    def dim(self) -> int:
        return 2 * self.num_hashes + (2 if self.track_hits else 0)

    def _effective_lookback(self, input_ids: Tensor) -> Tensor:
        """Per-position lookback depth. -1 means inactive (hash = 0)."""
        B, S = input_ids.shape
        device = input_ids.device
        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, S)

        if self.boundary is None:
            return positions  # full window always

        elif self.boundary == HashBoundary.WORD:
            is_sep = self.is_separator[input_ids]
            # Use -1 sentinel so first word (before any separator) is included
            last_sep = (
                torch.where(is_sep, positions, torch.full_like(positions, -1))
                .cummax(dim=1)
                .values
            )
            # pos_in_word: 1 at first byte of word, 2 at second, ...
            # 0 at separator itself
            pos_in_word = positions - last_sep
            return pos_in_word - 1  # -1 at separator → inactive

        elif self.boundary == HashBoundary.DIGIT:
            is_dig = self.is_digit_buf[input_ids]
            run_start = torch.zeros_like(is_dig)
            run_start[:, 0] = is_dig[:, 0]
            run_start[:, 1:] = is_dig[:, 1:] & ~is_dig[:, :-1]
            last_start = (
                torch.where(run_start, positions, torch.full_like(positions, -1))
                .cummax(dim=1)
                .values
            )
            pos_in_run = positions - last_start
            return torch.where(is_dig, pos_in_run, torch.full_like(positions, -1))

        elif self.boundary == HashBoundary.CODEPOINT:
            is_mb = self.is_multibyte[input_ids]
            is_lead = self.is_lead_byte[input_ids]
            last_lead = (
                torch.where(is_lead, positions, torch.full_like(positions, -1))
                .cummax(dim=1)
                .values
            )
            pos_in_cp = positions - last_lead
            return torch.where(is_mb, pos_in_cp, torch.full_like(positions, -1))

        raise ValueError(f"Unknown boundary: {self.boundary!r}")

    def forward(self, input_ids: Tensor, dtype: torch.dtype) -> Tensor:
        B, S = input_ids.shape
        device = input_ids.device
        P = self.MODULUS

        byte_vals = self.byte_value[input_ids]  # (B, S) long
        eff_lb = self._effective_lookback(input_ids)  # (B, S) long

        # Gather last `window` byte values: gathered[b, t, k] = byte_vals[b, t-k]
        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        offsets = torch.arange(self.window, device=device)  # (W,)
        gather_pos = (positions.unsqueeze(-1) - offsets).clamp(min=0)  # (B, S, W)
        gathered = byte_vals.gather(1, gather_pos.reshape(B, -1)).reshape(
            B, S, self.window
        )

        # Mask: zero out bytes outside segment boundary
        mask = eff_lb.unsqueeze(-1) >= offsets  # (B, S, W) bool
        gathered = gathered * mask

        # Compute independent polynomial hashes
        parts = []
        hit_bucket = None
        for h in range(self.num_hashes):
            powers = getattr(self, f"powers_{h}")  # (W,)
            hash_val = ((gathered * powers) % P).sum(dim=-1) % P  # (B, S) long
            # Project to small range for stable sin/cos encoding
            q = self.PROJ_PRIMES[h]
            projected = hash_val % q
            angle = projected.float() * (2 * math.pi / q)
            parts.append(torch.stack([angle.sin(), angle.cos()], dim=-1))
            if h == 0 and self.track_hits:
                hit_bucket = projected  # reuse first hash's projection

        # Hit count: how many previous positions share the same hash bucket
        if self.track_hits:
            Q = self.hit_bucket_size
            bucket = hit_bucket % Q  # (B, S), values in [0, Q)
            one_hot = F.one_hot(bucket, Q).float()  # (B, S, Q)
            cumcount = one_hot.cumsum(dim=1)  # (B, S, Q), causal
            hits = (
                cumcount.gather(2, bucket.unsqueeze(-1)).squeeze(-1) - 1
            )  # exclude self
            # Segment count so far (for fraction): cumsum of active positions
            is_active = (eff_lb >= 0).float()
            seg_count = is_active.cumsum(dim=1).clamp(min=1)
            hit_frac = hits / seg_count
            hit_log = hits.clamp(min=0).log1p()
            parts.append(torch.stack([hit_log, hit_frac], dim=-1))

        return torch.cat(parts, dim=-1).to(dtype=dtype)


_NUMBER_WORDS: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


class PairwiseOp(StrEnum):
    """Operations available in DigitComputeComponent for pairwise number features."""

    # Commutative arithmetic
    ADD = "add"
    MUL = "mul"
    # Non-commutative arithmetic (a=oldest/first, b=most_recent/second)
    SUB = "sub"  # a - b
    RSUB = "rsub"  # b - a
    DIV = "div"  # a / |b|
    RDIV = "rdiv"  # b / |a|
    MOD = "mod"  # fmod(a, |b|)
    RMOD = "rmod"  # fmod(b, |a|)
    # Comparisons
    GT = "gt"  # a > b
    EQ = "eq"  # a == b
    LT = "lt"  # a < b


_PAIRWISE_ARITH_OPS = frozenset(
    {
        PairwiseOp.ADD,
        PairwiseOp.MUL,
        PairwiseOp.SUB,
        PairwiseOp.RSUB,
        PairwiseOp.DIV,
        PairwiseOp.RDIV,
        PairwiseOp.MOD,
        PairwiseOp.RMOD,
    }
)
_PAIRWISE_CMP_OPS = frozenset({PairwiseOp.GT, PairwiseOp.EQ, PairwiseOp.LT})


class DigitComputeComponent(nn.Module):
    """Pairwise arithmetic features from recent numbers in the byte stream.

    Detects numbers in two forms:
    1. **Digit runs:** contiguous ASCII digits (0-9), assembled into integers.
    2. **Number words:** lowercased letter runs matched against a dictionary
       (zero..twenty).

    For each C(k,2) pair of the last k detected numbers, computes a
    configurable subset of:
    - Arithmetic ops: ``add``, ``sub``, ``mul``, ``div``, ``mod``
      — each represented as [sign, next_digit_onehot]
    - Comparisons: ``gt``, ``eq``, ``lt`` — each as 0/1

    Each arithmetic op is encoded as:
    - **sign** (1 dim): +1.0 (positive), -1.0 (negative), 0.0 (zero result)
    - **next-digit one-hot** (12 dims): over {0,1,...,9, '.', END}.
      The digit index is determined by the "active number" — the length
      of the contiguous digit/decimal-point run ending at the current
      token position. If not in a digit run, index=0 (first digit).
      This makes the output directly useful for next-token prediction:
      the component tells the model exactly which digit to emit next.

    dim per pair = n_arith * 13 + n_cmp

    Args:
        tok: byte tokenizer for byte value lookup.
        k: number of recent numbers to track (default 2).
        number_words: dict mapping lowercase words to their numeric value.
            Defaults to zero..twenty.
        ops: set of ``PairwiseOp`` values to include. Defaults to all.
    """

    # Max digits per input number (float64 significant digits)
    _MAX_INPUT_DIGITS = 15

    def __init__(
        self,
        tok: EfficientByteTokenizer,
        k: int = 2,
        number_words: dict[str, int] | None = None,
        ops: set[PairwiseOp] | None = None,
    ):
        super().__init__()
        self.k = k
        self._num_pairs = k * (k - 1) // 2

        # Resolve and store selected ops (preserving canonical order)
        selected = frozenset(ops) if ops is not None else frozenset(PairwiseOp)
        unknown = selected - frozenset(PairwiseOp)
        if unknown:
            raise ValueError(f"Unknown ops: {unknown}")
        self._arith_ops = tuple(
            op for op in PairwiseOp if op in selected and op in _PAIRWISE_ARITH_OPS
        )
        self._cmp_ops = tuple(
            op for op in PairwiseOp if op in selected and op in _PAIRWISE_CMP_OPS
        )

        # Per arithmetic op: 1 (sign) + 12 (next-digit one-hot)
        self._arith_dim = 1 + self.NEXT_DIGIT_VOCAB
        # Per pair: n_arith arith ops + n_cmp comparisons
        self._pair_dim = len(self._arith_ops) * self._arith_dim + len(self._cmp_ops)
        self._dim = self._num_pairs * self._pair_dim
        self._eps = 1e-8
        self._number_words = number_words if number_words is not None else _NUMBER_WORDS

        V = tok.vocab_size
        # digit_value: token_id → 0-9 or -1
        digit_value = torch.full((V,), -1, dtype=torch.long)
        # lowercase_byte: token_id → lowercase ASCII byte or 0 (not a letter)
        lowercase_byte = torch.zeros(V, dtype=torch.long)
        for tid in range(V):
            info = tok.token_info(tid)
            if info is None:
                continue
            if info.has(ByteCategory.DIGIT):
                digit_value[tid] = info.byte_value - 0x30
            if info.has(ByteCategory.LETTER):
                lb = info.lowercase_byte
                lowercase_byte[tid] = lb if lb is not None else info.byte_value
        # is_decimal: token_id → True if byte is '.' (0x2E)
        is_decimal = torch.zeros(V, dtype=torch.bool)
        for tid in range(V):
            info = tok.token_info(tid)
            if info is not None and info.byte_value == 0x2E:
                is_decimal[tid] = True
        self.register_buffer("digit_value", digit_value)
        self.register_buffer("lowercase_byte", lowercase_byte)
        self.register_buffer("is_decimal", is_decimal)

    @property
    def dim(self) -> int:
        return self._dim

    @staticmethod
    def signed_log1p(x: Tensor) -> Tensor:
        return x.sign() * torch.log1p(x.abs())

    def _try_word_number(self, word_bytes: list[int]) -> int | None:
        """Check if a sequence of lowercase bytes matches a number word."""
        word = bytes(word_bytes).decode("ascii", errors="replace")
        return self._number_words.get(word)

    # Next-digit one-hot vocabulary: {0..9, '.', END}
    NEXT_DIGIT_VOCAB = 12
    DIGIT_IDX_DOT = 10
    DIGIT_IDX_END = 11

    def _result_to_string(self, value: float) -> str:
        """Convert arithmetic result magnitude to string for digit lookup."""
        av = abs(value)
        if av == int(av) and av < 1e15:
            return str(int(av))
        s = f"{av}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s

    def _encode_next_digit(
        self,
        value: float,
        active_len: int,
        out: Tensor,
        b: int,
        t: int,
        off: int,
    ) -> None:
        """Write sign + next-digit one-hot into the output tensor.

        Layout at out[b, t, off:off + 1 + NEXT_DIGIT_VOCAB]:
          [0]    sign: +1.0 (positive), -1.0 (negative), 0.0 (zero)
          [1..12] one-hot over {0,1,...,9, '.', END}

        ``active_len`` is the length of the current contiguous digit/decimal
        run ending at token t. It indexes into the result string to pick
        the next character.
        """
        # Sign
        if value > 0:
            out[b, t, off] = 1.0
        elif value < 0:
            out[b, t, off] = -1.0
        # else 0.0 (already zero-initialized)

        # Next digit one-hot
        s = self._result_to_string(value)
        if active_len < len(s):
            ch = s[active_len]
            if ch == ".":
                idx = self.DIGIT_IDX_DOT
            elif ch.isdigit():
                idx = int(ch)
            else:
                idx = self.DIGIT_IDX_END
        else:
            idx = self.DIGIT_IDX_END
        out[b, t, off + 1 + idx] = 1.0

    def _compute_pairs(
        self,
        ring: list[float],
        ring_idx: int,
        ring_count: int,
        k: int,
        eps: float,
        out: Tensor,
        b: int,
        t: int,
        active_len: int,
    ) -> None:
        """Compute pairwise features and write to output tensor."""
        if ring_count < 2:
            return
        n_avail = min(ring_count, k)
        # Oldest first so a=first_seen, b=second_seen (natural left-to-right order)
        nums = [ring[(ring_idx - 1 - i) % k] for i in range(n_avail)][::-1]
        ad = self._arith_dim  # dims per arithmetic op (1 + NEXT_DIGIT_VOCAB)
        pd = self._pair_dim  # dims per pair
        Op = PairwiseOp

        pair_idx = 0
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                a, bv = nums[i], nums[j]
                b_abs = abs(bv) + eps
                base = pair_idx * pd

                # Arithmetic ops: [sign, next_digit_onehot]
                a_abs = abs(a) + eps
                arith_fn = {
                    Op.ADD: a + bv,
                    Op.MUL: a * bv,
                    Op.SUB: a - bv,
                    Op.RSUB: bv - a,
                    Op.DIV: a / b_abs,
                    Op.RDIV: bv / a_abs,
                    Op.MOD: math.fmod(a, b_abs),
                    Op.RMOD: math.fmod(bv, a_abs),
                }
                for op_i, op in enumerate(self._arith_ops):
                    val = arith_fn[op]
                    off = base + op_i * ad
                    self._encode_next_digit(val, active_len, out, b, t, off)

                # Comparisons (1 dim each, after arithmetic)
                cmp_off = base + len(self._arith_ops) * ad
                cmp_fn = {
                    Op.GT: float(a > bv),
                    Op.EQ: float(a == bv),
                    Op.LT: float(a < bv),
                }
                for cmp_i, op in enumerate(self._cmp_ops):
                    out[b, t, cmp_off + cmp_i] = cmp_fn[op]
                pair_idx += 1

    def forward(self, input_ids: Tensor, dtype: torch.dtype) -> Tensor:
        B, S = input_ids.shape
        device = input_ids.device
        k = self.k

        dv = self.digit_value[input_ids]  # (B, S), -1 or 0..9
        lb = self.lowercase_byte[input_ids]  # (B, S), 0 or lowercase ASCII
        id = self.is_decimal[input_ids]  # (B, S), bool for '.' tokens
        out = torch.zeros(B, S, self._dim, device=device, dtype=dtype)
        if self._num_pairs == 0:
            return out

        eps = self._eps
        max_input_digits = self._MAX_INPUT_DIGITS

        for b in range(B):
            ring = [0.0] * k
            ring_idx = 0
            ring_count = 0
            # Digit run state (for number detection → ring buffer)
            digit_num = 0.0
            digit_len = 0
            digit_has_dot = False
            digit_frac_mul = 0.0
            in_digit_run = False
            # Word run state
            word_bytes: list[int] = []
            in_word_run = False
            # Active number run state (for next-digit encoding)
            active_len = 0
            active_has_dot = False

            for t in range(S):
                d = dv[b, t].item()
                l = lb[b, t].item()
                is_dot = id[b, t].item()
                pushed = False

                # A '.' continues a digit run if one is active and has no dot yet
                dot_continues_run = is_dot and in_digit_run and not digit_has_dot

                # --- Finish any run that ended ---
                if d >= 0:
                    # Digit: finish word run if active
                    if in_word_run:
                        val = self._try_word_number(word_bytes)
                        if val is not None:
                            ring[ring_idx % k] = float(val)
                            ring_idx += 1
                            ring_count += 1
                            pushed = True
                        in_word_run = False
                        word_bytes = []
                elif dot_continues_run:
                    # Decimal point inside digit run — don't finish anything
                    pass
                elif l > 0:
                    # Letter: finish digit run if active
                    if in_digit_run:
                        ring[ring_idx % k] = digit_num
                        ring_idx += 1
                        ring_count += 1
                        pushed = True
                        in_digit_run = False
                else:
                    # Neither digit, dot-in-run, nor letter: finish both
                    if in_digit_run:
                        ring[ring_idx % k] = digit_num
                        ring_idx += 1
                        ring_count += 1
                        pushed = True
                        in_digit_run = False
                    if in_word_run:
                        val = self._try_word_number(word_bytes)
                        if val is not None:
                            ring[ring_idx % k] = float(val)
                            ring_idx += 1
                            ring_count += 1
                            pushed = True
                        in_word_run = False
                        word_bytes = []

                # --- Extend or start current run ---
                if d >= 0:
                    if not in_digit_run:
                        digit_num = 0.0
                        digit_len = 0
                        digit_has_dot = False
                        digit_frac_mul = 0.0
                        in_digit_run = True
                    if digit_len < max_input_digits:
                        if digit_has_dot:
                            digit_num += d * digit_frac_mul
                            digit_frac_mul *= 0.1
                        else:
                            digit_num = digit_num * 10 + d
                    digit_len += 1
                elif dot_continues_run:
                    digit_has_dot = True
                    digit_frac_mul = 0.1
                    digit_len += 1
                elif l > 0:
                    if not in_word_run:
                        word_bytes = []
                        in_word_run = True
                    word_bytes.append(l)

                # --- Track active digit/decimal run for next-digit index ---
                if d >= 0:
                    active_len += 1
                elif is_dot and not active_has_dot:
                    active_len += 1
                    active_has_dot = True
                else:
                    active_len = 0
                    active_has_dot = False

                self._compute_pairs(
                    ring, ring_idx, ring_count, k, eps, out, b, t, active_len
                )

        return out
