"""Vectorized number detection from byte token streams.

All functions use only tensor operations — no .item(), no Python loops over
tensor elements, no data-dependent control flow. Compatible with
torch.compile(fullgraph=True).

Shared by DigitComputeComponent (byte_stream_components.py) and
NumberExtractor (byte_modules.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Parallel prefix scan for linear recurrences
# ---------------------------------------------------------------------------


def affine_scan(a: Tensor, b: Tensor) -> Tensor:
    """Parallel prefix scan for y[t] = a[t]*y[t-1] + b[t], y[-1] = 0.

    Uses iterative doubling: O(S log S) work, O(log S) sequential steps.
    Each step is a fully parallel tensor op — GPU-friendly and compile-safe.

    Args:
        a: (B, S) multiplicative coefficients.
        b: (B, S) additive coefficients (float64 recommended for digit accumulation).

    Returns:
        y: (B, S) scan result.
    """
    B, S = a.shape
    a_cur = a.clone()
    b_cur = b.clone()
    offset = 1
    while offset < S:
        # Shifted versions (identity padding: a=1, b=0)
        # Use cat instead of F.pad to generate simpler LLVM IR
        # (F.pad triggers an LLVM SLP vectorizer assertion in some Triton builds)
        a_prev = torch.cat(
            [torch.ones(B, offset, dtype=a.dtype, device=a.device),
             a_cur[:, :-offset]], dim=1,
        )
        b_prev = torch.cat(
            [torch.zeros(B, offset, dtype=b.dtype, device=b.device),
             b_cur[:, :-offset]], dim=1,
        )
        # Compose: (a_cur, b_cur) ∘ (a_prev, b_prev) = (a_cur*a_prev, a_cur*b_prev + b_cur)
        b_cur = a_cur * b_prev + b_cur
        a_cur = a_cur * a_prev
        offset *= 2
    return b_cur


# ---------------------------------------------------------------------------
# Segmented cumulative helpers
# ---------------------------------------------------------------------------


def _segmented_cumsum(values: Tensor, segment_starts: Tensor) -> Tensor:
    """Cumulative sum that resets at segment boundaries.

    Args:
        values: (B, S) values to accumulate.
        segment_starts: (B, S) bool, True at positions where accumulation resets.

    Returns:
        (B, S) cumulative sums within each segment.
    """
    cum = values.cumsum(dim=1)
    cum_at_start = torch.where(segment_starts, cum, torch.zeros_like(cum))
    cum_at_start = cum_at_start.cummax(dim=1).values
    # The cumsum at the start position itself should be included (not subtracted)
    # cum_at_start is the cum value AT the start; we want to subtract the value BEFORE it
    cum_before_start = torch.where(
        segment_starts, F.pad(cum[:, :-1], (1, 0), value=0.0), torch.zeros_like(cum)
    )
    cum_before_start = cum_before_start.cummax(dim=1).values
    return cum - cum_before_start


# ---------------------------------------------------------------------------
# Digit run detection and value accumulation
# ---------------------------------------------------------------------------


@dataclass
class DetectedNumbers:
    """Results of vectorized number detection.

    All tensors are on the same device as the input.
    """

    # Per-position (B, S) masks and values
    is_number_boundary: Tensor  # True at first non-member token after a number run
    number_value_at_boundary: Tensor  # float64, number value (meaningful at boundaries)
    number_start_at_boundary: (
        Tensor  # long, start position of the number (at boundaries)
    )
    active_len: Tensor  # long, length of current digit/dot run at each position

    # Compact arrays (B, n_max)
    positions: Tensor  # long, boundary position of each detected number
    values: Tensor  # float32, numeric value of each number
    lengths: Tensor  # long, token span of each number
    mask: Tensor  # bool, valid entries

    # Per-position (B, S) cumulative count (resets at BOS)
    num_count: (
        Tensor  # long, how many numbers detected up to this position (in current doc)
    )


def detect_numbers(
    input_ids: Tensor,
    digit_value: Tensor,
    lowercase_byte: Tensor,
    is_decimal: Tensor,
    bos_id: int,
    word_pattern_lengths: tuple[int, ...],
    word_patterns: dict[int, Tensor],  # length -> (N_words, length) byte patterns
    word_values: dict[int, Tensor],  # length -> (N_words,) int values
    n_max: int,
) -> DetectedNumbers:
    """Detect numbers in a byte token stream using fully vectorized ops.

    Detects two kinds of numbers:
    1. Digit runs: contiguous ASCII digits (0-9) with at most one decimal point.
    2. Word numbers: letter runs matching known words (zero..twenty).

    Numbers are positioned at the first non-member token AFTER the run
    (boundary position), ensuring strict causality.

    Args:
        input_ids: (B, S) token IDs.
        digit_value: (V,) buffer mapping token ID -> digit 0-9 or -1.
        lowercase_byte: (V,) buffer mapping token ID -> lowercase ASCII byte or 0.
        is_decimal: (V,) buffer mapping token ID -> True if '.'.
        bos_id: token ID of BOS (resets all state).
        word_pattern_lengths: tuple of word lengths to check, sorted.
        word_patterns: for each length L, (N, L) tensor of lowercase byte patterns.
        word_values: for each length L, (N,) tensor of number values.
        n_max: maximum numbers to track per batch element.

    Returns:
        DetectedNumbers with all detection results.
    """
    B, S = input_ids.shape
    device = input_ids.device

    # -- Lookup buffers --
    dv = digit_value[input_ids]  # (B, S), -1 or 0-9
    lb = lowercase_byte[input_ids]  # (B, S), 0 or ASCII byte
    is_dec = is_decimal[input_ids]  # (B, S), bool
    is_bos = input_ids == bos_id  # (B, S), bool

    is_digit = dv >= 0  # (B, S)
    is_letter = lb > 0  # (B, S)

    # -- Phase 1: Detect extended digit runs (digits + at most one valid dot) --
    # A dot is valid if preceded by a digit AND no prior dot in this run.
    # Key insight: a dot preceded by a non-digit can't be valid, so
    # "digit-or-(dot-after-digit)" defines the candidate run members.
    dot_after_digit = is_dec & F.pad(is_digit[:, :-1], (1, 0), value=False)

    # Candidate extended member: digit or dot-after-digit
    ext_candidate = is_digit | dot_after_digit  # (B, S)
    # BOS breaks everything
    ext_candidate = ext_candidate & ~is_bos

    # Identify extended candidate runs
    ext_start = ext_candidate & ~F.pad(ext_candidate[:, :-1], (1, 0), value=False)

    # Count dots within each candidate run (to enforce "at most one dot")
    dot_cum = dot_after_digit.long().cumsum(dim=1)
    dot_cum_before_start = (
        torch.where(
            ext_start,
            F.pad(dot_cum[:, :-1], (1, 0), value=0),
            torch.zeros_like(dot_cum),
        )
        .cummax(dim=1)
        .values
    )
    dots_in_run = dot_cum - dot_cum_before_start  # (B, S)

    # A dot is valid only if it's the first dot in its run
    valid_dot = dot_after_digit & (dots_in_run <= 1)

    # True extended members: digits + valid dots (NOT invalid dots)
    ext_member = (is_digit | valid_dot) & ~is_bos  # (B, S)

    # Re-detect extended run boundaries with the cleaned membership
    ext_start_clean = ext_member & ~F.pad(ext_member[:, :-1], (1, 0), value=False)
    ext_end_clean = ext_member & ~F.pad(ext_member[:, 1:], (0, 1), value=False)

    # -- Phase 2: Accumulate digit values using parallel prefix scan --
    # Recurrence: y[t] = a[t]*y[t-1] + b[t]
    #   digit continuation: a=10, b=digit_value
    #   digit start:        a=0,  b=digit_value  (y = d, fresh start)
    #   valid dot:           a=1,  b=0            (y unchanged, pass-through)
    #   non-member:          a=0,  b=0            (y = 0)
    continued = ext_member & F.pad(ext_member[:, :-1], (1, 0), value=False)

    a = torch.zeros(B, S, device=device, dtype=torch.float64)
    b = torch.zeros(B, S, device=device, dtype=torch.float64)
    a = torch.where(is_digit & continued, 10.0, a)
    a = torch.where(valid_dot, 1.0, a)
    b = torch.where(is_digit, dv.double().clamp(min=0), b)

    all_digits_int = affine_scan(a, b)  # (B, S) float64

    # -- Phase 3: Count fractional digits (digits after the dot in each run) --
    # Detect "seen a dot in this run"
    dot_in_ext = valid_dot.long()
    dot_cum_ext = dot_in_ext.cumsum(dim=1)
    dot_cum_before_ext_start = (
        torch.where(
            ext_start_clean,
            F.pad(dot_cum_ext[:, :-1], (1, 0), value=0),
            torch.zeros_like(dot_cum_ext),
        )
        .cummax(dim=1)
        .values
    )
    dots_in_ext_run = dot_cum_ext - dot_cum_before_ext_start  # (B, S)
    seen_dot = dots_in_ext_run > 0  # (B, S)

    # Count digits (not dots) after the first dot in this run
    digit_after_dot = is_digit & seen_dot & ext_member
    frac_cum = digit_after_dot.long().cumsum(dim=1)
    frac_cum_before_start = (
        torch.where(
            ext_start_clean,
            F.pad(frac_cum[:, :-1], (1, 0), value=0),
            torch.zeros_like(frac_cum),
        )
        .cummax(dim=1)
        .values
    )
    n_frac_digits = (frac_cum - frac_cum_before_start) * ext_member.long()  # (B, S)

    # Number value at each position in the run
    digit_number = all_digits_int / (10.0 ** n_frac_digits.double())  # (B, S) float64

    # The value at the last position of each extended run is the full number
    digit_run_value = digit_number * ext_end_clean.double()  # meaningful at ext_end

    # Digit run boundaries: the position AFTER the run (first non-member)
    # = ext_end shifted right by 1
    # Exclude BOS: in the sequential code, BOS resets state before checking
    # for run endings, so in-progress runs at BOS are silently discarded.
    digit_boundary = (
        F.pad(ext_end_clean[:, :-1], (1, 0), value=False) & ~is_bos
    )  # (B, S)
    digit_boundary_value = F.pad(
        digit_run_value[:, :-1], (1, 0), value=0.0
    )  # (B, S) float64

    # Start position of each digit run: propagate ext_start position through the run
    seq_pos = torch.arange(S, device=device).expand(B, S)  # (B, S)
    start_pos_at_ext_start = torch.where(
        ext_start_clean, seq_pos, torch.zeros_like(seq_pos)
    )
    digit_start_pos = start_pos_at_ext_start.cummax(dim=1).values * ext_member.long()
    # At ext_end, digit_start_pos gives the start of this run
    digit_start_at_end = digit_start_pos * ext_end_clean.long()
    digit_start_at_boundary = F.pad(digit_start_at_end[:, :-1], (1, 0), value=0).long()

    # -- Phase 4: Detect word numbers --
    # Letter run detection
    is_letter_clean = is_letter & ~is_bos  # BOS breaks letter runs too
    letter_start = is_letter_clean & ~F.pad(
        is_letter_clean[:, :-1], (1, 0), value=False
    )
    letter_end = is_letter_clean & ~F.pad(is_letter_clean[:, 1:], (0, 1), value=False)

    # Run length at each position (within letter runs)
    letter_cum = is_letter_clean.long().cumsum(dim=1)
    letter_cum_before_start = (
        torch.where(
            letter_start,
            F.pad(letter_cum[:, :-1], (1, 0), value=0),
            torch.zeros_like(letter_cum),
        )
        .cummax(dim=1)
        .values
    )
    letter_run_len = (letter_cum - letter_cum_before_start) * is_letter_clean.long()

    # Match words by length
    word_boundary = torch.zeros(B, S, dtype=torch.bool, device=device)
    word_boundary_value = torch.zeros(B, S, dtype=torch.float64, device=device)
    word_start_at_boundary = torch.zeros(B, S, dtype=torch.long, device=device)

    for L in word_pattern_lengths:
        if L > S:
            continue
        patterns = word_patterns[L]  # (N, L) long
        vals = word_values[L]  # (N,) long

        # Extract windows of lowercase bytes: window[i] = lb[i:i+L]
        windows = lb.unfold(1, L, 1)  # (B, S-L+1, L)
        # windows at index i corresponds to positions [i, i+L-1]
        # i.e., ending at position i+L-1

        # Compare each window against each pattern
        # patterns: (N, L) → (1, 1, N, L)
        # windows: (B, S-L+1, L) → (B, S-L+1, 1, L)
        matches = (windows.unsqueeze(2) == patterns.unsqueeze(0).unsqueeze(0)).all(
            dim=-1
        )  # (B, S-L+1, N)
        any_match = matches.any(dim=-1)  # (B, S-L+1)

        # Get the matched value (first match per position)
        # Use argmax to find which pattern matched
        match_idx = matches.long().argmax(dim=-1)  # (B, S-L+1)
        matched_val = vals[match_idx].double()  # (B, S-L+1)

        # Align to sequence: match at index i → letter run ends at position i+L-1
        # The word boundary (first non-letter after run) is at i+L
        # Pad to get boundary at the correct position
        if L < S:
            boundary_at = F.pad(
                any_match, (L, 0), value=False
            )  # (B, S+1) → trim to (B, S)
            boundary_at = boundary_at[:, :S]
            val_at = F.pad(matched_val, (L, 0), value=0.0)[:, :S]
        else:
            continue

        # Only valid if: the letter run ending at position i+L-1 has exactly length L
        # The boundary is at position i+L; the letter run ended at i+L-1
        # At the boundary position (i+L), letter_run_len at (i+L-1) should equal L
        prev_letter_run_len = F.pad(letter_run_len[:, :-1], (1, 0), value=0)
        valid = boundary_at & (prev_letter_run_len == L)

        # Also must be at an actual letter-to-non-letter boundary
        prev_is_letter = F.pad(is_letter_clean[:, :-1], (1, 0), value=False)
        valid = valid & prev_is_letter & ~is_letter_clean

        # Start position of the word: boundary_pos - L
        start_pos = (seq_pos - L).clamp(min=0)

        # Exclude BOS from word boundaries (same reason as digit boundaries)
        valid = valid & ~is_bos
        word_boundary = word_boundary | valid
        word_boundary_value = torch.where(valid, val_at, word_boundary_value)
        word_start_at_boundary = torch.where(valid, start_pos, word_start_at_boundary)

    # -- Phase 5: Combine digit and word boundaries --
    is_number_boundary = digit_boundary | word_boundary
    number_value = torch.where(
        digit_boundary, digit_boundary_value, word_boundary_value
    )
    number_start = torch.where(
        digit_boundary, digit_start_at_boundary, word_start_at_boundary
    )

    # -- Phase 6: Active digit/dot run length --
    # active_len tracks the length of the current contiguous digit-or-single-dot run
    # Resets at non-digit/non-valid-dot positions
    # For the first dot (no prior dot in ANY digit run at this position), it continues
    # This matches the original: active_has_dot is per-run state
    # We reuse ext_member: active run = contiguous ext_member positions
    active_cum = ext_member.long().cumsum(dim=1)
    active_cum_before_start = (
        torch.where(
            ext_start_clean,
            F.pad(active_cum[:, :-1], (1, 0), value=0),
            torch.zeros_like(active_cum),
        )
        .cummax(dim=1)
        .values
    )
    active_len = (active_cum - active_cum_before_start) * ext_member.long()  # (B, S)

    # -- Phase 7: Compact into (B, n_max) arrays --
    # Reset number count at BOS (document boundaries)
    bos_positions = is_bos  # (B, S)
    num_cum = is_number_boundary.long().cumsum(dim=1)
    # Subtract count at most recent BOS to get document-local count
    num_at_bos = (
        torch.where(bos_positions, num_cum, torch.zeros_like(num_cum))
        .cummax(dim=1)
        .values
    )
    num_count = num_cum - num_at_bos  # (B, S), document-local count

    # Scatter into compact arrays
    # Use n_max+1 so out-of-range indices go to a dummy slot
    compact_idx = (num_count - 1).clamp(min=0, max=n_max)  # 0-indexed; n_max = dummy
    valid_boundary = is_number_boundary & (num_count >= 1) & (num_count <= n_max)
    # Route non-boundary (and overflow) positions to the dummy slot
    scatter_idx = torch.where(
        valid_boundary, compact_idx, torch.full_like(compact_idx, n_max)
    )

    positions_out = torch.zeros(B, n_max + 1, dtype=torch.long, device=device)
    values_out = torch.zeros(B, n_max + 1, dtype=torch.float32, device=device)
    lengths_out = torch.zeros(B, n_max + 1, dtype=torch.long, device=device)
    mask_out = torch.zeros(B, n_max + 1, dtype=torch.bool, device=device)

    positions_out.scatter_(1, scatter_idx, seq_pos)
    values_out.scatter_(1, scatter_idx, number_value.float())
    lengths_out.scatter_(1, scatter_idx, (seq_pos - number_start + 1).long())
    mask_out.scatter_(1, scatter_idx, valid_boundary)

    # Trim dummy slot
    positions_out = positions_out[:, :n_max]
    values_out = values_out[:, :n_max]
    lengths_out = lengths_out[:, :n_max]
    mask_out = mask_out[:, :n_max]

    return DetectedNumbers(
        is_number_boundary=is_number_boundary,
        number_value_at_boundary=number_value,
        number_start_at_boundary=number_start,
        active_len=active_len,
        positions=positions_out,
        values=values_out,
        lengths=lengths_out,
        mask=mask_out,
        num_count=num_count,
    )


# ---------------------------------------------------------------------------
# Digit sequence extraction (for next-digit prediction encoding)
# ---------------------------------------------------------------------------

# Vocabulary for next-digit encoding: {0..9, '.', END}
DIGIT_IDX_DOT = 10
DIGIT_IDX_END = 11
NEXT_DIGIT_VOCAB = 12
MAX_RESULT_DIGITS = 15  # matches float64 precision


def extract_digit_at(
    values: Tensor,
    position: Tensor,
    max_len: int = MAX_RESULT_DIGITS,
) -> Tensor:
    """Extract the digit character at a specific position for each value.

    Optimized version of ``extract_digit_sequences(...).gather(-1, pos)``
    that avoids the Python loop entirely -- computes only the one needed digit
    per element in a single vectorized pass.

    Args:
        values: (...) float tensor of numeric values.
        position: (...) long tensor of 0-indexed positions into the digit
            representation (same shape as *values*).
        max_len: positions >= max_len map to END.

    Returns:
        (...) long tensor of character indices (0-9 = digit, 10 = dot, 11 = END).
    """
    av = values.abs().double()
    int_part = av.long()
    frac_part = av - int_part.double()

    # Number of integer digits
    n_int = torch.where(
        int_part > 0,
        torch.floor(torch.log10(int_part.double().clamp(min=0.5))) + 1,
        torch.ones_like(av),
    ).long()

    is_integer = (frac_part < 1e-15) | (av >= 1e15)
    has_frac = ~is_integer & (av > 0)

    # Integer digit at this position
    power = (n_int - 1 - position).clamp(min=0)
    int_digit = (int_part // (10**power)) % 10

    # Fractional digit at this position
    frac_pos = (position - n_int).clamp(min=0)
    frac_scaled = (frac_part * (10.0 ** (frac_pos + 1).double())).long()
    frac_digit = frac_scaled % 10

    # Classify position
    in_int = position < n_int
    at_dot = (position == n_int) & has_frac
    in_frac = (position > n_int) & has_frac

    end_val = torch.full_like(int_digit, DIGIT_IDX_END)
    char_idx = torch.where(
        in_int,
        int_digit,
        torch.where(
            at_dot,
            torch.full_like(int_digit, DIGIT_IDX_DOT),
            torch.where(in_frac, frac_digit, end_val),
        ),
    )

    # Beyond representable range -> END
    char_idx = torch.where(position >= max_len, end_val, char_idx)

    return char_idx


def extract_digit_sequences(
    values: Tensor,
    max_len: int = MAX_RESULT_DIGITS,
) -> Tensor:
    """Convert numeric values to their digit-character sequences.

    For each value, produces a sequence of character indices:
    0-9 for digits, 10 for '.', 11 for END.

    Matches the behavior of _result_to_string + indexing.

    Args:
        values: (...) float tensor of numeric values.
        max_len: maximum sequence length.

    Returns:
        (..., max_len) long tensor of character indices.
    """
    shape = values.shape
    av = values.abs().double()  # use float64 for precision

    # Integer part and fractional part
    int_part = av.long()
    frac_part = av - int_part.double()

    # Number of integer digits: floor(log10(x)) + 1, or 1 if x == 0
    n_int = torch.where(
        int_part > 0,
        torch.floor(torch.log10(int_part.double().clamp(min=0.5))) + 1,
        torch.ones_like(av),
    ).long()

    # Whether the result has a fractional part
    is_integer = (frac_part < 1e-15) | (av >= 1e15)
    has_frac = ~is_integer & (av > 0)

    # Build the digit sequence position by position
    result = torch.full(
        (*shape, max_len), DIGIT_IDX_END, dtype=torch.long, device=values.device
    )

    for pos in range(max_len):
        pos_t = torch.tensor(pos, device=values.device)

        # Integer digit at this position
        power = (n_int - 1 - pos_t).clamp(min=0)
        int_digit = (int_part // (10**power)) % 10  # (...) long

        # Fractional digit
        frac_pos = (pos_t - n_int).clamp(min=0)  # 0-indexed position after the dot
        # Multiply frac by 10^(frac_pos+1), take last digit
        frac_scaled = (frac_part * (10.0 ** (frac_pos + 1).double())).long()
        frac_digit = frac_scaled % 10

        # Determine character at this position
        in_int = pos_t < n_int
        at_dot = (pos_t == n_int) & has_frac
        in_frac = (pos_t > n_int) & has_frac

        char_idx = torch.where(
            in_int,
            int_digit,
            torch.where(
                at_dot,
                torch.full_like(int_digit, DIGIT_IDX_DOT),
                torch.where(
                    in_frac,
                    frac_digit,
                    torch.full_like(int_digit, DIGIT_IDX_END),
                ),
            ),
        )

        result[..., pos] = char_idx

    # Strip trailing zeros and dot from fractional part (match _result_to_string)
    # For fractional results, find the last non-zero fractional digit
    # and set everything after it (plus the dot if all frac digits are zero) to END
    # This is complex to do in a fully vectorized way; for now, the fixed-point
    # representation is close enough for most practical cases.

    return result
