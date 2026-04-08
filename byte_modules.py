from __future__ import annotations

import math
from collections import Counter
from enum import StrEnum

from regex import P
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from efficient_byte_tokenizer import ByteCategory, EfficientByteTokenizer
from multi_streams import CompressedView

from modules import CastedLinear


NUM_BYTE_CATEGORIES = 8
BYTE_CATEGORY_DEFS = [
    ByteCategory.BOS,
    ByteCategory.PAD,
    ByteCategory.DIGIT,
    ByteCategory.LETTER,
    ByteCategory.SEPARATOR,
    ByteCategory.PUNCTUATION,
    ByteCategory.SYMBOL,
    ByteCategory.MULTIBYTE,
]


class CategoryAttnBias(nn.Module):
    """Token→category LUT shared by both catmask modes.

    - bias mode: computes (B, H, S, S) additive attention bias from (H, C, C) logits.
    - lora mode: provides cat_ids for the LoRA gather (computation in SharedBlock).
    """

    def __init__(self, tok: EfficientByteTokenizer):
        super().__init__()
        V = tok.vocab_size
        token_to_cat = torch.zeros(V, dtype=torch.long)
        for ci, cat in enumerate(BYTE_CATEGORY_DEFS):
            mask = tok.mask(cat)
            for tid in range(V):
                if mask[tid]:
                    token_to_cat[tid] = ci
        self.register_buffer("token_to_cat", token_to_cat)

    def get_cat_ids(self, input_ids: Tensor) -> Tensor:
        """(B, S) token IDs → (B, S) category indices."""
        return self.token_to_cat[input_ids]

    def precompute_bias(self, input_ids: Tensor, dtype) -> Tensor:
        """Precompute (B, 1, S, C) one-hot for reuse across layers."""
        cat_ids = self.token_to_cat[input_ids]
        cat_oh = F.one_hot(cat_ids, NUM_BYTE_CATEGORIES).to(dtype=dtype)
        return cat_oh.unsqueeze(1)  # (B, 1, S, C)

    def bias_forward(self, cat_oh_expanded: Tensor, cat_attn_logits: Tensor) -> Tensor:
        """Compute (B, H, S, S) bias from precomputed (B, 1, S, C) and per-layer (H, C, C)."""
        q_contrib = cat_oh_expanded @ cat_attn_logits.unsqueeze(0)  # (B, H, S, C)
        return q_contrib @ cat_oh_expanded.transpose(-1, -2)  # (B, H, S, S)


class StructuralBoundaryBias(nn.Module):
    """Detects word/sentence/paragraph boundaries for structural attention bias.

    - bias mode: produces (B, H, S, S) additive bias from same-word/sentence/paragraph features.
    - lora mode: provides structural category IDs (position-within-word × position-within-sentence).
    """

    def __init__(
        self, tok: EfficientByteTokenizer, word_bins: int = 4, sent_bins: int = 4
    ):
        super().__init__()
        V = tok.vocab_size
        # Build boundary LUTs from tokenizer
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
        self.num_features = 3 if self.has_newline else 2

        # LoRA: binning for position-within-word and position-within-sentence
        self.word_bins = word_bins
        self.sent_bins = sent_bins
        self.num_struct_cats = word_bins * sent_bins
        word_edges = torch.tensor([2, 4, 8], dtype=torch.long)[: word_bins - 1]
        sent_edges = torch.tensor([4, 16, 64], dtype=torch.long)[: sent_bins - 1]
        self.register_buffer("word_bin_edges", word_edges)
        self.register_buffer("sent_bin_edges", sent_edges)

    def get_boundary_ids(self, input_ids: Tensor):
        """(B, S) → (word_id, sentence_id, paragraph_id or None)."""
        word_id = self.is_separator[input_ids].cumsum(dim=1)
        sentence_id = self.is_sentence_end[input_ids].cumsum(dim=1)
        paragraph_id = (
            self.is_newline[input_ids].cumsum(dim=1) if self.has_newline else None
        )
        return word_id, sentence_id, paragraph_id

    def precompute_bias(self, input_ids: Tensor) -> Tensor:
        """Precompute (B, F, S, S) structural features for reuse across layers."""
        word_id, sentence_id, paragraph_id = self.get_boundary_ids(input_ids)
        same_word = (word_id[:, :, None] == word_id[:, None, :]).float()
        same_sentence = (sentence_id[:, :, None] == sentence_id[:, None, :]).float()
        features = [same_word, same_sentence]
        if paragraph_id is not None:
            same_paragraph = (
                paragraph_id[:, :, None] == paragraph_id[:, None, :]
            ).float()
            features.append(same_paragraph)
        return torch.stack(features, dim=1)  # (B, F, S, S)

    def bias_forward(
        self, struct_features: Tensor, struct_bias_weights: Tensor
    ) -> Tensor:
        """Compute (B, H, S, S) bias from precomputed features and per-layer (H, F) weights."""
        return torch.einsum("bfqk,hf->bhqk", struct_features, struct_bias_weights)

    def get_struct_cat_ids(self, input_ids: Tensor) -> Tensor:
        """(B, S) → (B, S) structural category indices for LoRA mode."""
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)
        # Position within current word
        is_sep = self.is_separator[input_ids].bool()
        last_sep_pos = (
            torch.where(is_sep, positions, torch.zeros_like(positions))
            .cummax(dim=1)
            .values
        )
        pos_in_word = positions - last_sep_pos
        # Position within current sentence
        is_sent = self.is_sentence_end[input_ids].bool()
        last_sent_pos = (
            torch.where(is_sent, positions, torch.zeros_like(positions))
            .cummax(dim=1)
            .values
        )
        pos_in_sent = positions - last_sent_pos
        # Bin
        word_bin = torch.bucketize(pos_in_word, self.word_bin_edges)
        sent_bin = torch.bucketize(pos_in_sent, self.sent_bin_edges)
        return word_bin * self.sent_bins + sent_bin


class DistanceBias(nn.Module):
    """T5-style log-bucketed relative distance bias for attention.

    - bias mode: produces (1, H, S, S) additive distance bias.
    - lora mode: provides position bin IDs for per-position LoRA.
    """

    def __init__(self, num_buckets: int = 32, max_distance: int = 128):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance

        # Precompute distance-to-bucket mapping
        distances = torch.arange(max_distance + 1, dtype=torch.long)
        max_exact = num_buckets // 2
        is_small = distances < max_exact
        # Log-spaced buckets for distances >= max_exact
        log_ratio = torch.log(distances.float().clamp(min=1) / max_exact) / math.log(
            max_distance / max_exact
        )
        val_if_large = (
            max_exact + (log_ratio * (num_buckets - max_exact)).long()
        ).clamp(min=max_exact, max=num_buckets - 1)
        bucket_ids = torch.where(is_small, distances, val_if_large)
        self.register_buffer("distance_to_bucket", bucket_ids)
        # For LoRA: same bucketing on absolute positions
        self.register_buffer("position_to_bin", bucket_ids)

    def precompute_bias(self, seq_len: int, device) -> Tensor:
        """Precompute (S, S) bucket IDs for reuse across layers."""
        positions = torch.arange(seq_len, device=device)
        rel_dist = (positions[:, None] - positions[None, :]).clamp(
            min=0, max=self.max_distance
        )
        return self.distance_to_bucket[rel_dist]  # (S, S)

    def bias_forward(self, bucket_ids: Tensor, dist_bias_weights: Tensor) -> Tensor:
        """Compute (1, H, S, S) bias from precomputed (S, S) bucket IDs and per-layer weights."""
        return dist_bias_weights[:, bucket_ids].unsqueeze(0)  # (1, H, S, S)

    def get_pos_bin_ids(self, seq_len: int, device) -> Tensor:
        """(S,) → (1, S) position bin IDs for LoRA mode."""
        positions = torch.arange(seq_len, device=device).clamp(max=self.max_distance)
        return self.position_to_bin[positions].unsqueeze(0)  # (1, S)


class UTF8Prior(nn.Module):
    """Precomputed UTF-8 structural prior masks (no learnable parameters).

    Given input_ids, computes position-dependent masks that zero out tokens
    impossible under UTF-8 encoding rules.  All operations are parallel
    (bounded lookback of 3, no sequential scan).

    Two masks are returned:
      cat_mask   (B, S, num_categories) — additive 0/-inf for structured head level-0
      token_mask (B, S, V)              — additive 0/-inf for final logits/log-probs
    """

    # Byte-type enum (plain ints — torch.compile friendly)
    BT_BOS = 0
    BT_PAD = 1
    BT_ASCII = 2  # digit, letter, separator, punctuation, symbol
    BT_LEAD_2 = 3
    BT_LEAD_3 = 4
    BT_LEAD_4 = 5
    BT_CONT = 6

    # State enum
    ST_READY = 0  # expect ASCII / leading / special (no continuation)
    ST_EXPECT_CONT = 1  # expect continuation byte
    ST_UNSYNCED = 2  # unknown state (chunk boundary) — no constraint

    def __init__(self, tok, num_categories=8, multibyte_cat_idx=7):
        super().__init__()
        V = tok.vocab_size
        self.num_categories = num_categories
        NEG_INF = float("-inf")

        # ---- token_id → byte type & byte value ----
        token_byte_type = torch.zeros(V, dtype=torch.long)
        token_byte_value = torch.zeros(V, dtype=torch.long)
        for tid in range(V):
            info = tok.token_info(tid)
            if info is None:
                token_byte_type[tid] = self.BT_BOS if tid == tok.bos_id else self.BT_PAD
            elif info.has(ByteCategory.MB_CONTINUATION):
                token_byte_type[tid] = self.BT_CONT
                token_byte_value[tid] = info.byte_value
            elif info.has(ByteCategory.MB_LEAD_2):
                token_byte_type[tid] = self.BT_LEAD_2
                token_byte_value[tid] = info.byte_value
            elif info.has(ByteCategory.MB_LEAD_3):
                token_byte_type[tid] = self.BT_LEAD_3
                token_byte_value[tid] = info.byte_value
            elif info.has(ByteCategory.MB_LEAD_4):
                token_byte_type[tid] = self.BT_LEAD_4
                token_byte_value[tid] = info.byte_value
            else:
                token_byte_type[tid] = self.BT_ASCII
                token_byte_value[tid] = info.byte_value
        self.register_buffer("token_byte_type", token_byte_type)
        self.register_buffer("token_byte_value", token_byte_value)

        # ---- lead type → expected continuation count ----
        lead_expected = torch.zeros(7, dtype=torch.long)
        lead_expected[self.BT_LEAD_2] = 1
        lead_expected[self.BT_LEAD_3] = 2
        lead_expected[self.BT_LEAD_4] = 3
        self.register_buffer("lead_expected", lead_expected)

        # ---- state → category mask (3, num_categories) ----
        state_cat_mask = torch.zeros(3, num_categories)
        # READY: all categories valid (no constraint)
        # EXPECT_CONT: only multibyte (idx 7) valid
        for ci in range(num_categories):
            if ci != multibyte_cat_idx:
                state_cat_mask[self.ST_EXPECT_CONT, ci] = NEG_INF
        # UNSYNCED: all valid
        self.register_buffer("state_cat_mask", state_cat_mask)

        # ---- state → token mask (3, V) ----
        cont_np = tok.mask(ByteCategory.MB_CONTINUATION)
        state_token_mask = torch.zeros(3, V)
        # READY: forbid continuation bytes
        for tid in range(V):
            if cont_np[tid]:
                state_token_mask[self.ST_READY, tid] = NEG_INF
        # EXPECT_CONT: only continuation bytes valid
        for tid in range(V):
            if not cont_np[tid]:
                state_token_mask[self.ST_EXPECT_CONT, tid] = NEG_INF
        # UNSYNCED: all valid
        self.register_buffer("state_token_mask", state_token_mask)

        # ---- special lead byte constraints (first cont only) ----
        # After 0xE0: first cont must be 0xA0–0xBF (prevent overlong 3-byte)
        # After 0xED: first cont must be 0x80–0x9F (prevent surrogates)
        # After 0xF0: first cont must be 0x90–0xBF (prevent overlong 4-byte)
        special_e0 = state_token_mask[self.ST_EXPECT_CONT].clone()
        special_ed = state_token_mask[self.ST_EXPECT_CONT].clone()
        special_f0 = state_token_mask[self.ST_EXPECT_CONT].clone()
        for tid in range(V):
            info = tok.token_info(tid)
            if info and info.has(ByteCategory.MB_CONTINUATION):
                bv = info.byte_value
                if bv < 0xA0:
                    special_e0[tid] = NEG_INF
                if bv > 0x9F:
                    special_ed[tid] = NEG_INF
                if bv < 0x90:
                    special_f0[tid] = NEG_INF
        self.register_buffer("special_e0", special_e0)
        self.register_buffer("special_ed", special_ed)
        self.register_buffer("special_f0", special_f0)

    def forward(self, input_ids):
        """Compute UTF-8 structural prior masks from input_ids.

        The mask at position t constrains the *prediction* at position t
        (i.e. target token t), based on the UTF-8 state after consuming
        input_ids[t].  Only causal information (positions ≤ t) is used.

        Args:
            input_ids: (B, S) token IDs with BOS at position 0.

        Returns:
            cat_mask:   (B, S, num_categories) additive mask (0.0 or -inf)
            token_mask: (B, S, V) additive mask (0.0 or -inf)
        """
        B, S = input_ids.shape
        device = input_ids.device

        # Step 1: classify each input token
        byte_type = self.token_byte_type[input_ids]  # (B, S) long
        byte_val = self.token_byte_value[input_ids]  # (B, S) long

        # Step 2: continuation count via bounded lookback (max 3)
        is_cont = byte_type == self.BT_CONT  # (B, S) bool
        c1 = is_cont
        c2 = torch.zeros(B, S, device=device, dtype=torch.bool)
        c2[:, 1:] = is_cont[:, 1:] & is_cont[:, :-1]
        c3 = torch.zeros(B, S, device=device, dtype=torch.bool)
        c3[:, 2:] = is_cont[:, 2:] & is_cont[:, 1:-1] & is_cont[:, :-2]
        cont_count = c1.long() + c2.long() + c3.long()  # (B, S), 0-3

        # Step 3: find lead byte via lookback
        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        lead_pos = (positions - cont_count).clamp(min=0)
        lead_type = byte_type.gather(1, lead_pos)

        # Step 4: remaining continuations expected
        non_cont_remaining = self.lead_expected[byte_type]
        cont_remaining = self.lead_expected[lead_type] - cont_count
        remaining = torch.where(is_cont, cont_remaining, non_cont_remaining)

        # Step 5: state assignment
        is_valid_lead = (
            (lead_type == self.BT_LEAD_2)
            | (lead_type == self.BT_LEAD_3)
            | (lead_type == self.BT_LEAD_4)
        )
        is_special = (byte_type == self.BT_BOS) | (byte_type == self.BT_PAD)
        unsynced = is_special | (is_cont & ((lead_pos <= 0) | ~is_valid_lead))
        state = torch.where(
            unsynced,
            self.ST_UNSYNCED,
            torch.where(remaining > 0, self.ST_EXPECT_CONT, self.ST_READY),
        )  # (B, S) long

        # Step 6: gather base masks by state
        cat_mask = self.state_cat_mask[state]  # (B, S, num_categories)
        token_mask = self.state_token_mask[state]  # (B, S, V)

        # Step 7: special lead byte refinements (first cont after E0/ED/F0)
        # These apply when byte_type[t] is LEAD_3/LEAD_4 and byte value is special
        is_e0 = (byte_type == self.BT_LEAD_3) & (byte_val == 0xE0)
        is_ed = (byte_type == self.BT_LEAD_3) & (byte_val == 0xED)
        is_f0 = (byte_type == self.BT_LEAD_4) & (byte_val == 0xF0)
        token_mask = torch.where(is_e0.unsqueeze(-1), self.special_e0, token_mask)
        token_mask = torch.where(is_ed.unsqueeze(-1), self.special_ed, token_mask)
        token_mask = torch.where(is_f0.unsqueeze(-1), self.special_f0, token_mask)

        return cat_mask, token_mask


class ByteLogitHierarchy(nn.Module):
    """Zero-parameter module defining the byte-category tree structure.

    Holds level_indices, level_masks, margin matrices, and provides
    :meth:`assemble` to convert per-level logits into flat ``(B, S, V)``
    log-probs.

    The hierarchy decomposes token log-probability as a sum of
    log-normalized levels::

        log p(token) = log p(category) + sum(log fraction_sub_i) + log fraction_leaf

    Each level is independently log-softmax normalized.  Softcap is applied
    only to leaf-level logits.
    """

    def __init__(self, vocab_size: int, tok, logit_softcap: float):
        super().__init__()
        self.logit_softcap = logit_softcap
        self.vocab_size = vocab_size

        # ---- Gather token-ID sets from the tokenizer ----
        def _ids(cat):
            arr = tok.ids(cat)
            return set(arr.tolist()) if len(arr) > 0 else set()

        cat_defs = [
            ("bos", ByteCategory.BOS),
            ("pad", ByteCategory.PAD),
            ("digit", ByteCategory.DIGIT),
            ("letter", ByteCategory.LETTER),
            ("separator", ByteCategory.SEPARATOR),
            ("punctuation", ByteCategory.PUNCTUATION),
            ("symbol", ByteCategory.SYMBOL),
            ("multibyte", ByteCategory.MULTIBYTE),
        ]
        cat_id_sets = {name: _ids(cat) for name, cat in cat_defs}

        upper_ids = _ids(ByteCategory.UPPERCASE)
        lower_ids = _ids(ByteCategory.LOWERCASE)
        vowel_ids = _ids(ByteCategory.VOWEL)
        consonant_ids = _ids(ByteCategory.CONSONANT)
        mb_cont_ids = _ids(ByteCategory.MB_CONTINUATION)
        mb_leading_ids = _ids(ByteCategory.MB_LEADING)
        mb_lead2_ids = _ids(ByteCategory.MB_LEAD_2)
        mb_lead3_ids = _ids(ByteCategory.MB_LEAD_3)
        mb_lead4_ids = _ids(ByteCategory.MB_LEAD_4)

        # ---- Build levels (index/mask buffers) ----
        all_indices: list[Tensor] = []
        all_masks: list[Tensor] = []
        level_sizes: list[int] = []
        self._is_leaf: list[bool] = []

        def _add_level(head_size, token_to_idx, active_tids, leaf):
            idx = torch.zeros(vocab_size, dtype=torch.long)
            mask = torch.zeros(vocab_size, dtype=torch.float32)
            for tid, j in token_to_idx.items():
                idx[tid] = j
            for tid in active_tids:
                mask[tid] = 1.0
            level_sizes.append(head_size)
            all_indices.append(idx)
            all_masks.append(mask)
            self._is_leaf.append(leaf)

        # Level: category (8-way, all tokens)
        cat_map: dict[int, int] = {}
        all_tids: set[int] = set()
        for ci, (name, _) in enumerate(cat_defs):
            for tid in cat_id_sets[name]:
                cat_map[tid] = ci
                all_tids.add(tid)
        _add_level(len(cat_defs), cat_map, all_tids, leaf=False)

        # Level: letter case (upper=0, lower=1)
        letter_ids = cat_id_sets["letter"]
        if upper_ids and lower_ids:
            _add_level(
                2,
                {tid: (0 if tid in upper_ids else 1) for tid in letter_ids},
                letter_ids,
                leaf=False,
            )

        # Level: vowel/consonant for uppercase (vowel=0, consonant=1)
        if upper_ids and (vowel_ids & upper_ids) and (consonant_ids & upper_ids):
            _add_level(
                2,
                {tid: (0 if tid in vowel_ids else 1) for tid in upper_ids},
                upper_ids,
                leaf=False,
            )

        # Level: vowel/consonant for lowercase (vowel=0, consonant=1)
        if lower_ids and (vowel_ids & lower_ids) and (consonant_ids & lower_ids):
            _add_level(
                2,
                {tid: (0 if tid in vowel_ids else 1) for tid in lower_ids},
                lower_ids,
                leaf=False,
            )

        # Level: multibyte type (continuation=0, leading=1)
        mb_ids = cat_id_sets["multibyte"]
        if mb_cont_ids and mb_leading_ids:
            _add_level(
                2,
                {tid: (0 if tid in mb_cont_ids else 1) for tid in mb_ids},
                mb_ids,
                leaf=False,
            )

        # Level: multibyte lead type (lead2=0, lead3=1, lead4=2)
        lead_tids = mb_lead2_ids | mb_lead3_ids | mb_lead4_ids
        if len(lead_tids) > 1:
            lead_map: dict[int, int] = {}
            for tid in mb_lead2_ids:
                lead_map[tid] = 0
            for tid in mb_lead3_ids:
                lead_map[tid] = 1
            for tid in mb_lead4_ids:
                lead_map[tid] = 2
            _add_level(3, lead_map, lead_tids, leaf=False)

        # Leaf levels (skip groups with <=1 token — singletons need no leaf head)
        leaf_token_ids: list[int] = []

        def _add_leaf(group):
            if len(group) > 1:
                _add_level(
                    len(group),
                    {tid: i for i, tid in enumerate(sorted(group))},
                    group,
                    leaf=True,
                )
            leaf_token_ids.extend(sorted(group))

        _add_leaf(cat_id_sets["bos"])
        _add_leaf(cat_id_sets["pad"])
        _add_leaf(cat_id_sets["digit"])
        _add_leaf(cat_id_sets["separator"])
        _add_leaf(upper_ids & vowel_ids)
        _add_leaf(upper_ids & consonant_ids)
        _add_leaf(lower_ids & vowel_ids)
        _add_leaf(lower_ids & consonant_ids)
        _add_leaf(cat_id_sets["punctuation"])
        _add_leaf(cat_id_sets["symbol"])
        _add_leaf(mb_cont_ids)
        _add_leaf(mb_lead2_ids)
        _add_leaf(mb_lead3_ids)
        _add_leaf(mb_lead4_ids)

        # Assert every token appears in exactly one leaf group
        leaf_counts = Counter(leaf_token_ids)
        duplicates = {tid: cnt for tid, cnt in leaf_counts.items() if cnt > 1}
        if duplicates:
            raise ValueError(
                f"ByteLogitHierarchy: tokens appear in multiple leaves: {duplicates}"
            )
        leaf_set = set(leaf_token_ids)
        expected = set(range(vocab_size))
        missing = expected - leaf_set
        extra = leaf_set - expected
        if missing or extra:
            raise ValueError(
                f"ByteLogitHierarchy leaf coverage error: "
                f"missing token IDs {sorted(missing)}, "
                f"extra token IDs {sorted(extra)}"
            )

        # ---- Store as module attributes ----
        self._level_sizes = level_sizes
        self.register_buffer(
            "level_indices", torch.stack(all_indices)
        )  # (num_levels, V)
        self.register_buffer("level_masks", torch.stack(all_masks))  # (num_levels, V)

        # ---- Precompute n-gram marginalization matrices (static) ----
        # margin_i: (V, H_i) maps token probs → level-output probs via matmul.
        # margin_i[t, j] = 1.0 iff token t is active at level i and maps to output j.
        for i, hs in enumerate(level_sizes):
            margin = torch.zeros(vocab_size, hs)
            margin.scatter_(1, all_indices[i].unsqueeze(1), all_masks[i].unsqueeze(1))
            self.register_buffer(f"margin_{i}", margin)

    @property
    def num_levels(self) -> int:
        return len(self._level_sizes)

    @property
    def level_sizes(self) -> list[int]:
        return list(self._level_sizes)

    @property
    def total_slots(self) -> int:
        return sum(self._level_sizes)

    def assemble(
        self,
        per_level_logits: list[Tensor],
        cat_prior: Tensor | None = None,
        token_prior: Tensor | None = None,
        ngram_logp: Tensor | None = None,
    ) -> Tensor:
        """Convert per-level logits into flat ``(B, S, V)`` log-probs.

        Args:
            per_level_logits: list of ``(B, S, H_i)`` tensors, one per level.
            cat_prior: ``(B, S, num_categories)`` additive bias for level-0 logits.
            token_prior: ``(B, S, V)`` additive mask for final log-probs (e.g. UTF-8).
            ngram_logp: ``(B, S, V)`` token-level n-gram log-probs.  Marginalized
                into per-level conditional priors and added to each level's logits
                before log_softmax.
        """
        first = per_level_logits[0]
        B, S = first.shape[0], first.shape[1]
        log_p = torch.zeros(
            B, S, self.vocab_size, device=first.device, dtype=first.dtype
        )
        ngram_probs = ngram_logp.float().exp() if ngram_logp is not None else None
        for i, logits in enumerate(per_level_logits):
            if i == 0 and cat_prior is not None:
                logits = logits + cat_prior
            if self._is_leaf[i]:
                logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
            if ngram_probs is not None:
                margin = getattr(self, f"margin_{i}")  # (V, H_i)
                level_probs = ngram_probs @ margin  # (B, S, H_i)
                level_prior = level_probs.clamp(min=1e-30).log()
                level_prior = level_prior - level_prior.logsumexp(dim=-1, keepdim=True)
                logits = logits + level_prior.to(dtype=logits.dtype)
            lp = F.log_softmax(logits, dim=-1)
            log_p = log_p + lp[..., self.level_indices[i]] * self.level_masks[i]
        if token_prior is not None:
            log_p = log_p + token_prior
            log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        return log_p


class StructuredLogitsAdapter(nn.Module):
    """Converts flat structured logits ``(B, S, total_slots)`` into flat
    token log-probs ``(B, S, V)`` by splitting into per-level chunks and
    assembling via :class:`ByteLogitHierarchy`.

    Use this to wrap any module that predicts in hierarchical space so its
    output can be written to the flat logit stream.
    """

    def __init__(self, hierarchy: ByteLogitHierarchy):
        super().__init__()
        self.hierarchy = hierarchy

    def forward(
        self,
        flat_logits: Tensor,
        cat_prior: Tensor | None = None,
        token_prior: Tensor | None = None,
        ngram_logp: Tensor | None = None,
    ) -> Tensor:
        chunks = flat_logits.split(self.hierarchy.level_sizes, dim=-1)
        return self.hierarchy.assemble(
            list(chunks),
            cat_prior=cat_prior,
            token_prior=token_prior,
            ngram_logp=ngram_logp,
        )


class StructuredOutputHead(nn.Module):
    """Hierarchical softmax output head based on ByteCategory tree.

    Projects hidden states through per-level linear heads, then assembles
    via :class:`StructuredLogitsAdapter` into flat ``(B, S, V)`` log-probs.

    Decomposes token log-probability as a sum of log-normalized levels::

        log p(token) = log p(category) + sum(log fraction_sub_i) + log fraction_leaf
    """

    def __init__(self, model_dim, vocab_size, tok, logit_softcap):
        super().__init__()
        self.vocab_size = vocab_size
        self.hierarchy = ByteLogitHierarchy(vocab_size, tok, logit_softcap)
        self.adapter = StructuredLogitsAdapter(self.hierarchy)
        heads = []
        for size in self.hierarchy.level_sizes:
            head = CastedLinear(model_dim, size, bias=False)
            head._zero_init = True
            heads.append(head)
        self.heads = nn.ModuleList(heads)

    def forward(self, x, cat_prior=None, token_prior=None, ngram_logp=None):
        """Return ``(B, S, V)`` log-probabilities assembled from the hierarchy.

        Args:
            x: ``(B, S, D)`` hidden states.
            cat_prior: ``(B, S, num_categories)`` additive mask for level-0 logits.
            token_prior: ``(B, S, V)`` additive mask for final log-probs (e.g. UTF-8).
            ngram_logp: ``(B, S, V)`` token-level n-gram log-probs.
        """
        flat = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.adapter(
            flat, cat_prior=cat_prior, token_prior=token_prior, ngram_logp=ngram_logp
        )


# ------------------
# Stream Compressors
# ------------------

_NUMBER_WORDS = {
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


class NumberExtractor(nn.Module):
    """Extracts detected numbers from the token stream into a CompressedView.

    Zero learnable parameters. Detects numbers as:
    1. Digit runs: contiguous ASCII digits (0-9), optionally with a single
       decimal point, assembled into floats.
    2. Number words: lowercased letter runs matched against a dictionary
       (zero..twenty by default).

    Returns a ``CompressedView`` with:
    - ``positions``: (B, N_max) position of the last token of each number
    - ``mask``: (B, N_max) valid entries
    - ``metadata["values"]``: (B, N_max) scalar value of each number
    - ``metadata["lengths"]``: (B, N_max) token count per number
    - ``metadata["active_len"]``: (B, S) current digit run length at each position
    """

    _MAX_INPUT_DIGITS = 15  # float64 significant digits

    def __init__(
        self,
        tok: EfficientByteTokenizer,
        n_max: int = 32,
        number_words: dict[str, int] | None = None,
    ):
        super().__init__()
        self.n_max = n_max
        self._number_words = number_words if number_words is not None else _NUMBER_WORDS

        V = tok.vocab_size
        digit_value = torch.full((V,), -1, dtype=torch.long)
        lowercase_byte = torch.zeros(V, dtype=torch.long)
        is_decimal = torch.zeros(V, dtype=torch.bool)
        for tid in range(V):
            info = tok.token_info(tid)
            if info is None:
                continue
            if info.has(ByteCategory.DIGIT):
                digit_value[tid] = info.byte_value - 0x30
            if info.has(ByteCategory.LETTER):
                lb = info.lowercase_byte
                lowercase_byte[tid] = lb if lb is not None else info.byte_value
            if info.byte_value == 0x2E:
                is_decimal[tid] = True
        self.register_buffer("digit_value", digit_value)
        self.register_buffer("lowercase_byte", lowercase_byte)
        self.register_buffer("is_decimal", is_decimal)

    def _try_word_number(self, word_bytes: list[int]) -> int | None:
        word = bytes(word_bytes).decode("ascii", errors="replace")
        return self._number_words.get(word)

    def forward(self, input_ids: Tensor) -> CompressedView:
        B, S = input_ids.shape
        device = input_ids.device
        n_max = self.n_max
        max_input_digits = self._MAX_INPUT_DIGITS

        dv = self.digit_value[input_ids]  # (B, S)
        lb = self.lowercase_byte[input_ids]  # (B, S)
        is_dec = self.is_decimal[input_ids]  # (B, S)

        # Output tensors
        positions = torch.zeros(B, n_max, dtype=torch.long, device=device)
        mask = torch.zeros(B, n_max, dtype=torch.bool, device=device)
        values = torch.zeros(B, n_max, dtype=torch.float32, device=device)
        lengths = torch.zeros(B, n_max, dtype=torch.long, device=device)
        active_len_out = torch.zeros(B, S, dtype=torch.long, device=device)

        for b in range(B):
            num_idx = 0  # next slot in output
            # Digit run state
            digit_num = 0.0
            digit_len = 0
            digit_has_dot = False
            digit_frac_mul = 0.0
            in_digit_run = False
            digit_start = 0
            # Word run state
            word_bytes: list[int] = []
            in_word_run = False
            word_start = 0
            # Active number run (for next-digit encoding)
            active_len = 0
            active_has_dot = False

            def _push_number(val: float, start: int, end: int) -> None:
                nonlocal num_idx
                if num_idx >= n_max:
                    return
                values[b, num_idx] = val
                positions[b, num_idx] = end
                lengths[b, num_idx] = end - start + 1
                mask[b, num_idx] = True
                num_idx += 1

            for t in range(S):
                d = dv[b, t].item()
                l = lb[b, t].item()
                is_dot = is_dec[b, t].item()

                dot_continues_run = is_dot and in_digit_run and not digit_has_dot

                # --- Finish any run that ended ---
                # Numbers are positioned at t (the boundary token), not t-1
                # (the last digit). This ensures strict causality: the number
                # only appears in the compressed view when the boundary is
                # visible, matching what happens in truncated sequences.
                if d >= 0:
                    if in_word_run:
                        val = self._try_word_number(word_bytes)
                        if val is not None:
                            _push_number(float(val), word_start, t)
                        in_word_run = False
                        word_bytes = []
                elif dot_continues_run:
                    pass
                elif l > 0:
                    if in_digit_run:
                        _push_number(digit_num, digit_start, t)
                        in_digit_run = False
                else:
                    if in_digit_run:
                        _push_number(digit_num, digit_start, t)
                        in_digit_run = False
                    if in_word_run:
                        val = self._try_word_number(word_bytes)
                        if val is not None:
                            _push_number(float(val), word_start, t)
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
                        digit_start = t
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
                        word_start = t
                    word_bytes.append(l)

                # --- Track active digit/decimal run ---
                if d >= 0:
                    active_len += 1
                elif is_dot and not active_has_dot:
                    active_len += 1
                    active_has_dot = True
                else:
                    active_len = 0
                    active_has_dot = False
                active_len_out[b, t] = active_len

            # NOTE: we intentionally do NOT flush in-progress digit/word runs
            # at end-of-sequence. Only numbers terminated by a boundary token
            # (non-digit after digits, non-letter after letters) are emitted.
            # This ensures strict causality: the compressed view at position t
            # is identical regardless of what tokens follow after t.

        return CompressedView(
            positions=positions,
            mask=mask,
            metadata={
                "values": values,
                "lengths": lengths,
                "active_len": active_len_out,
            },
        )
