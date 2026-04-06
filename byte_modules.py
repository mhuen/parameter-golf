from __future__ import annotations

import math
from collections import Counter

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from efficient_byte_tokenizer import ByteCategory, EfficientByteTokenizer

from modules import (
    CastedLinear,
    sincos_encode,
)

NUM_BYTE_CATEGORIES = 8
_BYTE_CATEGORY_DEFS = [
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
        for ci, cat in enumerate(_BYTE_CATEGORY_DEFS):
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


class StructuredOutputHead(nn.Module):
    """Hierarchical softmax output head based on ByteCategory tree.

    Decomposes token log-probability as a sum of log-normalized levels:
      log p(token) = log p(category) + sum(log fraction_sub_i) + log fraction_leaf

    Each level is independently log-softmax normalized.  Softcap is applied
    only to leaf-level logits.
    """

    def __init__(self, model_dim, vocab_size, tok, logit_softcap):
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

        # ---- Build levels (heads + index/mask buffers) ----
        heads = []
        all_indices = []
        all_masks = []
        self._is_leaf: list[bool] = []

        def _add_level(head_size, token_to_idx, active_tids, leaf):
            idx = torch.zeros(vocab_size, dtype=torch.long)
            mask = torch.zeros(vocab_size, dtype=torch.float32)
            for tid, j in token_to_idx.items():
                idx[tid] = j
            for tid in active_tids:
                mask[tid] = 1.0
            head = CastedLinear(model_dim, head_size, bias=False)
            head._zero_init = True
            heads.append(head)
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
                f"StructuredOutputHead: tokens appear in multiple leaves: {duplicates}"
            )
        leaf_set = set(leaf_token_ids)
        expected = set(range(vocab_size))
        missing = expected - leaf_set
        extra = leaf_set - expected
        if missing or extra:
            raise ValueError(
                f"StructuredOutputHead leaf coverage error: "
                f"missing token IDs {sorted(missing)}, "
                f"extra token IDs {sorted(extra)}"
            )

        # ---- Store as module attributes ----
        self.heads = nn.ModuleList(heads)
        self.register_buffer(
            "level_indices", torch.stack(all_indices)
        )  # (num_levels, V)
        self.register_buffer("level_masks", torch.stack(all_masks))  # (num_levels, V)

        # ---- Precompute n-gram marginalization matrices (static) ----
        # margin_i: (V, H_i) maps token probs → level-output probs via matmul.
        # margin_i[t, j] = 1.0 iff token t is active at level i and maps to output j.
        for i, head in enumerate(heads):
            H = head.out_features
            margin = torch.zeros(vocab_size, H)
            margin.scatter_(1, all_indices[i].unsqueeze(1), all_masks[i].unsqueeze(1))
            self.register_buffer(f"margin_{i}", margin)

    def forward(self, x, cat_prior=None, token_prior=None, ngram_logp=None):
        """Return (B, S, V) log-probabilities assembled from the hierarchy.

        Args:
            x: (B, S, D) hidden states.
            cat_prior: (B, S, num_categories) additive mask for level-0 logits.
            token_prior: (B, S, V) additive mask for final log-probs (e.g. UTF-8).
            ngram_logp: (B, S, V) token-level n-gram log-probs.  Marginalized into
                per-level conditional priors and added to each head's logits before
                log_softmax.
        """
        B, S, _ = x.shape
        log_p = torch.zeros(B, S, self.vocab_size, device=x.device, dtype=x.dtype)
        # Precompute n-gram probs once for all levels (float32 for log/exp precision)
        ngram_probs = ngram_logp.float().exp() if ngram_logp is not None else None
        for i, head in enumerate(self.heads):
            logits = head(x)
            if i == 0 and cat_prior is not None:
                logits = logits + cat_prior
            if self._is_leaf[i]:
                logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
            # Per-level n-gram prior: marginalize token probs → level conditional
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
            # Renormalize: nll_loss expects valid log-probs summing to 1
            log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        return log_p


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
        for ci, cat in enumerate(_BYTE_CATEGORY_DEFS):
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
        state = torch.where(
            case_type == 1, 1.0, torch.where(case_type == 2, -1.0, 0.0)
        )
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
        state = torch.where(
            vc_type == 1, 1.0, torch.where(vc_type == 2, -1.0, 0.0)
        )
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
        opens = self.is_open[input_ids].cumsum(dim=1)   # causal
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
        boundary: "word" (reset at separators), "digit" (digit runs only),
                  "codepoint" (multibyte sequences only), or None (last N bytes).

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
        boundary: str | None = "word",
    ):
        super().__init__()
        self.window = window
        self.num_hashes = num_hashes
        self.boundary = boundary
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
        if boundary == "word":
            is_sep = torch.zeros(V, dtype=torch.bool)
            sep_mask = tok.mask(ByteCategory.SEPARATOR)
            for tid in range(V):
                if sep_mask[tid]:
                    is_sep[tid] = True
            self.register_buffer("is_separator", is_sep)
        elif boundary == "digit":
            is_dig = torch.zeros(V, dtype=torch.bool)
            d_mask = tok.mask(ByteCategory.DIGIT)
            for tid in range(V):
                if d_mask[tid]:
                    is_dig[tid] = True
            self.register_buffer("is_digit_buf", is_dig)
        elif boundary == "codepoint":
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
        return 2 * self.num_hashes

    def _effective_lookback(self, input_ids: Tensor) -> Tensor:
        """Per-position lookback depth. -1 means inactive (hash = 0)."""
        B, S = input_ids.shape
        device = input_ids.device
        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, S)

        if self.boundary is None:
            return positions  # full window always

        elif self.boundary == "word":
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

        elif self.boundary == "digit":
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

        elif self.boundary == "codepoint":
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
        gathered = byte_vals.gather(1, gather_pos.reshape(B, -1)).reshape(B, S, self.window)

        # Mask: zero out bytes outside segment boundary
        mask = eff_lb.unsqueeze(-1) >= offsets  # (B, S, W) bool
        gathered = gathered * mask

        # Compute independent polynomial hashes
        parts = []
        for h in range(self.num_hashes):
            powers = getattr(self, f"powers_{h}")  # (W,)
            hash_val = ((gathered * powers) % P).sum(dim=-1) % P  # (B, S) long
            # Project to small range for stable sin/cos encoding
            q = self.PROJ_PRIMES[h]
            angle = (hash_val % q).float() * (2 * math.pi / q)
            parts.append(torch.stack([angle.sin(), angle.cos()], dim=-1))

        return torch.cat(parts, dim=-1).to(dtype=dtype)
