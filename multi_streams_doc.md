# Multi-Stream GPT: Residual Stream Architecture Reference

This document traces how the four residual streams in `MultiStreamGPT` are initialized,
normalized, read from, and written to across all layers. The goal is to evaluate whether
normalization and scaling are set up correctly for stable training.

**Key source files:**
- `multi_stream_gpt.py` -- model definition, forward flow
- `multi_streams.py` -- StreamDef, MultiStreamBuilder, StreamConfig
- `multi_stream_attention.py` -- MultiStreamBlock, conv, MLP, mixing
- `modules.py` -- RMSNorm, CenterLastDim, SoftcapLinear, GatedCausalConv, StreamComponent, ContinuousRotation
- `byte_modules.py` -- ByteLogitHierarchy, UTF8Prior, NumberExtractor
- `byte_stream_components.py` -- STRUCTURAL stream components
- `train_multi_stream_bytes.py` -- training hyperparameters, optimizer setup

---

## 1. Architecture Overview

Instead of a single residual stream, MultiStreamGPT maintains **four parallel streams**
that flow through the network:

| Property | LOGIT | TOKENS | CONTEXT | STRUCTURAL |
|----------|-------|--------|---------|------------|
| **Writable** | Yes | No (read-only) | Yes | No (read-only) |
| **Purpose** | Next-token prediction logits | Current token identity | Learned working memory | Precomputed byte features |

All sub-layers can **read** any stream. Only **writable** streams (LOGIT, CONTEXT) receive
updates. Read-only streams (TOKENS, STRUCTURAL) are computed once from `input_ids` and
pass through unchanged.

### Forward Pass Flow

```
input_ids (B, S)
    |
    v
[1. Build Streams]  -->  LOGIT=zeros(208)  TOKENS=onehot(208)  CONTEXT=zeros(64)  STRUCTURAL=components(159)
    |
    v
[2. Bigram Prior]   -->  LOGIT += bigram_logits[input_ids]      (trainable V x V lookup)
    |
    v
[2b. Scale Down]    -->  LOGIT *= 0.1                           (logit_stream_normalization_factor=10)
    |
    v
[2c. Init Norms]    -->  CenterLastDim(LOGIT), RMSNorm(CONTEXT)
    |
    v
[3. PreConv x2]     -->  writes to CONTEXT only (LOGIT untouched)
    |
    v
[4. Block x6]       -->  each: Attn -> (Conv?) -> (Arith?) -> MLP
    |                     writes to LOGIT + CONTEXT (last block MLP: LOGIT only)
    v
[5. Scale Up]       -->  LOGIT *= 10
    |
    v
[5b. Softcap]       -->  SoftcapLinear(cap=30, knee=24) on LOGIT
    |
    v
[6. UTF-8 Prior]    -->  (inference only) adds 0 / -inf mask to LOGIT
    |
    v
logits (B, S, 208)
```

---

## 2. Stream Summary

| Stream | Dim | Read-Only | Norm | Init | Scale at Entry to Blocks | Role |
|--------|-----|-----------|------|------|--------------------------|------|
| **LOGIT** | 208 (=vocab_size) | No | `CenterLastDim` | zeros + bigram prior, /10, centered | ~[-0.5, +0.5] centered | Next-token logits |
| **TOKENS** | 208 | Yes | None | `F.one_hot(input_ids, 208)` | {0, 1}, L2=1.0 | Token identity |
| **CONTEXT** | 64 | No | `RMSNorm` | `torch.zeros(B, S, 64)` | 0 (all zeros) | Working memory |
| **STRUCTURAL** | 159 (actual) | Yes | None | Concatenated components | calibrated (mean≈0, std≈1 on ~46 dims) | Byte-level features |

> **STRUCTURAL dim:** The inline comments in `build_multi_stream_components()` have been
> corrected to reflect the actual dimensions with current ROTATION encoding defaults.

---

## 3. Stream Initialization Details

### 3.1 LOGIT Stream

| Step | Operation | Approximate Scale | Reference |
|------|-----------|-------------------|-----------|
| Init | `torch.zeros(B, S, 208)` via `source=StreamSource.ZEROS` | 0 | `multi_streams.py` StreamDef |
| +Bigram | `+= bigram_logits[input_ids]` | ~[-5, +5] (from data-init log-freq ratios) | `multi_stream_gpt.py:527` |
| /10 | `*= 1/logit_stream_normalization_factor` | ~[-0.5, +0.5] | `multi_stream_gpt.py:532` |
| CenterLastDim | `x - x.mean(dim=-1, keepdim=True)` | ~[-0.5, +0.5], zero-mean per position | `multi_stream_gpt.py:539` |

**Bigram prior data initialization** (`train_multi_stream_bytes.py:834-847`): Before training,
bigram log-probabilities are computed from the training data (with smoothing=0.1) and loaded
into `BigramPriorLayer.bigram_logits`. Typical magnitudes are ~[-5, +5].

**CenterLastDim** (`modules.py:24`): Subtracts per-position mean. Lossless for cross-entropy
(softmax is shift-invariant) but prevents mean drift. **Does NOT normalize magnitude** --
only recenters.

### 3.2 TOKENS Stream

- `F.one_hot(input_ids, vocab_size).to(dtype)` -- values exactly {0, 1}
- L2 norm = 1.0 per position (exactly one active dimension)
- Never modified (read-only), no normalization applied

### 3.3 CONTEXT Stream

- `torch.zeros(B, S, 64)` via `source=StreamSource.ZEROS`
- `RMSNorm` applied immediately: `F.rms_norm(x, (64,), eps=None)`
- When `x = 0`: `0 / sqrt(0 + eps) = 0`. PyTorch default eps prevents NaN.
- **CONTEXT stays at zero** until the first preconv layer writes to it.

**RMSNorm** (`modules.py:15`): `x / sqrt(mean(x^2) + eps)`. No learnable parameters.
After normalization, the RMS of the last dimension is 1.0.

### 3.4 STRUCTURAL Stream

Built by `CompositeStream` concatenating 19 components. Each component computes
deterministic features from `input_ids` only (no learnable parameters).

| Component | Dims | Value Range | Encoding |
|-----------|------|-------------|----------|
| `DocBoundaryComponent(num_freqs=2)` | 4 | [-1, +1] | sin/cos of BOS cumsum |
| `SinCosPositionComponent(num_freqs=10)` | 20 | [-1, +1] | sin/cos of absolute position |
| `ByteCategoryComponent` | 2 | [-1, +1] | Rotation codebook (8 categories on unit circle) |
| `MultiByteStateComponent(id_freqs=3)` | 8 | {-1,+1} binary + [0,3] int + [-1,+1] sincos | in_sequence, remaining, codepoint_id |
| `CaseComponent` | 2 | {-1,0,+1} + **[0, ~7]** | case_state + log1p(run_length) |
| `VowelConsonantComponent` | 2 | {-1,0,+1} + **[0, ~7]** | vc_state + log1p(run_length) |
| `ColumnPositionComponent(num_freqs=3)` | 6 | [-1, +1] | sin/cos of column position |
| `RepeatedByteComponent` | 2 | {-1,+1} + **[0, ~7]** | is_repeat + log1p(run_length) |
| `PunctuationDepthComponent` | 3 | [-1, +1] + {-1,+1} | ContinuousRotation(bracket_depth) + quote_state |
| `ByteCategoryStatsComponent` | 6 | [0, 1] | running category fractions |
| `DigitSequenceComponent(id_freqs=5)` | 13 | {-1,+1} + [-1,+1] | in_digit + rotation + sincos |
| `DigitComputeComponent` | 27 | {-1,0,+1} + [-1,+1] + {0,1} | sign + rotation digits + comparisons |
| `ByteHashComponent` (x6) | 36 | [-1,+1] + **[0, ~7]** + [0,1] | sin/cos hashes + log1p(hits) + hit_frac |
| `BoundaryComponent(3,3,2,2,2,2)` | 28 | [-1, +1] | sin/cos word/sent/para position+ID |
| **Total** | **159** | | |

**Scale concerns** (bolded above):
- **log1p run lengths**: CaseComponent, VowelConsonantComponent, RepeatedByteComponent,
  ByteHashComponent hit_log -- can reach ~7 for seq_len=1024 (`log1p(1023) ~ 6.9`).
  Most other features are in [-1, +1].
- **PunctuationDepthComponent bracket_depth**: now encoded via `ContinuousRotation(cap)`
  with `min_depth=-3, max_depth=8`, mapping depths to a 2D unit rotation. Bounded to
  [-1, +1]. Depths beyond the range saturate at boundary angles.

**Calibration:** `CompositeStream.calibrate()` standardizes ~46 of 159 dims (those marked
`standardizable=True`) to mean≈0, std≈1 using a two-pass mean/std computed from training
data. Non-standardizable dims (rotation pairs, sin/cos) are left unchanged (scale=1,
shift=0). Calibration is run once during training init at `args.train_seq_len`.
A noise floor (std ≤ 1e-5 → skip) and scale cap (max 1000×) guard against degenerate dims.

---

## 4. Forward Pass Trace with Scale Analysis

Reference: `MultiStreamGPT.forward()` at `multi_stream_gpt.py:504-565`.

### Step 1: Build Streams (`multi_stream_gpt.py:521`)

```python
streams, compressed, views = self.builder(input_ids, dtype=dtype)
```

| Stream | Value | Scale |
|--------|-------|-------|
| LOGIT | `zeros(B, S, 208)` | 0 |
| TOKENS | `one_hot(input_ids, 208)` | {0, 1} |
| CONTEXT | `zeros(B, S, 64)` | 0 |
| STRUCTURAL | `cat(components)` shape `(B, S, 159)` | calibrated, mostly [-1, +1] |

Also produces `compressed` (streams gathered at number positions) and `views` (CompressedView
metadata) for arithmetic attention.

### Step 2: Bigram Prior (`multi_stream_gpt.py:527`)

```python
streams[LOGIT] = streams[LOGIT] + self.bigram_prior(input_ids)
```

Adds `(V, V)` trainable lookup. After data initialization, typical values ~[-5, +5].

### Step 2b: Scale Down (`multi_stream_gpt.py:531-534`)

```python
streams[LOGIT] = streams[LOGIT] * (1.0 / self.logit_stream_normalization_factor)
```

Default factor = 10.0. LOGIT values go from ~[-5, +5] to ~[-0.5, +0.5].

**Purpose:** Bring LOGIT to roughly unit scale matching CONTEXT (RMS=1 after norm) and
STRUCTURAL (mostly [-1, +1]). Without this, the LOGIT stream (naturally at logit scale)
would dominate concatenated inputs to Q/K/V projections.

### Step 2c: Init Norms (`multi_stream_gpt.py:537-539`)

```python
for s in self._stream_config.streams:
    if s.key in self.init_norms:
        streams[s.name] = self.init_norms[s.key](streams[s.name])
```

- LOGIT: `CenterLastDim` -- subtracts mean, values still ~[-0.5, +0.5]
- CONTEXT: `RMSNorm` -- zeros stay zeros (0/sqrt(eps) = 0)

Establishes the "always normalized" invariant before entering blocks.

### Step 3: Pre-Conv Layers (`multi_stream_gpt.py:542-543`)

```python
streams = self.preconv_layers(streams)  # 2 layers
```

See `MultiStreamCausalConvLayers` (`multi_stream_attention.py:826`).

**Input streams (default):** LOGIT + CONTEXT + STRUCTURAL (excludes TOKENS).
**Output streams (default):** CONTEXT only (excludes LOGIT).

Each conv layer:
1. Pre-norm all streams: `CenterLastDim(LOGIT)`, `RMSNorm(CONTEXT)`
2. Concatenate input streams: dim = 208 + 64 + 159 = 431
3. `GatedCausalConv(dim=431, kernel_size=4, groups=6)`:
   - `gate = sigmoid(conv_gate(x) * std_repair)` -- in [0, 1]
   - `value = SiLU(conv_value(x) * std_repair)` -- std_repair = `1/sqrt(fan_in)`
   - `value_softcap(value)` -- caps at +/-30 (identity for |x|<24)
   - `out = gate * value` -- bounded by softcap
4. Alpha/beta write-back to CONTEXT only:
   - `mixed = sigmoid(beta)*normed_CONTEXT + sigmoid(alpha)*conv_delta`
   - At init: alpha=-3.0 (sigmoid~0.047), beta=5.0 (sigmoid~0.993)
   - So ~5% of conv output mixes into CONTEXT
5. Post-norm: `RMSNorm(CONTEXT)` -- re-establishes unit RMS

After 2 preconv layers, CONTEXT holds initial local n-gram patterns at RMS~1.
LOGIT is unchanged (not in conv output streams).

### Step 4: Attention Blocks (`multi_stream_gpt.py:546-547`)

```python
for block in self.blocks:  # 6 blocks
    streams = block(streams, compressed=compressed, views=views)
```

Each `MultiStreamBlock` has up to 4 sub-layers (see [Section 5](#5-sub-layer-write-back-pattern)
and [Section 6](#6-module-by-module-stream-interaction)):

1. **Attention** -- reads all streams, writes to LOGIT + CONTEXT
2. **Conv** (optional) -- reads configurable streams, writes to configurable writable streams
3. **Arithmetic attention** (optional) -- reads all streams, writes primarily to LOGIT
4. **MLP** -- reads all streams, writes to LOGIT + CONTEXT (last block: LOGIT only)

All sub-layers follow the same alpha/beta write-back pattern with pre/post normalization.

### Step 5: Scale Up + Softcap (`multi_stream_gpt.py:549-553`)

```python
streams[LOGIT] *= self.logit_stream_normalization_factor  # *= 10
streams[LOGIT] = self.logit_softcap(streams[LOGIT])       # SoftcapLinear(30)
```

- Scale up reverses the /10 from step 2b.
- `SoftcapLinear(cap=30, knee=24)`: exactly identity for |x| < 24, smoothly
  asymptotes to +/-30 beyond. C1-continuous at knee. No learnable parameters.
  (`modules.py:576`)

### Step 6: UTF-8 Prior -- inference only (`multi_stream_gpt.py:559-563`)

```python
if self.utf8_prior is not None and not self.training:
    _cat_mask, token_mask = self.utf8_prior(input_ids)
    streams[LOGIT] = streams[LOGIT] + token_mask.to(dtype=streams[LOGIT].dtype)
```

Adds 0 (valid token) or -inf (impossible token) based on UTF-8 byte state.
**Disabled during training** to avoid destabilizing CenterLastDim + 10x scale-up
(bimodal 0/-inf distribution causes NaN).

### Scale Summary Table

| Stage | LOGIT | CONTEXT | TOKENS | STRUCTURAL |
|-------|-------|---------|--------|------------|
| After builder | 0 | 0 | {0,1} | calibrated, mostly [-1,+1] |
| After bigram | ~[-5, +5] | 0 | {0,1} | same |
| After /10 | ~[-0.5, +0.5] | 0 | {0,1} | same |
| After init norms | centered ~[-0.5, +0.5] | 0 | {0,1} | same |
| After preconv | same (untouched) | RMS~1 | {0,1} | same |
| Inside blocks (invariant) | centered, ~O(1) | RMS~1 | {0,1} | same |
| After x10 | ~O(10) | RMS~1 | -- | -- |
| After softcap | capped to [-30, +30] | -- | -- | -- |
| After UTF-8 (eval) | finite or -inf | -- | -- | -- |

---

## 5. Sub-Layer Write-Back Pattern

All sub-layers in `MultiStreamBlock` (attention, conv, arith, MLP) and
`MultiStreamCausalConvLayers` use the same universal pattern:

```python
# 1. Pre-norm (read normalization)
normed = pre_norm(stream)          # CenterLastDim for LOGIT, RMSNorm for CONTEXT

# 2. Compute update delta (reads all normed streams, produces delta for writable streams)
delta = sub_layer(all_normed_streams)

# 3. Alpha/beta mix (per-dimension learned interpolation)
mixed = sigmoid(beta) * normed + sigmoid(alpha) * delta

# 4. Post-norm (re-normalize to maintain invariant)
output = post_norm(mixed)          # same norm type as pre_norm
```

Reference: `MultiStreamBlock.forward()` at `multi_stream_attention.py:1476-1550`.

### Key Properties

1. **Not a standard residual connection.** Standard transformers use `x + delta`.
   Here the residual base is `normed` (not the raw `x`), and both terms have independent
   learned scales (alpha, beta).

2. **Pre-norm and post-norm use the same norm instance** within each sub-layer.
   The norm parameters (if any -- RMSNorm and CenterLastDim are parameterless) are shared.

3. **Alpha and beta are per-dimension vectors** (shape `(stream_dim,)`). Each dimension
   of each stream has its own mixing coefficients. The model can learn per-dimension:
   additive (beta~1), interpolating (alpha+beta~1), replacement (alpha~1, beta~0),
   or amplification (alpha+beta > 1).

4. **Read-only streams** (TOKENS, STRUCTURAL) skip the write-back entirely --
   they pass through unchanged.

### Alpha/Beta Init Values

| Sub-layer | alpha_init | sigmoid(alpha) | beta_init | sigmoid(beta) | Init behavior |
|-----------|------------|----------------|-----------|---------------|---------------|
| Attention | -1.0 | 0.27 | 5.0 | 0.993 | ~27% update + ~99% residual |
| MLP | -3.0 | 0.047 | 5.0 | 0.993 | ~5% update + ~99% residual |
| PreConv | -3.0 | 0.047 | 5.0 | 0.993 | ~5% update + ~99% residual |
| Block Conv | -3.0 | 0.047 | 5.0 | 0.993 | ~5% update + ~99% residual |
| Arith Attn | -3.0 | 0.047 | 5.0 | 0.993 | ~5% update + ~99% residual |

Reference: `MultiStreamBlock.__init__()` at `multi_stream_attention.py:1351-1358`.

**Init noise:** Small Gaussian noise (`std=0.01`) is added to all alpha/beta/gate
parameters at model init for symmetry breaking (`multi_stream_gpt.py:495-500`).

**Note:** alpha + beta is ~1.04 at init (not constrained to sum to 1). This means
there is a slight amplification at init (~4% above unity) which is absorbed by the
post-norm. The post-norm after each sub-layer keeps the invariant: LOGIT stays
centered (zero-mean), CONTEXT stays at RMS=1.

---

## 6. Module-by-Module Stream Interaction

### 6.1 CausalMultiStreamAttention (concat mode, default)

**File:** `multi_stream_attention.py:424`

Used when `mixing_config is None` (default).

- **Reads:** All 4 streams, concatenated into a single vector.
  Input dim = 208 + 208 + 64 + 159 = 639.
- **Q/K/V:** Single projection from concatenated input:
  - `W_q`: 638 -> `num_heads * head_dim` (8 * 16 = 128)
  - `W_k`: 638 -> `num_kv_heads * head_dim` (4 * 16 = 64)
  - `W_v`: 638 -> `num_kv_heads * head_dim` (4 * 16 = 64)
- **QK normalization:**
  1. `q = RMSNorm(q)` -- per-head, output RMS = 1
  2. `k = RMSNorm(k)` -- per-head, output RMS = 1
  3. Optional `k = LearnableShift(k)` -- learnable fractional causal shift
  4. `q = q * q_gain` -- per-head scalar, init = 0.0 + noise(0.01)
- **Attention:** `F.scaled_dot_product_attention(q, k, v, is_causal=True)`
  With `q_gain_init=0.0`, attention logits are ~0 initially, giving near-uniform attention.
- **Output:** Per writable stream, gated projection from attention output:
  - `value = W_o_value(attn_flat)` -- projects to stream dim
  - `value = value_softcap(value)` -- caps at +/-30 (identity for |x|<24)
  - `gate = sigmoid(W_o_gate(attn_flat))` -- in [0, 1]
  - `update = value * gate`
- **Write mode:** Returns deltas only (`skip_residual=True`). The caller
  (`MultiStreamBlock`) handles the alpha/beta residual mix.

### 6.2 CausalMultiStreamAttentionViaMixing (mixing mode)

**File:** `multi_stream_attention.py:71`

Used when `mixing_config is not None`.

- **Reads:** All 4 streams, each with **independent** Q/K/V projections per stream.
- **Mixing:** Learned weights combine per-stream projections:
  - `MixingMode.GLU` (default): `primary * sigmoid(gate)` where primary and gate are
    each softmax-weighted sums of per-stream projections. Enables conjunctive (AND) patterns.
  - `MixingMode.ADDITIVE`: softmax-weighted sum only. Enables disjunctive (OR) patterns.
  - V always uses additive mixing.
- **Mixing source:**
  - `STATIC`: single learned parameter vector per head
  - `DYNAMIC`: input-dependent via linear projection from concatenated streams
  - `DYNAMIC_BOTTLENECK` (default): bottleneck projection (concat -> bottleneck_dim -> mix_logits)
- **Output:** Same gated per-stream pattern as concat mode.

### 6.3 MultiStreamMLP

**File:** `multi_stream_attention.py:604`

- **Reads:** All streams (configurable via `input_stream_ids`), concatenated.
  Input dim = sum of selected stream dims (default: all = 639).
- **Up-projection:** `fc_up`: 638 -> `hidden_dim` (default = 2 * writable_dim = 2 * 272 = 544)
- **Activation:** `leaky_relu(x, slope=0.5).square()` -- squared activation for self-gating.
- **Down-projection per stream:**
  - `value = proj_value(h)` -- projects to stream dim (or hierarchy total_slots for LOGIT)
  - `value = value_softcap(value)` -- caps at +/-30
  - `gate = sigmoid(proj_gate(h))` -- in [0, 1]
  - `update = value * gate`
  - If hierarchy active on LOGIT: `assemble_logits(chunks)` converts hierarchy slots to 208-dim
- **Write targets:**
  - Normal blocks: LOGIT + CONTEXT
  - Last block (`mlp_output_stream_ids=[LOGIT]`): LOGIT only (saves parameters)

### 6.4 MultiStreamCausalConv / MultiStreamCausalConvLayers

**Conv:** `multi_stream_attention.py:728`
**Conv Layers:** `multi_stream_attention.py:826`

- **Reads:** Configurable via `input_stream_ids`. Default: LOGIT + CONTEXT + STRUCTURAL
  (excludes TOKENS). Concatenated: dim = 208 + 64 + 159 = 431.
- **Writes:** Configurable via `output_stream_ids`. Default: CONTEXT only
  (excludes LOGIT). In `MultiStreamBlock`, block conv can write to different streams.
- **Mechanism:** `GatedCausalConv` (`modules.py:495`):
  - Causal left-padding: `output[t]` depends on `input[t-k+1..t]`
  - Std-dev repair: `1/sqrt(fan_in)` normalizes conv output variance
  - Gated mode: `gate * value` where gate=sigmoid, value=SiLU + value_softcap
- **Channel shuffle:** When `groups > 1` and multiple layers, each layer gets a different
  `channel_shift` applied via `torch.roll` before convolution. Provides cross-group
  information flow without extra parameters.
- **Output scale:** Conv output is naturally ~O(1) due to:
  - Std-dev repair (`1/sqrt(fan_in)`)
  - SiLU activation (bounded below by -0.28, soft-linear above)
  - Sigmoid gate (bounds to [0, 1])
  - Value softcap (caps at +/-30)
  - Then alpha~0.05 further attenuates before mixing with beta~0.99 * residual

### 6.5 CausalArithmeticMultiStreamAttention

**File:** `multi_stream_attention.py:948`

- **Reads:**
  - Full-length streams (all 4, concatenated) for Q and op-selector
  - Compressed streams at detected number positions for K
  - Compressed views: metadata (number values, lengths) for arithmetic computation
- **Writes:** Primarily LOGIT via gated scatter:
  - Pre-computes arithmetic results for all number pairs (non-differentiable)
  - Attention weights determine which pair result to use at each position
  - Result is scattered to vocab-space digit positions
  - Gate (scalar per stream, init=-2.0, sigmoid~0.12) controls contribution
  - Non-LOGIT writable streams receive zero deltas (no-op after alpha/beta mixing)

### 6.6 ByteLogitHierarchy

**File:** `byte_modules.py:386`

Not a separate sub-layer -- modifies how MLP and conv output projections target the
LOGIT stream when `structured_output_logits=True`.

Instead of projecting directly to 208 dims, sub-layers project to `total_slots`
(sum of all hierarchical level sizes). The hierarchy decomposes token prediction into:

```
Level 0 (8-way):  bos | pad | digit | letter | separator | punctuation | symbol | multibyte
Level 1 (2-way):  uppercase | lowercase              (for letters)
Level 2 (2-way):  upper_vowel | upper_consonant       (for uppercase)
Level 3 (2-way):  lower_vowel | lower_consonant       (for lowercase)
Level 4 (2-way):  continuation | leading               (for multibyte)
Level 5 (3-way):  lead2 | lead3 | lead4                (for leading bytes)
Leaf levels:      fine-grained per terminal group
```

`assemble_logits()` scatters per-level logits to 208-dim token space.
`SoftcapLinear(30)` is applied to leaf levels only.

This reduces the parameter count of output projections (total_slots << 208 for deep
hierarchies) while imposing the correct categorical structure.

---

## 7. Normalization Analysis and Concerns

### 7.1 CONTEXT Stream: Zeros Through RMSNorm

**Issue:** CONTEXT starts as all zeros. `F.rms_norm(zeros)` computes `0 / sqrt(0 + eps) = 0`.

**Verdict:** Not numerically dangerous (PyTorch's default eps prevents division by zero),
but means the first preconv layers operate on zero CONTEXT. The first nonzero values in
CONTEXT come from the preconv layers (alpha~0.05 of a gated conv output), then get
RMSNorm'd to RMS~1. This is by design -- CONTEXT is a blank slate that gets seeded by
local n-gram patterns from the conv.

### 7.2 CenterLastDim Does Not Control Magnitude

**Issue:** `CenterLastDim` only recenters (`x - mean(x)`). It does NOT normalize to unit
scale. If activations grow in magnitude, CenterLastDim won't prevent it.

**Mitigations in place:**
1. **10x scale factor** brackets the forward pass -- LOGIT operates at 1/10 its natural
   scale inside blocks, limiting how much magnitude is fed to/from sub-layers.
2. **Value softcap (30.0)** inside every sub-layer caps the magnitude of update deltas
   before gating.
3. **Sigmoid gates** naturally bound the gate channel to [0, 1], limiting update magnitude
   to the softcap value.
4. **Alpha/beta mixing** with post-norm re-centers after each sub-layer.
5. **SoftcapLinear(30)** at the output absolutely bounds final logits.

**Residual risk:** If alpha grows large and sub-layer outputs are consistently near the
softcap boundary, the LOGIT stream magnitude could grow across blocks. The post-norm
CenterLastDim only removes the mean, not the variance. In bfloat16, large intermediate
values (>~100) lose precision. However, value_softcap inside sub-layers + sigmoid gating
make this unlikely in practice.

### 7.3 STRUCTURAL Stream: Calibrated Standardization

All 19 STRUCTURAL components inherit from `StreamComponent` (`modules.py`).  Each
component declares a `standardizable_mask` — a per-dim bool tuple indicating which
dims can be z-normalized without breaking geometric structure (rotation pairs and
sin/cos must remain untouched).

`CompositeStream.calibrate()` computes per-dim mean/std from training data, then
applies `scale = (1/std).clamp(max=1000)`, `shift = -mean * scale` on standardizable
dims only.  At runtime, `forward()` applies one fused `x * scale + shift` over the
full 159-dim concatenation.  ~46 of 159 dims are standardizable.

**Remaining scale heterogeneity** (non-standardizable dims):

| Feature | Scale | Notes |
|---------|-------|-------|
| Sin/cos pairs (SinCosPosition, ColumnPosition, Boundary, etc.) | [-1, +1] | Unit circle, consistent |
| Rotation pairs (ByteCategory, PunctuationDepth depth, DigitSequence/Compute) | [-1, +1] | Unit norm vectors |

**Standardizable dims** (post-calibration, mean≈0, std≈1):

| Feature | Raw Scale | Notes |
|---------|-----------|-------|
| Case/VC/Repeat run lengths | [0, ~7] (log1p) | Was 7x larger than sin/cos before calibration |
| ByteHash hit_log + hit_frac | [0, ~7] + [0, 1] | Now centered and scaled |
| ByteCategoryStats fractions | [0, 1], std 0.01-0.03 | Heavily upscaled |
| DigitCompute signs/comparisons | {-1, 0, +1} sparse | Centered; sparse dims may drift between data splits |
| PunctuationDepth quote_state | {-1, +1} biased | ~99% outside quotes |
| MultiByteState binary/scalar | {-1,+1} + [0, 3] | First 2 dims standardized |

**PunctuationDepth bracket_depth** was formerly an unbounded integer cumsum.
It is now encoded via `ContinuousRotation(min_depth=-3, max_depth=8, mode="cap")`,
producing a 2D unit rotation bounded to [-1, +1].

### 7.4 q_gain Initialization at 0.0

**Issue:** `q_gain_init=0.0` means `Q *= 0.0` initially, producing zero attention logits
(uniform attention over all positions).

**Impact:**
- Initial attention output is a uniform average of values
- Combined with gated output (`sigmoid(W_o_gate(attn_flat))` ~ near 0.5 at init),
  initial attention contribution is ~0.5 * uniform_avg
- With `attn_alpha` sigmoid ~0.27, initial attention adds ~14% of a uniform average
- Symmetry breaking relies on `init_noise_std=0.01` (adds `N(0, 0.01)` to q_gain)
  and the fact that different heads see different gradient signals

**Verdict:** Intentional soft start. The model initially passes most information through
the residual (beta~0.99) and gradually learns to attend.

### 7.5 UTF-8 Prior Disabled During Training

**Background:** The UTF-8 prior adds 0 or -inf to LOGIT based on byte validity. This
was explicitly disabled during training (`multi_stream_gpt.py:556-558`) because:

1. CenterLastDim on a vector with some -inf entries would compute a mean dominated by
   the -inf values, collapsing all entries toward +inf.
2. After the 10x scale-up, these extreme values would cause NaN in bfloat16.

**Comment in source:**
> Skipped during training to avoid injecting a bimodal distribution that
> destabilises CenterLastDim + the ×factor scale-up (see NaN analysis).

### 7.6 Overall Assessment

The normalization scheme is **layered and deliberate**:

1. **LOGIT**: CenterLastDim prevents mean drift, 10x factor controls relative scale,
   value_softcap + sigmoid gates prevent individual update explosions,
   SoftcapLinear(30) bounds final output.
2. **CONTEXT**: RMSNorm after every sub-layer maintains unit RMS. This is robust.
3. **Read-only streams**: Fixed scale, no drift possible.
4. **Alpha/beta gating**: Starts near-identity, smooth learnable transition to
   arbitrary mixing. Post-norm after every mix maintains invariants.

The main areas that could benefit from attention:
- **STRUCTURAL sparse features**: After calibration, ~46 standardizable dims are
  well-behaved at the training seq_len, but sparse features (digit signs, quote state)
  can drift between data splits or at different sequence lengths.
- **LOGIT magnitude control**: CenterLastDim doesn't bound variance. If training shows
  growing LOGIT magnitudes inside blocks, consider RMSNorm (but note it would no longer
  be lossless for cross-entropy -- the model would need to learn the softmax temperature
  through the 10x factor).

---

## 8. Training Configuration Reference

Reference: `train_multi_stream_bytes.py`

### Default Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `num_layers` | 6 | Attention blocks |
| `num_heads` / `num_kv_heads` | 8 / 4 | GQA with 2:1 ratio |
| `multi_head_dim` | 128 | = 8 heads * 16 dim/head |
| `context_dim` | 64 | CONTEXT stream width |
| `logit_softcap` | 30.0 | Output SoftcapLinear cap |
| `value_softcap` | 30.0 | Sub-layer value SoftcapLinear cap |
| `logit_stream_normalization_factor` | 10.0 | Scale-down/up bracket |
| `num_preconv_layers` | 2 | Pre-attention conv layers |
| `preconv_kernel_size` | 4 | Local context window |
| `preconv_groups` | 6 | Grouped convolution |
| `structured_output_logits` | True | Use ByteLogitHierarchy |
| `init_noise_std` | 0.01 | Symmetry breaking noise on alpha/beta/gate |

### Optimizer Groups

| Group | Optimizer | LR | What's Included |
|-------|-----------|-----|-----------------|
| Bigram | Adam | 0.01 | `bigram_logits` (V x V) |
| Builder | Adam | 0.01 | Stream component params (mostly buffers) |
| Matrix (Muon) | Muon | 0.04 | All 2D+ weight matrices (W_q, W_k, W_v, W_o, MLP up/down) |
| Scalar | Adam | 0.04 | 1D params: alpha, beta, q_gain, gates, shift_logit, mix_logits |
| Conv | Adam | 0.01 | Conv1d weight tensors |

Muon momentum: warmed from 0.85 to 0.95 over 500 steps.

### Other Training Details

- **Precision:** bfloat16 autocast during forward; fp32 for control params (alpha, beta, gates)
  via `restore_low_dim_params_to_fp32`.
- **Gradient clipping:** disabled by default (`grad_clip_norm=0.0`)
- **LR schedule:** warmdown -- constant LR until `iterations - warmdown_iters`, then linear
  decay to 0. Also supports wallclock-based warmdown.
- **Warmup:** 20 steps of forward passes to prime optimizer state, then reset model weights
  to initial values.
- **Compilation:** `torch.compile(dynamic=False, fullgraph=True)`
- **Batch size:** `train_batch_tokens=524288`, `train_seq_len=1024`
  (effective batch = 512 sequences, distributed across grad_accum steps and DDP workers)

---

