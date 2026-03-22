This record replaces all RMS normalization in the baseline `train_gpt_mlx.py` with
analytically-derived scaling factors that maintain variance ~1 and mean ~0 throughout
the forward pass.

Hypothesis: RMS norm may add unwanted regularization by forcing unit RMS at every
normalization point. By removing it and instead using fixed scaling derived from the
statistical properties of each operation, the model may have more freedom to learn
optimal activation magnitudes.

Variance-preserving scaling rules applied:
- **Matmuls** (`CastedLinear`): sum over `in_dim` elements → scale by `1/sqrt(in_dim)`.
  Weights initialized with unit variance (std=1) instead of Glorot.
- **relu^2 activation**: for x~N(0,1), E[(relu(x)^2)^2] = 3/2. Scale by sqrt(2/3)
  before the output projection so it sees unit second moment.
- **Residual additions** (x + y): adding two ~unit-variance signals doubles variance.
  Scale by `1/sqrt(2)` after each add.
- **Skip connections**: same `1/sqrt(2)` scaling.
- **Embedding**: initialized with std=1.0 (up from 0.005) so lookup gives var~1 directly.
- **LM head** (tied embedding matmul): reduces over `dim` → scale by `1/sqrt(dim)`.
- **Final output**: learnable per-dim gain (replaces final RMS norm).

Components removed:
- `rms_norm()` function
- `RMSNormNoWeight` module
- `attn_norm`, `mlp_norm` in each `Block`
- `final_norm` in `GPT`

Components kept unchanged:
- Zero-init output projections (attn.proj, mlp.proj)
- Learnable `attn_scale`, `mlp_scale`, `resid_mix`, `skip_weights`, `q_gain`
- Muon + Adam optimizer split
- All training infrastructure (data loading, quantization, eval)

Configuration (same as baseline unless noted):
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied embeddings with `TIED_EMBED_INIT_STD=1.0` (changed from 0.005)
- Batching: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`

Status: Not yet trained — code-only snapshot for experimentation.

Included files:
- `train_gpt_mlx.py` (code snapshot)
- `submission.json` (leaderboard metadata)
