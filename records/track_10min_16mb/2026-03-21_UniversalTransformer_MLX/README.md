Universal Transformer experiment on the 10-minute record track.

Transformer blocks (attention + SwiGLU MLP) are shared across groups of layers via
a configurable BLOCK_PATTERN. Only per-layer scalars (attn_scale, mlp_scale, resid_mix)
are unique per application. This is ALBERT-style weight sharing.

Configuration (default):
- Architecture: Universal Transformer with grouped block sharing, SwiGLU MLP
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=21 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Block sharing: 7 shared blocks, 3 layers per block (`BLOCK_PATTERN=0,0,0,1,1,1,...,6,6,6`)
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Model params: `~17,087,544` (matches baseline budget)

Key architecture details:
- `SharedBlock`: attention + SwiGLU + norms, shared across layers assigned to the same block
- `LayerScalars`: 21 independent sets of (attn_scale, mlp_scale, resid_mix)
- SwiGLU MLP: gate + up projections with SiLU gating, then down projection (2x expansion)
- Encoder/decoder U-net skip connections (10 encoder + 11 decoder layers)
- Same width as baseline (512) with 2.3x depth (21 vs 9) via weight sharing

BLOCK_PATTERN examples:
- `""` — single shared block for all layers (extreme ALBERT-style sharing)
- `"0,0,0,1,1,1,..."` — grouped sharing, 3 layers per block (default)
- `"0,1,2,...,20"` — no sharing, each layer has its own block
- `"0,1,0,1,..."` — alternating/interleaved patterns

Parameter budget comparison:
- Baseline GPT (9 layers, per-layer everything): ~17.1M params
- Universal (21 layers, 7 shared blocks, 3 per block): ~17.1M params

Status: Not yet trained — code-only snapshot for experimentation.
