Universal Transformer experiment on the 10-minute record track.

Transformer blocks (attention + relu^2 MLP) are shared across groups of layers via
a configurable BLOCK_PATTERN. Only per-layer scalars (attn_scale, mlp_scale, resid_mix)
are unique per application. This is ALBERT-style weight sharing.

Configuration (default):
- Architecture: Universal Transformer with grouped block sharing, relu^2 MLP
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=27 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Block sharing: 9 shared blocks, 3 layers per block (`BLOCK_PATTERN=0,0,0,1,1,1,...,8,8,8`)
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Model params: `~17,101,384` (matches baseline budget)

Key architecture details:
- `SharedBlock`: attention + relu^2 MLP + norms, shared across layers assigned to the same block
- `LayerScalars`: 27 independent sets of (attn_scale, mlp_scale, resid_mix)
- relu^2 MLP: fc projection, relu, square, then output projection (2x expansion)
- Encoder/decoder U-net skip connections (13 encoder + 14 decoder layers)
- Same width as baseline (512) with 3x depth (27 vs 9) via weight sharing

BLOCK_PATTERN examples:
- `""` — single shared block for all layers (extreme ALBERT-style sharing)
- `"0,0,0,1,1,1,..."` — grouped sharing, 3 layers per block (default)
- `"0,1,2,...,26"` — no sharing, each layer has its own block
- `"0,1,0,1,..."` — alternating/interleaved patterns

Parameter budget comparison:
- Baseline GPT (9 layers, per-layer everything): ~17.1M params
- Universal (27 layers, 9 shared blocks, 3 per block): ~17.1M params

Status: Not yet trained — code-only snapshot for experimentation.
