Universal Transformer experiment on the 10-minute record track.

A single transformer block (attention + SwiGLU MLP) is applied 24 times. Only per-layer
scalars (attn_scale, mlp_scale, resid_mix) are unique per application. This is the
extreme end of ALBERT-style weight sharing.

Configuration:
- Architecture: Universal Transformer (fully shared block) with SwiGLU MLP
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=24 MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=8 MLP_MULT=3`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Model params: `~13,742,096`
- Estimated compressed size (int8+zlib): ~9.2 MB

Key architecture details:
- `SharedBlock`: single instance containing CausalSelfAttention + SwiGLU + norms
- `LayerScalars`: 24 independent sets of (attn_scale, mlp_scale, resid_mix)
- SwiGLU MLP: gate + up projections with SiLU gating, then down projection (3x expansion)
- Encoder/decoder U-net skip connections (12 encoder + 12 decoder layers)
- 2x wider than baseline (1024 vs 512) with 2.7x depth (24 vs 9)

Parameter budget comparison:
- Baseline GPT (9 layers, per-layer everything): ~17.1M params
- Shared-MLP (16 layers, shared MLP only): ~14.7M params
- Universal (24 layers, fully shared block): ~13.7M params

Status: Not yet trained — code-only snapshot for experimentation.
