This record captures a shared-MLP deep attention experiment on the 10-minute record track.

All transformer layers share a single MLP (fc + proj weights), while each layer has its own
attention block. This is inspired by ALBERT-style cross-layer parameter sharing, applied
selectively: only the MLP is shared, attention remains unique per layer.

Configuration:
- Architecture: GPT with shared MLP across all layers
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=16 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Model params: `14,717,056` (GPT baseline: `17,059,912`)
- Estimated compressed size (int8+zlib): ~9.8 MB
- Batching: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`

Key architecture differences vs GPT baseline:
- Single shared MLP (3x expansion) used by all 16 layers, passed as argument to Block.__call__()
- 16 layers (vs 9 baseline) — nearly 2x depth with unique attention per layer
- Each layer still has independent scalars: attn_scale, mlp_scale, resid_mix, q_gain
- Encoder/decoder U-net skip connections (8 encoder + 8 decoder layers)
- MLP output projection zero-initialized (same as baseline attention proj)

Status: Not yet trained — code-only snapshot for experimentation.

Included files:
- `train_shared_mlp_mlx.py` (code snapshot)
- `submission.json` (leaderboard metadata)
