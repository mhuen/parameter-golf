This record replaces transformer attention blocks with causal depthwise-separable 1D
convolutions of increasing kernel sizes, while preserving the residual stream structure.

Configuration:
- Architecture: Causal convolutional LM (no attention, no RoPE)
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 MLP_MULT=2`
- Kernel sizes per layer: `[3, 5, 7, 9, 13, 17, 21, 25, 31]`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Batching: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`

Key architecture differences vs GPT baseline:
- Attention replaced by causal depthwise 1D convolution (each channel has its own kernel)
- Causality enforced via left-padding by (kernel_size - 1)
- Pointwise relu^2 MLP retained for channel mixing (identical to baseline)
- Both conv and MLP sub-blocks write to the residual stream with learned scale vectors
- Encoder/decoder U-net skip connections preserved (4 encoder + 5 decoder layers)
- resid_mix (blending current residual with x0) preserved per block
- Much fewer params per block than attention (depthwise conv is dim * kernel_size vs 4 * dim^2)
- No Q/K/V projections, no RoPE, no attention heads

Status: Not yet trained — code-only snapshot for experimentation.

Included files:
- `train_conv_mlx.py` (code snapshot)
- `submission.json` (leaderboard metadata)
