This record captures an xLSTM (mLSTM) experiment on the 10-minute record track.

The model replaces transformer self-attention with the mLSTM (matrix LSTM) architecture from Beck et al. 2024, implemented in pure MLX (no CUDA/Triton kernels needed). Training infrastructure (data loading, Muon+Adam optimizer, int8+zlib quantization) is adapted from the baseline `train_gpt_mlx.py`.

Configuration:
- Architecture: xLSTM with mLSTM blocks (pre-norm mLSTM + pre-norm SiLU-gated FFN)
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=6 MODEL_DIM=512 NUM_HEADS=4 QK_DIM_FACTOR=0.5 V_DIM_FACTOR=1.0 FFN_MULT=2.0`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Model params: `16,287,280` (GPT baseline: `17,059,912`)
- Estimated compressed size (int8+zlib): `5.46 MB` (GPT baseline: ~15.8 MB)
- Batching: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`

Key architecture differences vs GPT baseline:
- mLSTM layer replaces causal self-attention with gated linear recurrence using q/k/v + scalar input/forget gates per head
- Training uses parallel form (materialized S×S gating matrix), efficient for seq_len=1024
- SiLU-gated FFN (SwiGLU) instead of relu^2 MLP
- Simple residual blocks (no encoder/decoder skip connections)
- No RoPE (positional info comes from the recurrent gating structure)

Status: Not yet trained — code-only snapshot for experimentation.

Included files:
- `train_xlstm_mlx.py` (code snapshot)
- `submission.json` (leaderboard metadata)
