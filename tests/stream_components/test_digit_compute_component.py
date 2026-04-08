"""Tests for DigitComputeComponent BOS reset."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import torch

from efficient_byte_tokenizer import EfficientByteTokenizer
from byte_stream_components import DigitComputeComponent

tok = EfficientByteTokenizer()


@torch.no_grad()
def test_shape():
    comp = DigitComputeComponent(tok, k=2)
    ids = np.concatenate([[tok.bos_id], tok.encode("12+34")])
    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    out = comp(ids, dtype=torch.float32)
    assert out.shape == (1, ids.shape[1], comp.dim)


@torch.no_grad()
def test_bos_reset():
    """Ring buffer and run state reset at document boundary."""
    comp = DigitComputeComponent(tok, k=2)
    doc1 = tok.encode("99+88")
    doc2 = tok.encode("12+34")
    multi = np.concatenate([[tok.bos_id], doc1, [tok.bos_id], doc2])
    ids_multi = torch.tensor(multi, dtype=torch.long).unsqueeze(0)
    out_multi = comp(ids_multi, dtype=torch.float32)

    solo = np.concatenate([[tok.bos_id], doc2])
    ids_solo = torch.tensor(solo, dtype=torch.long).unsqueeze(0)
    out_solo = comp(ids_solo, dtype=torch.float32)

    doc2_start = 1 + len(doc1)
    torch.testing.assert_close(
        out_multi[0, doc2_start:], out_solo[0], atol=1e-5, rtol=1e-5
    )
