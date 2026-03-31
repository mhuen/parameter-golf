"""Explore FA4 score_mod capabilities and function signatures."""

import inspect
import os

# 1) Function signatures
from flash_attn.cute import flash_attn_func, flash_attn_varlen_func

print("=== flash_attn_func ===")
print(inspect.signature(flash_attn_func))
print()
print("=== flash_attn_varlen_func ===")
print(inspect.signature(flash_attn_varlen_func))
print()

# 2) Find all score_mod references in the FA4 package
import flash_attn.cute

pkg_dir = os.path.dirname(flash_attn.cute.__file__)
print(f"=== score_mod references in {pkg_dir} ===")
for root, dirs, files in os.walk(pkg_dir):
    for f in sorted(files):
        if f.endswith(".py"):
            path = os.path.join(root, f)
            with open(path) as fh:
                content = fh.read()
            if "score_mod" in content:
                for i, line in enumerate(content.split("\n")):
                    if "score_mod" in line:
                        print(f"{path}:{i+1}: {line.rstrip()}")
print()

# 3) Check if there's a bias/attn_bias parameter
print("=== Checking for bias support ===")
for name, fn in [("flash_attn_func", flash_attn_func), ("flash_attn_varlen_func", flash_attn_varlen_func)]:
    sig = inspect.signature(fn)
    for param_name in sig.parameters:
        if "bias" in param_name.lower():
            print(f"  {name} has parameter: {param_name}")
    # Also check the underlying FlashAttnFunc.forward if accessible
print()

# 4) Quick smoke test: trivial score_mod (no aux_tensor indexing)
import torch

print("=== Smoke test: trivial score_mod (multiply by 1.0) ===")
try:
    B, S, H, D = 2, 64, 4, 32
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)

    def trivial_score_mod(scores):
        return scores * 1.0

    out, lse = flash_attn_func(q, k, v, causal=True, score_mod=trivial_score_mod)
    print(f"  OK — output shape: {out.shape}")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
print()

# 5) Test: score_mod that adds a constant
print("=== Smoke test: score_mod adds constant ===")
try:
    def add_constant_score_mod(scores):
        return scores + 0.1

    out, lse = flash_attn_func(q, k, v, causal=True, score_mod=add_constant_score_mod)
    print(f"  OK — output shape: {out.shape}")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
print()

# 6) Test: score_mod with aux_tensors — just access shape, no indexing
print("=== Smoke test: score_mod with aux_tensors (no indexing) ===")
try:
    bias_tensor = torch.ones(H, device="cuda", dtype=torch.bfloat16)

    def aux_no_index_score_mod(scores, aux_tensors=None):
        # Just return scores unchanged — test that aux_tensors can be passed
        return scores

    out, lse = flash_attn_func(
        q, k, v, causal=True,
        score_mod=aux_no_index_score_mod,
        aux_tensors=[bias_tensor],
    )
    print(f"  OK — output shape: {out.shape}")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
print()

# 7) Test: varlen with trivial cu_seqlens (no score_mod)
print("=== Smoke test: varlen with trivial cu_seqlens ===")
try:
    total = B * S
    q_flat = q.reshape(total, H, D)
    k_flat = k.reshape(total, H, D)
    v_flat = v.reshape(total, H, D)
    cu = torch.arange(0, (B + 1) * S, S, device="cuda", dtype=torch.int32)

    out, lse = flash_attn_varlen_func(
        q_flat, k_flat, v_flat,
        cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=S, max_seqlen_k=S,
        causal=True,
    )
    print(f"  OK — output shape: {out.shape}")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
print()

# 8) Print score_mod function signature if discoverable from FA4 source
print("=== Looking for score_mod signature/docstring ===")
interface_path = os.path.join(pkg_dir, "interface.py")
if os.path.exists(interface_path):
    with open(interface_path) as fh:
        lines = fh.readlines()
    for i, line in enumerate(lines):
        if "score_mod" in line and ("def " in line or "Args:" in lines[max(0,i-3):i+1] or "param" in line.lower() or ":" in line):
            # Print surrounding context
            start = max(0, i - 2)
            end = min(len(lines), i + 3)
            for j in range(start, end):
                print(f"  {j+1}: {lines[j].rstrip()}")
            print("  ---")
