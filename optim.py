"""Muon optimizer and Newton-Schulz orthogonalization utilities."""

import torch
import torch.distributed as dist
from torch import Tensor

# Try to import Dao-AILab's optimized Gram Newton-Schulz (requires Hopper/Blackwell + CUDA 12.9+)
try:
    from gram_newton_schulz import GramNewtonSchulz, POLAR_EXPRESS_COEFFICIENTS

    _gram_ns_op = GramNewtonSchulz(
        ns_coefficients=POLAR_EXPRESS_COEFFICIENTS,
        gram_newton_schulz_reset_iterations=[2],
    )
    _GRAM_NS_LIB = True
except ImportError:
    _GRAM_NS_LIB = False

# Polar Express coefficients for pure-PyTorch Gram NS fallback.
_GRAM_NS_COEFFS = [
    (8.123737, -22.232240, 16.373715),
    (4.026529, -2.776323, 0.514551),
    (3.870284, -2.739120, 0.520999),
    (3.253351, -2.343223, 0.481420),
    (2.300652, -1.668904, 0.418807),
]


def _zeropower_standard_ns5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


def _zeropower_gram_ns5(G: Tensor, eps: float = 1e-7) -> Tensor:
    """Stabilized Gram Newton-Schulz: iterates on the n×n Gram matrix instead of
    the full n×m rectangle.  ~42-58 % fewer FLOPs for rectangular matrices.
    Falls back to standard NS for square matrices (no benefit)."""
    if G.size(0) == G.size(1):
        return _zeropower_standard_ns5(G, eps=eps)
    X = G.half()  # Gram NS uses fp16, not bf16
    X /= X.norm() + eps
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    n = X.size(0)
    R = X @ X.T
    Q = torch.eye(n, device=X.device, dtype=X.dtype)
    for t in range(5):
        if t == 2:  # restart after iteration 2 for numerical stability
            X = Q @ X
            R = X @ X.T
            Q = torch.eye(n, device=X.device, dtype=X.dtype)
        a, b, c = _GRAM_NS_COEFFS[t]
        Z = b * R + c * R @ R
        Q = Q @ Z + a * Q
        RZ = R @ Z + a * R
        R = Z @ RZ + a * RZ
    X = Q @ X
    return X.T if transposed else X


def zeropower_via_newtonschulz5(
    G: Tensor, steps: int = 10, eps: float = 1e-7, gram_ns: bool = False
) -> Tensor:
    """Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration."""
    if gram_ns and G.size(0) != G.size(1):
        if _GRAM_NS_LIB:
            return _gram_ns_op(G)
        return _zeropower_gram_ns5(G, eps=eps)
    return _zeropower_standard_ns5(G, steps=steps, eps=eps)


class Muon(torch.optim.Optimizer):
    """Muon optimizer: Newton-Schulz orthogonalized momentum for matrix-shaped params.

    Args:
        reshape_3d: If True, 3D+ params are reshaped to 2D before NS by stacking
            along dim-0: (K, M, N) → (K*M, N). This applies NS jointly across
            all slices, encouraging inter-slice diversity (useful for Kronecker/
            Monarch factored weights).
    """

    def __init__(
        self,
        params,
        lr: float,
        momentum: float,
        backend_steps: int,
        nesterov: bool = True,
        gram_ns: bool = False,
        reshape_3d: bool = False,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                backend_steps=backend_steps,
                nesterov=nesterov,
                gram_ns=gram_ns,
                reshape_3d=reshape_3d,
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            gram_ns = group["gram_ns"]
            reshape_3d = group["reshape_3d"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(
                total_params, device=params[0].device, dtype=torch.bfloat16
            )

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    # Optionally reshape 3D+ to 2D for NS
                    orig_shape = g.shape
                    if reshape_3d and g.ndim > 2:
                        g = g.reshape(-1, g.shape[-1])
                    g = zeropower_via_newtonschulz5(
                        g, steps=backend_steps, gram_ns=gram_ns
                    )
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    if g.shape != orig_shape:
                        g = g.reshape(orig_shape)
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss
