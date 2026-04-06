"""Benchmark: factored einsum vs materialize-once-then-dense for Kronecker & Monarch."""
import math, time, torch, torch.nn as nn, torch.nn.functional as F

torch.set_grad_enabled(True)  # we want to test backward too

def _balanced_factors(n):
    s = int(math.sqrt(n))
    while n % s != 0:
        s -= 1
    return s, n // s

def _nearest_divisor(n, target):
    best = 1
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            for d in (i, n // i):
                if abs(d - target) < abs(best - target):
                    best = d
    return best

# ── Kronecker ──────────────────────────────────────────────────────────────
class KroneckerFactored(nn.Module):
    """Current approach: einsum reshape trick, never materializes W."""
    def __init__(self, in_f, out_f, num_terms=4):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f
        self.p_in, self.q_in = _balanced_factors(in_f)
        self.p_out, self.q_out = _balanced_factors(out_f)
        self.A = nn.Parameter(torch.randn(num_terms, self.p_out, self.p_in))
        self.B = nn.Parameter(torch.randn(num_terms, self.q_out, self.q_in))
        scale = (in_f * out_f) ** -0.5
        nn.init.normal_(self.A, std=scale)
        nn.init.normal_(self.B, std=scale)

    def forward(self, x):
        leading = x.shape[:-1]
        x = x.reshape(-1, self.q_in, self.p_in)
        out = torch.einsum("kim,bmn,kjn->bij", self.B.to(x.dtype), x, self.A.to(x.dtype))
        return out.reshape(*leading, self.out_f)


class KroneckerMaterialize(nn.Module):
    """Materialize W = Σ_k A_k⊗B_k, then dense matmul."""
    def __init__(self, in_f, out_f, num_terms=4):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f
        self.p_in, self.q_in = _balanced_factors(in_f)
        self.p_out, self.q_out = _balanced_factors(out_f)
        self.A = nn.Parameter(torch.randn(num_terms, self.p_out, self.p_in))
        self.B = nn.Parameter(torch.randn(num_terms, self.q_out, self.q_in))
        scale = (in_f * out_f) ** -0.5
        nn.init.normal_(self.A, std=scale)
        nn.init.normal_(self.B, std=scale)

    def _materialize(self):
        # W = Σ_k kron(A_k, B_k)  — shape (out_f, in_f)
        A, B = self.A.to(torch.bfloat16), self.B.to(torch.bfloat16)
        # kron via outer products: (K, p_out, p_in) x (K, q_out, q_in)
        # -> (K, p_out, q_out, p_in, q_in) -> (K, out_f, in_f) -> sum over K
        W = torch.einsum("kij,kmn->kimjn", A, B)  # (K, p_out, q_out, p_in, q_in)
        K = A.shape[0]
        W = W.reshape(K, self.out_f, self.in_f).sum(0)  # (out_f, in_f)
        return W

    def forward(self, x):
        W = self._materialize()
        return F.linear(x, W)


class KroneckerMaterializeCached(nn.Module):
    """Materialize once, cache, reuse across multiple forward calls."""
    def __init__(self, in_f, out_f, num_terms=4):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f
        self.p_in, self.q_in = _balanced_factors(in_f)
        self.p_out, self.q_out = _balanced_factors(out_f)
        self.A = nn.Parameter(torch.randn(num_terms, self.p_out, self.p_in))
        self.B = nn.Parameter(torch.randn(num_terms, self.q_out, self.q_in))
        scale = (in_f * out_f) ** -0.5
        nn.init.normal_(self.A, std=scale)
        nn.init.normal_(self.B, std=scale)
        self._W_cache = None

    def materialize(self):
        """Call once before forward pass(es). Stays in autograd graph."""
        A, B = self.A.to(torch.bfloat16), self.B.to(torch.bfloat16)
        W = torch.einsum("kij,kmn->kimjn", A, B)
        K = A.shape[0]
        self._W_cache = W.reshape(K, self.out_f, self.in_f).sum(0)

    def forward(self, x):
        return F.linear(x, self._W_cache)


# ── Monarch ────────────────────────────────────────────────────────────────
class MonarchFactored(nn.Module):
    """Current approach: two einsum block-diagonal matmuls."""
    def __init__(self, in_f, out_f, nblocks=0):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f
        if nblocks <= 0:
            nblocks = _nearest_divisor(min(in_f, out_f), int(math.sqrt(min(in_f, out_f))))
        self.nblocks = nblocks
        self.blk_in = in_f // nblocks
        self.blk_out2 = out_f // self.blk_in
        self.w1 = nn.Parameter(torch.randn(nblocks, self.blk_in, self.blk_in))
        self.w2 = nn.Parameter(torch.randn(self.blk_in, nblocks, self.blk_out2))
        scale = (in_f * out_f) ** -0.25
        nn.init.normal_(self.w1, std=scale)
        nn.init.normal_(self.w2, std=scale)

    def forward(self, x):
        leading = x.shape[:-1]
        x = x.reshape(-1, self.nblocks, self.blk_in)
        x = torch.einsum("bni,nij->bnj", x, self.w1.to(x.dtype))
        x = x.transpose(1, 2).contiguous()
        x = torch.einsum("bin,ino->bio", x, self.w2.to(x.dtype))
        return x.reshape(*leading, self.out_f)


class MonarchMaterialize(nn.Module):
    """Materialize full W from Monarch factors, then dense matmul."""
    def __init__(self, in_f, out_f, nblocks=0):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f
        if nblocks <= 0:
            nblocks = _nearest_divisor(min(in_f, out_f), int(math.sqrt(min(in_f, out_f))))
        self.nblocks = nblocks
        self.blk_in = in_f // nblocks
        self.blk_out2 = out_f // self.blk_in
        self.w1 = nn.Parameter(torch.randn(nblocks, self.blk_in, self.blk_in))
        self.w2 = nn.Parameter(torch.randn(self.blk_in, nblocks, self.blk_out2))
        scale = (in_f * out_f) ** -0.25
        nn.init.normal_(self.w1, std=scale)
        nn.init.normal_(self.w2, std=scale)

    def _materialize(self):
        w1, w2 = self.w1.to(torch.bfloat16), self.w2.to(torch.bfloat16)
        n, b = self.nblocks, self.blk_in
        # Stage 1: block-diagonal -> (out_f_1, in_f) sparse structure
        # W1 full: block_diag(w1[0], w1[1], ...) shape (n*b, n*b) = (in_f, in_f)
        W1 = torch.zeros(n * b, n * b, dtype=w1.dtype, device=w1.device)
        for i in range(n):
            W1[i*b:(i+1)*b, i*b:(i+1)*b] = w1[i]
        # Permutation P: reshape (n, b) -> (b, n) -> flatten
        # P[j*n + i] = i*b + j  (maps position (i,j) in (n,b) to j*n+i in (b,n))
        perm = torch.arange(n * b, device=w1.device).reshape(n, b).T.reshape(-1)
        P = torch.zeros(n * b, n * b, dtype=w1.dtype, device=w1.device)
        P[torch.arange(n * b, device=w1.device), perm] = 1.0
        # Stage 2: block-diagonal with blocks w2[i] shape (n, blk_out2)
        # W2 full: block_diag(w2[0], ..., w2[b-1]) shape (b*blk_out2, b*n) = (out_f, in_f)
        W2 = torch.zeros(b * self.blk_out2, b * n, dtype=w2.dtype, device=w2.device)
        for i in range(b):
            W2[i*self.blk_out2:(i+1)*self.blk_out2, i*n:(i+1)*n] = w2[i].T
        # Full: W = W2 @ P @ W1
        W = W2 @ P @ W1
        return W

    def forward(self, x):
        W = self._materialize()
        return F.linear(x, W)


class MonarchMaterializeFast(nn.Module):
    """Materialize full W using einsum (no loops), then dense matmul."""
    def __init__(self, in_f, out_f, nblocks=0):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f
        if nblocks <= 0:
            nblocks = _nearest_divisor(min(in_f, out_f), int(math.sqrt(min(in_f, out_f))))
        self.nblocks = nblocks
        self.blk_in = in_f // nblocks
        self.blk_out2 = out_f // self.blk_in
        self.w1 = nn.Parameter(torch.randn(nblocks, self.blk_in, self.blk_in))
        self.w2 = nn.Parameter(torch.randn(self.blk_in, nblocks, self.blk_out2))
        scale = (in_f * out_f) ** -0.25
        nn.init.normal_(self.w1, std=scale)
        nn.init.normal_(self.w2, std=scale)

    def _materialize(self):
        w1, w2 = self.w1.to(torch.bfloat16), self.w2.to(torch.bfloat16)
        # w1: (nblocks, blk_in, blk_in) — stage 1 block-diagonal
        # w2: (blk_in, nblocks, blk_out2) — stage 2 block-diagonal
        # Full W[out_row, in_col] where:
        #   in_col is indexed as (block_n, pos_j) -> n*blk_in + j
        #   After stage 1: intermediate[n, j'] = Σ_j w1[n, j', j] * x[n, j]
        #   After shuffle: intermediate becomes [j', n]
        #   After stage 2: out[j', o] = Σ_n w2[j', n, o] * intermediate[j', n]
        #   So out[j', o] = Σ_n w2[j', n, o] * Σ_j w1[n, j', j] * x[n, j]
        #   = Σ_{n,j} w2[j', n, o] * w1[n, j', j] * x[n*blk_in + j]
        #
        # Output indexed as (j', o) -> j'*blk_out2 + o
        # Input indexed as (n, j) -> n*blk_in + j
        # W[(j'*blk_out2 + o), (n*blk_in + j)] = w2[j', n, o] * w1[n, j', j]
        #
        # Einsum: W_4d[j', o, n, j] = w2[j', n, o] * w1[n, j', j]
        W = torch.einsum("jno,njk->jonk", w2, w1)
        # W shape: (blk_in, blk_out2, nblocks, blk_in) -> reshape to (out_f, in_f)
        W = W.reshape(self.out_f, self.in_f)
        return W

    def forward(self, x):
        W = self._materialize()
        return F.linear(x, W)


# ── Dense baseline ─────────────────────────────────────────────────────────
class DenseLinear(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f))
        nn.init.normal_(self.weight, std=(in_f * out_f) ** -0.5)

    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))


# ── Benchmark ──────────────────────────────────────────────────────────────
def bench(name, layer, x, n_iters=200, n_warmup=30, n_reuse=3):
    """n_reuse simulates weight sharing: multiple forward calls per materialization."""
    # Warmup
    for _ in range(n_warmup):
        if hasattr(layer, 'materialize'):
            layer.materialize()
        for _ in range(n_reuse):
            y = layer(x)
        loss = y.sum()
        loss.backward()
        for p in layer.parameters():
            if p.grad is not None:
                p.grad.zero_()

    # Timed
    fwd_times = []
    bwd_times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        if hasattr(layer, 'materialize'):
            layer.materialize()
        for _ in range(n_reuse):
            y = layer(x)
        t1 = time.perf_counter()
        loss = y.sum()
        loss.backward()
        t2 = time.perf_counter()
        fwd_times.append(t1 - t0)
        bwd_times.append(t2 - t1)
        for p in layer.parameters():
            if p.grad is not None:
                p.grad.zero_()

    fwd_ms = 1000 * sum(fwd_times) / len(fwd_times)
    bwd_ms = 1000 * sum(bwd_times) / len(bwd_times)
    tot_ms = fwd_ms + bwd_ms
    n_params = sum(p.numel() for p in layer.parameters())
    print(f"  {name:40s}  fwd={fwd_ms:6.2f}ms  bwd={bwd_ms:6.2f}ms  "
          f"total={tot_ms:6.2f}ms  params={n_params:,}")


def main():
    device = "cpu"
    dtype = torch.bfloat16

    # Typical model dimensions
    configs = [
        ("attn_qkv", 512, 512),
        ("mlp_up",   512, 2048),
        ("mlp_down", 2048, 512),
    ]

    batch_seq = 8 * 512  # batch_size * seq_len

    for label, in_f, out_f in configs:
        print(f"\n{'='*80}")
        print(f" {label}: ({in_f} -> {out_f})  input=({batch_seq}, {in_f})  n_reuse=3 (weight sharing)")
        print(f"{'='*80}")
        x = torch.randn(batch_seq, in_f, dtype=dtype, device=device)

        dense = DenseLinear(in_f, out_f).to(device)
        k_fac = KroneckerFactored(in_f, out_f).to(device)
        k_mat = KroneckerMaterialize(in_f, out_f).to(device)
        k_cache = KroneckerMaterializeCached(in_f, out_f).to(device)
        m_fac = MonarchFactored(in_f, out_f).to(device)
        m_mat = MonarchMaterialize(in_f, out_f).to(device)
        m_fast = MonarchMaterializeFast(in_f, out_f).to(device)

        bench("Dense (baseline)", dense, x)
        bench("Kronecker factored (einsum)", k_fac, x)
        bench("Kronecker materialize (per call)", k_mat, x)
        bench("Kronecker materialize (cached, 1x)", k_cache, x)
        bench("Monarch factored (einsum)", m_fac, x)
        bench("Monarch materialize (loop)", m_mat, x)
        bench("Monarch materialize (einsum)", m_fast, x)

    # Also test with n_reuse=1 (no weight sharing benefit)
    print(f"\n\n{'#'*80}")
    print(f" Same but n_reuse=1 (no weight sharing)")
    print(f"{'#'*80}")
    for label, in_f, out_f in configs:
        print(f"\n{'='*80}")
        print(f" {label}: ({in_f} -> {out_f})  n_reuse=1")
        print(f"{'='*80}")
        x = torch.randn(batch_seq, in_f, dtype=dtype, device=device)

        dense = DenseLinear(in_f, out_f).to(device)
        k_fac = KroneckerFactored(in_f, out_f).to(device)
        k_cache = KroneckerMaterializeCached(in_f, out_f).to(device)
        m_fac = MonarchFactored(in_f, out_f).to(device)
        m_fast = MonarchMaterializeFast(in_f, out_f).to(device)

        bench("Dense (baseline)", dense, x, n_reuse=1)
        bench("Kronecker factored (einsum)", k_fac, x, n_reuse=1)
        bench("Kronecker materialize (cached, 1x)", k_cache, x, n_reuse=1)
        bench("Monarch factored (einsum)", m_fac, x, n_reuse=1)
        bench("Monarch materialize (einsum)", m_fast, x, n_reuse=1)


if __name__ == "__main__":
    main()
