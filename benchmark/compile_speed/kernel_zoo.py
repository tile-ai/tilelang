"""A zoo of realistic inference kernels for the compile-speed benchmark.

Each family mirrors a kernel that a real transformer inference stack compiles
(GEMM projections, GQA flash-attention, RMSNorm, fused SiLU-gate, softmax), built
from the corresponding ``examples/`` template so it actually compiles. Every
family is swept over real model shapes (Qwen2.5 / Llama-3 dims), and each
``(family, shape, tile)`` combination is a distinct translation unit — a genuine
cold-cache miss that runs the full lowering + nvcc pipeline.

``build_zoo()`` returns a list of ``(name, prim_func)`` pairs; the benchmark
compiles them with ``par_compile``. Keep the kernels light: the point is *many
diverse* compiles (the parallel-AOT / autotune regime), not a few heavy ones.
"""

import tilelang.language as T

# --- Realistic inference shapes ------------------------------------------------
# (hidden, intermediate) pairs from common open models.
MODEL_DIMS = [
    (2048, 5632),  # Llama-3.2-1B-ish
    (3584, 18944),  # Qwen2.5-7B
    (4096, 14336),  # Llama-3-8B
    (5120, 13824),  # Qwen2.5-14B-ish
]
# (num_q_heads, num_kv_heads, head_dim) GQA configs.
ATTN_HEADS = [(32, 8, 128), (28, 4, 128), (40, 8, 128)]
SEQ_TILES = [64, 128]
GEMM_TILES = [(64, 64, 32), (128, 128, 32), (128, 64, 64)]


def gemm(M, N, K, block_M, block_N, block_K, dtype="float16"):
    """Dense projection GEMM (q/k/v/o/gate/up/down proj). examples/gemm."""

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def gqa_attention(batch, heads, kv_heads, seq_len, dim, block_M, block_N):
    """GQA flash-attention forward. examples/flash_attention/example_gqa_fwd."""
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, kv_heads, dim]
    group = heads // kv_heads
    dtype, accum = "float16", "float32"

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=128) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum)
            scores_max = T.alloc_fragment([block_M], accum)
            scores_max_prev = T.alloc_fragment([block_M], accum)
            scores_scale = T.alloc_fragment([block_M], accum)
            scores_sum = T.alloc_fragment([block_M], accum)
            logsum = T.alloc_fragment([block_M], accum)
            kv_head = by // group

            T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum))
            for k in T.Pipelined(T.ceildiv(seq_len, block_N), num_stages=1):
                T.copy(K[bz, k * block_N : (k + 1) * block_N, kv_head, :], K_shared)
                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True)
                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2((scores_max_prev[i] - scores_max[i]) * scale)
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]
                T.copy(V[bz, k * block_N : (k + 1) * block_N, kv_head, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o)
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

    return main


def rms_norm(M, N, block_M, dtype="float32"):
    """RMSNorm over the hidden dim. examples/norm/rms_norm.py."""

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, block_M), threads=128) as bx:
            A_local = T.alloc_fragment((block_M, N), dtype)
            A_pow = T.alloc_fragment((block_M, N), dtype)
            A_sum = T.alloc_fragment((block_M,), dtype)
            T.copy(A[bx * block_M, 0], A_local)
            for i, j in T.Parallel(block_M, N):
                A_pow[i, j] = A_local[i, j] * A_local[i, j]
            T.reduce_sum(A_pow, A_sum, dim=1)
            for i in T.Parallel(block_M):
                A_sum[i] = T.rsqrt(A_sum[i] / N + 1e-6)
            for i, j in T.Parallel(block_M, N):
                A_local[i, j] *= A_sum[i]
            T.copy(A_local, B[bx * block_M, 0])

    return main


def silu_gate(M, N, block_M, block_N, dtype="float16"):
    """Fused SwiGLU activation: out = silu(gate) * up. FFN elementwise epilogue."""

    @T.prim_func
    def main(
        Gate: T.Tensor((M, N), dtype),
        Up: T.Tensor((M, N), dtype),
        Out: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            g = T.alloc_fragment((block_M, block_N), dtype)
            u = T.alloc_fragment((block_M, block_N), dtype)
            T.copy(Gate[by * block_M, bx * block_N], g)
            T.copy(Up[by * block_M, bx * block_N], u)
            for i, j in T.Parallel(block_M, block_N):
                x = g[i, j].astype("float32")
                g[i, j] = (x * (1.0 / (1.0 + T.exp2(-x * 1.44269504)))).astype(dtype) * u[i, j]
            T.copy(g, Out[by * block_M, bx * block_N])

    return main


def softmax(M, N, block_M, dtype="float32"):
    """Row-wise softmax (logits / router). Online max-sum reduction."""

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, block_M), threads=128) as bx:
            row = T.alloc_fragment((block_M, N), dtype)
            row_max = T.alloc_fragment((block_M,), dtype)
            row_sum = T.alloc_fragment((block_M,), dtype)
            T.copy(A[bx * block_M, 0], row)
            T.reduce_max(row, row_max, dim=1, clear=True)
            for i, j in T.Parallel(block_M, N):
                row[i, j] = T.exp(row[i, j] - row_max[i])
            T.reduce_sum(row, row_sum, dim=1)
            for i, j in T.Parallel(block_M, N):
                row[i, j] = row[i, j] / row_sum[i]
            T.copy(row, B[bx * block_M, 0])

    return main


def build_zoo(scale=1, salt=0):
    """Return ``[(name, prim_func), ...]`` — a diverse, realistic kernel set.

    ``scale`` replicates the sweep to grow the count for larger boxes without
    changing the kernel mix. ``salt`` shifts the row/token count so a second call
    compiles *fresh* kernels (distinct cache keys) rather than hitting the
    process-global in-memory cache from a prior run. Both perturb the *M* (row)
    dimension only — never the tile dims, which carry alignment constraints.
    """
    zoo = []
    for rep in range(scale):
        # Perturb rows/tokens (M / seq_len), which is tile-alignment-agnostic, so
        # replicas and salted re-runs stay distinct cache keys without producing
        # illegal tile shapes.
        m = 256 + 128 * (rep + salt)
        for hidden, inter in MODEL_DIMS:
            for bm, bn, bk in GEMM_TILES:
                zoo.append((f"gemm_qkv_{hidden}_{bm}x{bn}x{bk}_{m}", gemm(m, hidden, hidden, bm, bn, bk)))
                zoo.append((f"gemm_up_{hidden}x{inter}_{bm}x{bn}x{bk}_{m}", gemm(m, inter, hidden, bm, bn, bk)))
        for h, kvh, hd in ATTN_HEADS:
            for bm in SEQ_TILES:
                zoo.append((f"gqa_{h}x{kvh}x{hd}_bm{bm}_{m}", gqa_attention(1, h, kvh, m, hd, bm, 64)))
        for hidden, inter in MODEL_DIMS:
            zoo.append((f"rmsnorm_{hidden}_{m}", rms_norm(m, hidden, 32)))
            zoo.append((f"silu_{inter}_{m}", silu_gate(m, inter, 32, 64)))
            zoo.append((f"softmax_{hidden}_{m}", softmax(m, hidden, 32)))
    return zoo


if __name__ == "__main__":
    zoo = build_zoo()
    from collections import Counter

    fams = Counter(name.split("_")[0] for name, _ in zoo)
    print(f"zoo size: {len(zoo)} kernels")
    for fam, n in sorted(fams.items()):
        print(f"  {fam:10s} {n}")
