import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench
import argparse


@tilelang.jit(
    out_idx=[3, 4],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flashattn_fwd(batch, heads, seq_len, dim, is_causal, block_M, block_N):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    shape = [batch, seq_len, heads, dim]
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def flash_fwd(
        Q: T.Tensor(shape, dtype),  # type: ignore
        K: T.Tensor(shape, dtype),  # type: ignore
        V: T.Tensor(shape, dtype),  # type: ignore
        Output: T.Tensor(shape, dtype),  # type: ignore
        lse: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=128) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv((bx + 1) * block_M, block_N) if is_causal else T.ceildiv(seq_len, block_N)
            for k in T.Pipelined(loop_range, num_stages=1):
                T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared)
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(bx * block_M + i >= k * block_N + j, 0, -T.infinity(acc_s.dtype))
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(k * block_N + j >= seq_len, -T.infinity(acc_s.dtype), 0)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared)
                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.copy(acc_s, acc_s_cast)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])
            for i in T.Parallel(block_M):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
            T.copy(logsum, lse[bz, by, bx * block_M : (bx + 1) * block_M])

    return flash_fwd


@tilelang.jit(
    out_idx=[2],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flashattn_bwd_preprocess(batch, heads, seq_len, dim):
    dtype = T.float16
    accum_dtype = T.float32
    shape = [batch, seq_len, heads, dim]
    blk = 32

    @T.prim_func
    def flash_bwd_prep(
        O: T.Tensor(shape, dtype),  # type: ignore
        dO: T.Tensor(shape, dtype),  # type: ignore
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, blk), batch) as (bx, by, bz):
            o = T.alloc_fragment([blk, blk], dtype)
            do = T.alloc_fragment([blk, blk], dtype)
            acc = T.alloc_fragment([blk, blk], accum_dtype)
            delta = T.alloc_fragment([blk], accum_dtype)
            T.clear(acc)
            for k in range(T.ceildiv(dim, blk)):
                T.copy(O[bz, by * blk : (by + 1) * blk, bx, k * blk : (k + 1) * blk], o)
                T.copy(dO[bz, by * blk : (by + 1) * blk, bx, k * blk : (k + 1) * blk], do)
                for i, j in T.Parallel(blk, blk):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, Delta[bz, bx, by * blk : (by + 1) * blk])

    return flash_bwd_prep


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
)
def flashattn_bwd(batch, heads, seq_len, dim, is_causal, block_M, block_N):
    sm_scale = (1.0 / dim) ** 0.5
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    shape = [batch, seq_len, heads, dim]
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def flash_bwd(
        Q: T.Tensor(shape, dtype),  # type: ignore
        K: T.Tensor(shape, dtype),  # type: ignore
        V: T.Tensor(shape, dtype),  # type: ignore
        dO: T.Tensor(shape, dtype),  # type: ignore
        lse: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
        dQ: T.Tensor(shape, accum_dtype),  # type: ignore
        dK: T.Tensor(shape, dtype),  # type: ignore
        dV: T.Tensor(shape, dtype),  # type: ignore
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, block_M), batch, threads=256) as (bx, by, bz):
            K_shared = T.alloc_shared([block_M, dim], dtype)
            dsT_shared = T.alloc_shared([block_M, block_N], dtype)
            # should not store K to local if dim is large
            # K_local = T.alloc_fragment([block_M, dim], dtype)
            # K_local_T = T.alloc_fragment([block_M, dim], dtype)
            # V_local = T.alloc_fragment([block_M, dim], dtype)
            q = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_M, dim], dtype)
            qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
            dsT = T.alloc_fragment([block_M, block_N], accum_dtype)
            qkT_cast = T.alloc_fragment([block_M, block_N], dtype)
            dsT_cast = T.alloc_fragment([block_M, block_N], dtype)
            lse_shared = T.alloc_shared([block_N], accum_dtype)
            delta = T.alloc_shared([block_N], accum_dtype)
            do = T.alloc_shared([block_N, dim], dtype)
            dv = T.alloc_fragment([block_M, dim], accum_dtype)
            dk = T.alloc_fragment([block_M, dim], accum_dtype)
            dq = T.alloc_fragment([block_N, dim], accum_dtype)
            dv_shared = T.alloc_shared([block_M, dim], dtype)
            dk_shared = T.alloc_shared([block_M, dim], dtype)
            dq_shared = T.alloc_shared([block_N, dim], accum_dtype)

            T.copy(K[bz, by * block_M : (by + 1) * block_M, bx, :], K_shared)
            T.copy(V[bz, by * block_M : (by + 1) * block_M, bx, :], V_shared)
            T.clear(dv)
            T.clear(dk)
            loop_st = T.floordiv(by * block_M, block_N) if is_causal else 0
            loop_ed = T.ceildiv(seq_len, block_N)
            for k in T.Pipelined(loop_st, loop_ed, num_stages=2):
                T.copy(Q[bz, k * block_N : (k + 1) * block_N, bx, :], q)
                T.clear(qkT)
                T.gemm(K_shared, q, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, wg_wait=-1)
                T.copy(dO[bz, k * block_N : (k + 1) * block_N, bx, :], do)
                T.clear(dsT)
                T.gemm(V_shared, do, dsT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, wg_wait=-1)
                T.wait_wgmma(1)

                T.copy(lse[bz, bx, k * block_N : (k + 1) * block_N], lse_shared)
                for i, j in T.Parallel(block_M, block_N):
                    qkT[i, j] = T.exp2(qkT[i, j] * scale - lse_shared[j])
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        qkT[i, j] = T.if_then_else(by * block_M + i <= k * block_N + j, qkT[i, j], 0)
                # We don't need to handle OOB positions for non-causal cases,
                # since OOB values won't affect other positions here.
                T.wait_wgmma(0)
                T.copy(qkT, qkT_cast)
                T.gemm(qkT_cast, do, dv, policy=T.GemmWarpPolicy.FullRow, wg_wait=-1)

                T.copy(Delta[bz, bx, k * block_N : (k + 1) * block_N], delta)

                for i, j in T.Parallel(block_M, block_N):
                    dsT_cast[i, j] = qkT[i, j] * (dsT[i, j] - delta[j]) * sm_scale
                T.gemm(dsT_cast, q, dk, policy=T.GemmWarpPolicy.FullRow, wg_wait=1)

                T.copy(dsT_cast, dsT_shared)
                T.clear(dq)
                T.gemm(dsT_shared, K_shared, dq, transpose_A=True, wg_wait=1)
                T.wait_wgmma(0)
                T.copy(dq, dq_shared)
                T.atomic_add(dQ[bz, k * block_N : (k + 1) * block_N, bx, :], dq_shared)
            T.copy(dv, dv_shared)
            T.copy(dk, dk_shared)
            T.copy(dv_shared, dV[bz, by * block_M : (by + 1) * block_M, bx, :])
            T.copy(dk_shared, dK[bz, by * block_M : (by + 1) * block_M, bx, :])

    return flash_bwd


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal):
        BATCH, N_CTX, H, D_HEAD = q.shape
        block_M = 64
        block_N = 64 if D_HEAD <= 128 else 32
        mod = flashattn_fwd(BATCH, H, N_CTX, D_HEAD, causal, block_M, block_N)
        o, lse = mod(q, k, v)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        BATCH, N_CTX, H, D_HEAD = q.shape

        def maybe_contiguous(x):
            if x.stride(-1) != 1:
                return x.contiguous()
            return x

        do, q, k, v, o = [maybe_contiguous(x) for x in (do, q, k, v, o)]
        block_M = 128
        block_N = 128 if D_HEAD <= 64 else 32
        mod_prep = flashattn_bwd_preprocess(BATCH, H, N_CTX, D_HEAD)
        delta = mod_prep(o, do)
        mod = flashattn_bwd(BATCH, H, N_CTX, D_HEAD, ctx.causal, block_M, block_N)
        shape = [BATCH, N_CTX, H, D_HEAD]
        dq = torch.zeros(shape, dtype=torch.float32, device=q.device)
        dk = torch.empty(shape, dtype=torch.float16, device=q.device)
        dv = torch.empty(shape, dtype=torch.float16, device=q.device)
        mod(q, k, v, do, lse, delta, dq, dk, dv)
        dq = dq.to(torch.float16)
        return dq, dk, dv, None


attention = _attention.apply


def ref_program(Q, K, V, is_causal):
    dim = Q.size(-1)
    scores = torch.einsum("bqhd,bkhd->bhqk", Q, K)
    scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
    if is_causal:
        seq_len = Q.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum("bhqk,bkhd->bqhd", attention_weights, V)
    return output


def main(
    BATCH: int = 8,
    H: int = 32,
    N_CTX: int = 1024,
    D_HEAD: int = 64,
    causal: bool = False,
):
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 5 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    Q = torch.empty(BATCH, N_CTX, H, D_HEAD, dtype=torch.half, device="cuda").normal_().requires_grad_()
    K = torch.empty_like(Q).normal_().requires_grad_()
    V = torch.empty_like(Q).normal_().requires_grad_()
    dO = torch.randn_like(Q)
    O = attention(Q, K, V, causal)
    O.backward(dO, retain_graph=True)
    dQ, Q.grad = Q.grad.clone(), None
    dK, K.grad = K.grad.clone(), None
    dV, V.grad = V.grad.clone(), None

    O_ref = ref_program(Q, K, V, causal)
    O_ref.backward(dO, retain_graph=True)
    dQ_ref, Q.grad = Q.grad.clone(), None
    dK_ref, K.grad = K.grad.clone(), None
    dV_ref, V.grad = V.grad.clone(), None

    assert torch.allclose(O, O_ref, rtol=1e-2, atol=1e-2)
    assert torch.allclose(dV, dV_ref, rtol=1e-2, atol=1e-2)
    assert torch.allclose(dK, dK_ref, rtol=1e-2, atol=1e-2)
    assert torch.allclose(dQ, dQ_ref, rtol=1e-2, atol=1e-2)
    print("All checks passed.✅")

    def run():
        O_ref.backward(dO, retain_graph=True)

    def run1():
        O.backward(dO, retain_graph=True)

    latency = do_bench(run, warmup=500)
    print("torch: {:.2f} ms".format(latency))
    print("torch: {:.2f} TFlops".format(total_flops / latency * 1e-9))
    latency = do_bench(run1, warmup=500)
    print("tilelang: {:.2f} ms".format(latency))
    print("tilelang: {:.2f} TFlops".format(total_flops / latency * 1e-9))


def run_regression_perf():
    BATCH = 1
    H = 32
    N_CTX = 256
    D_HEAD = 64
    causal = False
    device = "cuda"
    torch.manual_seed(0)
    block_M = 128
    block_N = 128 if D_HEAD <= 64 else 32
    Q = torch.randn(BATCH, N_CTX, H, D_HEAD, device=device, dtype=torch.half)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)
    O = torch.randn_like(Q)
    dO = torch.randn_like(Q)
    lse = torch.zeros(BATCH, H, N_CTX, device=device, dtype=torch.float32)
    with torch.no_grad():
        mod_prep = flashattn_bwd_preprocess(BATCH, H, N_CTX, D_HEAD)
        kernel = flashattn_bwd(BATCH, H, N_CTX, D_HEAD, causal, block_M, block_N)
    dQ = torch.zeros(BATCH, N_CTX, H, D_HEAD, device=device, dtype=torch.float32)
    dK = torch.zeros_like(Q, dtype=torch.float16)
    dV = torch.zeros_like(Q, dtype=torch.float16)
    Delta = mod_prep(O, dO)

    from tilelang.profiler import do_bench

    def run_kernel_only():
        kernel(Q, K, V, dO, lse, Delta, dQ, dK, dV)

    return do_bench(run_kernel_only, warmup=10, rep=100, backend="cupti")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--h", type=int, default=32, help="Number of heads")
    parser.add_argument("--n_ctx", type=int, default=1024, help="Context size")
    parser.add_argument("--d_head", type=int, default=64, help="Head dimension")
    parser.add_argument("--causal", type=bool, default=False, help="Causal flag")
    args = parser.parse_args()
    main(args.batch, args.h, args.n_ctx, args.d_head, args.causal)
