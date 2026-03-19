# ruff: noqa
import tilelang
from tilelang import language as T
import torch
from utils import assert_tensors_similar


@tilelang.jit
def preprocess(
    O,
    dO,
    block_ND: int = 32,
    num_stages: int = 5,
):
    B, S, H, D = T.const("B S H D")
    O: T.Tensor[[B, S, H, D], T.bfloat16]
    dO: T.Tensor[[B, S, H, D], T.bfloat16]

    dtype = T.bfloat16
    accum_dtype = T.float32

    Delta = T.empty([B, S, H], accum_dtype)

    with T.Kernel(H, T.ceildiv(S, block_ND), B) as (bx, by, bz):
        o = T.alloc_fragment([block_ND, block_ND], accum_dtype)
        do = T.alloc_fragment([block_ND, block_ND], accum_dtype)
        delta = T.alloc_fragment([block_ND], accum_dtype)
        acc = T.alloc_fragment([block_ND, block_ND], accum_dtype)
        T.clear(acc)
        for k in T.Pipelined(T.ceildiv(D, block_ND), num_stages=num_stages):
            T.copy(O[bz, by * block_ND : (by + 1) * block_ND, bx, k * block_ND : (k + 1) * block_ND], o)
            T.copy(dO[bz, by * block_ND : (by + 1) * block_ND, bx, k * block_ND : (k + 1) * block_ND], do)
            for i, j in T.Parallel(block_ND, block_ND):
                acc[i, j] += o[i, j] * do[i, j]
        T.reduce_sum(acc, delta, 1)
        T.copy(delta, Delta[bz, by * block_ND : (by + 1) * block_ND, bx])

    return Delta


@tilelang.jit
def postprocess(
    dKV,
    D: int = 512,
    D_tail: int = 64,
    kv_group: int = 1,
    block_N: int = 64,
    threads: int = 128,
):
    B, S_kv, _, _ = T.const("B S_kv _ _")
    dKV: T.Tensor[[B, S_kv, kv_group, D + D_tail], T.float32]

    dtype = T.bfloat16
    accum_dtype = T.float32
    dkv_shape = [B, S_kv, kv_group, D + D_tail]

    dKV_out = T.empty(dkv_shape, dtype)

    with T.Kernel(T.ceildiv(S_kv, block_N), kv_group, B, threads=threads) as (bx, by, bz):
        T.copy(
            dKV[bz, bx * block_N : (bx + 1) * block_N, by, :],
            dKV_out[bz, bx * block_N : (bx + 1) * block_N, by, :],
        )

    return dKV_out


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
    },
)
def bwd(
    Q,
    KV,
    dO,
    Indices,
    Lse,
    Delta,
    dKV,
    D: int = 512,
    D_tail: int = 64,
    kv_group: int = 1,
    sm_scale: float = None,
    is_causal: bool = True,
    block_size: int = 32,
    num_stages: int = 0,
    threads: int = 256,
):
    B, S, H, _ = T.const("B S H _")
    _, S_kv, _, _ = T.const("_ S_kv _ _")
    _, _, _, topk = T.const("_ _ _ topk")
    Q: T.Tensor[[B, S, H, D + D_tail], T.bfloat16]
    KV: T.Tensor[[B, S_kv, kv_group, D + D_tail], T.bfloat16]
    dO: T.Tensor[[B, S, H, D], T.bfloat16]
    Indices: T.Tensor[[B, S, kv_group, topk], T.int32]
    Lse: T.Tensor[[B, S, H], T.float32]
    Delta: T.Tensor[[B, S, H], T.float32]
    dKV: T.Tensor[[B, S_kv, kv_group, D + D_tail], T.float32]

    indices_dtype = T.int32
    dtype = T.bfloat16
    accum_dtype = T.float32

    assert is_causal == True, "non-casual is not supported now"
    assert topk % block_size == 0, "otherwise will load some index=0 thus causing wrong kv to be loaded"

    if sm_scale is None:
        sm_scale = (D + D_tail) ** (-0.5)
    sm_scale_mul_reciprocal_log2 = sm_scale * 1.44269504  # log2(e)

    H_kv = H // kv_group

    H_val = H_kv
    padded_H = max(tilelang.math.next_power_of_2(H_val), 16)
    block_H = min(64, padded_H)
    assert padded_H % block_H == 0
    NH = padded_H // block_H
    BS = block_size
    NS = tilelang.cdiv(topk, block_size)

    split_store = 2

    dQ = T.empty([B, S, H, D + D_tail], dtype)

    with T.Kernel(S, B, kv_group * NH, threads=threads) as (s_i, by, bz):
        Q_shared = T.alloc_shared([block_H, D], dtype)
        Q_tail_shared = T.alloc_shared([block_H, D_tail], dtype)
        KV_shared = T.alloc_shared([BS, D], dtype)
        KV_tail_shared = T.alloc_shared([BS, D_tail], dtype)
        dO_shared = T.alloc_shared([block_H, D], dtype)
        mask = T.alloc_fragment([BS], "bool")

        P_shared_cast = T.alloc_shared([block_H, BS], dtype)
        dP_shared_cast = T.alloc_shared([block_H, BS], dtype)
        dQ_shared = T.alloc_shared([block_H, D], dtype)
        dQ_tail_shared = T.alloc_shared([block_H, D_tail], dtype)

        acc_p = T.alloc_fragment([block_H, BS], accum_dtype)
        acc_dp = T.alloc_fragment([block_H, BS], accum_dtype)
        acc_dq = T.alloc_fragment([block_H, D], accum_dtype)
        acc_dq_tail = T.alloc_fragment([block_H, D_tail], accum_dtype)
        acc_dkv = T.alloc_fragment([BS, D], accum_dtype)
        acc_dkv_tail = T.alloc_fragment([BS, D_tail], accum_dtype)
        acc_dkv_shared = T.alloc_shared([BS // split_store, D], accum_dtype)
        acc_dkv_tail_shared = T.alloc_shared([BS // split_store, D_tail], accum_dtype)

        max_kv_i = s_i

        T.copy(Q[by, s_i, bz * block_H : (bz + 1) * block_H, :D], Q_shared)
        T.copy(Q[by, s_i, bz * block_H : (bz + 1) * block_H, D:], Q_tail_shared)
        T.copy(dO[by, s_i, bz * block_H : (bz + 1) * block_H, :D], dO_shared)

        T.clear(acc_dq)
        T.clear(acc_dq_tail)

        # Process each block of indices
        for i_i in T.Pipelined(NS, num_stages=num_stages):
            # Check which indices are valid
            for bi_i in T.Parallel(BS):
                mask[bi_i] = Indices[by, s_i, bz // NH, i_i * BS + bi_i] <= max_kv_i

            # Compute attention scores
            for h_i, bi_i in T.Parallel(block_H, BS):
                acc_p[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_p.dtype))

            # Load KV, V for this block of indices
            for bi_i, d_i in T.Parallel(BS, D):
                KV_shared[bi_i, d_i] = KV[by, Indices[by, s_i, bz // NH, i_i * BS + bi_i], bz // NH, d_i]

            T.gemm(Q_shared, KV_shared, acc_p, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

            for bi_i, d_i in T.Parallel(BS, D_tail):
                KV_tail_shared[bi_i, d_i] = KV[by, Indices[by, s_i, bz // NH, i_i * BS + bi_i], bz // NH, D + d_i]
            T.gemm(Q_tail_shared, KV_tail_shared, acc_p, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

            for h_i, bi_i in T.Parallel(block_H, BS):
                acc_p[h_i, bi_i] = T.exp2(acc_p[h_i, bi_i] * sm_scale_mul_reciprocal_log2 - Lse[by, s_i, bz * block_H + h_i])

            T.copy(acc_p, P_shared_cast)

            T.gemm(dO_shared, KV_shared, acc_dp, transpose_B=True, policy=T.GemmWarpPolicy.FullCol, clear_accum=True)

            for h_i, bi_i in T.Parallel(block_H, BS):
                acc_dp[h_i, bi_i] = acc_p[h_i, bi_i] * (acc_dp[h_i, bi_i] - Delta[by, s_i, bz * block_H + h_i]) * sm_scale

            T.copy(acc_dp, dP_shared_cast)
            T.gemm(dP_shared_cast, KV_shared, acc_dq, policy=T.GemmWarpPolicy.FullCol)
            T.gemm(dP_shared_cast, KV_tail_shared, acc_dq_tail, policy=T.GemmWarpPolicy.FullCol)

            T.gemm(dP_shared_cast, Q_shared, acc_dkv, transpose_A=True, policy=T.GemmWarpPolicy.FullCol, clear_accum=True)
            T.gemm(P_shared_cast, dO_shared, acc_dkv, transpose_A=True, policy=T.GemmWarpPolicy.FullCol)

            T.clear(acc_dkv_tail)
            T.gemm(dP_shared_cast, Q_tail_shared, acc_dkv_tail, transpose_A=True, policy=T.GemmWarpPolicy.FullCol)

            for s in range(split_store):
                for bi_i, d_i in T.Parallel(BS, D):
                    if bi_i < BS // split_store:
                        acc_dkv_shared[bi_i, d_i] = acc_dkv[bi_i + s * (BS // split_store), d_i]

                for bi_i, d_i in T.Parallel(BS, D_tail):
                    if bi_i < BS // split_store:
                        acc_dkv_tail_shared[bi_i, d_i] = acc_dkv_tail[bi_i + s * (BS // split_store), d_i]

                for bi_i, d_i in T.Parallel(BS // split_store, D // 4):
                    T.atomic_addx4(
                        dKV[by, Indices[by, s_i, bz // NH, i_i * BS + bi_i + s * (BS // split_store)], bz // NH, d_i * 4],
                        acc_dkv_shared[bi_i, d_i * 4],
                    )

                # Atomically update dKV, dKV_tail tensors
                for bi_i, d_i in T.Parallel(BS // split_store, D_tail // 4):
                    T.atomic_addx4(
                        dKV[by, Indices[by, s_i, bz // NH, i_i * BS + bi_i + s * (BS // split_store)], bz // NH, D + d_i * 4],
                        acc_dkv_tail_shared[bi_i, d_i * 4],
                    )

        # Store the accumulated dQ
        T.copy(acc_dq, dQ_shared)
        T.copy(acc_dq_tail, dQ_tail_shared)

        T.copy(dQ_shared, dQ[by, s_i, bz * block_H : (bz + 1) * block_H, :D])
        T.copy(dQ_tail_shared, dQ[by, s_i, bz * block_H : (bz + 1) * block_H, D:])

    return dQ


def sparse_mla_bwd(q, kv, o, do, indices, lse, sm_scale=None, is_casual=True, return_kernel=False, delta=None):
    assert q.is_contiguous()
    assert kv.is_contiguous()
    assert indices.is_contiguous()
    assert lse.is_contiguous()
    B, S, H, dim_plus_tail_dim = q.shape
    _, S_kv, kv_group, _ = kv.shape
    assert kv.shape[-1] == dim_plus_tail_dim
    assert kv.shape[0] == B
    # dim should be assigned
    D = 512

    D_tail = dim_plus_tail_dim - D
    topk = indices.shape[-1]
    assert indices.shape == (B, S, kv_group, topk)
    assert lse.shape == (B, S, H)

    # Run kernels
    if delta is None:
        delta = preprocess(o, do)
    dkv = torch.zeros_like(kv, dtype=torch.float32)
    dq = bwd(q, kv, do, indices, lse, delta, dkv, D, D_tail, kv_group, sm_scale, is_casual)
    dkv_out = postprocess(dkv, D, D_tail, kv_group)

    return dq, dkv_out


def ref_sparse_mla_bwd_interface(q, kv, o, do, indices, lse, sm_scale=None, is_casual=True):
    from sparse_mla_fwd import ref_sparse_mla_fwd_interface

    q = q.detach().clone()
    kv = kv.detach().clone()
    q.requires_grad = True
    kv.requires_grad = True
    o = ref_sparse_mla_fwd_interface(q, kv, indices, sm_scale, is_casual)
    o.backward(do)
    return q.grad, kv.grad


def test_sparse_mla_bwd(B=1, S=4096, SKV=8192, H=64, HKV=1, DQKV=576, DV=512, topk=2048, dtype=torch.bfloat16, check_correctness=True):
    # Prepare data
    q = torch.randn((B, S, H, DQKV), dtype=dtype, device="cuda").requires_grad_(True)
    kv = torch.randn((B, SKV, HKV, DQKV), dtype=dtype, device="cuda").requires_grad_(True)
    do = torch.randn((B, S, H, DV), dtype=dtype, device="cuda")

    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device="cuda")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]
                indices[b, t, h, : len(i_i)] = i_i

    # Forward
    from sparse_mla_fwd import sparse_mla_fwd_interface

    tl_out, tl_lse = sparse_mla_fwd_interface(q, kv, indices)

    tl_dq, tl_dkv = sparse_mla_bwd(q, kv, tl_out, do, indices, tl_lse)
    ref_dq, ref_dkv = ref_sparse_mla_bwd_interface(q, kv, None, do, indices, None)

    if check_correctness:
        assert_tensors_similar(tl_dq, ref_dq, eps=1e-4, name="dq")
        assert_tensors_similar(tl_dkv, ref_dkv, eps=1e-4, name="dkv")
        print("assert_tensors_similar passed")

    per_token_flop = 2 * sum(
        [
            H * DV * topk,
            H * DQKV * topk,
            H * DQKV * topk,
            H * DQKV * topk,
            H * DV * topk,
        ]
    )
    from tilelang.profiler import do_bench

    def fn():
        return sparse_mla_bwd(q, kv, tl_out, do, indices, tl_lse)

    ms = do_bench(fn, rep=100, warmup=250)
    print(f"Average time: {ms:.3f} ms")
    print(f"bwd io bandwidth = ", (B * S * max(DQKV * 2, DQKV + DV) * topk * 2) / (ms * 1e-3) / 1e12)
    print(f"bwd tflops = ", per_token_flop * S / (ms * 1e-3) / 1e12)


def run_regression_perf(B=1, S=4096, SKV=8192, H=64, HKV=1, DQKV=576, DV=512, topk=2048, dtype=torch.bfloat16):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    q = torch.randn((B, S, H, DQKV), dtype=dtype, device="cuda").requires_grad_(True)
    kv = torch.randn((B, SKV, HKV, DQKV), dtype=dtype, device="cuda").requires_grad_(True)
    do = torch.randn((B, S, H, DV), dtype=dtype, device="cuda")

    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device="cuda")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]
                indices[b, t, h, : len(i_i)] = i_i

    from sparse_mla_fwd import sparse_mla_fwd_interface

    tl_out, tl_lse = sparse_mla_fwd_interface(q, kv, indices)
    B, S, H, dim_plus_tail_dim = q.shape
    _, S_kv, kv_group, _ = kv.shape
    D = 512
    D_tail = dim_plus_tail_dim - D
    topk = indices.shape[-1]
    delta = preprocess(tl_out, do)
    dkv = torch.zeros_like(kv, dtype=torch.float32)

    from tilelang.profiler import do_bench

    def run_kernel_only():
        return bwd(q, kv, do, indices, tl_lse, delta, dkv, D, D_tail, kv_group, None, True)

    return do_bench(run_kernel_only, backend="cupti")


if __name__ == "__main__":
    test_sparse_mla_bwd(B=1, S=4096, SKV=8192, H=64, HKV=1, DQKV=576, DV=512, topk=2048, dtype=torch.bfloat16, check_correctness=True)
