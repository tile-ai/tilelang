# MXFP8 Block-Scaled GEMM on SM100 (Blackwell)
# 2CTA schedule with B-multicast (following DeepGEMM pattern).

import torch
import tilelang
import tilelang.language as T


@tilelang.jit
def mxfp8_blockscaled_gemm(
    A, B, SFA, SFB,
    block_M, block_N, block_K,
    in_dtype, out_dtype, accum_dtype,
    num_stages,
    sf_granularity_k=128,
):
    M, N, K = T.const("M, N, K")
    k_iters = T.ceildiv(K, block_K)
    sf_load_period = sf_granularity_k * 4 // block_K

    A: T.Tensor[[M, K], in_dtype]
    B: T.Tensor[[K, N], in_dtype]
    SFA: T.Tensor[[M, T.ceildiv(K, sf_granularity_k)], "uint8"]
    SFB: T.Tensor[[N, T.ceildiv(K, sf_granularity_k)], "uint8"]
    C = T.empty((M, N), out_dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128, cluster_dims=2) as (bx, by):
        half_N = block_N // 2
        cta_id = T.block_rank_in_cluster()
        T.assume(cta_id < 2)

        A_shared = T.alloc_shared((num_stages, block_M, block_K), in_dtype)
        B_shared = T.alloc_shared((num_stages, block_K, half_N), in_dtype)
        SFA_shared = T.alloc_shared((num_stages, block_M, 4), "uint8")
        SFB_shared = T.alloc_shared((num_stages, block_N, 4), "uint8")

        C_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
        SFA_tmem = T.alloc_tmem([block_M, block_M // 128 * 4], "uint8")
        SFB_tmem = T.alloc_tmem([block_M, block_N // 128 * 4], "uint8")

        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        C_shared = T.alloc_shared((block_M, block_N), out_dtype)

        loaded = T.alloc_cluster_barrier([32 * 2] * num_stages)
        with_sf_full = T.alloc_cluster_barrier([32 * 2] * num_stages)
        consumed = T.alloc_cluster_barrier([1] * num_stages)
        tmem_full = T.alloc_barrier([1])

        T.annotate_layout({
            SFA_shared: tilelang.layout.make_linear_layout(SFA_shared),
            SFB_shared: tilelang.layout.make_linear_layout(SFB_shared),
        })

        tx = T.get_thread_binding()
        warp_idx = tx // 32
        lane = tx % 32
        T.use_swizzle(8)

        if warp_idx == 0:
            for k in T.serial(k_iters):
                T.mbarrier_wait_parity(consumed[k % num_stages], ((k // num_stages) & 1) ^ 1)
                T.tma_copy(
                    A[by * block_M:(by + 1) * block_M, k * block_K:(k + 1) * block_K],
                    A_shared[k % num_stages, :, :],
                    barrier=loaded[k % num_stages],
                )
                T.tma_copy(
                    B[
                        k * block_K:(k + 1) * block_K,
                        (bx * block_N + cta_id * half_N):(bx * block_N + (cta_id + 1) * half_N),
                    ],
                    B_shared[k % num_stages, :, :],
                    barrier=loaded[k % num_stages],
                )
                T.mbarrier_arrive(loaded[k % num_stages], 0)

        elif warp_idx == 1 and cta_id == 0:
            for k in T.serial(k_iters):
                T.mbarrier_wait_parity(with_sf_full[k % num_stages], (k // num_stages) & 1)
                T.tcgen05_after_thread_sync()
                if k % sf_load_period == 0:
                    T.tcgen05_cp(SFA_shared[k % num_stages, :, :], SFA_tmem, use_2cta=True)
                    T.tcgen05_cp(SFB_shared[k % num_stages, :, :], SFB_tmem, use_2cta=True)
                    T.sync_warp()
                T.tcgen05_gemm_blockscaled(
                    A_shared[k % num_stages, :, :],
                    B_shared[k % num_stages, :, :],
                    C_tmem,
                    SFA_tmem,
                    SFB_tmem,
                    mbar=consumed[k % num_stages],
                    clear_accum=k == 0,
                    sf_a_id=k % sf_load_period,
                    sf_b_id=k % sf_load_period,
                    use_2cta=True,
                )
            T.tcgen05_mma_arrive(tmem_full, arrive_2cta=True)

        elif warp_idx == 2:
            for k in T.serial(k_iters):
                T.mbarrier_wait_parity(loaded[k % num_stages], (k // num_stages) & 1)
                if k % sf_load_period == 0:
                    sf_k_idx = k // sf_load_period
                    for i in T.serial(block_M // 32):
                        for j in T.serial(4):
                            SFA_shared[k % num_stages, i * 32 + lane, j] = SFA[
                                by * block_M + i * 32 + lane, sf_k_idx * 4 + j
                            ]
                    for i in T.serial(half_N // 32):
                        for j in T.serial(4):
                            SFB_shared[k % num_stages, cta_id * half_N + i * 32 + lane, j] = SFB[
                                bx * block_N + cta_id * half_N + i * 32 + lane, sf_k_idx * 4 + j
                            ]
                    T.sf_warp_transpose(SFA_shared[k % num_stages, :, :])
                    T.sf_warp_transpose(SFB_shared[k % num_stages, :, :])
                    T.fence_proxy_async()
                T.mbarrier_arrive(with_sf_full[k % num_stages], 0)

        T.mbarrier_wait_parity(tmem_full, 0)
        T.tcgen05_after_thread_sync()

        if cta_id == 0:
            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return C


def blockscaled_gemm_ref(a, b, sfa_unpacked, sfb_unpacked, sf_granularity_k=128, sf_granularity_n=128):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    sf_k_blocks = (K + sf_granularity_k - 1) // sf_granularity_k

    a_f32 = a.to(torch.float32)
    b_f32 = b.to(torch.float32)
    sfa_scales = torch.pow(2.0, sfa_unpacked.to(torch.float32) - 127.0)
    sfb_scales = torch.pow(2.0, sfb_unpacked.to(torch.float32) - 127.0)

    c = torch.zeros(M, N, device=a.device, dtype=torch.float32)
    n_blocks = (N + sf_granularity_n - 1) // sf_granularity_n
    for bi in range(sf_k_blocks):
        k_start = bi * sf_granularity_k
        k_end = min(k_start + sf_granularity_k, K)
        a_block = a_f32[:, k_start:k_end] * sfa_scales[:, bi:bi + 1]
        for ni in range(n_blocks):
            n_start = ni * sf_granularity_n
            n_end = min(n_start + sf_granularity_n, N)
            b_sub = b_f32[k_start:k_end, n_start:n_end] * sfb_scales[ni, bi]
            c[:, n_start:n_end] += a_block @ b_sub
    return c


def cosine_similarity(a, b):
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return (a_flat @ b_flat) / (a_flat.norm() * b_flat.norm())


def main():
    M, N, K = 512, 512, 512
    block_M, block_N, block_K = 128, 256, 128
    in_dtype, out_dtype, accum_dtype = T.float8_e4m3fn, T.bfloat16, T.float
    num_stages = 2
    sf_granularity_k = 128
    sf_granularity_n = 128

    a = torch.randn(M, K, device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)

    sf_k_blocks = (K + sf_granularity_k - 1) // sf_granularity_k
    sf_k_padded = ((sf_k_blocks + 3) // 4) * 4
    sf_n_blocks = (N + sf_granularity_n - 1) // sf_granularity_n

    sfa = torch.randint(117, 137, (M, sf_k_padded), device="cuda", dtype=torch.uint8)
    sfb_coarse = torch.randint(117, 137, (sf_n_blocks, sf_k_blocks), device="cuda", dtype=torch.uint8)
    sfb_unpacked = sfb_coarse.repeat_interleave(sf_granularity_n, dim=0)[:N].contiguous()

    c = mxfp8_blockscaled_gemm(
        a, b, sfa, sfb_unpacked,
        block_M, block_N, block_K,
        in_dtype, out_dtype, accum_dtype,
        num_stages,
        sf_granularity_k,
    )

    ref_c = blockscaled_gemm_ref(
        a, b, sfa[:, :sf_k_blocks], sfb_coarse, sf_granularity_k, sf_granularity_n
    ).to(torch.bfloat16)
    sim = cosine_similarity(c, ref_c)
    print(f"Output shape: {c.shape}, dtype: {c.dtype}")
    print(f"Cosine similarity: {sim.item():.6f}")
    assert sim > 0.99, f"Cosine similarity too low: {sim.item():.6f}"
    print("Correctness check passed.")


if __name__ == "__main__":
    main()
