# MXFP8 Block-Scaled GEMM on SM100 (Blackwell) — quantsize (M, N, K) = (1, 128, 128)
#
# Compared to the (1, 1, 128) variant:
#   - SFA: per-row per-K-block → [M, K/128]               (same)
#   - SFB: per-128-N per-K-block → [N/128, K/128]          (coarser along N)
#
# The kernel is identical — hardware always expects per-row scales in TMEM.
# We broadcast each SFB scale to 128 N-rows in the global tensor.

import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench

tilelang.disable_cache()


@tilelang.jit
def mxfp8_blockscaled_gemm(
    A, B, SFA, SFB,
    block_M, block_N, block_K,
    in_dtype, out_dtype, accum_dtype,
    num_stages,
    sf_granularity_k=128,
):
    """Block-scaled MXFP8 GEMM.

    A:   [M, K] in FP8
    B:   [K, N] in FP8
    SFA: [M, ceil(K / sf_granularity_k)] in uint8 (E8M0 scale factors for A)
    SFB: [N, ceil(K / sf_granularity_k)] in uint8 (E8M0 scale factors for B)
         (for quantsize 128 along N, every 128 consecutive N-rows share the same value)
    """
    M, N, K = T.const("M, N, K")

    k_iters = T.ceildiv(K, block_K)
    sf_load_period = sf_granularity_k * 4 // block_K

    A: T.Tensor[[M, K], in_dtype]
    B: T.Tensor[[K, N], in_dtype]
    SFA: T.Tensor[[M, T.ceildiv(K, sf_granularity_k)], "uint8"]
    SFB: T.Tensor[[N, T.ceildiv(K, sf_granularity_k)], "uint8"]
    C = T.empty((M, N), out_dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((num_stages, block_M, block_K), in_dtype)
        B_shared = T.alloc_shared((num_stages, block_K, block_N), in_dtype)

        # Scale factor shared memory — uint8 E8M0, 4 K-blocks per load
        SFA_shared = T.alloc_shared((num_stages, block_M, 4), "uint8")
        SFB_shared = T.alloc_shared((num_stages, block_N, 4), "uint8")

        C_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)

        # Scale factors in tensor memory (TMEM has 128 rows)
        # T.copy(SMEM→TMEM) auto-handles sf_warp_transpose + tcgen05.cp
        SFA_tmem = T.alloc_tmem([block_M, block_M // 128 * 4], "uint8")
        SFB_tmem = T.alloc_tmem([block_M, block_N // 128 * 4], "uint8")

        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        C_shared = T.alloc_shared((block_M, block_N), out_dtype)

        loaded = T.alloc_barrier([32] * num_stages)
        consumed = T.alloc_barrier([1] * num_stages)
        tmem_full = T.alloc_barrier([1])

        tx = T.get_thread_binding()
        T.use_swizzle(8)

        if tx < 32:
            for k in T.serial(k_iters):
                T.mbarrier_wait_parity(consumed[k % num_stages], ((k // num_stages) & 1) ^ 1)
                T.tma_copy(
                    A[by * block_M:(by + 1) * block_M, k * block_K:(k + 1) * block_K],
                    A_shared[k % num_stages, :, :],
                    barrier=loaded[k % num_stages],
                )
                T.tma_copy(
                    B[k * block_K:(k + 1) * block_K, bx * block_N:(bx + 1) * block_N],
                    B_shared[k % num_stages, :, :],
                    barrier=loaded[k % num_stages],
                )
                if k % sf_load_period == 0:
                    sf_k_idx = k // sf_load_period
                    T.copy(
                        SFA[by * block_M:(by + 1) * block_M, sf_k_idx * 4:(sf_k_idx + 1) * 4],
                        SFA_shared[k % num_stages, :, :],
                    )
                    T.copy(
                        SFB[bx * block_N:(bx + 1) * block_N, sf_k_idx * 4:(sf_k_idx + 1) * 4],
                        SFB_shared[k % num_stages, :, :],
                    )
                T.mbarrier_arrive(loaded[k % num_stages])

        elif tx < 64:
            for k in T.serial(k_iters):
                T.mbarrier_wait_parity(loaded[k % num_stages], (k // num_stages) & 1)

                # Copy SF from SMEM to TMEM (auto transpose + UTCCP)
                if k % sf_load_period == 0:
                    T.copy(SFA_shared[k % num_stages, :, :], SFA_tmem)
                    T.copy(SFB_shared[k % num_stages, :, :], SFB_tmem)

                T.sync_warp()

                T.blockscaled_gemm(
                    A_shared[k % num_stages, :, :],
                    B_shared[k % num_stages, :, :],
                    C_tmem,
                    SFA_tmem,
                    SFB_tmem,
                    mbar=consumed[k % num_stages],
                    clear_accum=k == 0,
                    sf_a_id=k % sf_load_period,
                    sf_b_id=k % sf_load_period,
                )

            T.tcgen05_mma_arrive(tmem_full)

        T.mbarrier_wait_parity(tmem_full, 0)
        T.sync_threads()

        T.copy(C_tmem, C_local)
        T.copy(C_local, C_shared)
        T.copy(C_shared, C[by * block_M, bx * block_N])

    return C


def blockscaled_gemm_ref(a, b, sfa_unpacked, sfb_unpacked, sf_granularity_k=128, sf_granularity_n=128):
    """Torch reference for block-scaled MXFP8 GEMM with quantsize (1, sf_granularity_n, sf_granularity_k).

    Args:
        a: [M, K] FP8 tensor
        b: [K, N] FP8 tensor
        sfa_unpacked: [M, sf_k_blocks] uint8 E8M0 scale factors for A
        sfb_unpacked: [N/sf_granularity_n, sf_k_blocks] uint8 E8M0 scale factors for B
        sf_granularity_k: K elements per scale factor block (default 128)
        sf_granularity_n: N elements per scale factor block (default 128)

    Returns:
        [M, N] float32 result
    """
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    sf_k_blocks = (K + sf_granularity_k - 1) // sf_granularity_k

    a_f32 = a.to(torch.float32)
    b_f32 = b.to(torch.float32)

    # E8M0 to float: 2^(exp - 127)
    sfa_scales = torch.pow(2.0, sfa_unpacked.to(torch.float32) - 127.0)  # [M, sf_k_blocks]
    sfb_scales = torch.pow(2.0, sfb_unpacked.to(torch.float32) - 127.0)  # [N/sf_granularity_n, sf_k_blocks]

    c = torch.zeros(M, N, device=a.device, dtype=torch.float32)
    n_blocks = (N + sf_granularity_n - 1) // sf_granularity_n
    for bi in range(sf_k_blocks):
        k_start = bi * sf_granularity_k
        k_end = min(k_start + sf_granularity_k, K)
        # Scale A: [M, block_k] * [M, 1]
        a_block = a_f32[:, k_start:k_end] * sfa_scales[:, bi:bi + 1]
        for ni in range(n_blocks):
            n_start = ni * sf_granularity_n
            n_end = min(n_start + sf_granularity_n, N)
            # Scale B sub-block: [block_k, block_n] * scalar
            b_sub = b_f32[k_start:k_end, n_start:n_end] * sfb_scales[ni, bi]
            c[:, n_start:n_end] += a_block @ b_sub
    return c


def cosine_similarity(a, b):
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return (a_flat @ b_flat) / (a_flat.norm() * b_flat.norm())


def main():
    M, N, K = 8192, 8192, 8192
    block_M, block_N, block_K = 128, 256, 128
    in_dtype, out_dtype, accum_dtype = T.float8_e4m3fn, T.bfloat16, T.float
    num_stages = 4
    sf_granularity_k = 128
    sf_granularity_n = 128  # quantsize along N

    a = torch.randn(M, K, device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)

    sf_k_blocks = (K + sf_granularity_k - 1) // sf_granularity_k
    sf_k_padded = ((sf_k_blocks + 3) // 4) * 4
    sf_n_blocks = (N + sf_granularity_n - 1) // sf_granularity_n

    # SFA: per-row per-K-block uint8 E8M0
    sfa = torch.randint(127 - 10, 127 + 10, (M, sf_k_padded), device="cuda", dtype=torch.uint8)

    # SFB: per-128-N per-K-block — broadcast to per-row
    sfb_coarse = torch.randint(127 - 10, 127 + 10, (sf_n_blocks, sf_k_blocks), device="cuda", dtype=torch.uint8)
    sfb_unpacked = sfb_coarse.repeat_interleave(sf_granularity_n, dim=0)[:N]
    # Pad to multiple of 4
    if sf_k_blocks % 4 != 0:
        pad = 4 - sf_k_blocks % 4
        sfb_unpacked = torch.nn.functional.pad(sfb_unpacked, (0, pad), value=127)
    sfb = sfb_unpacked.contiguous()

    c = mxfp8_blockscaled_gemm(
        a, b, sfa, sfb,
        block_M, block_N, block_K,
        in_dtype, out_dtype, accum_dtype,
        num_stages,
        sf_granularity_k,
    )
    print(mxfp8_blockscaled_gemm.get_kernel_source(
        a, b, sfa, sfb,
        block_M, block_N, block_K,
        in_dtype, out_dtype, accum_dtype,
        num_stages,
        sf_granularity_k,
    ))

    ref_c = blockscaled_gemm_ref(a, b, sfa[:, :sf_k_blocks], sfb_coarse, sf_granularity_k, sf_granularity_n).to(torch.bfloat16)
    sim = cosine_similarity(c, ref_c)
    print(f"Output shape: {c.shape}, dtype: {c.dtype}")
    print(f"Cosine similarity: {sim.item():.6f}")
    assert sim > 0.99, f"Cosine similarity too low: {sim.item():.6f}"

    tl_latency = do_bench(
        lambda: mxfp8_blockscaled_gemm(
            a, b, sfa, sfb,
            block_M, block_N, block_K,
            in_dtype, out_dtype, accum_dtype,
            num_stages,
            sf_granularity_k,
        ),
        backend="cupti",
    )
    print(f"Tilelang MXFP8 latency: {tl_latency} ms")
    print(f"TFLOPS: {2 * M * N * K / (tl_latency / 1e3) / 1e12:.2f}")


if __name__ == "__main__":
    main()
