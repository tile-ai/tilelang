import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench

try:
    from examples.gemm_sm100.gemm_mxfp8_blockscaled_1d1d import mxfp8_blockscaled_gemm as mxfp8_blockscaled_gemm_1cta
except ModuleNotFoundError:
    from gemm_mxfp8_blockscaled_1d1d import mxfp8_blockscaled_gemm as mxfp8_blockscaled_gemm_1cta


@tilelang.jit(execution_backend='cython')
def mxfp8_blockscaled_gemm_2cta(
    A,
    B,
    SFA,
    SFB,
    block_M,
    block_N,
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    sf_granularity_k=128,
):
    M, N, K = T.const("M, N, K")

    assert block_M == 128
    assert block_N == 256
    assert block_K == 128
    assert sf_granularity_k == 128

    half_N = block_N // 2
    k_iters = T.ceildiv(K, block_K)
    sf_k_blocks = T.ceildiv(K, sf_granularity_k)
    sf_k_padded = T.ceildiv(sf_k_blocks, 4) * 4
    sf_load_period = sf_granularity_k * 4 // block_K
    assert sf_load_period == 4

    A: T.Tensor[[M, K], in_dtype]
    B: T.Tensor[[K, N], in_dtype]
    SFA: T.Tensor[[M, sf_k_padded], "uint8"]
    SFB: T.Tensor[[N, sf_k_padded], "uint8"]
    C = T.empty((M, N), out_dtype)

    with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=128, cluster_dims=2) as (bx, by):
        cta_id = T.block_rank_in_cluster()
        T.assume(cta_id < 2)

        A_shared = T.alloc_shared((num_stages, block_M, block_K), in_dtype)
        B_shared = T.alloc_shared((num_stages, block_K, half_N), in_dtype)
        SFA_shared = T.alloc_shared((num_stages, block_M, 4), "uint8")
        SFB_shared = T.alloc_shared((num_stages, half_N, 4), "uint8")

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
                stage = k % num_stages
                parity = (k // num_stages) & 1
                T.mbarrier_wait_parity(consumed[stage], parity ^ 1)
                T.tma_copy(
                    A[bx * block_M:(bx + 1) * block_M, k * block_K:(k + 1) * block_K],
                    A_shared[stage, :, :],
                    barrier=loaded[stage],
                )
                T.tma_copy(
                    B[
                        k * block_K:(k + 1) * block_K,
                        (by * block_N + cta_id * half_N):(by * block_N + (cta_id + 1) * half_N),
                    ],
                    B_shared[stage, :, :],
                    barrier=loaded[stage],
                )
                T.mbarrier_arrive(loaded[stage], 0)

        elif warp_idx == 1 and cta_id == 0:
            for k in T.serial(k_iters):
                stage = k % num_stages
                parity = (k // num_stages) & 1
                T.mbarrier_wait_parity(with_sf_full[stage], parity)
                T.tcgen05_after_thread_sync()
                if k % sf_load_period == 0:
                    T.tcgen05_cp(SFA_shared[stage, :, :], SFA_tmem, use_2cta=True)
                    T.tcgen05_cp(SFB_shared[stage, :, :], SFB_tmem, use_2cta=True)
                    T.sync_warp()
                T.tcgen05_gemm_blockscaled(
                    A_shared[stage, :, :],
                    B_shared[stage, :, :],
                    C_tmem,
                    SFA_tmem,
                    SFB_tmem,
                    mbar=consumed[stage],
                    clear_accum=k == 0,
                    sf_a_id=k % sf_load_period,
                    sf_b_id=k % sf_load_period,
                    use_2cta=True,
                )
            T.tcgen05_mma_arrive(tmem_full, arrive_2cta=True)

        elif warp_idx == 2:
            for k in T.serial(k_iters):
                stage = k % num_stages
                parity = (k // num_stages) & 1
                T.mbarrier_wait_parity(loaded[stage], parity)
                if k % sf_load_period == 0:
                    sf_k_idx = k // sf_load_period
                    for i in T.serial(block_M // 32):
                        for j in T.serial(4):
                            SFA_shared[stage, i * 32 + lane, j] = SFA[
                                bx * block_M + i * 32 + lane,
                                sf_k_idx * 4 + j,
                            ]
                    for i in T.serial(half_N // 32):
                        for j in T.serial(4):
                            SFB_shared[stage, i * 32 + lane, j] = SFB[
                                by * block_N + cta_id * half_N + i * 32 + lane,
                                sf_k_idx * 4 + j,
                            ]
                    T.sf_warp_transpose(SFA_shared[stage, :, :])
                    T.sf_warp_transpose(SFB_shared[stage, :, :])
                    T.fence_proxy_async()
                T.mbarrier_arrive(with_sf_full[stage], 0)

        if cta_id == 0:
            T.mbarrier_wait_parity(tmem_full, 0)
            T.tcgen05_before_thread_sync()
            T.sync_threads()
            T.tcgen05_after_thread_sync()
            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[bx * block_M, by * block_N])

    return C


def blockscaled_gemm_ref(a, b, sfa_unpacked, sfb_unpacked, sf_granularity_k=128):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    sf_k_blocks = (K + sf_granularity_k - 1) // sf_granularity_k

    a_f32 = a.to(torch.float32)
    b_f32 = b.to(torch.float32)
    sfa_scales = torch.pow(2.0, sfa_unpacked.to(torch.float32) - 127.0)
    sfb_scales = torch.pow(2.0, sfb_unpacked.to(torch.float32) - 127.0)

    c = torch.zeros(M, N, device=a.device, dtype=torch.float32)
    for bi in range(sf_k_blocks):
        k_start = bi * sf_granularity_k
        k_end = min(k_start + sf_granularity_k, K)
        a_block = a_f32[:, k_start:k_end] * sfa_scales[:, bi:bi + 1]
        b_block = b_f32[k_start:k_end, :] * sfb_scales[:, bi:bi + 1].T
        c += a_block @ b_block
    return c


def cosine_similarity(a, b):
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return (a_flat @ b_flat) / (a_flat.norm() * b_flat.norm())


def make_scale_tensors(M, N, K, sf_granularity_k=128, *, all_ones=False):
    sf_k_blocks = (K + sf_granularity_k - 1) // sf_granularity_k
    sf_k_padded = ((sf_k_blocks + 3) // 4) * 4
    if all_ones:
        sfa = torch.full((M, sf_k_padded), 127, device="cuda", dtype=torch.uint8)
        sfb = torch.full((N, sf_k_padded), 127, device="cuda", dtype=torch.uint8)
    else:
        sfa = torch.randint(122, 132, (M, sf_k_padded), device="cuda", dtype=torch.uint8)
        sfb = torch.randint(122, 132, (N, sf_k_padded), device="cuda", dtype=torch.uint8)
    return sfa, sfb, sf_k_blocks


def check_kernel_source(a, b, sfa, sfb, block_M, block_N, block_K, in_dtype, out_dtype, accum_dtype, num_stages, sf_granularity_k):
    source = mxfp8_blockscaled_gemm_2cta.get_kernel_source(
        a,
        b,
        sfa,
        sfb,
        block_M,
        block_N,
        block_K,
        in_dtype,
        out_dtype,
        accum_dtype,
        num_stages,
        sf_granularity_k,
    )
    assert "tcgen05mma_blockscaled_ss<tl::DataType::kFloat8_e4m3, true>" in source
    assert "tcgen05_cp<true>" in source
    print("Verified 2CTA blockscaled kernel source.")
    return source


def run_correctness_case(M, N, K, *, num_stages=4, all_ones=False):
    block_M, block_N, block_K = 128, 256, 128
    sf_granularity_k = 128
    in_dtype, out_dtype, accum_dtype = T.float8_e4m3fn, T.bfloat16, T.float

    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
    sfa, sfb, sf_k_blocks = make_scale_tensors(M, N, K, sf_granularity_k, all_ones=all_ones)

    c = mxfp8_blockscaled_gemm_2cta(
        a,
        b,
        sfa,
        sfb,
        block_M,
        block_N,
        block_K,
        in_dtype,
        out_dtype,
        accum_dtype,
        num_stages,
        sf_granularity_k,
    )
    ref_c = blockscaled_gemm_ref(
        a,
        b,
        sfa[:, :sf_k_blocks],
        sfb[:, :sf_k_blocks],
        sf_granularity_k,
    ).to(torch.bfloat16)

    sim = cosine_similarity(c, ref_c)
    max_abs_err = (c.float() - ref_c.float()).abs().max().item()
    print(
        f"case M={M} N={N} K={K} all_ones={all_ones}: "
        f"cos={sim.item():.6f}, max_abs_err={max_abs_err:.6f}"
    )
    assert sim > 0.99, f"Cosine similarity too low: {sim.item():.6f}"
    return a, b, sfa, sfb, c


def run_single_shape_compare(M=8192, N=8192, K=8192, *, num_stages=4):
    block_M, block_N, block_K = 128, 256, 128
    sf_granularity_k = 128
    in_dtype, out_dtype, accum_dtype = T.float8_e4m3fn, T.bfloat16, T.float

    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
    sfa, sfb, sf_k_blocks = make_scale_tensors(M, N, K, sf_granularity_k)

    c_2cta = mxfp8_blockscaled_gemm_2cta(
        a,
        b,
        sfa,
        sfb,
        block_M,
        block_N,
        block_K,
        in_dtype,
        out_dtype,
        accum_dtype,
        num_stages,
        sf_granularity_k,
    )
    c_1cta = mxfp8_blockscaled_gemm_1cta(
        a,
        b,
        sfa,
        sfb,
        block_M,
        block_N,
        block_K,
        in_dtype,
        out_dtype,
        accum_dtype,
        num_stages,
        sf_granularity_k,
    )
    ref_c = blockscaled_gemm_ref(
        a,
        b,
        sfa[:, :sf_k_blocks],
        sfb[:, :sf_k_blocks],
        sf_granularity_k,
    ).to(torch.bfloat16)

    sim_2cta = cosine_similarity(c_2cta, ref_c)
    sim_1cta = cosine_similarity(c_1cta, ref_c)
    err_2cta = (c_2cta.float() - ref_c.float()).abs().max().item()
    err_1cta = (c_1cta.float() - ref_c.float()).abs().max().item()

    print(f"shape M={M} N={N} K={K}")
    print(f"2CTA cosine similarity: {sim_2cta.item():.6f}")
    print(f"2CTA max abs error: {err_2cta:.6f}")
    print(f"1CTA cosine similarity: {sim_1cta.item():.6f}")
    print(f"1CTA max abs error: {err_1cta:.6f}")

    check_kernel_source(
        a,
        b,
        sfa,
        sfb,
        block_M,
        block_N,
        block_K,
        in_dtype,
        out_dtype,
        accum_dtype,
        num_stages,
        sf_granularity_k,
    )

    latency_2cta = do_bench(
        lambda: mxfp8_blockscaled_gemm_2cta(
            a,
            b,
            sfa,
            sfb,
            block_M,
            block_N,
            block_K,
            in_dtype,
            out_dtype,
            accum_dtype,
            num_stages,
            sf_granularity_k,
        ),
        backend="cupti",
    )
    latency_1cta = do_bench(
        lambda: mxfp8_blockscaled_gemm_1cta(
            a,
            b,
            sfa,
            sfb,
            block_M,
            block_N,
            block_K,
            in_dtype,
            out_dtype,
            accum_dtype,
            num_stages,
            sf_granularity_k,
        ),
        backend="cupti",
    )
    print(f"2CTA latency: {latency_2cta} ms")
    print(f"2CTA TFLOPS: {2 * M * N * K / (latency_2cta / 1e3) / 1e12:.2f}")
    print(f"1CTA latency: {latency_1cta} ms")
    print(f"1CTA TFLOPS: {2 * M * N * K / (latency_1cta / 1e3) / 1e12:.2f}")


if __name__ == "__main__":
    run_single_shape_compare()
