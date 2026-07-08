# Persistent, warp-specialized SM100 GEMM using PersistentTileScheduler.
#
# Each warp role (TMA / tcgen5 MMA / epilogue) creates its own scheduler instance
# and drives a single ``while sched.valid()`` loop. The scheduler owns the only
# iteration clock: ``sched.current_iter`` drives pipeline phase /
# ``sched.current_iter & 1`` double-buffering; ``sched.m_idx`` /
# ``sched.n_idx`` are the tile coords. One clock, held by the scheduler -- no
# separate ``for w in range(waves)`` counter.

import torch
import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver
from tilelang.profiler import do_bench


@tilelang.jit
def gemm_persistent(
    A,
    B,
    block_M,
    block_N,
    store_block_N,  # block_N for C_shared
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    use_tma_store=True,
):
    M, N, K = T.const("M, N, K")

    A: T.Tensor[[M, K], in_dtype]
    B: T.Tensor[[K, N], in_dtype]
    C = T.empty((M, N), out_dtype)

    sm_num = driver.get_num_sms()
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N, block_N)
    assert K % (2 * block_K) == 0  # for simplicity
    k_blocks = T.ceildiv(K, block_K)
    group_size = 8
    assert n_blocks % (2 * group_size) == 0  # Please adjust group_size if not satisfied

    with T.Kernel(sm_num, threads=256) as (block_id):
        A_shared = T.alloc_shared((num_stages, block_M, block_K), in_dtype)
        B_shared = T.alloc_shared((num_stages, block_K, block_N), in_dtype)
        C_tmem_0 = T.alloc_tmem([block_M, block_N], accum_dtype)
        C_tmem_1 = T.alloc_tmem([block_M, block_N], accum_dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        C_local_cast = T.alloc_fragment((block_M, block_N), out_dtype)
        C_shared = T.alloc_shared((block_M, store_block_N), out_dtype)
        loaded = T.alloc_barrier([32] * num_stages)
        consumed = T.alloc_barrier([1] * num_stages)
        tmem_full = T.alloc_barrier([1] * 2)
        tmem_empty = T.alloc_barrier([128] * 2)

        tx = T.get_thread_binding()

        if tx < 32:  # warp 0: issue tma
            sched = T.PersistentTileScheduler(m_blocks, n_blocks, swizzle_size=group_size)
            sched.init(block_id)
            while sched.valid():
                bx, by = sched.m_idx, sched.n_idx

                for k in T.serial(k_blocks):
                    phase = sched.current_iter * k_blocks + k
                    T.mbarrier_wait_parity(consumed[phase % num_stages], ((phase // num_stages) & 1) ^ 1)
                    T.tma_copy(
                        A[bx * block_M : (bx + 1) * block_M, k * block_K : (k + 1) * block_K],
                        A_shared[phase % num_stages, :, :],
                        barrier=loaded[phase % num_stages],
                    )
                    T.tma_copy(
                        B[k * block_K : (k + 1) * block_K, by * block_N : (by + 1) * block_N],
                        B_shared[phase % num_stages, :, :],
                        barrier=loaded[phase % num_stages],
                    )
                    T.mbarrier_arrive(loaded[phase % num_stages])
                sched.next_tile()

        elif tx < 64:  # warp 1: issue tcgen5
            sched = T.PersistentTileScheduler(m_blocks, n_blocks, swizzle_size=group_size)
            sched.init(block_id)
            while sched.valid():
                T.mbarrier_wait_parity(tmem_empty[sched.current_iter & 1], ((sched.current_iter // 2) & 1) ^ 1)
                for k in T.serial(k_blocks):
                    phase = sched.current_iter * k_blocks + k
                    T.mbarrier_wait_parity(loaded[phase % num_stages], (phase // num_stages) & 1)
                    if sched.current_iter & 1 == 0:
                        T.tcgen05_gemm(
                            A_shared[k % num_stages, :, :],
                            B_shared[k % num_stages, :, :],
                            C_tmem_0,
                            mbar=consumed[k % num_stages],
                            clear_accum=k == 0,
                        )
                    else:
                        T.tcgen05_gemm(
                            A_shared[k % num_stages, :, :],
                            B_shared[k % num_stages, :, :],
                            C_tmem_1,
                            mbar=consumed[k % num_stages],
                            clear_accum=k == 0,
                        )
                T.tcgen05_mma_arrive(tmem_full[sched.current_iter & 1])
                sched.next_tile()

        elif 128 <= tx < 256:  # warp 4~7: epilogue
            sched = T.PersistentTileScheduler(m_blocks, n_blocks, swizzle_size=group_size)
            sched.init(block_id)
            while sched.valid():
                bx, by = sched.m_idx, sched.n_idx

                T.mbarrier_wait_parity(tmem_full[sched.current_iter & 1], (sched.current_iter // 2) & 1)
                if (sched.current_iter & 1) == 0:
                    T.copy(C_tmem_0, C_local)
                else:
                    T.copy(C_tmem_1, C_local)
                T.mbarrier_arrive(tmem_empty[sched.current_iter & 1])

                if use_tma_store:
                    for i in T.unroll(T.ceildiv(block_N, store_block_N)):
                        T.copy(C_local[:, i * store_block_N : (i + 1) * store_block_N], C_shared)
                        T.copy(C_shared, C[bx * block_M, by * block_N + i * store_block_N])
                else:
                    T.copy(C_local, C_local_cast)
                    T.copy(C_local_cast, C[bx * block_M, by * block_N])
                sched.next_tile()
    return C


@tilelang.jit
def gemm_persistent_2cta(
    A,
    B,
    block_M,
    block_N,
    store_block_N,  # block_N for C_shared
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    use_tma_store=True,
):
    M, N, K = T.const("M, N, K")

    A: T.Tensor[[M, K], in_dtype]
    B: T.Tensor[[K, N], in_dtype]
    C = T.empty((M, N), out_dtype)

    sm_num = driver.get_num_sms()
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N, block_N)
    assert K % (2 * block_K) == 0  # for simplicity
    k_blocks = T.ceildiv(K, block_K)
    group_size = 8  # in cluster
    assert n_blocks % (2 * group_size) == 0  # Please adjust group_size if not satisfied
    cluster_size = 2

    with T.ClusterKernel(sm_num, threads=256, cluster_dims=2) as (block_id):
        A_shared = T.alloc_shared((num_stages, block_M, block_K), in_dtype)
        B_shared = T.alloc_shared((num_stages, block_K, block_N // 2), in_dtype)
        C_tmem_0 = T.alloc_tmem([block_M, block_N], accum_dtype)
        C_tmem_1 = T.alloc_tmem([block_M, block_N], accum_dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        C_local_cast = T.alloc_fragment((block_M, block_N), out_dtype)
        C_shared = T.alloc_shared((block_M, store_block_N), out_dtype)
        loaded = T.alloc_cluster_barrier([32 * 2] * num_stages)
        consumed = T.alloc_cluster_barrier([1] * num_stages)
        tmem_full = T.alloc_cluster_barrier([1] * 2)
        tmem_empty = T.alloc_cluster_barrier([128 * 2] * 2)

        tx = T.get_thread_binding()
        cta_id = T.block_rank_in_cluster()
        T.assume(cta_id < 2)  # todo: automatically assume this

        if tx < 32:  # warp 0: issue tma
            sched = T.PersistentTileScheduler(m_blocks, n_blocks, swizzle_size=group_size, cluster_size=cluster_size)
            sched.init(block_id // cluster_size)
            while sched.valid():
                bx, by = sched.m_idx * cluster_size + cta_id, sched.n_idx

                for k in T.serial(k_blocks):
                    phase = sched.current_iter * k_blocks + k
                    T.mbarrier_wait_parity(consumed[phase % num_stages], ((phase // num_stages) & 1) ^ 1)
                    T.tma_copy(
                        A[bx * block_M : (bx + 1) * block_M, k * block_K : (k + 1) * block_K],
                        A_shared[phase % num_stages, :, :],
                        barrier=loaded[phase % num_stages],
                    )

                    T.tma_copy(
                        B[k * block_K : (k + 1) * block_K, (by * 2 + cta_id) * block_N // 2 : (by * 2 + cta_id + 1) * block_N // 2],
                        B_shared[phase % num_stages, :, :],
                        barrier=loaded[phase % num_stages],
                    )
                    T.mbarrier_arrive(loaded[phase % num_stages], 0)
                sched.next_tile()

        elif tx < 64 and cta_id == 0:  # warp 1: issue tcgen5
            sched = T.PersistentTileScheduler(m_blocks, n_blocks, swizzle_size=group_size, cluster_size=cluster_size)
            sched.init(block_id // cluster_size)
            while sched.valid():
                T.mbarrier_wait_parity(tmem_empty[sched.current_iter & 1], ((sched.current_iter // 2) & 1) ^ 1)
                for k in T.serial(k_blocks):
                    phase = sched.current_iter * k_blocks + k
                    T.mbarrier_wait_parity(loaded[phase % num_stages], (phase // num_stages) & 1)
                    if sched.current_iter & 1 == 0:
                        T.tcgen05_gemm(
                            A_shared[phase % num_stages, :, :],
                            B_shared[phase % num_stages, :, :],
                            C_tmem_0,
                            mbar=consumed[phase % num_stages],
                            clear_accum=k == 0,
                            use_2cta=True,
                        )
                    else:
                        T.tcgen05_gemm(
                            A_shared[phase % num_stages, :, :],
                            B_shared[phase % num_stages, :, :],
                            C_tmem_1,
                            mbar=consumed[phase % num_stages],
                            clear_accum=k == 0,
                            use_2cta=True,
                        )
                T.tcgen05_mma_arrive(tmem_full[sched.current_iter & 1], arrive_2cta=True)
                sched.next_tile()

        elif 128 <= tx < 256:  # warp 4~7: epilogue
            sched = T.PersistentTileScheduler(m_blocks, n_blocks, swizzle_size=group_size, cluster_size=cluster_size)
            sched.init(block_id // cluster_size)
            while sched.valid():
                bx, by = sched.m_idx * cluster_size + cta_id, sched.n_idx

                T.mbarrier_wait_parity(tmem_full[sched.current_iter & 1], (sched.current_iter // 2) & 1)
                if (sched.current_iter & 1) == 0:
                    T.copy(C_tmem_0, C_local)
                else:
                    T.copy(C_tmem_1, C_local)
                T.mbarrier_arrive(tmem_empty[sched.current_iter & 1], 0)

                if use_tma_store:
                    for i in T.unroll(T.ceildiv(block_N, store_block_N)):
                        T.copy(C_local[:, i * store_block_N : (i + 1) * store_block_N], C_shared)
                        T.copy(C_shared, C[bx * block_M, by * block_N + i * store_block_N])
                else:
                    T.copy(C_local, C_local_cast)
                    T.copy(C_local_cast, C[bx * block_M, by * block_N])
                sched.next_tile()

    return C


def main():
    M, N, K = 8192, 8192, 8192
    block_M, block_N, block_K = 128, 256, 64
    store_block_N = 64
    in_dtype, out_dtype, accum_dtype = T.bfloat16, T.bfloat16, T.float
    enable_2cta_tcgen5mma = True
    num_stages = 6 if enable_2cta_tcgen5mma else 4  # Each cta only needs to load half of B, enabling larger stages
    kernel = gemm_persistent_2cta if enable_2cta_tcgen5mma else gemm_persistent

    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    print(kernel.get_kernel_source(a, b, block_M, block_N, store_block_N, block_K, in_dtype, out_dtype, accum_dtype, num_stages))
    c = kernel(a, b, block_M, block_N, store_block_N, block_K, in_dtype, out_dtype, accum_dtype, num_stages)

    ref_c = (a.to(torch.float) @ b.to(torch.float)).to(torch.bfloat16)
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("All checks passed. ✅")

    tl_latency = do_bench(
        lambda: kernel(a, b, block_M, block_N, store_block_N, block_K, in_dtype, out_dtype, accum_dtype, num_stages),
        _n_warmup=50,
        _n_repeat=50,
        backend="cupti",
    )
    torch_latency = do_bench(lambda: a @ b, backend="cupti")
    print(f"Tilelang latency: {tl_latency} ms")
    print(f"Flops: {2 * M * N * K / (tl_latency / 1e3) / 1e12} TFLOPS")
    print(f"Torch latency: {torch_latency} ms")
    print(f"Flops: {2 * M * N * K / (torch_latency / 1e3) / 1e12} TFLOPS")


if __name__ == "__main__":
    main()
