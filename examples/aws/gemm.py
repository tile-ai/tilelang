import argparse

import torch
import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver
from tilelang.profiler import do_bench


@tilelang.jit
def gemm(
    A,
    B,
    block_M,
    block_N,
    store_block_N,
    block_K,
    dtype,
    accum_dtype,
    num_stages,
    num_persistent_stages,
):
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((N, K), dtype)
    C = T.empty((M, N), dtype)

    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N, block_N)
    k_blocks = T.ceildiv(K, block_K)
    sm_num = driver.get_num_sms()
    group_size = 8
    assert n_blocks % (2 * group_size) == 0
    assert K % (2 * block_K) == 0
    assert block_N % store_block_N == 0

    with T.Kernel(sm_num, threads=128) as block_id:
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_N, block_K), dtype)
        C_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
        # mbar = T.alloc_barrier(1)
        C_local = T.alloc_fragment((block_M, store_block_N), accum_dtype)
        C_shared = T.alloc_shared((block_M, store_block_N), dtype)

        sched = T.PersistentTileScheduler(m_blocks, n_blocks, swizzle_size=group_size, name="sched")

        # The complete, unambiguous schedule to transform `gemm` to `gemm_ws`.
        #
        # Ops and loops carry stable ids (a T.WSID entry in their
        # annotations); the schedule refers to those ids. Data
        # dependencies are not declared — the pass queries each tile
        # op's reads/writes itself.
        #
        # Each pipeline is a full/empty barrier pair protecting `depth`
        # versions of its buffers. Sync instructions carry a
        # software-pipeline stage; instructions between an
        # acquire/commit (or wait/release) pair run at that stage's
        # iteration offset.
        T.annotate_ws_schedule(
            T.WSSchedule(
                # This overrides the number of threads.
                num_warps=8,
                # The partition of warps, each with a distinct role.
                roles=[
                    T.WSRole("TMA", warps_lo=0, warps_hi=1, max_nreg=40),
                    T.WSRole("MMA", warps_lo=1, warps_hi=2, max_nreg=40),
                    T.WSRole("Epilogue", warps_lo=4, warps_hi=8, max_nreg=224),
                ],
                # Each pipeline has 2 sets of barriers: full and empty.
                # The producer waits for the empty barrier and signals the full barrier.
                # The consumer waits for the full barrier and signals the empty barrier.
                # The pipeline depth is the number of versions of the buffers.
                # Multiple buffers can share one pipeline.
                pipelines=[
                    # This pipeline synchronizes between the TMA warp and the MMA warp, protecting the multi-versioned A_shared and B_shared buffers.
                    T.WSPipeline("smem", [A_shared, B_shared], depth=num_stages),
                    # This pipeline synchronizes between the MMA warp and the Epilogue warps, protecting the multi-versioned C_tmem buffer.
                    T.WSPipeline("tmem", [C_tmem], depth=num_persistent_stages),
                ],
                # A scope is a scheduled loop: a serial loop named by its
                # ws op id, or a T.ws_op-wrapped while loop (the persistent
                # loop below); the root scope is implicit. Each scope body
                # gives one sequence of ops per role. When a role's sync
                # stages do not cancel out, the software pipeline's
                # prologue and epilogue are unrolled automatically.
                scopes=[
                    T.WSScope(
                        "loop_k",
                        {
                            "TMA": [
                                # Wait for the smem_empty barrier.
                                # Before this, A_shared and B_shared are not usable, because there are multiple versions of them, and the buffers are protected by the pipeline.
                                # After this, the buffers are visible, and the acquired version is bound to each buffer.
                                T.WSSync.producer_acquire("smem", stage=0),
                                # Now A_shared will refer to the acquired version of A_shared, for every op.
                                # Here, copy_A_g2s will write to that version of A_shared.
                                # During schedule materialization, TileLang will first find that the operand A_shared being written to is bound to pipeline smem, so it will bind an smem_full barrier to the op. It will be eventually lowered to a cp.async.bulk.tensor.mbarrier::complete_tx::bytes, decrementing the transaction count of the barrier.
                                "copy_A_g2s",
                                # Similarly, copy_B_g2s will write to the acquired version of B_shared.
                                "copy_B_g2s",
                                # Signal the smem_full barrier.
                                # After this, the buffers are not usable, and again protected by the pipeline.
                                T.WSSync.producer_commit("smem", stage=0),
                            ],
                            "MMA": [
                                # Wait for the smem_full barrier at stage num_stages - 1.
                                # After this, the corresponding version of A_shared and B_shared is bound to the variables.
                                T.WSSync.consumer_wait("smem", stage=num_stages - 1),
                                # During schedule materialization, TileLang will first find that the operands A_shared and B_shared being read from are bound to pipeline smem, so it will bind an smem_empty barrier to the op. It will be eventually lowered to a tcgen05.commit.mbarrier::arrive::one, incrementing the arrival count of the barrier.
                                "gemm_C",
                                # Signal the smem_empty barrier.
                                # After this, the buffers are not usable, and again protected by the pipeline.
                                T.WSSync.consumer_release("smem", stage=num_stages - 1),
                                # After materialization, the loop would look like:
                                #   for k in range(k_blocks):
                                #       if k >= num_stages - 1:
                                #           ...body(k - num_stages + 1)
                                #   body(k_blocks - num_stages + 1)
                                #   ...
                                #   body(k_blocks - 1)
                                # But that is exactly equivalent to
                                #   for k in range(k_blocks):
                                #       body(k)
                                # So that is the loop we see: this role's
                                # stages agree, so the offsets cancel.
                            ],
                            "Epilogue": [],
                        },
                    ),
                    # This is the persistent loop.
                    T.WSScope(
                        "loop_wave",
                        {
                            "TMA": [
                                "tile_idx",
                                "loop_k",
                                "sched_next",
                            ],
                            "MMA": [
                                T.WSSync.producer_acquire("tmem", stage=0),
                                "loop_k",
                                T.WSSync.producer_commit("tmem", stage=0),
                                "sched_next",
                            ],
                            "Epilogue": [
                                "tile_idx",
                                T.WSSync.consumer_wait("tmem", stage=0),
                                # The unrolled loop is one op; it reads the
                                # acquired version of C_tmem.
                                "loop_i",
                                T.WSSync.consumer_release("tmem", stage=0),
                                "sched_next",
                            ],
                        },
                    ),
                    # The root scope is implicit. Nonetheless, the ops inside
                    # can still be scheduled.
                    T.WSScope(
                        T.WSScope.ROOT,
                        {
                            "TMA": ["sched_init", "loop_wave"],
                            "MMA": ["sched_init", "loop_wave"],
                            "Epilogue": ["sched_init", "loop_wave"],
                        },
                    ),
                ],
            )
        )

        with T.ws_op("sched_init"):
            sched.init(block_id)

        with T.ws_op("loop_wave"):
            while sched.valid():
                with T.ws_op("tile_idx"):
                    bx = sched.m_idx
                    by = sched.n_idx

                for k in T.Pipelined(
                    k_blocks,
                    num_stages=num_stages,
                    annotations={T.WSID: "loop_k"},
                ):
                    T.copy(
                        A[bx * block_M, k * block_K],
                        A_shared,
                        annotations={T.WSID: "copy_A_g2s"},
                    )
                    T.copy(
                        B[by * block_N, k * block_K],
                        B_shared,
                        annotations={T.WSID: "copy_B_g2s"},
                    )
                    T.gemm(
                        A_shared,
                        B_shared,
                        C_tmem,
                        transpose_B=True,
                        clear_accum=k == 0,
                        # mbar=mbar,
                        annotations={T.WSID: "gemm_C"},
                    )
                    # T.mbarrier_wait_parity(mbar, k % 2)

                for i in T.unroll(
                    T.ceildiv(block_N, store_block_N),
                    annotations={T.WSID: "loop_i"},
                ):
                    T.copy(C_tmem[:, i * store_block_N : (i + 1) * store_block_N], C_local)
                    T.copy(C_local, C_shared)
                    T.copy(C_shared, C[bx * block_M, by * block_N + i * store_block_N])

                with T.ws_op("sched_next"):
                    sched.next_tile()

    return C


@tilelang.jit
def gemm_ws(
    A,
    B,
    block_M,
    block_N,
    store_block_N,
    block_K,
    dtype,
    accum_dtype,
    num_stages,
    num_persistent_stages,
):
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((N, K), dtype)
    C = T.empty((M, N), dtype)

    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N, block_N)
    k_blocks = T.ceildiv(K, block_K)
    sm_num = driver.get_num_sms()
    group_size = 8
    assert n_blocks % (2 * group_size) == 0
    assert K % (2 * block_K) == 0
    assert block_N % store_block_N == 0

    with T.Kernel(sm_num, threads=256) as block_id:
        A_shared = T.alloc_shared((num_stages, block_M, block_K), dtype)
        B_shared = T.alloc_shared((num_stages, block_N, block_K), dtype)
        C_tmem = T.alloc_tmem([num_persistent_stages, block_M, block_N], accum_dtype)
        C_local = T.alloc_fragment((block_M, store_block_N), accum_dtype)
        C_shared = T.alloc_shared((block_M, store_block_N), dtype)
        smem_full = T.alloc_barrier([32] * num_stages)
        smem_empty = T.alloc_barrier([1] * num_stages)
        tmem_full = T.alloc_barrier([1] * num_persistent_stages)
        tmem_empty = T.alloc_barrier([128] * num_persistent_stages)

        # Scheduler state is per thread: init once, then every role runs
        # its own `while sched.valid()` loop at its own pace, reading
        # sched.current_iter as the wave clock.
        sched = T.PersistentTileScheduler(m_blocks, n_blocks, swizzle_size=group_size, name="sched")
        sched.init(block_id)

        tx = T.get_thread_binding()

        if tx < 32:  # warp 0: TMA
            T.set_max_nreg(40, 0)
            while sched.valid():
                bx = sched.m_idx
                by = sched.n_idx

                for k in T.serial(k_blocks):
                    phase = sched.current_iter * k_blocks + k

                    T.mbarrier_wait_parity(smem_empty[phase % num_stages], ((phase // num_stages) & 1) ^ 1)
                    T.tma_copy(
                        A[
                            bx * block_M : (bx + 1) * block_M,
                            k * block_K : (k + 1) * block_K,
                        ],
                        A_shared[phase % num_stages, :, :],
                        barrier=smem_full[phase % num_stages],
                    )
                    T.tma_copy(
                        B[
                            by * block_N : (by + 1) * block_N,
                            k * block_K : (k + 1) * block_K,
                        ],
                        B_shared[phase % num_stages, :, :],
                        barrier=smem_full[phase % num_stages],
                    )
                    T.mbarrier_arrive(smem_full[phase % num_stages])
                sched.next_tile()
        elif tx < 64:  # warp 1: MMA
            T.set_max_nreg(40, 0)
            while sched.valid():
                wave_id = sched.current_iter
                T.mbarrier_wait_parity(
                    tmem_empty[wave_id % num_persistent_stages],
                    ((wave_id // num_persistent_stages) & 1) ^ 1,
                )

                for k in T.serial(k_blocks):
                    phase = wave_id * k_blocks + k

                    T.mbarrier_wait_parity(smem_full[phase % num_stages], (phase // num_stages) & 1)
                    T.tcgen05_gemm(
                        A_shared[phase % num_stages, :, :],
                        B_shared[phase % num_stages, :, :],
                        C_tmem[wave_id % num_persistent_stages, :, :],
                        transpose_B=True,
                        mbar=None,
                        clear_accum=k == 0,
                    )

                    T.tcgen05_mma_arrive(smem_empty[phase % num_stages])

                T.tcgen05_mma_arrive(tmem_full[wave_id % num_persistent_stages])
                sched.next_tile()
        elif tx < 128:  # warps 2-3: idle register donors
            T.set_max_nreg(40, 0)
        elif 128 <= tx < 256:  # warps 4-7: epilogue
            T.set_max_nreg(224, 1)
            while sched.valid():
                bx = sched.m_idx
                by = sched.n_idx
                wave_id = sched.current_iter

                T.mbarrier_wait_parity(
                    tmem_full[wave_id % num_persistent_stages],
                    (wave_id // num_persistent_stages) & 1,
                )

                for i in T.unroll(T.ceildiv(block_N, store_block_N)):
                    T.copy(
                        C_tmem[
                            wave_id % num_persistent_stages,
                            :,
                            i * store_block_N : (i + 1) * store_block_N,
                        ],
                        C_local,
                    )
                    T.copy(C_local, C_shared)
                    T.copy(C_shared, C[bx * block_M, by * block_N + i * store_block_N])

                T.mbarrier_arrive(tmem_empty[wave_id % num_persistent_stages])
                sched.next_tile()

    return C


KERNELS = {"gemm": gemm, "gemm_ws": gemm_ws}


def main(
    kernel="gemm",
    M=8192,
    N=8192,
    K=8192,
    block_M=128,
    block_N=256,
    store_block_N=64,
    block_K=64,
    num_stages=4,
    num_persistent_stages=2,
):
    dtype, accum_dtype = T.bfloat16, T.float
    kernel_fn = KERNELS[kernel]
    kwargs = {
        "block_M": block_M,
        "block_N": block_N,
        "store_block_N": store_block_N,
        "block_K": block_K,
        "dtype": dtype,
        "accum_dtype": accum_dtype,
        "num_stages": num_stages,
        "num_persistent_stages": num_persistent_stages,
    }

    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    c = kernel_fn(a, b, **kwargs)

    ref_c = (a.to(torch.float) @ b.to(torch.float).T).to(torch.bfloat16)
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("All checks passed. ✅")

    tl_latency = do_bench(lambda: kernel_fn(a, b, **kwargs), backend="cupti")
    torch_latency = do_bench(lambda: a @ b.T, backend="cupti")
    print(f"Tilelang latency: {tl_latency} ms")
    print(f"Flops: {2 * M * N * K / (tl_latency / 1e3) / 1e12} TFLOPS")
    print(f"Torch latency: {torch_latency} ms")
    print(f"Flops: {2 * M * N * K / (torch_latency / 1e3) / 1e12} TFLOPS")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--kernel", choices=sorted(KERNELS), default="gemm")
    p.add_argument("--m", type=int, default=8192)
    p.add_argument("--n", type=int, default=8192)
    p.add_argument("--k", type=int, default=8192)
    p.add_argument("--block_m", type=int, default=128)
    p.add_argument("--block_n", type=int, default=256)
    p.add_argument(
        "--store_block_n",
        type=int,
        default=64,
        help="epilogue store slice width",
    )
    p.add_argument("--block_k", type=int, default=64)
    p.add_argument("--num_stages", type=int, default=4)
    p.add_argument("--num_persistent_stages", type=int, default=2)
    args = p.parse_args()
    main(
        kernel=args.kernel,
        M=args.m,
        N=args.n,
        K=args.k,
        block_M=args.block_m,
        block_N=args.block_n,
        store_block_N=args.store_block_n,
        block_K=args.block_k,
        num_stages=args.num_stages,
        num_persistent_stages=args.num_persistent_stages,
    )
