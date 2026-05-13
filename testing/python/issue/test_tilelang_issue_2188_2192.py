"""Regression tests for GitHub issues #2188 and #2192.

#2188 – Feature: native T.named_barrier_arrive (bar.arrive) API
#2192 – Bug: unexpected __syncthreads() inserted between T.ws producer/consumer
"""

import torch
import tilelang
import tilelang.language as T


# ---------------------------------------------------------------------------
# Issue #2188: T.named_barrier_arrive
# ---------------------------------------------------------------------------

def test_named_barrier_arrive_api():
    """T.named_barrier_arrive generates bar.arrive (tl::__named_barrier_arrive)."""

    @tilelang.jit(out_idx=[1])
    def kernel_fn():
        @T.prim_func
        def kernel(inp: T.Tensor((1,), T.int32), out: T.Tensor((1,), T.int32)):
            with T.Kernel(1, threads=64) as _:
                tx = T.get_thread_binding()
                slot = T.alloc_shared((1,), T.int32)

                ready_barrier = 10
                empty_barrier = 11
                participant_threads = 64

                if tx < 32:
                    if tx == 0:
                        slot[0] = inp[0] + 1
                    T.named_barrier_arrive(ready_barrier, participant_threads)
                    T.sync_threads(empty_barrier, participant_threads)
                else:
                    T.sync_threads(ready_barrier, participant_threads)
                    if tx == 32:
                        out[0] = slot[0]
                    T.named_barrier_arrive(empty_barrier, participant_threads)

        return kernel

    jit_kernel = kernel_fn()
    source = jit_kernel.get_kernel_source()

    # Verify the intrinsic is emitted
    assert "__named_barrier_arrive<10, 64>" in source, (
        "T.named_barrier_arrive should emit tl::__named_barrier_arrive<id, count>()")
    assert "__named_barrier_arrive<11, 64>" in source

    # Functional correctness
    inp = torch.tensor([41], device="cuda", dtype=torch.int32)
    out = jit_kernel(inp)
    torch.testing.assert_close(out.cpu(), torch.tensor([42], dtype=torch.int32))


def test_named_barrier_arrive_no_spurious_syncthreads():
    """T.named_barrier_arrive branches must not get CTA-wide __syncthreads()."""

    @tilelang.jit(out_idx=[1])
    def kernel_fn():
        @T.prim_func
        def kernel(inp: T.Tensor((1,), T.int32), out: T.Tensor((1,), T.int32)):
            with T.Kernel(1, threads=64) as _:
                tx = T.get_thread_binding()
                slot = T.alloc_shared((1,), T.int32)

                ready_barrier = 10
                empty_barrier = 11
                participant_threads = 64

                if tx < 32:
                    if tx == 0:
                        slot[0] = inp[0] + 1
                    T.named_barrier_arrive(ready_barrier, participant_threads)
                    T.sync_threads(empty_barrier, participant_threads)
                else:
                    T.sync_threads(ready_barrier, participant_threads)
                    if tx == 32:
                        out[0] = slot[0]
                    T.named_barrier_arrive(empty_barrier, participant_threads)

        return kernel

    jit_kernel = kernel_fn()
    source = jit_kernel.get_kernel_source()

    # There must be no CTA-wide __syncthreads() inside the producer/consumer
    # if-branches. The only allowed __syncthreads() is the one emitted by the
    # fence_barrier_init sequence, which appears before the if-branches.
    lines = source.split("\n")
    in_if_branch = False
    spurious_syncs = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("if (") and ("threadIdx" in stripped):
            in_if_branch = True
        if in_if_branch and "__syncthreads();" in stripped:
            # A naked __syncthreads() (no args) inside a threadIdx branch is spurious
            spurious_syncs.append(line)
    assert not spurious_syncs, (
        f"Spurious __syncthreads() found inside threadIdx-guarded branches:\n"
        + "\n".join(spurious_syncs)
    )


# ---------------------------------------------------------------------------
# Issue #2192: no spurious __syncthreads() between T.ws blocks
# ---------------------------------------------------------------------------

def test_ws_no_spurious_syncthreads_in_loop():
    """T.ws producer/consumer pipeline must not get __syncthreads() per iteration."""

    num_stages = 2
    block_M, block_N, block_K = 128, 128, 64
    M, N, K = 1024, 1024, 1024
    mbarrier_list = [128, 128] * num_stages

    @tilelang.jit(out_idx=[2])
    def matmul(M, N, K, block_M, block_N, block_K,
               dtype=T.float16, accum_dtype=T.float32):
        @T.prim_func
        def main(A: T.Tensor[(M, K), dtype],
                 B: T.Tensor[(K, N), dtype],
                 C: T.Tensor[(M, N), dtype]):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M),
                          threads=256) as (bx, by):
                A_shared = T.alloc_shared((num_stages, block_M, block_K), dtype)
                B_shared = T.alloc_shared((num_stages, block_K, block_N), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                mbars = T.alloc_barrier(mbarrier_list)

                with T.ws(0):
                    T.clear(C_local)

                for ko in range(T.ceildiv(K, block_K)):
                    with T.ws(1):
                        T.mbarrier_wait_parity(
                            mbarrier=mbars[ko % num_stages + num_stages],
                            parity=((ko // num_stages) % num_stages) ^ 1)
                        T.tma_copy(
                            A[by * block_M:(by + 1) * block_M,
                              ko * block_K:(ko + 1) * block_K],
                            A_shared[ko % num_stages, :, :],
                            barrier=mbars[ko % num_stages])
                        T.tma_copy(
                            B[ko * block_K:(ko + 1) * block_K,
                              bx * block_N:(bx + 1) * block_N],
                            B_shared[ko % num_stages, :, :],
                            barrier=mbars[ko % num_stages])
                        T.mbarrier_arrive(mbarrier=mbars[ko % num_stages])
                    with T.ws(0):
                        T.mbarrier_wait_parity(
                            mbarrier=mbars[ko % num_stages],
                            parity=(ko // num_stages) % num_stages)
                        T.gemm(A_shared[ko % num_stages, :, :],
                               B_shared[ko % num_stages, :, :], C_local)
                        T.mbarrier_arrive(
                            mbarrier=mbars[ko % num_stages + num_stages])

                with T.ws(0):
                    T.copy(C_local, C[by * block_M, bx * block_N])

        return main

    jit_kernel = matmul(M, N, K, block_M, block_N, block_K)
    source = jit_kernel.get_kernel_source()

    # Locate the main pipeline loop and check there is no __syncthreads()
    # inside it between the two warp-specialized branches.
    in_loop = False
    loop_depth = 0
    spurious_syncs = []
    for line in source.split("\n"):
        stripped = line.strip()
        # Detect the start of the for-loop (ko loop)
        if not in_loop and stripped.startswith("for (int ko"):
            in_loop = True
            loop_depth = 0
        if in_loop:
            loop_depth += stripped.count("{") - stripped.count("}")
            if loop_depth <= 0 and in_loop and stripped == "}":
                in_loop = False
                continue
            if "__syncthreads();" in stripped:
                spurious_syncs.append(line)

    assert not spurious_syncs, (
        f"Unexpected __syncthreads() found inside pipeline loop:\n"
        + "\n".join(spurious_syncs)
    )

    # Functional correctness
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = jit_kernel(a, b)
    ref_c = a @ b
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    test_named_barrier_arrive_api()
    print("test_named_barrier_arrive_api PASSED")
    test_named_barrier_arrive_no_spurious_syncthreads()
    print("test_named_barrier_arrive_no_spurious_syncthreads PASSED")
    test_ws_no_spurious_syncthreads_in_loop()
    print("test_ws_no_spurious_syncthreads_in_loop PASSED")
    print("\nAll issue 2188/2192 regression tests passed.")
