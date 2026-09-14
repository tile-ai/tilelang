"""Demonstrate T.annotate_unlimit_memory to bypass the auto-scheduler's
memory capacity limit for a specific scope.

By default, the auto-scheduler enforces per-scope memory limits (e.g. 216 KB
for shared/UB, 512 KB for L1).  When you know a scope has spare headroom,
use T.annotate_unlimit_memory to remove the constraint so Z3 can fit larger
buffer configurations.

This example schedules a GEMM with a manually-annotated L1 buffer that needs
5 versions.  With num_stages=3 and block_K=512, the L1 usage would exceed the
default 512 KB L1 limit.
"""

import argparse

import tilelang
import tilelang.ascend.language as T


def gemm_with_unlimit(M, N, K, block_M, block_N, block_K, dtype="float16"):
    """GEMM with large L1 buffering -- needs unlimit to schedule."""

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, block_N)) as bx:
            # Remove the L1 capacity limit so the auto-scheduler can fit
            # 5 versions of the large L1 buffers.
            T.annotate_unlimit_memory("shared.l1")

            a_l1 = T.alloc_l1((block_M, block_K), dtype)
            b_l1 = T.alloc_l1((block_N, block_K), dtype)
            c_res = T.alloc_l0c((block_M, block_N), "float32")
            temp = T.alloc_shared((block_M, block_N // 2), "float32")
            # Manually request 5 versions of a_l1 for deeper pipelining
            T.annotate_buffer_versions({a_l1: 5})

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[bx * block_M : (bx + 1) * block_M, k * block_K : (k + 1) * block_K], a_l1)
                T.copy(B[bx * block_N : (bx + 1) * block_N, k * block_K : (k + 1) * block_K], b_l1)
                T.gemm(a_l1, b_l1, c_res, transpose_B=True, clear_accum=(k == 0))

            T.dual_copy(c_res, temp)
            T.dual_copy(temp, C[bx * block_M : (bx + 1) * block_M, bx * block_N : (bx + 1) * block_N])

    return main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile the annotate_unlimit_memory example with the Ascend backend.")
    parser.add_argument(
        "--target",
        choices=("ascend",),
        default="ascend",
        help="Compilation backend (default: ascend).",
    )
    args = parser.parse_args()

    M, N, K = 4096, 4096, 4096
    block_M, block_N, block_K = 256, 256, 512
    dtype = "float16"

    # L1 usage: a_l1=256*512*2=256KB (×5=1280KB), b_l1=256*512*2=256KB (×2=512KB)
    # Total exceeds the default 512 KB L1 limit, so unlimit is required.
    a_l1_size = block_M * block_K * 2  # float16
    b_l1_size = block_N * block_K * 2
    print(f"a_l1 ({block_M}x{block_K} fp16): {a_l1_size} bytes ({a_l1_size / 1024:.0f} KB) x5 versions")
    print(f"b_l1 ({block_N}x{block_K} fp16): {b_l1_size} bytes ({b_l1_size / 1024:.0f} KB) x2 versions")
    print("Default L1 limit: 512 KB -- annotate_unlimit_memory required\n")

    program = gemm_with_unlimit(M, N, K, block_M, block_N, block_K, dtype)
    # tilelang.lower expects the caller to hold the target scope; passes that
    # consult Target.current() read the Ascend vector capabilities through it.
    from tvm.target import Target

    with Target(args.target):
        mod = tilelang.lower(program, target=args.target)
    print(f'Compilation succeeded with target="{args.target}" and T.annotate_unlimit_memory("shared.l1").')
