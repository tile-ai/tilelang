"""Example: SimtVF with multiple __ubuf__ buffers on Ascend NPU.

This example demonstrates using multiple T.alloc_shared to allocate multiple
Unified Buffer (UB) memory regions, which triggers the MergeSharedMemoryAllocations
pass to merge them into a single buf_dyn_shmem.

The kernel computes: E = (A * B + C) * D

VF1: computes A * B → stores to temp1
VF2: computes temp1 + C → stores to temp2
VF3: computes temp2 * D → stores to E
"""

import tilelang
import tilelang.ascend.language as T


def ubuf_multi(N):
    """Multi-VF kernel with multiple shared UB buffers.

    Multiple alloc_shared calls trigger the MergeSharedMemoryAllocations pass,
    which merges all shared.dyn buffers into a single buf_dyn_shmem.
    """

    @T.prim_func
    def main(
        A: T.Buffer((N,), "float32"),
        B: T.Buffer((N,), "float32"),
        C: T.Buffer((N,), "float32"),
        D: T.Buffer((N,), "float32"),
        E: T.Buffer((N,), "float32"),
    ):
        with T.Kernel(1) as _:
            # Multiple alloc_shared - triggers merge into buf_dyn_shmem
            temp1 = T.alloc_shared((N,), "float32")
            temp2 = T.alloc_shared((N,), "float32")

            # VF1: A * B → temp1
            with T.SimtVF(threads=128):
                for i in T.Parallel(N):
                    temp1[i] = A[i] * B[i]
            T.ascend_pipe_barrier("PIPE_V")

            # VF2: temp1 + C → temp2
            with T.SimtVF(threads=(16, 16)):
                tx = T.get_thread_binding()
                for i in T.Parallel(N):
                    temp2[i] = temp1[i] + C[i]
                if tx == 0:
                    temp1[tx] = 0.0
            T.ascend_pipe_barrier("PIPE_V")

            # VF3: temp2 * D → E
            with T.SimtVF(threads=256):
                for i in T.Parallel(N):
                    E[i] = temp2[i] * D[i]

    return main


def ref_program(a, b, c, d):
    """Reference implementation: E = (A * B + C) * D"""
    return (a * b + c) * d


if __name__ == "__main__":
    import torch

    N = 256
    dtype = torch.float32
    device = torch.device("npu")

    # Create and compile kernel
    print(f"Compiling ubuf_multi kernel (N={N})...")
    program = ubuf_multi(N)
    kernel = tilelang.compile(program, out_idx=-1)
    print("Compilation succeeded!")

    # Create test data
    a = torch.randn(N, dtype=dtype, device=device)
    b = torch.randn(N, dtype=dtype, device=device)
    c = torch.randn(N, dtype=dtype, device=device)
    d = torch.randn(N, dtype=dtype, device=device)

    # Run kernel
    print("Running kernel on NPU...")
    e = kernel(a, b, c, d)
    torch.npu.synchronize()

    # Print generated source code
    print("\n--- Generated Ascend Source ---")
    print(kernel.get_kernel_source())

    # Verify
    expected = ref_program(a, b, c, d)
    if not torch.equal(e, expected):
        max_diff = torch.max(torch.abs(e - expected)).item()
        raise AssertionError(f"Results mismatch! Max diff: {max_diff}")
    print("\nVerification passed!")
