"""Example: SimtVF vector_add on Ascend NPU."""

import tilelang
import tilelang.language as T


def vector_add(N):
    """Simple vector add kernel using SimtVF.

    SimtVF (SIMT Vector Fixed) is Ascend's programming model for SIMT-style
    parallelism. Threads are managed internally by the SimtVF region.
    """

    @T.prim_func
    def main(
        A: T.Buffer((N,), "float32"),
        B: T.Buffer((N,), "float32"),
        C: T.Buffer((N,), "float32"),
    ):
        with T.Kernel(1) as _, T.SimtVF(threads=128):
            for i in T.Parallel(N):
                C[i] = A[i] + B[i]

    return main


def ref_program(a, b):
    """Reference implementation using torch."""
    return a + b


if __name__ == "__main__":
    import torch

    N = 1024
    dtype = torch.float32
    device = torch.device("npu")

    # Create and compile kernel
    print(f"Compiling vector_add kernel (N={N})...")
    program = vector_add(N)
    kernel = tilelang.compile(program, target="ascend", out_idx=-1)
    print("Compilation succeeded!")

    # Create test data
    a = torch.randn(N, dtype=dtype, device=device)
    b = torch.randn(N, dtype=dtype, device=device)

    # Run kernel
    print("Running kernel on NPU...")
    c = kernel(a, b)
    torch.npu.synchronize()

    # Verify (float32 add should be exact)
    expected = ref_program(a, b)
    if not torch.equal(c, expected):
        max_diff = torch.max(torch.abs(c - expected)).item()
        raise AssertionError(f"Results mismatch! Max diff: {max_diff}")
    print("Verification passed!")
