"""GPU smoke test for built tilelang wheels.

Runs against an *installed* tilelang (wheel), not the source tree: copy this
file somewhere outside the repository before running it, or run it with the
repository root absent from ``sys.path``, so ``import tilelang`` resolves to
site-packages. It compiles and launches a small fp16 GEMM on the detected
backend (CUDA or ROCm) and checks the result against torch.

Exits non-zero on any failure.
"""

import sys

import torch

import tilelang
import tilelang.language as T


def report_environment() -> None:
    print("=" * 60)
    print("python:", sys.version.split()[0])
    print("torch:", torch.__version__, "| cuda:", torch.version.cuda, "| hip:", getattr(torch.version, "hip", None))
    assert torch.cuda.is_available(), "torch reports no GPU; a GPU is required for the wheel smoke test"
    print("gpu:", torch.cuda.get_device_name(0))
    print("tilelang:", tilelang.__version__)

    from tilelang.backend.target import determine_target

    print("detected target:", determine_target(return_object=True))
    print("=" * 60)


@tilelang.jit
def matmul(A, B, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


def main() -> None:
    report_environment()

    kernel = matmul.compile(M=1024, N=1024, K=1024, block_M=128, block_N=128, block_K=32)

    a = torch.randn(1024, 1024).cuda().half()
    b = torch.randn(1024, 1024).cuda().half()

    c = kernel(a, b)
    ref_c = a @ b

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("correctness: PASS")

    latency = kernel.get_profiler().do_bench()
    print(f"latency: {latency:.4f} ms")
    print("WHEEL SMOKE TEST: ALL PASS")


if __name__ == "__main__":
    main()
