import argparse
import tilelang.language as T
from tilelang.autotuner import AutoTuner


def ref_program(A, B):
    return A @ B.T


def get_configs():
    configs = [
        {
            "block_M": 128,
            "block_N": 128,
            "block_K": 64,
            "num_stages": 2,
            "thread_num": 256,
            "enable_rasteration": True,  # keep param name for backward-compat
        }
    ]
    return configs


def run(M, N, K):
    def kernel(
        block_M=None,
        block_N=None,
        block_K=None,
        num_stages=None,
        thread_num=None,
        enable_rasteration=None,
    ):
        dtype = T.float16
        accum_dtype = T.float32

        @T.prim_func
        def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_N, block_K), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                C_shared = T.alloc_shared((block_M, block_N), dtype)
                T.use_swizzle(panel_size=10, enable=enable_rasteration)
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    T.gemm(
                        A_shared,
                        B_shared,
                        C_local,
                        transpose_B=True,
                    )
                T.copy(C_local, C_shared)
                T.copy(C_shared, C[by * block_M, bx * block_N])

        return main

    autotuner = (
        AutoTuner.from_kernel(kernel=kernel, configs=get_configs())
        .set_compile_args(
            out_idx=[-1],
            target="auto",
        )
        .set_profile_args(
            ref_prog=ref_program,
        )
    )
    return autotuner.run(warmup=3, rep=20)


if __name__ == "__main__":
    # Parse command-line arguments for matrix dimensions
    parser = argparse.ArgumentParser(description="Autotuned MatMul Benchmark")
    parser.add_argument("--m", type=int, default=16384, help="Matrix dimension M")
    parser.add_argument("--n", type=int, default=16384, help="Matrix dimension N")
    parser.add_argument("--k", type=int, default=16384, help="Matrix dimension K")
    args = parser.parse_args()

    M, N, K = args.m, args.n, args.k

    # Compute total floating-point operations to measure throughput
    total_flops = 2 * M * N * K

    result = run(M, N, K)

    print(f"Latency: {result.latency}")
    print(f"TFlops: {total_flops / result.latency * 1e-9:.3f}")
    print(f"Config: {result.config}")

    print(f"Reference TFlops: {total_flops / result.ref_latency * 1e-9:.3f}")
