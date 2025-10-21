import argparse
import itertools
import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import AutoTuner


def ref_program(x, y):
    return x + y


@tilelang.jit(out_idx=[-1])
def elementwise_add(M, N, block_M, block_N, in_dtype, out_dtype, threads):

    @T.prim_func
    def elem_add(A: T.Tensor((M, N), in_dtype), B: T.Tensor((M, N), in_dtype), C: T.Tensor(
        (M, N), out_dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), in_dtype)
            B_shared = T.alloc_shared((block_M, block_N), in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), out_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)

            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(B[by * block_M, bx * block_N], B_shared)
            for (local_y, local_x) in T.Parallel(block_M, block_N):
                C_local[local_y, local_x] = A_shared[local_y, local_x] + B_shared[local_y, local_x]
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return elem_add


def get_configs(M, N):
    block_M = [64, 128, 256]
    block_N = [64, 128, 256]
    threads = [64, 128, 256]
    configs = list(itertools.product(block_M, block_N, threads))
    return [{"block_M": bm, "block_N": bn, "threads": th} for bm, bn, th in configs]


def get_best_config(M, N):

    def kernel(block_M=None, block_N=None, threads=None):
        return elementwise_add(M, N, block_M, block_N, "float32", "float32", threads)

    autotuner = AutoTuner.from_kernel(
        kernel=kernel, configs=get_configs(M, N)).set_compile_args(
            out_idx=[-1],
            target="cuda",
        ).set_profile_args(
            supply_type=tilelang.TensorSupplyType.Auto,
            ref_prog=ref_program,
            skip_check=False,
        )
    return autotuner.run(warmup=3, rep=20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--use_autotune", action="store_true", default=False)
    args, _ = parser.parse_known_args()
    M, N = args.m, args.n

    a = torch.randn(M, N, dtype=torch.float32, device="cuda")
    b = torch.randn(M, N, dtype=torch.float32, device="cuda")

    if args.use_autotune:
        result = get_best_config(M, N)
        kernel = result.kernel
    else:
        # Default config
        config = {"block_M": 32, "block_N": 32, "threads": 128}
        kernel = elementwise_add(M, N, **config, in_dtype="float32", out_dtype="float32")

    out = kernel(a, b)
    torch.testing.assert_close(out, ref_program(a, b), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    main()
