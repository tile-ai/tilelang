import argparse
import torch
import tilelang
import tilelang.language as T


def ref_program(x, y):
    return x + y


@tilelang.jit
def elementwise_add(A, B, block_M, block_N, in_dtype, out_dtype, threads):
    M, N = T.const("M, N")

    A: T.Tensor((M, N), in_dtype)
    B: T.Tensor((M, N), in_dtype)
    C = T.empty((M, N), out_dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_N), in_dtype)
        B_shared = T.alloc_shared((block_M, block_N), in_dtype)
        C_local = T.alloc_fragment((block_M, block_N), out_dtype)
        C_shared = T.alloc_shared((block_M, block_N), out_dtype)

        T.copy(A[by * block_M, bx * block_N], A_shared)
        T.copy(B[by * block_M, bx * block_N], B_shared)
        for local_y, local_x in T.Parallel(block_M, block_N):
            C_local[local_y, local_x] = A_shared[local_y, local_x] + B_shared[local_y, local_x]
        T.copy(C_local, C_shared)
        T.copy(C_shared, C[by * block_M, bx * block_N])

    return C


def main(M=1024, N=1024):
    a = torch.randn(M, N, dtype=torch.float32, device="cuda")
    b = torch.randn(M, N, dtype=torch.float32, device="cuda")

    out = elementwise_add(a, b, block_M=32, block_N=32, threads=128, in_dtype=T.float32, out_dtype=T.float32)

    torch.testing.assert_close(out, ref_program(a, b), rtol=1e-2, atol=1e-2)


def run_regression_perf():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    args, _ = parser.parse_known_args()
    M, N = args.m, args.n
    a = torch.randn(M, N, dtype=torch.float32, device="cuda")
    b = torch.randn(M, N, dtype=torch.float32, device="cuda")
    config = {"block_M": 32, "block_N": 32, "threads": 128}
    from tilelang.profiler import do_bench

    return do_bench(lambda: elementwise_add(a, b, **config, in_dtype="float32", out_dtype="float32"), backend="cupti")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    args, _ = parser.parse_known_args()
    main(args.m, args.n)
