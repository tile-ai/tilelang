import tilelang
import tilelang.language as T
import torch
import tilelang.testing


# add decorator @tilelang.jit if you want to return a torch function
# @tilelang.jit
def tilelang_copy(M, N, block_M, block_N, dtype="float16"):

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            for i, j in T.Parallel(block_M, block_N):
                B[by * block_M + i, bx * block_N + j] = A[by * block_M + i, bx * block_N + j]

    return main


def run_tilelang_copy(M=1024, N=1024, block_M=128, block_N=128, dtype="float16"):
    program = tilelang_copy(M, N, block_M, block_N, dtype)
    kernel = tilelang.compile(
        program,
        out_idx=[1],
        target="cuda",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True
        })
    a = torch.randn(M, N, device="cuda", dtype=getattr(torch, dtype))
    b = kernel(a)
    torch.testing.assert_close(b, a, rtol=1e-2, atol=1e-2)


def test_tilelang_copy():
    run_tilelang_copy(M=1024, N=1024, block_M=128, block_N=128)
    run_tilelang_copy(M=1024, N=576, block_M=32, block_N=576)
    run_tilelang_copy(M=1024, N=576, block_M=32, block_N=576, dtype="float")


def tilelang_copy_with_stride(M, N, NN, block_M, block_N, dtype="float16"):

    @T.prim_func
    def main(
            A: T.StridedTensor((M, N), (NN, 1), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            for i, j in T.Parallel(block_M, block_N):
                B[by * block_M + i, bx * block_N + j] = A[by * block_M + i, bx * block_N + j]

    return main


def run_tilelang_copy_with_stride(M=1024,
                                  N=1024,
                                  NN=2048,
                                  block_M=128,
                                  block_N=128,
                                  dtype="float16"):
    if isinstance(NN, int):
        assert NN > N, "NN must be greater than N"
    program = tilelang_copy_with_stride(M, N, NN, block_M, block_N, dtype)
    kernel = tilelang.compile(
        program,
        out_idx=[1],
        target="cuda",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        })
    if isinstance(NN, T.Var):
        NN = N * 2
    a = torch.randn(M, NN, device="cuda", dtype=getattr(torch, dtype))
    b = kernel(a[:, :N])
    torch.testing.assert_close(b, a[:, :N], rtol=1e-2, atol=1e-2)


def test_tilelang_copy_with_stride():
    run_tilelang_copy_with_stride(M=1024, N=1024, NN=2048, block_M=128, block_N=128)
    run_tilelang_copy_with_stride(M=1024, N=1024, NN=T.dynamic("NN"), block_M=128, block_N=128)


def tilelang_copy_bufferload(num_tokens, dtype="float16"):

    @T.prim_func
    def main(
            indices: T.Tensor((num_tokens,), "int32"),
            x: T.Tensor((num_tokens,), dtype),
    ):
        with T.Kernel(num_tokens, threads=32) as pid:
            idx = T.alloc_local([1], "int32")
            T.copy(indices[pid], idx[0])
            x[idx[0]] = x[idx[0]] + 1

    return main


def run_tilelang_copy_bufferload(num_tokens=128, dtype="float16"):
    program = tilelang_copy_bufferload(num_tokens, dtype)
    # test compilation only
    tilelang.compile(
        program,
        out_idx=[1],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True
        })


def test_tilelang_copy_bufferload():
    run_tilelang_copy_bufferload(num_tokens=128)


def tilelang_copy_buffer_load_with_parallel(M, N, block_M, block_N, dtype="float16"):

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            for i, j in T.Parallel(block_M, block_N):
                T.copy(A[by * block_M + i, bx * block_N + j], B[by * block_M + i, bx * block_N + j])

    return main


def run_tilelang_copy_buffer_load_with_parallel(M=1024,
                                                N=1024,
                                                block_M=128,
                                                block_N=128,
                                                dtype="float16"):
    program = tilelang_copy_buffer_load_with_parallel(M, N, block_M, block_N, dtype)
    kernel = tilelang.compile(
        program,
        out_idx=[1],
        target="cuda",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True
        })
    a = torch.randn(M, N, device="cuda", dtype=getattr(torch, dtype))
    b = kernel(a)
    torch.testing.assert_close(b, a, rtol=1e-2, atol=1e-2)


def test_tilelang_copy_buffer_load_with_parallel():
    run_tilelang_copy_buffer_load_with_parallel(M=1024, N=1024, block_M=128, block_N=128)


if __name__ == "__main__":
    tilelang.testing.main()
