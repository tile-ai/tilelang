import tilelang
import tilelang.language as T
import torch
import tilelang.testing

print(torch.__version__)


# add decorator @tilelang.jit if you want to return a torch function
# @tilelang.jit
def tilelang_copy(M, N, block_M, block_N, src_dtype=T.float16, dst_dtype=T.float16):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), src_dtype),
        B: T.Tensor((M, N), dst_dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            T.copy(
                A[by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N],
                B[by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N],
            )

    return main


def run_tilelang_copy(M=1024, N=1024, block_M=128, block_N=128, dtype=T.float16):
    program = tilelang_copy(M, N, block_M, block_N, src_dtype=dtype, dst_dtype=dtype)
    kernel = tilelang.compile(
        program,
        out_idx=[1],
        target="cuda",
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True, tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True},
    )
    source = kernel.get_kernel_source()
    print(source)
    a = torch.randn(M, N, device="cuda", dtype=getattr(torch, dtype))
    b = kernel(a)
    torch.testing.assert_close(b, a, rtol=1e-2, atol=1e-2)


def test_tilelang_copy():
    run_tilelang_copy(M=1024, N=1024, block_M=128, block_N=128)
    run_tilelang_copy(M=1024, N=576, block_M=32, block_N=576)
    run_tilelang_copy(M=1024, N=576, block_M=32, block_N=576, dtype=T.float32)


def tilelang_copy_with_stride(M, N, NN, block_M, block_N, dtype=T.float16):
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


def run_tilelang_copy_with_stride(M=1024, N=1024, NN=2048, block_M=128, block_N=128, dtype=T.float16):
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
        },
    )
    if isinstance(NN, T.Var):
        NN = N * 2
    a = torch.randn(M, NN, device="cuda", dtype=getattr(torch, dtype))
    b = kernel(a[:, :N])
    torch.testing.assert_close(b, a[:, :N], rtol=1e-2, atol=1e-2)


def test_tilelang_copy_with_stride():
    run_tilelang_copy_with_stride(M=1024, N=1024, NN=2048, block_M=128, block_N=128)
    run_tilelang_copy_with_stride(M=1024, N=1024, NN=T.dynamic("NN"), block_M=128, block_N=128)


def tilelang_copy_bufferload(num_tokens, dtype=T.float16):
    @T.prim_func
    def main(
        indices: T.Tensor((num_tokens,), T.int32),
        x: T.Tensor((num_tokens,), dtype),
    ):
        with T.Kernel(num_tokens, threads=32) as pid:
            idx = T.alloc_local([1], T.int32)
            T.copy(indices[pid], idx[0])
            x[idx[0]] = x[idx[0]] + 1

    return main


def run_tilelang_copy_bufferload(num_tokens=128, dtype=T.float16):
    program = tilelang_copy_bufferload(num_tokens, dtype)
    # test compilation only
    tilelang.compile(
        program,
        out_idx=[1],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True, tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True},
    )


def test_tilelang_copy_bufferload():
    run_tilelang_copy_bufferload(num_tokens=128)


def tilelang_copy_buffer_load_with_parallel(M, N, block_M, block_N, dtype=T.float16):
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


def run_tilelang_copy_buffer_load_with_parallel(M=1024, N=1024, block_M=128, block_N=128, dtype=T.float16):
    program = tilelang_copy_buffer_load_with_parallel(M, N, block_M, block_N, dtype)
    kernel = tilelang.compile(
        program,
        out_idx=[1],
        target="cuda",
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True, tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True},
    )
    a = torch.randn(M, N, device="cuda", dtype=getattr(torch, dtype))
    b = kernel(a)
    torch.testing.assert_close(b, a, rtol=1e-2, atol=1e-2)


def test_tilelang_copy_buffer_load_with_parallel():
    run_tilelang_copy_buffer_load_with_parallel(M=1024, N=1024, block_M=128, block_N=128)


def run_tilelang_copy_fp8_e8m0(M=1024, N=1024, block_M=128, block_N=128, src_dtype=T.float8_e8m0fnu, dst_dtype=T.float8_e8m0fnu):
    program = tilelang_copy(M, N, block_M, block_N, src_dtype=src_dtype, dst_dtype=dst_dtype)
    kernel = tilelang.compile(
        program,
        out_idx=[1],
    )
    source = kernel.get_kernel_source()
    assert "fp8_e8_t" in source
    dummy_input = torch.randint(0, 100, (M, N), device="cuda", dtype=torch.int8).view(torch.float8_e8m0fnu)
    output = kernel(dummy_input)
    assert output is not None


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
def test_tilelang_copy_fp8_e8m0():
    run_tilelang_copy_fp8_e8m0(src_dtype=T.float8_e8m0fnu, dst_dtype=T.float8_e8m0fnu)


def run_tilelang_copy_fp4(M=1024, N=1024, block_M=128, block_N=128, src_dtype=T.float4_e2m1fn, dst_dtype=T.float4_e2m1fn):
    program = tilelang_copy(M, N, block_M, block_N, src_dtype=src_dtype, dst_dtype=dst_dtype)
    kernel = tilelang.compile(
        program,
        out_idx=[1],
    )
    source = kernel.get_kernel_source()
    assert "fp4_e2_t" in source
    # For FP4, use same shape as kernel expects, since int8 is used as storage type
    dummy_input = torch.randint(0, 100, (M, N), device="cuda", dtype=torch.int8)
    output = kernel(dummy_input)
    assert output is not None


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
def test_tilelang_copy_fp4():
    run_tilelang_copy_fp4(src_dtype=T.float4_e2m1fn, dst_dtype=T.float4_e2m1fn)
    run_tilelang_copy_fp4(src_dtype=T.float4_e2m1fn, dst_dtype=T.float16)
    run_tilelang_copy_fp4(src_dtype=T.float4_e2m1fn, dst_dtype=T.bfloat16)


if __name__ == "__main__":
    tilelang.testing.main()
