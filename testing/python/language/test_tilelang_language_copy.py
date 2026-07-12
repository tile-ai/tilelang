import tilelang
import tilelang.language as T
import torch
import tilelang.testing
from tilelang import tvm
from tilelang.layout import make_gemm_fragment_8x8_transposed

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
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
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


def run_tilelang_copy_cross_dtype(M=256, N=256, block_M=128, block_N=128, src_dtype=T.float16, dst_dtype=T.bfloat16):
    program = tilelang_copy(M, N, block_M, block_N, src_dtype=src_dtype, dst_dtype=dst_dtype)
    kernel = tilelang.compile(program, out_idx=[1])
    a = torch.randn(M, N, device="cuda", dtype=getattr(torch, src_dtype))
    b = kernel(a)
    torch.testing.assert_close(b, a.to(getattr(torch, dst_dtype)), rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda
def test_tilelang_copy_cross_dtype():
    run_tilelang_copy_cross_dtype(src_dtype=T.float16, dst_dtype=T.bfloat16)
    run_tilelang_copy_cross_dtype(src_dtype=T.bfloat16, dst_dtype=T.float16)


def tilelang_copy_oob_safe_value(
    M,
    N,
    padded_M,
    padded_N,
    block_M,
    block_N,
    src_dtype=T.float32,
    dst_dtype=T.float32,
    pad_value=None,
):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), src_dtype),
        B: T.Tensor((padded_M, padded_N), dst_dtype),
    ):
        with T.Kernel(T.ceildiv(padded_N, block_N), T.ceildiv(padded_M, block_M), threads=128) as (bx, by):
            if pad_value is not None:
                T.annotate_safe_value({A: pad_value})
            T.copy(
                A[by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N],
                B[by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N],
            )

    return main


def run_tilelang_copy_oob_safe_value(
    M=70,
    N=75,
    block_M=32,
    block_N=32,
    src_dtype=T.float32,
    dst_dtype=T.float32,
    pad_value=None,
):
    padded_M = tilelang.cdiv(M, block_M) * block_M
    padded_N = tilelang.cdiv(N, block_N) * block_N
    program = tilelang_copy_oob_safe_value(
        M,
        N,
        padded_M,
        padded_N,
        block_M,
        block_N,
        src_dtype,
        dst_dtype,
        pad_value,
    )
    kernel = tilelang.compile(
        program,
        out_idx=[1],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    a = torch.randn(M, N, device="cuda", dtype=getattr(torch, src_dtype))
    b = kernel(a)
    torch_dst_dtype = getattr(torch, dst_dtype)
    fallback_value = 0 if pad_value is None else pad_value
    ref_b = torch.full((padded_M, padded_N), fallback_value, device="cuda", dtype=torch_dst_dtype)
    ref_b[:M, :N] = a.to(torch_dst_dtype)
    torch.testing.assert_close(b[:M, :N], ref_b[:M, :N], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(b[M:, :N], ref_b[M:, :N], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(b[:M, N:], ref_b[:M, N:], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(b[M:, N:], ref_b[M:, N:], rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda
def test_tilelang_copy_oob_safe_value_uses_annotation():
    run_tilelang_copy_oob_safe_value(pad_value=-99)


@tilelang.testing.requires_cuda
def test_tilelang_copy_oob_safe_value_defaults_to_zero():
    run_tilelang_copy_oob_safe_value(pad_value=None)


@tilelang.testing.requires_cuda
def test_tilelang_copy_oob_safe_value_casts_to_destination_dtype():
    run_tilelang_copy_oob_safe_value(src_dtype=T.float32, dst_dtype=T.float16, pad_value=-99)


def tilelang_elementwise_copy_oob_safe_value(M, N, padded_M, padded_N, dtype=T.float32, pad_value=-99):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((padded_M, padded_N), dtype),
    ):
        with T.Kernel(1, threads=128):
            T.annotate_safe_value({A: pad_value})
            for i, j in T.Parallel(padded_M, padded_N):
                T.copy(A[i, j], B[i, j])

    return main


@tilelang.testing.requires_cuda
def test_tilelang_elementwise_copy_oob_safe_value_non_regression():
    M, N = 70, 75
    padded_M, padded_N = 96, 96
    program = tilelang_elementwise_copy_oob_safe_value(M, N, padded_M, padded_N)
    kernel = tilelang.compile(
        program,
        out_idx=[1],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    a = torch.randn(M, N, device="cuda", dtype=torch.float32)
    b = kernel(a)
    ref_b = torch.full((padded_M, padded_N), -99, device="cuda", dtype=torch.float32)
    ref_b[:M, :N] = a
    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


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
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
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
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
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
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    a = torch.randn(M, N, device="cuda", dtype=getattr(torch, dtype))
    b = kernel(a)
    torch.testing.assert_close(b, a, rtol=1e-2, atol=1e-2)


def test_tilelang_copy_buffer_load_with_parallel():
    run_tilelang_copy_buffer_load_with_parallel(M=1024, N=1024, block_M=128, block_N=128)


def tilelang_copy_shape_mismatched(M, N, src_dtype=T.float16, dst_dtype=T.float16):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), src_dtype),
        B: T.Tensor((M, N), dst_dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(1, threads=128):
            T.copy(A[:, :2], B[:, :3])

    return main


def run_tilelang_copy_shape_mismatched(M=1024, N=1024, dtype=T.float16):
    program = tilelang_copy_shape_mismatched(M, N, src_dtype=dtype, dst_dtype=dtype)
    kernel = tilelang.compile(
        program,
        out_idx=[1],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    a = torch.randn(M, N, device="cuda", dtype=getattr(torch, dtype))
    b = kernel(a)
    torch.testing.assert_close(b[:, :1], a[:, :1], rtol=1e-2, atol=1e-2)


def test_tilelang_copy_shape_mismatched():
    run_tilelang_copy_shape_mismatched(M=128, N=128)


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
    dummy_input = torch.randint(0, 100, (M, N // 2), device="cuda", dtype=torch.int8)
    output = kernel(dummy_input)
    if src_dtype == dst_dtype:
        assert torch.allclose(output.view(torch.int8), dummy_input)
    assert output is not None


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
def test_tilelang_copy_fp4():
    run_tilelang_copy_fp4(src_dtype=T.float4_e2m1fn, dst_dtype=T.float4_e2m1fn)
    run_tilelang_copy_fp4(src_dtype=T.float4_e2m1fn, dst_dtype=T.float16)
    run_tilelang_copy_fp4(src_dtype=T.float4_e2m1fn, dst_dtype=T.bfloat16)


@tilelang.testing.requires_cuda
def test_tilelang_copy_uses_stmatrix_m16n8_for_sm100_int8_shared_store():
    @T.prim_func
    def main(A: T.Tensor((16, 8), T.int8)):
        with T.Kernel(1, threads=32):
            frag = T.alloc_fragment((16, 8), T.int8)
            smem = T.alloc_shared((16, 8), T.int8)

            T.annotate_layout({frag: make_gemm_fragment_8x8_transposed().repeat([2, 1], repeat_on_thread=False)})

            for i, j in T.Parallel(16, 8):
                frag[i, j] = A[i, j]

            T.copy(frag, smem)

    target = {"kind": "cuda", "arch": "sm_100a"}
    with tvm.target.Target(target):
        artifact = tilelang.lower(main, target=target)

    assert "tl::ptx_stmatrix_m16n8_x1_trans" in artifact.kernel_source


if __name__ == "__main__":
    tilelang.testing.main()
