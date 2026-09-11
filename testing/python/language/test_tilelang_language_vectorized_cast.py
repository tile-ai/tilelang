import pytest
import torch
import tilelang.testing
import tilelang.language as T
from tilelang import tvm
from tvm import tirx


@tilelang.jit
def vectorized_cast_kernel(M: int, dtype_A: str, dtype_B: str):
    assert M % 256 == 0

    @T.prim_func
    def main(
        A: T.Tensor[(M,), dtype_A],  # noqa: F821
        B: T.Tensor[(M,), dtype_B],  # noqa: F821
    ):
        with T.Kernel(1, threads=128):
            T.copy(A, B)

    return main


@tilelang.jit
def parallel_vectorized_cast_kernel(M: int, dtype_A: str, dtype_B: str):
    assert M % 256 == 0

    @T.prim_func
    def main(
        A: T.Tensor[(M,), dtype_A],  # noqa: F821
        B: T.Tensor[(M,), dtype_B],  # noqa: F821
    ):
        with T.Kernel(1, threads=128):
            A_local = T.alloc_fragment((M,), dtype_A)
            B_local = T.alloc_fragment((M,), dtype_B)

            T.copy(A, A_local)
            for i in T.Parallel(M):
                B_local[i] = A_local[i]
            T.copy(B_local, B)

    return main


def run_vectorized_cast(src_dtype: T.dtype, dst_dtype: T.dtype, check_str: str, lanes: int = 2):
    """Run the vectorized cast kernel and check the correctness.
    Args:
        src_dtype: The source data type.
        dst_dtype: The destination data type.
        check_str: Used to ensure vectorized cast is used.
        lanes: The number of lanes of the source and destination data types.
    """

    M = 128 * lanes
    kernel = vectorized_cast_kernel(M, src_dtype, dst_dtype)
    kernel_parallel = parallel_vectorized_cast_kernel(M, src_dtype, dst_dtype)

    code = kernel.get_kernel_source()
    code_parallel = kernel_parallel.get_kernel_source()
    assert check_str in code and check_str in code_parallel, f"Cast {src_dtype} to {dst_dtype} with {lanes=} is not vectorized!"

    # Requires torch >= 2.8
    if src_dtype == T.float8_e8m0fnu or dst_dtype == T.float8_e8m0fnu:
        return

    if src_dtype == T.float4_e2m1fn or dst_dtype == T.float4_e2m1fn:
        return

    A_float = torch.randn(M, dtype=torch.float32, device="cuda")
    A = A_float.to(src_dtype.as_torch())

    A = A_float.to(src_dtype.as_torch())
    B = torch.zeros(M, dtype=dst_dtype.as_torch(), device="cuda")
    C = torch.zeros(M, dtype=dst_dtype.as_torch(), device="cuda")

    kernel(A, B)
    kernel_parallel(A, C)

    torch.testing.assert_close(A.to(dst_dtype.as_torch()), B)
    torch.testing.assert_close(A.to(dst_dtype.as_torch()), C)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "src_dtype, dst_dtype, check_str, lanes",
    [
        (T.float32, T.float16, "__float22half2_rn", 2),
        (T.float32, T.float16, "__float22half2_rn", 4),
        (T.float16, T.float32, "__half22float2", 2),
        (T.float16, T.float32, "__half22float2", 4),
        (T.float32, T.bfloat16, "__float22bfloat162_rn", 2),
        (T.float32, T.bfloat16, "__float22bfloat162_rn", 4),
        (T.bfloat16, T.float32, "__bfloat1622float2", 2),
        (T.bfloat16, T.float32, "__bfloat1622float2", 4),
    ],
)
def test_vectorized_cast(src_dtype, dst_dtype, check_str, lanes):
    run_vectorized_cast(src_dtype, dst_dtype, check_str, lanes)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 9)
@pytest.mark.parametrize(
    "src_dtype, dst_dtype, check_str, lanes",
    [
        # FP8 <-> FP32
        (T.float32, T.float8_e4m3fn, "__nv_cvt_float2_to_fp8x2", 2),
        (T.float32, T.float8_e4m3fn, "__nv_cvt_float2_to_fp8x2", 4),
        (T.float32, T.float8_e5m2, "__nv_cvt_float2_to_fp8x2", 2),
        (T.float32, T.float8_e5m2, "__nv_cvt_float2_to_fp8x2", 4),
        (T.float8_e4m3fn, T.float32, "__tl_cvt_fp8x2_to_float2", 2),
        (T.float8_e4m3fn, T.float32, "__tl_cvt_fp8x2_to_float2", 4),
        (T.float8_e5m2, T.float32, "__tl_cvt_fp8x2_to_float2", 2),
        (T.float8_e5m2, T.float32, "__tl_cvt_fp8x2_to_float2", 4),
        # FP8 <-> Half
        (T.float8_e4m3fn, T.float16, "__tl_cvt_fp8x2_to_half2", 2),
        (T.float8_e4m3fn, T.float16, "__tl_cvt_fp8x2_to_half2", 4),
        (T.float8_e5m2, T.float16, "__tl_cvt_fp8x2_to_half2", 2),
        (T.float16, T.float8_e4m3fn, "__tl_cvt_half2_to_fp8x2", 2),
        (T.float16, T.float8_e4m3fn, "__tl_cvt_half2_to_fp8x2", 4),
        (T.float16, T.float8_e5m2, "__tl_cvt_half2_to_fp8x2", 2),
        # E8M0 <-> BFloat16
        (T.float8_e8m0fnu, T.bfloat16, "__tl_cvt_e8m0x2_to_bfloat162", 2),
        (T.bfloat16, T.float8_e8m0fnu, "__tl_cvt_bfloat162_to_e8m0x2", 2),
        # Float -> E8M0
        (T.float32, T.float8_e8m0fnu, "__tl_cvt_float2_to_e8m0x2", 2),
        # Double -> E8M0
        (T.float64, T.float8_e8m0fnu, "__tl_cvt_double2_to_e8m0x2", 2),
    ],
)
def test_vectorized_cast_fp8(src_dtype, dst_dtype, check_str, lanes):
    run_vectorized_cast(src_dtype, dst_dtype, check_str, lanes)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
@pytest.mark.parametrize(
    "src_dtype, dst_dtype, check_str, lanes",
    [
        # FP4 <-> Half
        (T.float4_e2m1fn, T.float16, "__tl_cvt_fp4x2_to_half2", 2),
        (T.float16, T.float4_e2m1fn, "__tl_cvt_half2_to_fp4x2", 2),
        # FP4 <-> Float
        (T.float4_e2m1fn, T.float32, "__tl_cvt_fp4x2_to_float2", 2),
        (T.float32, T.float4_e2m1fn, "__tl_cvt_float2_to_fp4x2", 2),
        # FP4 <-> Double
        (T.float4_e2m1fn, T.float64, "__tl_cvt_fp4x2_to_double2", 2),
        (T.float64, T.float4_e2m1fn, "__tl_cvt_double2_to_fp4x2", 2),
        # FP4 <-> BFloat16
        (T.float4_e2m1fn, T.bfloat16, "__tl_cvt_fp4x2_to_bfloat162", 2),
        (T.bfloat16, T.float4_e2m1fn, "__tl_cvt_bfloat162_to_fp4x2", 2),
    ],
)
def test_vectorized_cast_fp4(src_dtype, dst_dtype, check_str, lanes):
    run_vectorized_cast(src_dtype, dst_dtype, check_str, lanes)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
@pytest.mark.parametrize(
    "src_dtype, dst_dtype, check_str, lanes",
    [
        # FP8 <-> BFloat16 (PTX cvt, Blackwell+)
        (T.float8_e4m3fn, T.bfloat16, "__tl_cvt_e4m3x2_to_bfloat162", 2),
        (T.float8_e4m3fn, T.bfloat16, "__tl_cvt_e4m3x2_to_bfloat162", 4),
        (T.float8_e5m2, T.bfloat16, "__tl_cvt_e5m2x2_to_bfloat162", 2),
        (T.bfloat16, T.float8_e4m3fn, "__tl_cvt_bfloat162_to_fp8x2", 2),
        (T.bfloat16, T.float8_e4m3fn, "__tl_cvt_bfloat162_to_fp8x2", 4),
        (T.bfloat16, T.float8_e5m2, "__tl_cvt_bfloat162_to_fp8x2", 2),
    ],
)
def test_vectorized_cast_fp8_bf16(src_dtype, dst_dtype, check_str, lanes):
    run_vectorized_cast(src_dtype, dst_dtype, check_str, lanes)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("lanes", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("destination", ["float8_e4m3", "float8_e4m3fn"])
def test_fp4_fp32_fp8_codegen(lanes, destination):
    build = tvm.get_global_func("target.build.tilelang_cuda_without_compile", allow_missing=True)
    if build is None:
        pytest.skip("TileLang was built without the CUDA code generator")
    suffix = f"x{lanes}" if lanes > 1 else ""
    value = tirx.Var("value", "float4_e2m1fn" + suffix)
    converted = tirx.Cast(destination + suffix, tirx.Cast("float32" + suffix, value))
    func = tirx.PrimFunc([value], tirx.Evaluate(converted))
    func = func.with_attr("global_symbol", "fp4_fp8_cast")
    func = func.with_attr("calling_conv", tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    source = build(tvm.IRModule({"fp4_fp8_cast": func}), tvm.target.Target("cuda")).inspect_source()
    assert source.count("ConvertE2M1x4ToE4M3x4(") == max(1, lanes // 4)
    assert "__tl_cvt_fp4x2_to_float2" not in source
    assert "__nv_cvt_float2_to_fp8x2" not in source

    if lanes == 2:
        for annotation_owner in ("inner", "outer"):
            annotations = {"sat": tirx.IntImm("bool", 1)}
            intermediate = tirx.Cast("float32" + suffix, value, annotations if annotation_owner == "inner" else None)
            annotated = tirx.Cast(destination + suffix, intermediate, annotations if annotation_owner == "outer" else None)
            annotated_func = func.with_body(tirx.Evaluate(annotated))
            annotated_source = build(tvm.IRModule({"fp4_fp8_cast": annotated_func}), tvm.target.Target("cuda")).inspect_source()
            assert "ConvertE2M1x4ToE4M3x4" not in annotated_source
            assert "__tl_cvt_fp4x2_to_float2" in annotated_source
            assert "__nv_cvt_float2_to_fp8x2" in annotated_source


def fp4_fp32_fp8_kernel(elements, lanes):
    tile = 128 * lanes

    @T.prim_func
    def main(A: T.Tensor((elements,), "float4_e2m1fn"), B: T.Tensor((elements,), "float8_e4m3fn")):
        with T.Kernel(elements // tile, threads=128) as block:
            for i in T.Parallel(tile):
                index = block * tile + i
                B[index] = T.Cast("float8_e4m3fn", T.Cast("float32", A[index]))

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 9)
@pytest.mark.skipif(not hasattr(torch, "float4_e2m1fn_x2"), reason="PyTorch packed FP4 dtype is unavailable")
@pytest.mark.parametrize("lanes", [1, 2, 4, 8])
def test_fp4_fp32_fp8_exact(lanes):
    # Every four-nibble word exercises magnitude, sign, and lane ordering.
    elements = 65536 * 4
    kernel = tilelang.compile(fp4_fp32_fp8_kernel(elements, lanes), pass_configs={"tirx.disable_vectorize": lanes == 1})
    assert "ConvertE2M1x4ToE4M3x4" in kernel.get_kernel_source()
    words = torch.arange(65536, dtype=torch.int32, device="cuda")[:, None]
    packed = ((words >> torch.tensor([0, 8], device="cuda")) & 255).to(torch.uint8).flatten()
    nibbles = ((words >> torch.tensor([0, 4, 8, 12], device="cuda")) & 15).flatten().long()
    values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device="cuda",
    )
    expected = values[nibbles].to(torch.float8_e4m3fn).view(torch.uint8)
    output = torch.empty(elements, dtype=torch.float8_e4m3fn, device="cuda")
    kernel(packed.view(torch.float4_e2m1fn_x2), output)
    assert torch.equal(output.view(torch.uint8), expected)


if __name__ == "__main__":
    tilelang.testing.main()
