"""Tests for CUDA-specific T.fma and T.fmul intrinsics."""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.cuda.target import normalize_cutedsl_target
from tilelang.engine.lower import lower


ROWS = 32
WIDTH = 4


def _make_scalar_program(dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((ROWS,), dtype),
        B: T.Tensor((ROWS,), dtype),
        C: T.Tensor((ROWS,), dtype),
        D: T.Tensor((ROWS,), dtype),
    ):
        with T.Kernel(1, threads=ROWS):
            i = T.get_thread_binding()
            D[i] = T.fma(A[i], B[i], T.fmul(C[i], B[i]))

    return main


def _make_vectorized_program(dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((ROWS, WIDTH), dtype),
        B: T.Tensor((ROWS, WIDTH), dtype),
        C: T.Tensor((ROWS, WIDTH), dtype),
        D: T.Tensor((ROWS, WIDTH), dtype),
    ):
        with T.Kernel(1, threads=ROWS):
            for i, j in T.Parallel(ROWS, WIDTH):
                D[i, j] = T.fma(A[i, j], B[i, j], T.fmul(C[i, j], B[i, j]))

    return main


def _lower_cuda_source(program, arch: str) -> str:
    target = {"kind": "cuda", "arch": arch}
    with tvm.transform.PassContext(), tvm.target.Target(target):
        artifact = tilelang.lower(program, target=target)
    assert artifact.kernel_source is not None
    return artifact.kernel_source


def _lower_cutedsl_source(program) -> str:
    if not tvm.runtime.enabled("cuda"):
        pytest.skip("TileLang CuTeDSL codegen requires TVM built with CUDA support.")
    from tilelang.jit.adapter.cutedsl.checks import check_cutedsl_available

    try:
        check_cutedsl_available()
    except ImportError as err:
        pytest.skip(f"CuTeDSL is not available: {err}")
    build_cutedsl = tvm.ffi.get_global_func("target.build.tilelang_cutedsl_without_compile", allow_missing=True)
    if build_cutedsl is None:
        pytest.skip("TileLang CuTeDSL backend is not enabled in this build.")

    target = normalize_cutedsl_target({"kind": "cutedsl", "arch": "sm_90"})
    assert target is not None
    with target:
        artifact = lower(program.with_attr("global_symbol", "main"), target=target)
    assert artifact.kernel_source is not None
    return artifact.kernel_source


def test_fma_fmul_build_registered_intrinsics():
    x = tvm.tirx.Var("x", "float32")
    y = tvm.tirx.Var("y", "float32")
    z = tvm.tirx.Var("z", "float32")

    assert T.fmul(x, y).op.name == "tl.fmul"
    assert T.fma(x, y, z).op.name == "tl.fma"


@pytest.mark.parametrize("intrinsic", [T.fma, T.fmul])
def test_fma_fmul_reject_integer_inputs(intrinsic):
    x = tvm.tirx.Var("x", "int32")
    args = (x, x, x) if intrinsic is T.fma else (x, x)
    with pytest.raises(TypeError, match="only supports floating-point inputs"):
        intrinsic(*args)


def test_fma_fmul_reject_mixed_dtypes():
    x = tvm.tirx.Var("x", "float32")
    y = tvm.tirx.Var("y", "float16")
    with pytest.raises(ValueError, match="same dtype"):
        T.fmul(x, y)
    with pytest.raises(ValueError, match="same dtype"):
        T.fma(x, x, y)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    ("dtype", "expected_mul", "expected_fma"),
    [
        (T.float32, "__fmul_rn", "__fmaf_rn"),
        (T.float64, "__dmul_rn", "__fma_rn"),
        (T.float16, "__hmul_rn", "__hfma"),
        (T.bfloat16, "__hmul_rn", "__hfma"),
    ],
)
def test_fma_fmul_scalar_cuda_codegen(dtype, expected_mul, expected_fma):
    source = _lower_cuda_source(_make_scalar_program(dtype), "sm_90")
    assert expected_mul in source
    assert expected_fma in source


@tilelang.testing.requires_cuda
def test_fma_fmul_vectorized_cuda_codegen_sm90():
    source = _lower_cuda_source(_make_vectorized_program(), "sm_90")

    assert "float4" in source
    assert source.count("__fmul_rn(") >= WIDTH
    assert source.count("__fmaf_rn(") >= WIDTH
    assert "tl::mul2" not in source
    assert "tl::fma2" not in source


@tilelang.testing.requires_cuda
def test_fma_fmul_vectorized_cuda_codegen_sm100():
    source = _lower_cuda_source(_make_vectorized_program(), "sm_100")

    assert source.count("tl::mul2(") >= WIDTH // 2
    assert source.count("tl::fma2(") >= WIDTH // 2


def test_fma_fmul_vectorized_cutedsl_codegen():
    source = _lower_cutedsl_source(_make_vectorized_program())

    assert source.count("tl.ieee_fmul(") >= WIDTH
    assert source.count("tl.ieee_fmaf(") >= WIDTH


@tilelang.testing.requires_cuda
def test_fma_fmul_vectorized_cuda_result():
    kernel = tilelang.compile(_make_vectorized_program(), out_idx=[3], target="cuda")
    a = torch.randn((ROWS, WIDTH), device="cuda", dtype=torch.float32)
    b = torch.randn((ROWS, WIDTH), device="cuda", dtype=torch.float32)
    c = torch.randn((ROWS, WIDTH), device="cuda", dtype=torch.float32)

    expected = torch.addcmul(c * b, a, b)
    actual = kernel(a, b, c)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
