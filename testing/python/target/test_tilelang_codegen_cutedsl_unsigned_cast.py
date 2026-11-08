import pytest

import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.engine.lower import lower
from tilelang.cuda.target import normalize_cutedsl_target


def _lower_cutedsl(program):
    if not tvm.runtime.enabled("cuda"):
        pytest.skip("TileLang CuTeDSL codegen requires TVM built with CUDA support.")

    build_cutedsl = tvm.ffi.get_global_func("target.build.tilelang_cutedsl_without_compile", allow_missing=True)
    if build_cutedsl is None:
        pytest.skip("TileLang CuTeDSL backend is not enabled in this build.")

    target = normalize_cutedsl_target({"kind": "cutedsl", "arch": "sm_80"})
    assert target is not None

    with target:
        return lower(program.with_attr("global_symbol", "main"), target=target)


def test_cutedsl_unsigned_widening_cast_zero_extends():
    """CuTeDSL normalizes unsigned elements to signed; the widening cast must
    still zero-extend (uint8 200 -> int32 200, not -56)."""

    @T.prim_func
    def prog(A: T.Tensor((1,), "uint8"), B: T.Tensor((1,), "int32")):
        with T.Kernel(1, threads=1):
            B[0] = T.cast(A[0], "int32")

    artifact = _lower_cutedsl(prog)

    assert "dtype=cutlass.Uint8" in artifact.kernel_source
    assert "cutlass.Int32(" in artifact.kernel_source


def test_cutedsl_unsigned_to_float_cast_zero_extends():
    @T.prim_func
    def prog(A: T.Tensor((1,), "uint8"), B: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=1):
            B[0] = T.cast(A[0], "float32")

    artifact = _lower_cutedsl(prog)

    assert "dtype=cutlass.Uint8" in artifact.kernel_source
    assert "cutlass.Float32(" in artifact.kernel_source


def test_cutedsl_unsigned_to_bfloat16_cast_zero_extends():
    @T.prim_func
    def prog(A: T.Tensor((1,), "uint8"), B: T.Tensor((1,), "bfloat16")):
        with T.Kernel(1, threads=1):
            B[0] = T.cast(A[0], "bfloat16")

    artifact = _lower_cutedsl(prog)

    assert "dtype=cutlass.Uint8" in artifact.kernel_source
    assert "cutlass.BFloat16(" in artifact.kernel_source


def test_cutedsl_unsigned_to_float_cast_preserves_width():
    """uint32 -> float32 has no widening width, but still needs the unsigned
    reinterpretation (0xFFFFFFFF must become 4294967296.0, not -1.0)."""

    @T.prim_func
    def prog(A: T.Tensor((1,), "uint32"), B: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=1):
            B[0] = T.cast(A[0], "float32")

    artifact = _lower_cutedsl(prog)

    assert "dtype=cutlass.Uint32" in artifact.kernel_source
    assert "cutlass.Float32(" in artifact.kernel_source


def test_cutedsl_unsigned_vector_cast_zero_extends():
    @T.prim_func
    def prog(A: T.Tensor((256,), "uint8"), B: T.Tensor((256,), "int32")):
        with T.Kernel(1, threads=64):
            for i in T.Parallel(256):
                B[i] = T.cast(A[i], "int32")

    artifact = _lower_cutedsl(prog)

    source = artifact.kernel_source
    assert "dtype=cutlass.Uint8" in source
    assert "tl.cast_tensor(" in source


def test_cutedsl_unsigned_view_keeps_tir_element_type():
    """The pointer loses the unsigned type; the view must be re-wrapped with
    the TIR element type so widening is a zero-extension."""

    @T.prim_func
    def prog(A: T.Tensor((1,), "uint8"), B: T.Tensor((1,), "int32")):
        with T.Kernel(1, threads=1):
            B[0] = T.cast(A[0], "int32")

    artifact = _lower_cutedsl(prog)

    assert "dtype=cutlass.Uint8" in artifact.kernel_source


def test_cutedsl_unsigned_operations_keep_unsigned_view():
    """Comparisons, shifts and division must not see a signed view either."""

    @T.prim_func
    def prog(A: T.Tensor((1,), "uint8"), B: T.Tensor((1,), "int32")):
        with T.Kernel(1, threads=1):
            B[0] = (
                T.cast(A[0] > 127, "int32")
                + T.cast(A[0] >> 7, "int32")
                + T.cast(A[0] // 3, "int32")
            )

    artifact = _lower_cutedsl(prog)

    source = artifact.kernel_source
    assert "dtype=cutlass.Uint8" in source
    assert ">>" in source
    assert "//" in source


def test_cutedsl_signed_widening_cast_is_unchanged():
    """A signed source must keep sign-extension semantics."""

    @T.prim_func
    def prog(A: T.Tensor((1,), "int8"), B: T.Tensor((1,), "int32")):
        with T.Kernel(1, threads=1):
            B[0] = T.cast(A[0], "int32")

    artifact = _lower_cutedsl(prog)

    assert "dtype=cutlass.Uint8" not in artifact.kernel_source


if __name__ == "__main__":
    tilelang.testing.main()
