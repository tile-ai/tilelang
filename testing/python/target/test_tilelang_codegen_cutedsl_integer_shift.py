import pytest

import tilelang.language as T
import tilelang.testing
import tilelang
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


def test_cutedsl_codegen_promotes_narrow_shift_before_wide_cast():
    @T.prim_func
    def prog(A: T.Tensor((1,), "uint16"), B: T.Tensor((1,), "uint32")):
        with T.Kernel(1, threads=1):
            local = T.alloc_local((1,), "uint16")
            local[0] = A[0]
            B[0] = T.cast(local[0] << 20, T.uint32)

    artifact = _lower_cutedsl(prog)

    assert "cutlass.Uint32(local[0]) << cutlass.Uint16(20)" in artifact.kernel_source
    assert "cutlass.Uint32(cutlass.Uint16((local[0] << cutlass.Uint16(20))))" not in artifact.kernel_source


def test_cutedsl_codegen_does_not_promote_shift_across_signedness():
    @T.prim_func
    def prog(A: T.Tensor((1,), "int8"), B: T.Tensor((1,), "uint16")):
        with T.Kernel(1, threads=1):
            local = T.alloc_local((1,), "int8")
            local[0] = A[0]
            B[0] = T.cast(local[0] >> 1, T.uint16)

    artifact = _lower_cutedsl(prog)

    assert "cutlass.Uint16(local[0]) >>" not in artifact.kernel_source


def test_cutedsl_codegen_vector_div_uses_lane_path_with_fastmath():
    @T.prim_func
    def prog(A: T.Tensor((4,), "float32"), B: T.Tensor((4,), "float32"), C: T.Tensor((4,), "float32")):
        with T.Kernel(1, threads=1):
            idx = T.Ramp(0, 1, 4)
            denom = B[idx] + T.Broadcast(T.float32(1), 4)
            C[idx] = A[idx] / denom

    config_key = tilelang.PassConfigKey.TL_ENABLE_FAST_MATH.value
    with tvm.transform.PassContext(config={config_key: True}):
        artifact = _lower_cutedsl(prog)

    assert "fastmath=True" not in artifact.kernel_source
    assert "tl.divf(" in artifact.kernel_source


def test_cutedsl_codegen_canonicalizes_floor_math_call():
    @T.prim_func
    def prog(A: T.Tensor((1,), "float32"), B: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=1):
            B[0] = T.floor(A[0] / T.float32(2))

    artifact = _lower_cutedsl(prog)

    assert "tl.floor(" in artifact.kernel_source
    assert "floorf(" not in artifact.kernel_source


def test_cutedsl_codegen_local_vector_slice_uses_lane_access():
    @T.prim_func
    def prog(A: T.Tensor((4,), "float32"), C: T.Tensor((4,), "float32")):
        with T.Kernel(1, threads=1):
            local = T.alloc_local((4,), "float32")
            idx = T.Ramp(0, 1, 4)
            local[idx] = A[idx]
            C[idx] = local[idx] + T.Broadcast(T.float32(1), 4)

    artifact = _lower_cutedsl(prog)

    assert "tl.make_tensor_at_offset(local.iterator" not in artifact.kernel_source
    assert "local[0]" in artifact.kernel_source
    assert "local[3]" in artifact.kernel_source


if __name__ == "__main__":
    tilelang.testing.main()
