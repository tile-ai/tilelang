"""CuTeDSL codegen: vector values are tensors and loop bounds are statements-free.

Both regressions were found by ``examples/dequantize_gemm`` and
``examples/blocksparse_attention``: a vectorized shift whose amount is a ramp
printed a Python tuple, and a loop bound that lowered to a ``Let`` printed its
binding inside ``range(...)``.
"""

import pytest

import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.engine.lower import lower
from tilelang.cuda.target import normalize_cutedsl_target

pytestmark = tilelang.testing.requires_cuda.marks()


def _lower_cutedsl(prog):
    build_cutedsl = tvm.ffi.get_global_func("target.build.tilelang_cutedsl_without_compile", allow_missing=True)
    if build_cutedsl is None:
        pytest.skip("TileLang CuTeDSL backend is not enabled in this build")
    target = normalize_cutedsl_target({"kind": "cutedsl", "arch": "sm_90"})
    with target:
        return lower(prog.with_attr("global_symbol", "main"), target=target)


def test_cutedsl_codegen_vector_shift_takes_tensor_operand():
    @T.prim_func
    def prog(A: T.Tensor((64,), "uint8"), B: T.Tensor((64,), "uint8")):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(32):
                for v in T.vectorized(2):
                    B[i * 2 + v] = A[i] >> T.Cast("uint8", v * 4)

    source = _lower_cutedsl(prog).kernel_source
    # The shift amount is lane-dependent, i.e. a ramp. CuTeDSL vector values are
    # tensors, so it must be materialized instead of printed as a tuple.
    assert "_ramp" in source
    assert ">> (" not in source


def test_cutedsl_codegen_hoists_loop_bound_lets():
    @T.prim_func
    def prog(A: T.Tensor((64,), "int32"), B: T.Tensor((64,), "int32"), N: T.int32):
        with T.Kernel(1, threads=32):
            for k in T.serial(T.floormod(64, N)):
                for i in T.Parallel(32):
                    B[i] = B[i] + A[k * 32 + i]

    source = _lower_cutedsl(prog).kernel_source
    # A floormod with an unknown-sign divisor lowers to a Let expression. Its
    # binding is a statement, so it must precede the loop header rather than be
    # spliced inside `range(...)`.
    loop_lines = [line for line in source.splitlines() if "in range(" in line]
    assert loop_lines
    for line in loop_lines:
        assert line.rstrip().endswith("):"), line


if __name__ == "__main__":
    tilelang.testing.main()
