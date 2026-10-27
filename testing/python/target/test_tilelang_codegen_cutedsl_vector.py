"""CuTeDSL codegen: vector values are tensors and loop bounds are statements-free.

Both regressions were found by ``examples/dequantize_gemm`` and
``examples/blocksparse_attention``: a vectorized shift whose amount is a ramp
printed a Python tuple, and a loop bound that lowered to a ``Let`` printed its
binding inside ``range(...)``.

The vector buffers themselves are scalar-backed, including the ``shared.dyn``
scope that ``T.alloc_shared`` defaults to.
"""

import pytest
import torch

import tilelang
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


def test_cutedsl_boolean_vector_cast_keeps_the_lane_count():
    """A ``boolx2`` value casts straight to two ``int32`` lanes.

    Booleans report 8 bits, so the 32-bit alignment used for narrow *float*
    vector casts padded the result to four lanes and stored a two-lane value
    into a four-element tensor.
    """
    try:
        from tilelang.jit.adapter.cutedsl.checks import check_cutedsl_available

        check_cutedsl_available()
    except (ImportError, ModuleNotFoundError, RuntimeError, AssertionError) as err:
        pytest.skip(f"CuTeDSL is not available: {err}")

    n = 64

    @T.prim_func
    def prog(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "int32")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding(0)
            for j in T.vectorized(2):
                B[tx * 2 + j] = T.cast(A[tx * 2 + j] > T.float32(0.5), "int32")

    source = _lower_cutedsl(prog).kernel_source
    assert "make_rmem_tensor((4,), cutlass.Int32)" not in source

    kernel = tilelang.compile(prog, target="cutedsl", out_idx=[1])
    a = torch.randn(n, device="cuda")
    assert torch.equal(kernel(a), (a > 0.5).to(torch.int32))


def test_cutedsl_shared_dyn_vector_buffer_roundtrip():
    """A vector-typed ``shared.dyn`` buffer is scalar-backed, so indices scale.

    ``T.alloc_shared`` defaults to ``shared.dyn``, whose allocation declares the
    *element* type. The buffer-reference helpers only lane-scaled ``shared`` and
    ``local``, so a ``shared.dyn`` access kept the vector index against a scalar
    pointer and touched the wrong bytes.
    """
    try:
        from tilelang.jit.adapter.cutedsl.checks import check_cutedsl_available

        check_cutedsl_available()
    except (ImportError, ModuleNotFoundError, RuntimeError, AssertionError) as err:
        pytest.skip(f"CuTeDSL is not available: {err}")

    n = 64

    @T.prim_func
    def prog(A: T.Tensor((n,), "uint8"), B: T.Tensor((n,), "uint8")):
        with T.Kernel(1, threads=32):
            S = T.alloc_shared((32,), "uint8x2")
            tx = T.get_thread_binding(0)
            S[tx] = A[T.Ramp(tx * 2, 1, 2)]
            B[T.Ramp(tx * 2, 1, 2)] = S[tx]

    source = _lower_cutedsl(prog).kernel_source
    assert "get_dyn_smem(cutlass.Uint8" in source

    kernel = tilelang.compile(prog, target="cutedsl", out_idx=[1])
    a = torch.randint(0, 256, (n,), dtype=torch.uint8, device="cuda")
    assert torch.equal(kernel(a), a)


if __name__ == "__main__":
    tilelang.testing.main()
