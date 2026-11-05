"""Execute the CUDA scalar callback's native IR with an independent CPU oracle.

This validates callback construction, captures and thread ownership without
claiming CUDA device execution. The input tile-op is parsed and lowered by
the real C++ operator bridge; each generated thread slice then runs via C.
"""

import itertools
import sys

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

import tilelang.cpu.op  # noqa: F401
import tilelang.cuda.op  # noqa: F401
import tilelang.language as T
from tilelang.tileop._lowering import close_lowered_function
from tilelang.tileop.gemm.registry import resolve_gemm_impl
from tilelang.testing.packed import compile_packed_function


def _tensor(name, shape):
    return ir.Var(name, T.Tensor(shape, "float32").as_type())


def test_scalar_callback_registration_loads_only_selected_emitter():
    target = Target({"kind": "cuda", "arch": "sm_70"})
    selected = resolve_gemm_impl("cuda.fma", target)
    assert selected.__name__ == "GemmFMA"
    assert selected.__module__ == "tilelang.cuda.op.gemm.gemm_fma"
    assert "tvm" not in sys.modules


def test_metal_padding_uses_native_tensor_dtype_and_preserves_all_addresses():
    from tilelang.metal.op.gemm.gemm_metal import _make_padded_layout

    tensor = _tensor("A", (4, 8))
    layout = _make_padded_layout(tensor)
    for i, j in itertools.product(range(4), range(8)):
        assert int(layout.map_forward_index([i, j])[0]) == i * 12 + j


@pytest.mark.parametrize("trans_a,trans_b,clear", list(itertools.product([False, True], repeat=3)))
def test_fma_native_callback_respects_nonzero_thread_interval(trans_a, trans_b, clear):
    m, n, k = 3, 5, 4
    a_shape = (k, m) if trans_a else (m, k)
    b_shape = (n, k) if trans_b else (k, n)
    a, b, c = _tensor("A", a_shape), _tensor("B", b_shape), _tensor("C", (m, n))
    call = T.gemm(a, b, c, transpose_A=trans_a, transpose_B=trans_b, clear_accum=clear)
    cpu_target = Target("c")
    with cpu_target:
        lowered = tvm_ffi.get_global_func("tl.testing.LowerNativeOperator")(call, cpu_target, {})
    target = Target({"kind": "cuda", "arch": "sm_70"})
    thread = ir.Var("thread", ir.PrimType("int32"))
    implementation = resolve_gemm_impl("cuda.fma", target)(lowered["operator"])
    with target:
        layout_map = implementation.infer_layout(target, 4)
        function = implementation.lower(layout_map, target, ir.Range.from_min_extent(7, 4), thread, ir.const(0))
    closed, captures = close_lowered_function(function)
    assert tirx.analysis.check_well_formed(closed)
    assert not tirx.analysis.undefined_vars(closed)
    executable = compile_packed_function(closed, target=cpu_target)
    generator = np.random.default_rng(29)
    a_value = generator.integers(-5, 6, size=a_shape).astype("float32") / 8
    b_value = generator.integers(-5, 6, size=b_shape).astype("float32") / 8
    c_value = np.full((m, n), 3.25, dtype="float32")
    oracle = (a_value.T if trans_a else a_value) @ (b_value.T if trans_b else b_value)
    if not clear:
        oracle += c_value
    values = {
        a: tvm_tirx.runtime.tensor(a_value),
        b: tvm_tirx.runtime.tensor(b_value),
        c: tvm_tirx.runtime.tensor(c_value),
    }
    for thread_id in range(7, 11):
        arguments = [thread_id if value.same_as(thread) else values[value] for value in captures]
        executable(*arguments)
    np.testing.assert_allclose(values[c].numpy(), oracle, rtol=1e-6, atol=1e-6)
