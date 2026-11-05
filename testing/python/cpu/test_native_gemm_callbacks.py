"""CPU GEMM's actual C++ dispatch, Python lowering and closed native Function.

The test compiler includes testing/cpp/native_operator_probe.cc. That bridge
invokes production ParseOperator, InferLayout and Lower methods; no operator
is mocked. The closed callback function is additionally compiled and run via C.
"""

import itertools

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

import tilelang.cpu.op  # noqa: F401
import tilelang.language as T
from tilelang.testing.packed import compile_packed_function


def tensor(name, shape):
    return ir.Var(name, T.Tensor(shape, "float32").as_type())


def region(value, begins, shape):
    return ir.TensorRegion(
        value, [ir.Range.from_min_extent(begin, extent) for begin, extent in zip(begins, shape, strict=True)], ir.TensorRegionType()
    )


def lower(call):
    target = Target("c")
    with target:
        lowered = tvm_ffi.get_global_func("tl.testing.LowerNativeOperator")(call, target, {})
        function, captures = tvm_ffi.get_global_func("tl.gemm.lower")(
            lowered["operator"], {}, target, ir.Range(1), ir.const(0), ir.const(0)
        )
    assert tirx.analysis.check_well_formed(function)
    assert not tirx.analysis.undefined_vars(function)
    scope = lowered["statement"]
    assert isinstance(scope, ir.RegionStmt) and scope.op.name == "tl.attr"
    metadata = tvm_ffi.get_global_func("tl.DecodeAnnotations")(scope.args[0], scope.attrs["annotation_schema"])
    assert metadata["key"] == "lexical_alloc_scope"
    assert metadata["name"] == "tl.gemm"
    # C++ renewed parameters and substituted them back to these original
    # captures; no new free identity may escape the inlining boundary.
    container = tirx.Function(captures, scope, ir.TupleType([]))
    assert tirx.analysis.check_well_formed(container)
    assert not tirx.analysis.undefined_vars(container)
    return function, captures


def execute(function, captures, values):
    executable = compile_packed_function(function, target="c")
    runtime = {key: tvm_tirx.runtime.tensor(value) if isinstance(value, np.ndarray) else value for key, value in values.items()}
    executable(*[runtime[key] for key in captures])
    return runtime


@pytest.mark.parametrize("trans_a,trans_b,clear", list(itertools.product([False, True], repeat=3)))
def test_cpu_gemm_callback_executes_all_transpose_and_clear_modes(trans_a, trans_b, clear):
    m, n, k = 3, 5, 4
    a_shape = (k, m) if trans_a else (m, k)
    b_shape = (n, k) if trans_b else (k, n)
    a, b, c = tensor("A", a_shape), tensor("B", b_shape), tensor("C", (m, n))
    call = T.gemm(a, b, c, transpose_A=trans_a, transpose_B=trans_b, clear_accum=clear)
    function, captures = lower(call)
    generator = np.random.default_rng(29)
    a_value = generator.integers(-5, 6, size=a_shape).astype("float32") / 8
    b_value = generator.integers(-5, 6, size=b_shape).astype("float32") / 8
    c_value = np.full((m, n), 3.25, dtype="float32")
    oracle = (a_value.T if trans_a else a_value) @ (b_value.T if trans_b else b_value)
    if not clear:
        oracle += c_value
    actual = execute(function, captures, {a: a_value, b: b_value, c: c_value})[c].numpy()
    np.testing.assert_allclose(actual, oracle, rtol=1e-6, atol=1e-6)


def test_cpu_gemm_clear_only_writes_the_requested_output_region():
    m, n, k = 3, 5, 4
    a, b, c = tensor("A", (5, 7)), tensor("B", (7, 8)), tensor("C", (7, 11))
    call = T.gemm(region(a, (1, 2), (m, k)), region(b, (2, 1), (k, n)), region(c, (2, 3), (m, n)), clear_accum=True)
    function, captures = lower(call)
    a_value = np.arange(35, dtype="float32").reshape(5, 7) / 8
    b_value = np.arange(56, dtype="float32").reshape(7, 8) / 8
    c_value = np.full((7, 11), -97.0, dtype="float32")
    oracle = c_value.copy()
    oracle[2:5, 3:8] = a_value[1:4, 2:6] @ b_value[2:6, 1:6]
    actual = execute(function, captures, {a: a_value, b: b_value, c: c_value})[c].numpy()
    np.testing.assert_array_equal(actual, oracle)
