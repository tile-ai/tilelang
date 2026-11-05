"""Source location consumers use native fields and exact setter semantics."""

import tvm_ffi
from tvm_tirx import ir, tirx
from tilelang.testing.packed import compile_packed_function

import tilelang.ir as tl_ir
import tilelang.language as T


def function():
    parameter = ir.Var("A", tirx.TensorType((4,), "int32"))
    statement = ir.TensorStore(parameter, [ir.const(0)], ir.const(1))
    return tirx.Function([parameter], [statement], ir.TupleType([])), parameter, statement


def test_native_location_setters_replace_and_clear_without_changing_hash():
    kernel, tensor, statement = function()
    original_hash = tvm_ffi.structural_hash(kernel)
    first = tl_ir.make_span("first.py", 7)
    second = tl_ir.make_span("second.py", 19)
    for node, setter, getter in (
        (statement, tl_ir.set_stmt_span, tl_ir.get_stmt_span),
        (tensor, tl_ir.set_buffer_span, tl_ir.get_buffer_span),
        (kernel, tl_ir.set_prim_func_span, tl_ir.get_prim_func_span),
    ):
        setter(node, first)
        assert getter(node).same_as(first)
        setter(node, second)
        assert getter(node).same_as(second)
        assert isinstance(node.loc, ir.SourceLoc)
        setter(node, None)
        assert getter(node) is None and isinstance(node.loc, ir.UnknownLoc)
        setter(None, first)
        assert getter(None) is None
    assert tvm_ffi.structural_hash(kernel) == original_hash


def test_callsite_formatting_uses_known_callee_then_caller():
    callee = ir.SourceLoc(ir.SourceName("macro.py"), 11, 3, 11, 17)
    caller = ir.SourceLoc(ir.SourceName("kernel.py"), 52, 1, 52, 18)
    location = ir.CallSiteLoc(callee, caller)
    assert tl_ir.span_to_location(location) == ("macro.py", 11)
    format_location = tvm_ffi.get_global_func("tl.ir.SpanToString")
    assert format_location(location) == "macro.py:11:3"
    unknown_callee = ir.CallSiteLoc(ir.UnknownLoc(), caller)
    assert tl_ir.span_to_location(unknown_callee) == ("kernel.py", 52)
    assert format_location(unknown_callee) == "kernel.py:52:1"
    assert tl_ir.span_to_location(ir.UnknownLoc()) is None
    assert format_location(ir.UnknownLoc()) == ""


def test_stamp_preserves_known_nested_location_and_counts_native_definitions():
    kernel, tensor, statement = function()
    known = tl_ir.make_span("inner.py", 3)
    inherited = tl_ir.make_span("outer.py", 21)
    tl_ir.set_stmt_span(statement, known)
    before_hash = tvm_ffi.structural_hash(kernel)
    tl_ir.stamp_stmt_spans(kernel.body, inherited)
    assert kernel.body.loc.same_as(inherited)
    assert statement.loc.same_as(known)
    assert isinstance(tensor.loc, ir.UnknownLoc)
    assert tl_ir.span_coverage(kernel) == {"stmts": [2, 2], "buffers": [0, 1]}
    tl_ir.set_buffer_span(tensor, inherited)
    assert tl_ir.span_coverage(kernel) == {"stmts": [2, 2], "buffers": [1, 1]}
    assert tvm_ffi.structural_hash(kernel) == before_hash


def test_policy_python_binding_is_registered_only_when_requested():
    cls = tl_ir.GemmWarpPolicy
    assert cls is tl_ir.GemmWarpPolicy
    assert cls.__tvm_ffi_type_info__.type_key == "tl.GemmWarpPolicy"
    assert hasattr(cls, "compute_warp_partition")


def test_select_construction_supplies_a_native_location():
    condition = ir.Var("condition", ir.PrimType("bool"))
    implicit = T.Select(condition, 0, 1)
    assert isinstance(implicit.loc, ir.UnknownLoc)
    explicit = tl_ir.make_span("select.py", 12)
    select = T.Select(condition, 0, 1, span=explicit)
    assert select.loc.same_as(explicit)
    assert select.condition.same_as(condition)
    assert int(select.true_value.value) == 0
    assert int(select.false_value.value) == 1


def test_select_default_location_compiles_both_results():
    import numpy as np
    import tvm_tirx

    @T.prim_func
    def kernel(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
        for i in T.serial(4):
            B[i] = T.Select(A[i] > 0, 7, -3)

    assert tirx.analysis.check_well_formed(kernel)
    data = np.array([-1, 0, 1, 2], dtype="int32")
    output = tvm_tirx.runtime.tensor(np.zeros(4, dtype="int32"), tvm_tirx.cpu())
    compile_packed_function(kernel, tilelang_owned=False)(tvm_tirx.runtime.tensor(data, tvm_tirx.cpu()), output)
    np.testing.assert_array_equal(output.numpy(), [-3, -3, 7, 7])
