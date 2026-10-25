import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

from tilelang._compat import location
from tilelang._compat.tensor import TensorView

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def test_native_source_location_fields_and_formatting():
    span = location.make_span("kernel.py", 27)
    assert isinstance(span, ir.SourceLoc)
    assert (span.start_line, span.end_line, span.start_column, span.end_column) == (27, 27, 1, 1 << 20)
    assert location.span_to_location(span) == ("kernel.py", 27)
    assert tvm_ffi.get_global_func("tl.ir.SpanToString")(span) == "kernel.py:27:1"
    assert tvm_ffi.get_global_func("tl.ir.SpanToString")(ir.UNKNOWN_LOC) == ""


def test_native_callsite_location_uses_callee_then_caller():
    callee = location.make_span("callee.py", 4)
    caller = location.make_span("caller.py", 12)
    callsite = ir.CallSiteLoc(callee, caller)
    assert location.span_to_location(callsite) == ("callee.py", 4)
    assert tvm_ffi.get_global_func("tl.ir.SpanToString")(callsite) == "callee.py:4:1"
    fallback = ir.CallSiteLoc(ir.UNKNOWN_LOC, caller)
    assert location.span_to_location(fallback) == ("caller.py", 12)
    assert tvm_ffi.get_global_func("tl.ir.SpanToString")(fallback) == "caller.py:12:1"


def test_location_stamping_preserves_identity_and_existing_locations():
    tensor = TensorView.create("output", [8], "float32")
    index = ir.Var("index", "int32")
    store = tensor.store([index], 1)
    explicit = location.make_span("store.py", 9)
    location.set_stmt_span(store, explicit)
    loop = ir.For(index, 0, 8, 1, ir.ForKind.DEFAULT, store)
    original = ir.load_json(ir.save_json(loop))
    outer = location.make_span("loop.py", 3)
    location.stamp_stmt_spans(loop, outer)
    assert loop.body.seq[0].same_as(store)
    assert location.get_stmt_span(store).same_as(explicit)
    assert location.get_stmt_span(loop).same_as(outer)
    assert location.get_stmt_span(loop.body).same_as(explicit)
    ir.assert_structural_equal(original, loop, map_free_vars=True)
    location.stamp_stmt_spans(loop, ir.UNKNOWN_LOC)
    assert location.get_stmt_span(loop).same_as(outer)


def test_native_tensor_and_function_locations():
    tensor = TensorView.create("input", [8], "float32")
    before = tensor.tensor
    span = location.make_span("parameter.py", 10)
    location.set_buffer_span(tensor, span)
    assert tensor.tensor.same_as(before)
    assert location.get_buffer_span(tensor.tensor).same_as(span)
    function = tirx.Function([tensor.tensor], None, ret_type=ir.TupleType([]))
    location.set_prim_func_span(function, span)
    assert location.get_prim_func_span(function).same_as(span)
    assert location.span_coverage(function) == {"stmts": [0, 0], "buffers": [1, 1]}
    location.set_buffer_span(tensor, None)
    location.set_prim_func_span(function, None)
    assert location.get_buffer_span(tensor) is None
    assert location.get_prim_func_span(function) is None


def test_native_location_coverage_counts_tensor_identity_once():
    parameter = TensorView.create("parameter", [8], "float32")
    local = TensorView.create("scratch", [8], "float32", scope="local")
    body = ir.SeqStmt([local.allocate(), local.store([0], parameter.load([0])), parameter.store([1], local.load([0]))])
    function = tirx.Function([parameter.tensor], body, ret_type=ir.TupleType([]))
    span = location.make_span("body.py", 6)
    location.set_buffer_span(local, span)
    location.stamp_stmt_spans(body, span)
    coverage = location.span_coverage(function)
    assert coverage["buffers"] == [1, 2]
    assert coverage["stmts"] == [4, 4]


def test_absent_native_locations_are_safe():
    assert location.get_stmt_span(None) is None
    assert location.get_buffer_span(None) is None
    assert location.get_prim_func_span(None) is None
    assert location.span_to_location(None) is None
    location.set_stmt_span(None, None)
    location.set_buffer_span(None, None)
    location.set_prim_func_span(None, None)
    location.stamp_stmt_spans(None, None)
