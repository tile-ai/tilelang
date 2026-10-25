import tilelang
from tilelang import tvm
from tilelang._compat.tvm import arith
from tilelang._compat.tvm import tirx
from tilelang._compat.tirx import op
import tilelang.language as T
import tilelang.transform
from tvm_tirx import tirx as native_tirx


def test_access_ptr_builds_tl_access_ptr_from_bufferload_1d():
    buf = tirx.decl_buffer((64,), "uint8", name="A")
    load = tirx.BufferLoad(buf, [tirx.IntImm("int32", 16)])

    ptr = T.access_ptr(load, "r", 16)

    assert isinstance(ptr, tirx.Call)
    assert ptr.op.same_as(op.Op.get("tl.access_ptr"))
    assert len(ptr.args) == 3
    # args: (base_load, extent, rw_mask)
    assert isinstance(ptr.args[0], tirx.BufferLoad)
    assert isinstance(ptr.args[1], tirx.IntImm)
    assert int(ptr.args[1].value) == 16
    assert isinstance(ptr.args[2], tirx.IntImm)
    assert int(ptr.args[2].value) == 1


def test_access_ptr_defaults_to_element_extent_for_bufferload():
    buf = tirx.decl_buffer((64,), "float16", name="A")
    load = tirx.BufferLoad(buf, [tirx.IntImm("int32", 7)])

    ptr = T.access_ptr(load, "rw")

    assert isinstance(ptr, tirx.Call)
    assert ptr.op.same_as(op.Op.get("tl.access_ptr"))
    assert isinstance(ptr.args[0], tirx.BufferLoad)
    assert isinstance(ptr.args[1], tirx.IntImm)
    assert int(ptr.args[1].value) == 1
    assert isinstance(ptr.args[2], tirx.IntImm)
    assert int(ptr.args[2].value) == 3


def test_access_ptr_multiplies_extents_for_2d_load():
    buf = tirx.decl_buffer((8, 8), "float16", name="A")
    load = tirx.BufferLoad(buf, [tirx.IntImm("int32", 2), tirx.IntImm("int32", 3)])

    ptr = T.access_ptr(load, "w", 4, 5)

    assert isinstance(ptr, tirx.Call)
    assert ptr.op.same_as(op.Op.get("tl.access_ptr"))
    assert isinstance(ptr.args[0], tirx.BufferLoad)
    # extent = 4*5 = 20
    assert isinstance(ptr.args[1], tirx.IntImm)
    assert int(ptr.args[1].value) == 20
    assert isinstance(ptr.args[2], tirx.IntImm)
    assert int(ptr.args[2].value) == 2


def test_lower_access_ptr_rewrites_to_native_address():
    buf = tirx.decl_buffer((8, 8), "float16", name="A")
    load = tirx.BufferLoad(buf, [tirx.IntImm("int32", 2), tirx.IntImm("int32", 3)])
    ptr = T.access_ptr(load, "w", 4, 5)

    func = tirx.PrimFunc([buf], tirx.Evaluate(ptr))
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    lowered = tilelang.transform.LowerAccessPtr()(mod)
    with tvm.target.Target("c"):
        lowered = native_tirx.transform.LowerTensorLayouts()(lowered)

    calls: list[tirx.Call] = []

    def _collect(expression):
        if isinstance(expression, tirx.Call):
            calls.append(expression)

    tirx.stmt_functor.post_order_visit(lowered["main"].body, _collect)
    assert any(call.op.same_as(op.Op.get("tirx.address_of")) for call in calls)
    assert not any(call.op.same_as(op.Op.get("tl.access_ptr")) for call in calls)

    acc = [call for call in calls if call.op.same_as(op.Op.get("tirx.address_of"))][0]
    assert len(acc.args) == 1
    analyzer = arith.Analyzer()
    offset = analyzer.simplify(acc.args[0].indices[0])
    extent = analyzer.simplify(acc.attrs["tl.access_extent"])
    assert isinstance(offset, tirx.IntImm)
    assert int(offset.value) == 19
    assert isinstance(extent, tirx.IntImm)
    assert int(extent.value) == 20
    assert isinstance(acc.attrs["tl.access_mask"], tirx.IntImm)
    assert int(acc.attrs["tl.access_mask"].value) == 2


def test_lower_access_ptr_preserves_buffer_element_offset():
    elem_offset = tirx.IntImm("int32", 5)
    buf = tirx.decl_buffer((8, 8), "float16", name="A", elem_offset=elem_offset)
    load = tirx.BufferLoad(buf, [tirx.IntImm("int32", 2), tirx.IntImm("int32", 3)])
    ptr = T.access_ptr(load, "r", 1)

    func = tirx.PrimFunc([buf], tirx.Evaluate(ptr))
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    lowered = tilelang.transform.LowerAccessPtr()(mod)
    with tvm.target.Target("c"):
        lowered = native_tirx.transform.LowerTensorLayouts()(lowered)

    calls = []
    tirx.stmt_functor.post_order_visit(
        lowered["main"].body,
        lambda expr: calls.append(expr) if isinstance(expr, tirx.Call) else None,
    )
    acc = [call for call in calls if call.op.same_as(op.Op.get("tirx.address_of"))][0]
    load = acc.args[0]
    offset = arith.Analyzer().simplify(load.source.ty.layout.apply(*load.indices, shape=load.source.shape)["m"])
    assert isinstance(offset, tirx.IntImm)
    assert int(offset.value) == 24
