import tilelang
import tvm
from tvm import arith
from tvm import tirx
from tvm.tirx import op
import tilelang.language as T
import tilelang.transform


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


def test_lower_access_ptr_rewrites_to_tvm_access_ptr():
    buf = tirx.decl_buffer((8, 8), "float16", name="A")
    load = tirx.BufferLoad(buf, [tirx.IntImm("int32", 2), tirx.IntImm("int32", 3)])
    ptr = T.access_ptr(load, "w", 4, 5)

    func = tirx.PrimFunc([buf.data], tirx.Evaluate(ptr), buffer_map={buf.data: buf})
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    lowered = tilelang.transform.LowerAccessPtr()(mod)

    calls: list[tirx.Call] = []

    def _collect(e):
        if isinstance(e, tirx.Call):
            calls.append(e)

    tirx.stmt_functor.post_order_visit(lowered["main"].body, _collect)
    assert any(c.op.same_as(op.Op.get("tirx.tvm_access_ptr")) for c in calls)
    assert not any(c.op.same_as(op.Op.get("tl.access_ptr")) for c in calls)

    # Check the lowered tvm_access_ptr carries the expected linear offset/extents.
    acc = [c for c in calls if c.op.same_as(op.Op.get("tirx.tvm_access_ptr"))][0]
    assert len(acc.args) == 5
    analyzer = arith.Analyzer()
    offset = analyzer.simplify(acc.args[2])
    extent = analyzer.simplify(acc.args[3])
    assert isinstance(offset, tirx.IntImm)
    assert int(offset.value) == 19
    assert isinstance(extent, tirx.IntImm)
    assert int(extent.value) == 20
    assert isinstance(acc.args[4], tirx.IntImm)
    assert int(acc.args[4].value) == 2
