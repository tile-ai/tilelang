import tilelang
import tvm
from tvm import arith
from tvm import tir
from tvm.tir import op
import tilelang.language as T
import tilelang.transform


def test_access_ptr_builds_tl_access_ptr_from_bufferload_1d():
    buf = tir.decl_buffer((64,), "uint8", name="A")
    load = tir.BufferLoad(buf, [tir.IntImm("int32", 16)])

    ptr = T.access_ptr(load, "r", 16)

    assert isinstance(ptr, tir.Call)
    assert ptr.op.same_as(op.Op.get("tl.access_ptr"))
    assert len(ptr.args) == 3
    # args: (base_load, extent, rw_mask)
    assert isinstance(ptr.args[0], tir.BufferLoad)
    assert isinstance(ptr.args[1], tir.IntImm)
    assert int(ptr.args[1].value) == 16
    assert isinstance(ptr.args[2], tir.IntImm)
    assert int(ptr.args[2].value) == 1


def test_access_ptr_defaults_to_element_extent_for_bufferload():
    buf = tir.decl_buffer((64,), "float16", name="A")
    load = tir.BufferLoad(buf, [tir.IntImm("int32", 7)])

    ptr = T.access_ptr(load, "rw")

    assert isinstance(ptr, tir.Call)
    assert ptr.op.same_as(op.Op.get("tl.access_ptr"))
    assert isinstance(ptr.args[0], tir.BufferLoad)
    assert isinstance(ptr.args[1], tir.IntImm)
    assert int(ptr.args[1].value) == 1
    assert isinstance(ptr.args[2], tir.IntImm)
    assert int(ptr.args[2].value) == 3


def test_access_ptr_multiplies_extents_for_2d_load():
    buf = tir.decl_buffer((8, 8), "float16", name="A")
    load = tir.BufferLoad(buf, [tir.IntImm("int32", 2), tir.IntImm("int32", 3)])

    ptr = T.access_ptr(load, "w", 4, 5)

    assert isinstance(ptr, tir.Call)
    assert ptr.op.same_as(op.Op.get("tl.access_ptr"))
    assert isinstance(ptr.args[0], tir.BufferLoad)
    # extent = 4*5 = 20
    assert isinstance(ptr.args[1], tir.IntImm)
    assert int(ptr.args[1].value) == 20
    assert isinstance(ptr.args[2], tir.IntImm)
    assert int(ptr.args[2].value) == 2


def test_lower_access_ptr_rewrites_to_tvm_access_ptr():
    buf = tir.decl_buffer((8, 8), "float16", name="A")
    load = tir.BufferLoad(buf, [tir.IntImm("int32", 2), tir.IntImm("int32", 3)])
    ptr = T.access_ptr(load, "w", 4, 5)

    func = tir.PrimFunc([buf.data], tir.Evaluate(ptr), buffer_map={buf.data: buf})
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    lowered = tilelang.transform.LowerAccessPtr()(mod)

    calls: list[tir.Call] = []

    def _collect(e):
        if isinstance(e, tir.Call):
            calls.append(e)

    tir.stmt_functor.post_order_visit(lowered["main"].body, _collect)
    assert any(c.op.same_as(op.Op.get("tir.tvm_access_ptr")) for c in calls)
    assert not any(c.op.same_as(op.Op.get("tl.access_ptr")) for c in calls)

    # Check the lowered tvm_access_ptr carries the expected linear offset/extents.
    acc = [c for c in calls if c.op.same_as(op.Op.get("tir.tvm_access_ptr"))][0]
    assert len(acc.args) == 5
    analyzer = arith.Analyzer()
    offset = analyzer.simplify(acc.args[2])
    extent = analyzer.simplify(acc.args[3])
    assert isinstance(offset, tir.IntImm)
    assert int(offset.value) == 19
    assert isinstance(extent, tir.IntImm)
    assert int(extent.value) == 20
    assert isinstance(acc.args[4], tir.IntImm)
    assert int(acc.args[4].value) == 2
