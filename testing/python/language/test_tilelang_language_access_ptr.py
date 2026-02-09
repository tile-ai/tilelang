import tilelang  # noqa: F401  # sets up TVM Python path
from tvm import tir
from tvm.tir import op
import tilelang.language as T


def test_access_ptr_builds_tvm_access_ptr_from_bufferload_1d():
    buf = tir.decl_buffer((64,), "uint8", name="A")
    load = tir.BufferLoad(buf, [tir.IntImm("int32", 16)])

    ptr = T.access_ptr(load, "r", 16)

    assert isinstance(ptr, tir.Call)
    assert ptr.op.same_as(op.Op.get("tir.tvm_access_ptr"))
    assert len(ptr.args) == 5
    # args: (ptype, data, offset, extent, rw_mask)
    assert isinstance(ptr.args[2], tir.IntImm)
    assert int(ptr.args[2].value) == 16
    assert isinstance(ptr.args[3], tir.IntImm)
    assert int(ptr.args[3].value) == 16
    assert isinstance(ptr.args[4], tir.IntImm)
    assert int(ptr.args[4].value) == 1


def test_access_ptr_defaults_to_element_extent_for_bufferload():
    buf = tir.decl_buffer((64,), "float16", name="A")
    load = tir.BufferLoad(buf, [tir.IntImm("int32", 7)])

    ptr = T.access_ptr(load, "rw")

    assert isinstance(ptr, tir.Call)
    assert ptr.op.same_as(op.Op.get("tir.tvm_access_ptr"))
    assert isinstance(ptr.args[2], tir.IntImm)
    assert int(ptr.args[2].value) == 7
    assert isinstance(ptr.args[3], tir.IntImm)
    assert int(ptr.args[3].value) == 1
    assert isinstance(ptr.args[4], tir.IntImm)
    assert int(ptr.args[4].value) == 3


def test_access_ptr_linearizes_offset_and_multiplies_extents_for_2d_load():
    buf = tir.decl_buffer((8, 8), "float16", name="A")
    load = tir.BufferLoad(buf, [tir.IntImm("int32", 2), tir.IntImm("int32", 3)])

    ptr = T.access_ptr(load, "w", 4, 5)

    assert isinstance(ptr, tir.Call)
    assert ptr.op.same_as(op.Op.get("tir.tvm_access_ptr"))
    # offset = 2*8 + 3 = 19
    assert isinstance(ptr.args[2], tir.IntImm)
    assert int(ptr.args[2].value) == 19
    # extent = 4*5 = 20
    assert isinstance(ptr.args[3], tir.IntImm)
    assert int(ptr.args[3].value) == 20
    assert isinstance(ptr.args[4], tir.IntImm)
    assert int(ptr.args[4].value) == 2
