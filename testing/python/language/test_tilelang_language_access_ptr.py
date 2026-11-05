import tvm_ffi
from tvm_tirx import IRModule, ir, sym, tirx
from tvm_tirx.target import Target

import tilelang
import tilelang.language as T
import tilelang.transform


def _tensor(shape, dtype, *, name="A", elem_offset=0):
    strides = []
    stride = 1
    for extent in reversed(shape):
        strides.append(stride)
        stride *= extent
    strides.reverse()
    layout = tirx.TileLayout.from_iters(
        [tirx.layout.Iter(extent, stride, "m") for extent, stride in zip(shape, strides)],
        offset={"m": elem_offset},
    )
    return ir.Var(name, tirx.TensorType(shape, dtype, layout=layout))


def test_access_ptr_builds_tl_access_ptr_from_bufferload_1d():
    tensor = _tensor((64,), "uint8")
    load = ir.TensorLoad(tensor, [ir.IntImm("int32", 16)])
    ptr = T.access_ptr(load, "r", 16)

    assert isinstance(ptr, ir.Call)
    assert ptr.op.same_as(ir.Op.get("tl.access_ptr"))
    assert len(ptr.args) == 3
    # args: (base_load, extent, rw_mask)
    assert isinstance(ptr.args[0], ir.TensorLoad)
    assert isinstance(ptr.args[1], ir.IntImm)
    assert int(ptr.args[1].value) == 16
    assert isinstance(ptr.args[2], ir.IntImm)
    assert int(ptr.args[2].value) == 1
    assert ptr.ty == ir.PtrType(ir.PrimType("uint8"))


def test_access_ptr_defaults_to_element_extent_for_bufferload():
    tensor = _tensor((64,), "float16")
    load = ir.TensorLoad(tensor, [ir.IntImm("int32", 7)])
    ptr = T.access_ptr(load, "rw")

    assert isinstance(ptr, ir.Call)
    assert ptr.op.same_as(ir.Op.get("tl.access_ptr"))
    assert isinstance(ptr.args[0], ir.TensorLoad)
    assert isinstance(ptr.args[1], ir.IntImm)
    assert int(ptr.args[1].value) == 1
    assert isinstance(ptr.args[2], ir.IntImm)
    assert int(ptr.args[2].value) == 3


def test_access_ptr_multiplies_extents_for_2d_load():
    tensor = _tensor((8, 8), "float16")
    load = ir.TensorLoad(tensor, [ir.IntImm("int32", 2), ir.IntImm("int32", 3)])
    ptr = T.access_ptr(load, "w", 4, 5)

    assert isinstance(ptr, ir.Call)
    assert ptr.op.same_as(ir.Op.get("tl.access_ptr"))
    assert isinstance(ptr.args[0], ir.TensorLoad)
    assert isinstance(ptr.args[1], ir.IntImm)
    assert int(ptr.args[1].value) == 20
    assert isinstance(ptr.args[2], ir.IntImm)
    assert int(ptr.args[2].value) == 2


def _lowered_address(tensor, ptr):
    func = tirx.Function([tensor], ir.Evaluate(ptr), ret_type=ir.PrimType("void"))
    mod = IRModule.from_expr(func.with_attr("global_symbol", "main"))
    lowered = tilelang.transform.LowerAccessPtr()(mod)
    calls = []
    tvm_ffi.structural_walk(lowered["main"].body, (ir.Call, calls.append), order="post")
    assert not any(call.op.name in ("tl.access_ptr", "tl.ptr_access") for call in calls)
    (address,) = [call for call in calls if call.op.name == "tirx.address_of"]
    assert len(address.args) == 1
    assert address.args[0].source.same_as(tensor)
    assert [int(index) for index in address.args[0].indices] == [2, 3]

    # The address boundary keeps tensor coordinates until its owning layout
    # consumer runs. Check the same physical element address after flattening.
    with Target("llvm"):
        lowered = tirx.transform.LowerTensorLayouts()(lowered)
        lowered = tilelang.transform.FlattenBuffer()(lowered)
    calls = []
    tvm_ffi.structural_walk(lowered["main"].body, (ir.Call, calls.append), order="post")
    (address,) = [call for call in calls if call.op.name == "tirx.address_of"]
    load = address.args[0]
    assert isinstance(load, ir.TensorLoad)
    assert len(load.indices) == 1
    return sym.Analyzer().simplify(load.indices[0])


def test_lower_access_ptr_rewrites_to_native_address():
    tensor = _tensor((8, 8), "float16")
    ptr = T.access_ptr(ir.TensorLoad(tensor, [2, 3]), "w", 4, 5)
    # Extent and effect mask belong to the TileLang access contract, before
    # its final address lowering deliberately erases them.
    assert int(ptr.args[1]) == 20 and int(ptr.args[2]) == 2
    offset = _lowered_address(tensor, ptr)
    assert isinstance(offset, ir.IntImm)
    assert int(offset.value) == 19


def test_lower_access_ptr_preserves_buffer_element_offset():
    tensor = _tensor((8, 8), "float16", elem_offset=5)
    ptr = T.access_ptr(ir.TensorLoad(tensor, [2, 3]), "r", 1)
    offset = _lowered_address(tensor, ptr)
    assert isinstance(offset, ir.IntImm)
    assert int(offset.value) == 24
