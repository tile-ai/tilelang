import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import runtime, tirx
from tvm_tirx.target import Target

from tilelang._compat import ir, stmt_functor
from tilelang._compat.tensor import TensorView
from tilelang.transform import _tirx_passes
from tilelang.transform import LowerAccessPtr

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def native(name, *arguments):
    return tvm_ffi.get_global_func(f"tl.{name}")(*arguments)


def region_of(tensor, begins, extents):
    return tensor.region([ir.Range.from_min_extent(begin, extent) for begin, extent in zip(begins, extents)])


def lower_function(function):
    return LowerAccessPtr()(ir.IRModule({"main": function}))["main"]


def test_normalize_native_tensor_load_and_region():
    tensor = TensorView.create("tensor", [4, 8], "float32")
    region = native("NormalizeToBufferRegion", tensor.tensor)
    assert region.source.same_as(tensor.tensor)
    assert [int(axis.extent) for axis in region.region] == [4, 8]
    loaded = native("NormalizeToBufferRegion", tensor.load([1, 2]))
    assert [int(axis.begin) for axis in loaded.region] == [1, 2]
    assert [int(axis.extent) for axis in loaded.region] == [1, 1]
    assert native("NormalizeToBufferRegion", region).same_as(region)


def test_transport_regions_retain_masks_and_symbolic_extents():
    tensor = TensorView.create("tensor", [4, 8], "float32")
    extent = ir.Var("extent", "int32")
    region = region_of(tensor, [1, 2], [2, extent])
    wrapped = ir.Call("tl.region", [region, 1])
    assert isinstance(wrapped.ty, ir.TensorRegionType)
    decoded, mask = native("NormalizeToAccessRegion", wrapped, 3)
    assert decoded.same_as(region)
    assert mask == 1
    legacy_arguments = ir.Call("tl.region", [tensor.load([1, 2]), 2, 2, extent])
    decoded, mask = native("NormalizeToAccessRegion", legacy_arguments, 3)
    ir.assert_structural_equal(decoded, region)
    assert mask == 2
    assert decoded.region[1].extent.same_as(extent)


@pytest.mark.parametrize("mask", [-1, 4, 1024])
def test_invalid_region_masks_are_rejected(mask):
    tensor = TensorView.create("tensor", [8], "float32")
    with pytest.raises(ValueError, match="mask"):
        native("NormalizeToBufferRegion", ir.Call("tl.region", [tensor.load([0]), mask, 8]))


def test_region_load_vector_geometry():
    tensor = TensorView.create("tensor", [16], "float32")
    region = native("NormalizeToBufferRegion", tensor.load([ir.prim.Ramp(3, 1, 4)]))
    assert int(region.region[0].begin) == 3
    assert int(region.region[0].extent) == 4
    with pytest.raises(ValueError, match="stride-1"):
        native("NormalizeToBufferRegion", tensor.load([ir.prim.Ramp(3, 2, 4)]))
    with pytest.raises(ValueError, match="extent must match"):
        native("NormalizeToBufferRegion", ir.Call("tl.region", [tensor.load([ir.prim.Ramp(3, 1, 4)]), 1, 8]))


def test_typed_access_pointer_preserves_logical_metadata_and_location():
    tensor = TensorView.create("tensor", [4, 8], "float32", scope="shared")
    region = region_of(tensor, [1, 2], [2, 3])
    pointer = native("MakeAccessPtrFromRegion", region, 2, True)
    assert pointer.ty == ir.PtrType("float32", "shared")
    assert int(pointer.args[1]) == 6
    assert native("GetVarFromAccessPtr", pointer).same_as(tensor.tensor)
    lowered = native("LowerAccessPtrCall", pointer)
    assert lowered.op.name == "tirx.address_of"
    assert lowered.args[0].source.same_as(tensor.tensor)
    assert lowered.attrs["tl.access_region"].same_as(region)
    assert int(lowered.attrs["tl.access_mask"]) == 2
    assert int(lowered.attrs["tl.access_extent"]) == 6
    assert lowered.ty == pointer.ty
    lowered.validate()
    ir.assert_structural_equal(lowered, ir.load_json(ir.save_json(lowered)), map_free_vars=True)


def test_native_pointer_metadata_tracks_flattened_and_renewed_tensor_identity():
    tensor = TensorView.create("tensor", [4, 8], "float32", scope="shared")
    pointer = native("LowerAccessPtrCall", native("MakeAccessPtrFromRegion", region_of(tensor, [1, 2], [1, 4]), 1, True))
    function = tirx.Function([], [tensor.allocate(), ir.Evaluate(pointer)])
    renewed = ir.renew_def(function)
    module = _tirx_passes.FlattenTensor()(ir.IRModule({"main": renewed}))
    pointers = []
    stmt_functor.post_order_visit(
        module,
        lambda node: pointers.append(node)
        if isinstance(node, ir.Call) and isinstance(node.op, ir.Op) and node.op.name == "tirx.address_of"
        else None,
    )
    assert len(pointers) == 1
    lowered = pointers[0]
    assert lowered.args[0].source.same_as(lowered.attrs["tl.access_region"].source)
    assert len(lowered.attrs["tl.access_region"].region) == len(lowered.args[0].indices) == 1
    assert int(lowered.attrs["tl.access_region"].region[0].begin) == 10
    assert int(lowered.attrs["tl.access_region"].region[0].extent) == 4
    assert ir.check_well_formed(module)


@pytest.mark.parametrize("index", [ir.prim.Ramp(3, 2, 4), ir.prim.Broadcast(3, 4)])
def test_pointer_to_vectorized_load_uses_scalar_base(index):
    tensor = TensorView.create("tensor", [16], "float32")
    pointer = native("MakeAccessPtrFromBufferLoad", tensor.load([index]), 1)
    lowered = native("LowerAccessPtrCall", pointer)
    assert lowered.ty == ir.PtrType("float32", "global")
    assert int(lowered.args[0].indices[0]) == 3
    assert int(lowered.attrs["tl.access_extent"]) == 1


def test_access_pointer_validates_extent_mask_and_result_type():
    tensor = TensorView.create("tensor", [8], "float32")
    with pytest.raises(ValueError, match="extent"):
        ir.Call("tl.access_ptr", [tensor.load([0]), -1, 1]).validate()
    with pytest.raises(ValueError, match="mask"):
        ir.Call("tl.access_ptr", [tensor.load([0]), 8, 4]).validate()
    with pytest.raises(TypeError, match="pointer type"):
        ir.Call("tl.access_ptr", [tensor.load([0]), 8, 1], ty=ir.PtrType("int32")).validate()
    with pytest.raises(TypeError):
        ir.Call("tl.access_ptr", [tensor.load([0]), 8.0, 1]).validate()


def test_pointer_origin_follows_native_offsets():
    tensor = TensorView.create("tensor", [8], "float32")
    pointer = tensor.pointer_at([2])
    offset = ir.Call("tirx.ptr_byte_offset", [pointer, 4], ty=pointer.ty)
    assert native("GetVarFromAccessPtr", offset).same_as(tensor.tensor)
    assert native("GetVarFromAccessPtr", tensor.data).same_as(tensor.tensor)
    with pytest.raises(TypeError, match="tensor or typed pointer"):
        native("GetVarFromAccessPtr", ir.Var("not_a_pointer", "int32"))


def test_lower_access_ptr_preserves_native_function_contract():
    tensor = TensorView.create("tensor", [8], "float32")
    pointer = native("MakeAccessPtrFromRegion", region_of(tensor, [2], [4]), 1, False)
    original = tirx.Function([tensor.tensor], ir.Evaluate(pointer), ret_type=ir.TupleType([]))
    lowered = lower_function(original)
    assert original.body.seq[0].value.op.name == "tl.access_ptr"
    assert lowered.body.seq[0].value.op.name == "tirx.address_of"
    assert lowered.params[0].same_as(tensor.tensor)
    assert lower_function(lowered).same_as(lowered)
    assert ir.check_well_formed(lowered)
    declaration = tirx.Function([tensor.tensor], None, ret_type=ir.TupleType([]))
    assert lower_function(declaration).same_as(declaration)


def test_strided_offset_access_executes_on_cpu(compile_tilelang):
    source = TensorView.create("source", [80], "float32")
    output = TensorView.create("output", [1], "float32")
    view = TensorView.create("view", [4, 8], "float32", strides=[20, 2], elem_offset=3)
    pointer = native("MakeAccessPtrFromRegion", region_of(view, [2, 3], [1, 1]), 1, True)
    selected = TensorView.create("selected", [1], "float32")
    function = tirx.Function(
        [source.tensor, output.tensor],
        [view.declare(source.data), selected.declare(pointer), output.store(0, selected.load(0))],
        ret_type=ir.TupleType([]),
    ).with_attr("global_symbol", "strided_access")
    lowered = lower_function(function)
    assert ir.check_well_formed(lowered)
    executable = compile_tilelang(lowered, target="c")
    values = np.arange(80, dtype="float32")
    actual = runtime.empty((1,), "float32")
    executable(runtime.tensor(values), actual)
    np.testing.assert_array_equal(actual.numpy(), values[[3 + 2 * 20 + 3 * 2]])


@pytest.mark.gpu
def test_typed_pointer_alias_executes_on_cuda(compile_tilelang):
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("A CUDA device is required")
    source = TensorView.create("source", [32], "float32")
    output = TensorView.create("output", [32], "float32")
    thread = ir.Var("thread", "int32")
    pointer = native("MakeAccessPtrFromRegion", region_of(source, [thread], [1]), 1, False)
    selected = TensorView.create("selected", [1], "float32")
    body = ir.RegionStmt(
        "tirx.launch_thread",
        [ir.StringImm("threadIdx.x"), ir.const(32)],
        [thread],
        None,
        [selected.declare(pointer), output.store(thread, selected.load(0) + 3.0)],
    )
    function = tirx.Function([source.tensor, output.tensor], body, ret_type=ir.TupleType([])).with_attr("global_symbol", "pointer_kernel")
    lowered = lower_function(function)
    remaining = []
    stmt_functor.post_order_visit(
        lowered, lambda node: remaining.append(node) if isinstance(node, ir.Call) and node.op.name == "tl.access_ptr" else None
    )
    assert not remaining
    version = device.compute_version.replace(".", "")
    target = Target({"kind": "cuda", "arch": f"sm_{version}", "backend_config": {"cuda": {"compiler": "nvcc"}}}, host="c")
    executable = compile_tilelang(lowered, target=target)
    values = np.arange(32, dtype="float32")
    actual = runtime.empty((32,), "float32", device=device)
    executable(runtime.tensor(values, device=device), actual)
    np.testing.assert_array_equal(actual.numpy(), values + 3.0)
