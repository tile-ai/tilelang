import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, tirx
from tvm_tirx.ir import prim
from tvm_tirx.ir.transform import PassContext
from tvm_tirx.target import Target

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView
from tilelang.transform import _tirx_passes

pytestmark = pytest.mark.usefixtures("native_layouts")


def allocations(value):
    result = []
    stmt_functor.post_order_visit(
        value,
        lambda node: result.append(node)
        if isinstance(node, ir.Bind) and isinstance(node.value, ir.Call) and node.value.op.name == "tirx.alloc_tensor"
        else None,
    )
    return result


def sequential_storage():
    first = TensorView.create("first", [128], "float32")
    second = TensorView.create("second", [128], "float32")
    output = TensorView.create("output", [2], "float32")
    index = ir.Var("first_index", "int32")
    next_index = ir.Var("second_index", "int32")
    body = ir.SeqStmt(
        [
            first.allocate(),
            ir.For(index, 0, 128, 1, ir.ForKind.DEFAULT, first.store(index, prim.Cast("float32", index))),
            output.store(0, first.load(127)),
            second.allocate(),
            ir.For(next_index, 0, 128, 1, ir.ForKind.DEFAULT, second.store(next_index, prim.Cast("float32", next_index + 1))),
            output.store(1, second.load(127)),
        ]
    )
    function = (
        tirx.Function([output.tensor], body, ret_type=ir.TupleType([]))
        .with_attr("target", Target("c"))
        .with_attr("global_symbol", "storage_test")
    )
    return _tirx_passes.LowerTensorLayouts()(ir.IRModule({"main": function}))


@pytest.mark.parametrize("detect_inplace", [False, True])
def test_tilelang_storage_disables_lifetime_reuse_but_preserves_execution(detect_inplace):
    module = sequential_storage()
    with PassContext(config={"tl.storage_rewrite_detect_inplace": detect_inplace}):
        result = tvm_ffi.get_global_func("tl.transform.StorageRewrite")()(module)
    assert len(allocations(result)) == 2
    assert tirx.analysis.check_well_formed(result)
    executable = tirx.compile(result, target="c")
    output = runtime.empty((2,), "float32")
    executable(output)
    np.testing.assert_array_equal(output.numpy(), [127, 128])


def test_tilelang_storage_preserves_allocations_unlike_native_reuse():
    module = sequential_storage()
    reused = tirx.transform.StorageRewrite()(module)
    separate = _tirx_passes.StorageRewrite(enable_reuse=False, detect_inplace=False)(module)
    assert len(allocations(reused)) == 1
    assert len(allocations(separate)) == 2


def test_storage_rewrite_offsets_retained_pointer_regions():
    first = TensorView.create("first", [128], "float32", scope="shared.dyn")
    second = TensorView.create("second", [128], "float32", scope="shared.dyn")
    output = TensorView.create("output", [1], "float32")
    region = ir.TensorRegion(second.tensor, [ir.Range(0, 128)], ir.TensorRegionType())
    pointer = ir.Call(
        "tirx.address_of",
        [second.load(0)],
        attrs={"tl.access_region": region, "tl.access_mask": ir.const(1)},
    )
    first_region = ir.TensorRegion(first.tensor, [ir.Range(0, 128)], ir.TensorRegionType())
    first_pointer = ir.Call(
        "tirx.address_of",
        [first.load(0)],
        attrs={"tl.access_region": first_region, "tl.access_mask": ir.const(1)},
    )
    body = [
        first.allocate(),
        second.allocate(),
        first.store(0, 1),
        second.store(0, 2),
        ir.Evaluate(ir.Call("tirx.call_extern", ["consume", pointer, first_pointer], ty="void")),
        output.store(0, first.load(0) + second.load(0)),
    ]
    function = tirx.Function([output.tensor], body).with_attr("target", Target("cuda"))
    with Target("cuda"):
        module = _tirx_passes.LowerTensorLayouts()(ir.IRModule({"main": function}))
        module = _tirx_passes.FlattenTensor()(module)
        module = _tirx_passes.StorageRewrite()(module)
    calls = []
    stmt_functor.post_order_visit(
        module,
        lambda node: calls.append(node) if isinstance(node, ir.Call) and node.attrs and "tl.access_region" in node.attrs else None,
    )
    assert len(calls) == 2
    origins = []
    for call in calls:
        access = call.attrs["tl.access_region"]
        load = call.args[0]
        assert access.source.same_as(load.source)
        ir.assert_structural_equal(access.region[0].begin, load.indices[0])
        origins.append(int(access.region[0].begin))
        assert int(access.region[0].extent) == 128
    assert max(origins) > 0
    assert tirx.analysis.check_well_formed(module)


def test_pointer_type_rewrite_uses_native_implementation():
    module = sequential_storage()
    actual = tvm_ffi.get_global_func("tl.transform.PointerValueTypeRewrite")()(module)
    expected = tirx.transform.PointerValueTypeRewrite()(module)
    ir.assert_structural_equal(actual, expected)


@pytest.mark.parametrize("pass_name", ["StorageRewrite", "PointerValueTypeRewrite"])
def test_storage_metadata_keeps_its_logical_element_units(pass_name):
    from tilelang._compat import ir as tlir

    shared = TensorView.create("shared", [128], "float32", scope="shared")
    thread = ir.Var("thread", "int32")
    pointer = ir.Call(
        "tirx.ptr_byte_offset",
        [shared.tensor.data, ir.const(0)],
        attrs={
            "tl.access_region": ir.TensorRegion(shared.tensor, [ir.Range(0, 128)], ir.TensorRegionType()),
            "tl.access_extent": ir.const(128),
            "tl.access_mask": ir.const(1),
        },
    )
    consume = ir.Evaluate(ir.Call("tirx.call_extern", ["consume", pointer], ty="void"))
    body = [shared.allocate(), shared.store(prim.Ramp(thread * 4, 1, 4), prim.Broadcast(ir.const(1.0), 4)), consume]
    launch = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(32)], [thread], None, body)
    function = tirx.Function([], launch).with_attr("target", Target("cuda"))
    module = tvm_ffi.get_global_func(f"tl.transform.{pass_name}")()(ir.IRModule({"main": function}))
    assert tlir.check_well_formed(module)
    calls = []
    stmt_functor.post_order_visit(module, lambda node: calls.append(node) if isinstance(node, ir.Call) and node.attrs else None)
    access = next(call for call in calls if "tl.access_region" in call.attrs)
    allocation = allocations(module)[0].var
    assert access.attrs["tl.access_region"].source.same_as(allocation)
    assert allocation.ty.dtype == ir.PrimType("float32")
    assert int(access.attrs["tl.access_extent"]) == 128
