"""Allocation ownership preserves TMA alignment without changing reuse defaults."""

import uuid

import pytest
import tvm_ffi
from tvm_tirx import IRModule, ir, runtime, tirx
from tvm_tirx.target import Target

import tilelang


def tensor(name, count=17, scope="shared.dyn", alignment=64):
    return ir.Var(name, tirx.TensorType([count], "float32", scope=scope, data_alignment=alignment))


def allocate(value):
    return ir.Bind(value, ir.Call("tirx.alloc_tensor", [], ty_args=[value.ty]))


def store(value, data, index=0):
    return ir.TensorStore(value, [index], ir.const(data, "float32") if isinstance(data, int) else data)


def pair(*, scope="shared.dyn", live=True, first_size=17, second_alignment=64, required=None):
    first = tensor("ordinary", first_size, scope)
    second = tensor("xs", 128, scope, second_alignment)
    output = tensor("output", 2, "global")
    body = [allocate(first), store(first, 3)]
    if not live:
        body.append(store(output, ir.TensorLoad(first, [0])))
    body.extend([allocate(second), store(second, 5)])
    if live:
        body.append(store(output, ir.TensorLoad(first, [0])))
    body.append(store(output, ir.TensorLoad(second, [0]), 1))
    attributes = {"global_symbol": "main", "target": Target("c")}
    if required is not None:
        attributes["tl.smem_alignment_map"] = {"xs": required}
    return IRModule({"main": tirx.Function([output], body, attrs=ir.DictAttrs(attributes))})


def bindings(module):
    result = []
    tvm_ffi.structural_walk(module, (ir.Bind, result.append), order="pre")
    return result


def allocations(module):
    return [node for node in bindings(module) if isinstance(node.value, ir.Call) and node.value.op.name == "tirx.alloc_tensor"]


def scope_with_property(value):
    scope = runtime.MemoryScope.register("test.storage_alignment." + uuid.uuid4().hex)
    runtime.MemoryScope.def_attr("runtime.memory_alloc_property")[scope] = value
    return scope


@pytest.mark.parametrize("property_value", [0, 1, 2])
def test_tilelang_preserves_allocation_property_for_its_later_owner(property_value):
    original = pair(scope=scope_with_property(property_value))
    before = ir.save_json(original)
    after = tilelang.transform.StorageRewrite()(original)
    assert len(allocations(after)) == 2
    assert tirx.analysis.check_well_formed(after)
    assert ir.save_json(original) == before


@pytest.mark.parametrize("scope", ["shared", "shared.dyn"])
def test_tilelang_does_not_reuse_disjoint_lifetimes(scope):
    original = pair(scope=scope, live=False, first_size=128)
    with ir.transform.PassContext(config={"tl.storage_rewrite_detect_inplace": True}):
        after = tilelang.transform.StorageRewrite()(original)
    # This is lifetime reuse, not a pointwise in-place update. Its ownership
    # remains with the shared arena planner regardless of the inplace option.
    assert len(allocations(after)) == 2
    assert tirx.analysis.check_well_formed(after)


@pytest.mark.parametrize("width,required", [(24, 256), (32, 1024), (40, 256)])
@pytest.mark.parametrize("map_value", ["int", "intimm"])
def test_tilelang_defers_arena_offsets_until_alignment_metadata_is_consumed(width, required, map_value):
    value = required if map_value == "int" else ir.const(required, "int32")
    original = pair(first_size=width, required=value)
    before = ir.save_json(original)
    after_storage = tilelang.transform.StorageRewrite()(original)
    assert len(allocations(after_storage)) == 2
    assert any(node.var.name == "xs" for node in allocations(after_storage))
    assert "tl.smem_alignment_map" in after_storage["main"].attrs
    merged = tilelang.transform.MergeSharedMemoryAllocations(disable_reuse=True)(after_storage)
    arena = allocations(merged)
    assert len(arena) == 1
    assert arena[0].var.ty.data_alignment >= required
    view = next(node for node in bindings(merged) if node.var.name == "xs")
    assert view.value.op.name == "tirx.decl_tensor"
    pointer = view.value.args[0]
    assert pointer.op.name == "tirx.ptr_byte_offset"
    offset = int(pointer.args[1])
    assert offset == ((width * 4 + required - 1) // required) * required
    assert offset % required == 0 and offset >= width * 4
    assert int(arena[0].var.ty.shape[0]) >= offset + 128 * 4
    assert tirx.analysis.check_well_formed(merged)
    assert ir.save_json(original) == before


@pytest.mark.parametrize("bad_alignment", [0, 3, -1])
def test_deferred_invalid_alignment_is_not_erased_by_storage_planning(bad_alignment):
    original = pair(required=bad_alignment)
    prepared = tilelang.transform.StorageRewrite()(original)
    assert len(allocations(prepared)) == 2
    with pytest.raises(ValueError):
        tilelang.transform.MergeSharedMemoryAllocations(disable_reuse=True)(prepared)


def test_fork_only_storage_factory_is_not_needed():
    assert tvm_ffi.get_global_func("tirx.transform.StorageRewriteWithOptions", allow_missing=True) is None
    assert tvm_ffi.get_global_func("tl.transform.StorageRewrite") is not None
