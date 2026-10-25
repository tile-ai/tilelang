"""The native allocator consumes compatibility without assuming transitivity."""

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, sym, tirx

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView


def _nodes(root, kind):
    result = []
    stmt_functor.post_order_visit(root, lambda node: result.append(node) if isinstance(node, kind) else None)
    return result


def _tensor(name, size, scope="shared.dyn", dtype="int32", alignment=16):
    return ir.Var(name, tirx.TensorType([size], dtype, scope=scope, data_alignment=alignment))


def _allocate(tensor):
    return ir.Bind(tensor, ir.Call("tirx.alloc_tensor", [], ty_args=[tensor.ty]))


def _scope(body, aliases, aiv_storages=None):
    attrs = {"tl.buffer_alias_map": aliases}
    if aiv_storages is not None:
        attrs["tl.aiv_storage_map"] = aiv_storages
    return ir.RegionStmt("tirx.scope", [], [], attrs, body)


def _merge(body, parameters=(), *, disable_reuse=False, alignment=16, attrs=None):
    function = tirx.Function(list(parameters), body, ret_type=ir.TupleType([])).with_attr("global_symbol", "main")
    if attrs:
        function = function.with_attrs(attrs)
    assert tirx.analysis.check_well_formed(function)
    result = tvm_ffi.get_global_func("tl.transform.MergeUBAllocations")(alignment, disable_reuse)(ir.IRModule({"main": function}))["main"]
    assert tirx.analysis.check_well_formed(result)
    restored = ir.load_json(ir.save_json(result))
    ir.assert_structural_equal(result, restored)
    assert tirx.analysis.check_well_formed(restored)
    return result


def _allocations(function):
    return [
        binding
        for binding in _nodes(function, ir.Bind)
        if isinstance(binding.value, ir.Call) and binding.value.op.name == "tirx.alloc_tensor"
    ]


def _offsets(function):
    return {
        binding.var.name: int(sym.Analyzer().simplify(binding.value.args[0].args[1]))
        for binding in _nodes(function, ir.Bind)
        if isinstance(binding.value, ir.Call) and binding.value.op.name == "tirx.decl_tensor"
    }


@pytest.mark.parametrize("disable_reuse", [False, True])
def test_packer_respects_complete_allocation_conflicts(disable_reuse):
    first, second, third = [_tensor(name, size) for name, size in [("first", 64), ("second", 32), ("third", 32)]]
    contract = {first: [second, third], second: [first], third: [first]}
    tensors = [first, second, third]
    body = [_allocate(tensor) for tensor in tensors] + [ir.TensorStore(tensor, [0], index) for index, tensor in enumerate(tensors)]
    after = _merge(_scope(body, contract), disable_reuse=disable_reuse)
    assert not [region for region in _nodes(after, ir.RegionStmt) if "tl.buffer_alias_map" in region.attrs]
    offsets = _offsets(after)
    assert offsets["second"] + 128 <= offsets["third"] or offsets["third"] + 128 <= offsets["second"]
    if disable_reuse:
        for name in ("second", "third"):
            assert offsets["first"] + 256 <= offsets[name] or offsets[name] + 128 <= offsets["first"]
    else:
        assert any(offsets["first"] < offsets[name] + 128 and offsets[name] < offsets["first"] + 256 for name in ("second", "third"))
    assert len(_allocations(after)) == 1


@pytest.mark.parametrize("disable_reuse", [False, True])
def test_subcore_ub_allocations_have_independent_address_spaces(disable_reuse):
    common, left, right = [_tensor(name, size) for name, size in [("common", 32), ("left", 128), ("right", 64)]]
    tensors = [common, left, right]
    body = [_allocate(tensor) for tensor in tensors] + [ir.TensorStore(tensor, [0], index) for index, tensor in enumerate(tensors)]
    after = _merge(_scope(body, {tensor: [] for tensor in tensors}, {left: 0, right: 1}), disable_reuse=disable_reuse)
    offsets = _offsets(after)
    (allocation,) = _allocations(after)
    assert int(allocation.var.ty.shape[0]) == (32 + max(128, 64)) * 4
    assert offsets == {"common": 0, "left": 128, "right": 128}
    assert all("tl.aiv_storage_map" not in region.attrs for region in _nodes(after, ir.RegionStmt))


def test_ss_allocations_remain_disjoint_and_aligned_even_with_alias_contract():
    first, second = [_tensor(name, 2, "shared.ss", "uint32") for name in ("first", "second")]
    body = [
        _allocate(first),
        _allocate(second),
        ir.TensorStore(first, [0], ir.const(1, "uint32")),
        ir.TensorStore(second, [0], ir.const(2, "uint32")),
    ]
    after = _merge(_scope(body, {first: [second], second: [first]}))
    assert sorted(_offsets(after).values()) == [0, 32]
    (allocation,) = _allocations(after)
    assert int(allocation.var.ty.shape[0]) == 64
    assert allocation.var.ty.data_alignment == 32


@pytest.mark.parametrize(
    "scope,alignment", [("shared.dyn", 16), ("shared.l1", 512), ("shared.l0a", 512), ("shared.l0b", 512), ("shared.l0c", 1024)]
)
def test_scope_alignment_and_empty_contract_enable_sequential_allocation(scope, alignment):
    tensors = [_tensor(name, 3, scope, "int8") for name in ("first", "second")]
    after = _merge(_scope([_allocate(tensor) for tensor in tensors], {}))
    assert list(_offsets(after).values()) == [0, alignment]
    (allocation,) = _allocations(after)
    assert int(allocation.var.ty.shape[0]) == 2 * alignment
    assert allocation.var.ty.data_alignment == alignment


def test_absent_contract_preserves_allocations_and_other_metadata():
    tensors = [_tensor(name, 8) for name in ("first", "second")]
    body = ir.RegionStmt("tirx.scope", [], [], {"tl.aiv_storage_map": {tensors[0]: 0}}, [_allocate(tensor) for tensor in tensors])
    after = _merge(body)
    assert len(_allocations(after)) == 2
    assert "tl.aiv_storage_map" in after.body.seq[0].attrs


def test_function_level_contract_preserves_native_tensor_definitions():
    tensors = [_tensor(name, 8) for name in ("first", "second")]
    after = _merge([_allocate(tensor) for tensor in tensors], attrs={"tl.buffer_alias_map": {}})
    assert len(_allocations(after)) == 1
    assert "tl.buffer_alias_map" not in after.attrs
    assert all(binding.var.same_as(tensors[index]) for index, binding in enumerate(_nodes(after, ir.Bind)[1:]))


def test_arena_stays_inside_loop_that_defines_dynamic_allocation_sizes():
    index = ir.Var("index", "int64")
    tensors = [_tensor(name, index + 4) for name in ("first", "second")]
    body = ir.For(index, 0, 4, 1, ir.ForKind.DEFAULT, _scope([_allocate(tensor) for tensor in tensors], {}))
    after = _merge(body)
    assert isinstance(after.body.seq[0], ir.For)
    assert len(_allocations(after.body.seq[0].body)) == 1
    assert any(variable.same_as(index) for variable in _nodes(_allocations(after)[0].var.ty.shape[0], ir.Var))


def test_arena_is_inserted_before_sibling_scopes_without_escaping_device_entry():
    first, second = [_tensor(name, 4) for name in ("first", "second")]
    scopes = [_scope([_allocate(tensor)], {tensor: []}) for tensor in (first, second)]
    entry = ir.RegionStmt("tirx.device_entry", [], [], {}, scopes)
    after = _merge(entry)
    assert after.body.seq[0].op.name == "tirx.device_entry"
    assert isinstance(after.body.seq[0].body.seq[0], ir.Bind)
    assert len(_allocations(after)) == 1


def test_distinct_device_entries_do_not_share_an_arena():
    entries = []
    for ordinal in range(2):
        tensors = [_tensor(f"storage_{ordinal}_{position}", 4) for position in range(2)]
        entries.append(ir.RegionStmt("tirx.device_entry", [], [], {}, _scope([_allocate(tensor) for tensor in tensors], {})))
    after = _merge(entries)
    assert len(_allocations(after)) == 2
    assert all(isinstance(entry, ir.RegionStmt) and len(_allocations(entry)) == 1 for entry in after.body.seq)


def test_packed_fp4_and_vector_elements_have_physical_byte_sizes():
    first = _tensor("first", 5, dtype="float4_e2m1fn", alignment=1)
    second = _tensor("second", 5, dtype="int32x4", alignment=1)
    after = _merge(_scope([_allocate(first), _allocate(second)], {}), alignment=1)
    assert int(_allocations(after)[0].var.ty.shape[0]) == 83
    assert sorted(_offsets(after).values()) == [0, 80]


def test_strided_aliases_keep_geometry_and_byte_offsets_numerically():
    first = TensorView.create("first", [2, 3], "int32", scope="shared.dyn", strides=[8, 2], elem_offset=1)
    second = TensorView.create("second", [5], "int32", scope="shared.dyn")
    alias = TensorView.create("alias", [3], "int32", scope="shared.dyn")
    output = _tensor("output", 3, scope="global")
    shifted = ir.Call("tirx.ptr_byte_offset", [first.data, ir.const(4, "int64")], ty=first.data.ty)
    body = [first.allocate(), second.allocate(), alias.declare(shifted), first.store([1, 2], 37), alias.store(0, 19), second.store(4, 53)]
    body += [
        ir.TensorStore(output, [0], first.load([1, 2])),
        ir.TensorStore(output, [1], first.load([0, 0])),
        ir.TensorStore(output, [2], second.load(4)),
    ]
    after = _merge(_scope(body, {}), [output])
    assert _nodes(after, ir.TensorStore)[0].dest.same_as(first.tensor)
    assert _nodes(after, ir.TensorStore)[0].dest.ty.layout.same_as(first.tensor.ty.layout)
    module = tirx.compile(after, target="llvm")
    result = runtime.tensor(np.zeros(3, dtype="int32"))
    module(result)
    np.testing.assert_array_equal(result.numpy(), [37, 19, 53])


@pytest.mark.parametrize("alignment", [0, -1, 3, 1 << 28])
def test_invalid_arena_alignment_is_rejected(alignment):
    with pytest.raises(ValueError, match="positive power of two"):
        _merge([], alignment=alignment)
