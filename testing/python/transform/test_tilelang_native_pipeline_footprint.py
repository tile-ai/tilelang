"""Imprecise tensor footprints may pipeline only across disjoint storage effects."""

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, target, tirx
from tvm_tirx.ir import prim

import tilelang
from tilelang.testing.ir import collect_nodes
from tilelang.testing.packed import compile_packed_function


LLVM = target.Target({"kind": "llvm", "mcpu": "generic"})


def _tensor(name, size, dtype="int32", scope="global", *, alignment=0):
    return ir.Var(name, tirx.TensorType((size,), dtype, scope=scope, align=alignment, layout="default"))


def _allocate(tensor):
    return ir.Bind(tensor, ir.Call("tirx.alloc_tensor", [], ty_args=[tensor.ty]))


def _view(tensor, backing, byte_offset=0):
    pointer = tirx.tensor_data_ptr(backing)
    if byte_offset:
        pointer = tirx.ptr_byte_offset(pointer, ir.const(byte_offset, "int64"), ty=pointer.ty)
    return ir.Bind(tensor, ir.Call("tirx.decl_tensor", [pointer], ty_args=[tensor.ty]))


def _function(params, prefix, statements, *, extent=5, index=None, noalias=True):
    if index is None:
        index = ir.Var("iteration", "int32")
    loop = ir.For(index, 0, extent, 1, ir.ForKind.DEFAULT, ir.SeqStmt(statements), annotations={"num_stages": 2})
    function = tirx.Function(params, ir.SeqStmt([*prefix, loop])).with_attr({"global_symbol": "main", "target": LLVM})
    if noalias:
        function = function.with_attr("tirx.noalias", True)
    assert tirx.analysis.check_well_formed(function)
    return function


def _accesses(function, *, unknown_address=False):
    before = ir.save_json(function)
    result = tvm_ffi.get_global_func("tl.testing.CollectPipelineAccesses")(function)
    assert bool(result["unknown"])
    assert bool(result["unknown_address"]) == unknown_address
    assert ir.save_json(function) == before
    return result


def _plan(function):
    before = ir.save_json(function)
    result = tilelang.transform.PipelinePlanning()(ir.IRModule({"main": function}))
    assert ir.save_json(function) == before
    assert tirx.analysis.check_well_formed(result)
    original_loop = collect_nodes(function, ir.For)[0]
    planned_loop = collect_nodes(result["main"], ir.For)[0]
    # Planning only schedules statements. It must not narrow or replace an
    # unbounded/wrapping index by the collector's full-axis approximation.
    ir.assert_structural_equal(original_loop.body, planned_loop.body)
    assert "num_stages" not in planned_loop.annotations
    return result, planned_loop


def _assert_sequential(function):
    result, loop = _plan(function)
    assert "software_pipeline_stage" not in loop.annotations
    assert "software_pipeline_order" not in loop.annotations
    assert "software_pipeline_async_stages" not in loop.annotations
    assert "tl_pipelined_num_stages" not in loop.annotations
    return result


def _inject(planned):
    result = tilelang.transform.InjectSoftwarePipeline()(planned)
    assert tirx.analysis.check_well_formed(result)
    assert not tirx.analysis.undefined_vars(result["main"])
    return result["main"]


def _offset_copy(dtype="int64", extent=5, *, noalias=True, output_size=16, alignment=0):
    source = _tensor("source", 8, alignment=alignment)
    output = _tensor("output", output_size, alignment=alignment)
    offsets = _tensor("offsets", 1, dtype)
    temporary = _tensor("temporary", 1, scope="shared")
    offset = ir.Var("offset", dtype)
    index = ir.Var("iteration", "int32")
    destination = index.astype(dtype) + offset
    function = _function(
        [source, output, offsets],
        [_allocate(temporary), ir.Bind(offset, ir.TensorLoad(offsets, [0]))],
        [
            ir.TensorStore(temporary, [0], ir.TensorLoad(source, [index])),
            ir.TensorStore(output, [destination], ir.TensorLoad(temporary, [0]) + 3),
        ],
        extent=extent,
        index=index,
        noalias=noalias,
    )
    return function


@pytest.mark.parametrize("dtype", ["int64", "uint64"])
@pytest.mark.parametrize("extent, offset", [(1, 0), (5, 3)])
def test_disjoint_unbounded_output_keeps_pipeline_and_numeric_result(dtype, extent, offset):
    function = _offset_copy(dtype, extent)
    _accesses(function)
    planned, loop = _plan(function)
    stages = list(loop.annotations["software_pipeline_stage"])
    assert stages[0] < stages[1]
    assert int(loop.annotations["tl_pipelined_num_stages"]) == 2
    injected = _inject(planned)

    data = np.arange(8, dtype="int32") * 7 - 13
    expected = np.full(16, -99, dtype="int32")
    expected[offset : offset + extent] = data[:extent] + 3
    for candidate in (function, injected):
        executable = compile_packed_function(candidate, c_codegen="compute")
        output = runtime.tensor(np.full(16, -99, dtype="int32"))
        executable(runtime.tensor(data), output, runtime.tensor(np.array([offset], dtype=dtype)))
        np.testing.assert_array_equal(output.numpy(), expected)


@pytest.mark.parametrize("contract", ["unspecified", "explicit_may_alias"])
def test_shifted_external_alias_keeps_cross_iteration_order(contract):
    # Both pointer contracts permit the four-byte shift used by this witness.
    function = _offset_copy(extent=8, noalias=contract == "explicit_may_alias", output_size=8, alignment=4)
    if contract == "explicit_may_alias":
        function = function.with_attr("tl.may_alias_param_indices", [0, 1])
    _accesses(function)
    planned = _assert_sequential(function)
    injected = _inject(planned)

    # C points one element past A in the same allocation. Each sequential
    # store changes the next iteration's input. Prefetching it early does not
    # implement the original function, even though A and C have different Vars.
    initial = np.arange(9, dtype="int32") * 11 + 5
    expected = initial.copy()
    for index in range(8):
        expected[index + 1] = expected[index] + 3
    prefetched = initial.copy()
    saved = prefetched[:8].copy()
    prefetched[1:] = saved + 3
    assert not np.array_equal(prefetched, expected)

    for candidate in (function, injected):
        executable = compile_packed_function(candidate, c_codegen="compute")
        backing = initial.copy()
        # NumPy's DLPack views carry the shifted data pointer with byte_offset
        # zero, as required by this native function's tensor parameter ABI.
        source = tvm_ffi.from_dlpack(backing[:8])
        output = tvm_ffi.from_dlpack(backing[1:])
        executable(source, output, runtime.tensor(np.array([0], dtype="int64")))
        np.testing.assert_array_equal(backing, expected)


@pytest.mark.parametrize("alias_kind", ["same_tensor", "declared_view"])
def test_same_root_shifted_alias_preserves_numeric_result(alias_kind):
    backing = _tensor("backing", 9)
    offsets = _tensor("offsets", 1, "int64")
    temporary = _tensor("temporary", 1, scope="shared")
    offset = ir.Var("offset", "int64")
    index = ir.Var("iteration", "int32")
    prefix = [_allocate(temporary), ir.Bind(offset, ir.TensorLoad(offsets, [0]))]
    destination = backing
    position = index.astype("int64") + offset + 1
    if alias_kind == "declared_view":
        destination = _tensor("destination", 8)
        prefix.append(_view(destination, backing, 4))
        position = index.astype("int64") + offset
    function = _function(
        [backing, offsets],
        prefix,
        [
            ir.TensorStore(temporary, [0], ir.TensorLoad(backing, [index])),
            ir.TensorStore(destination, [position], ir.TensorLoad(temporary, [0]) + 3),
        ],
        index=index,
        extent=8,
    )
    _accesses(function)
    injected = _inject(_assert_sequential(function))
    initial = np.arange(9, dtype="int32") * 11 + 5
    expected = initial.copy()
    for index in range(8):
        expected[index + 1] = expected[index] + 3
    for candidate in (function, injected):
        executable = compile_packed_function(candidate, c_codegen="compute")
        actual = runtime.tensor(initial)
        executable(actual, runtime.tensor(np.array([0], dtype="int64")))
        np.testing.assert_array_equal(actual.numpy(), expected)


def _imprecise_index(kind):
    if kind in {"int64", "uint64"}:
        index = ir.Var("unbounded_index", kind)
        return index, 8, 1, [index]
    if kind == "wrapping_ramp":
        # Preserve [255, 0] with native 32-bit Ramp plus per-lane narrowing.
        ramp = prim.Ramp(ir.const(255, "int32"), ir.const(1, "int32"), 2)
        return prim.Cast(ir.PrimType("uint8x2"), ramp), 255, 2, []
    if kind == "wrapping_cast":
        ramp = prim.Ramp(ir.const(254, "int32"), ir.const(1, "int32"), 4)
        return prim.Cast(ir.PrimType("uint8x4"), ramp), 255, 4, []
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", ["int64", "uint64", "wrapping_ramp", "wrapping_cast"])
@pytest.mark.parametrize("hazard", ["read_write", "write_read", "write_write"])
def test_small_view_full_axis_never_proves_disjointness(kind, hazard):
    backing = _tensor("backing", 260)
    view = _tensor("small_view", 4)
    temporary = _tensor("temporary", 4, scope="shared")
    unknown, fixed, lanes, params = _imprecise_index(kind)
    temporary_index = 0 if lanes == 1 else prim.Ramp(ir.const(0), ir.const(1), lanes)
    if hazard == "read_write":
        statements = [
            ir.TensorStore(temporary, [temporary_index], ir.TensorLoad(view, [unknown])),
            ir.TensorStore(view, [fixed], ir.TensorLoad(temporary, [0]) + 1),
        ]
        region_kind = "reads"
    else:
        value = ir.TensorLoad(temporary, [0]) + 1
        if lanes > 1:
            value = prim.Broadcast(value, lanes)
        statements = [
            ir.TensorStore(temporary, [0], ir.TensorLoad(view, [fixed])),
            ir.TensorStore(view, [unknown], value),
        ]
        if hazard == "write_write":
            statements[0] = ir.TensorStore(view, [fixed], ir.const(17))
            value = ir.const(23) if lanes == 1 else prim.Broadcast(ir.const(23), lanes)
            statements[1] = ir.TensorStore(view, [unknown], value)
        region_kind = "writes"
    function = _function([backing, *params], [_view(view, backing), _allocate(temporary)], statements)
    accesses = _accesses(function)
    full_axis = [
        region
        for region in accesses[region_kind]
        if region.source.same_as(view) and int(region.region[0].begin) == 0 and int(region.region[0].extent) == 4
    ]
    assert full_axis
    assert fixed >= 4
    # The approximation [0, 4) misses a possible access at 8 or 255 in the
    # larger backing allocation. Storage identity must retain the dependency.
    _assert_sequential(function)


def test_imprecise_value_load_does_not_taint_known_shared_write():
    source = _tensor("source", 8)
    other = _tensor("other", 8)
    output = _tensor("output", 8)
    temporary = _tensor("temporary", 2, scope="shared")
    offset = ir.Var("offset", "int64")
    index = ir.Var("iteration", "int32")
    function = _function(
        [source, other, output, offset],
        [_allocate(temporary)],
        [
            ir.TensorStore(temporary, [0], ir.TensorLoad(source, [index])),
            ir.TensorStore(temporary, [1], ir.TensorLoad(other, [offset])),
            ir.TensorStore(output, [index], ir.TensorLoad(temporary, [0]) + ir.TensorLoad(temporary, [1])),
        ],
        index=index,
    )
    _accesses(function)
    planned, loop = _plan(function)
    stages = list(loop.annotations["software_pipeline_stage"])
    assert stages[0] == stages[1] < stages[2]
    _inject(planned)


def test_potential_read_read_alias_does_not_prevent_pipeline():
    source = _tensor("source", 8)
    other = _tensor("other", 8)
    temporary = _tensor("temporary", 1, scope="shared")
    offset = ir.Var("offset", "int64")
    index = ir.Var("iteration", "int32")
    function = _function(
        [source, other, offset],
        [_allocate(temporary)],
        [
            ir.TensorStore(temporary, [0], ir.TensorLoad(source, [index])),
            ir.Evaluate(ir.TensorLoad(other, [offset]) + ir.TensorLoad(temporary, [0])),
        ],
        index=index,
        noalias=False,
    )
    _accesses(function)
    planned, loop = _plan(function)
    stages = list(loop.annotations["software_pipeline_stage"])
    assert stages[0] < stages[1]
    _inject(planned)


@pytest.mark.parametrize("kind", ["raw_pointer", "opaque_global_state", "opaque_with_tensor"])
@pytest.mark.parametrize("paired_imprecise", [False, True])
def test_unknown_addresses_and_opaque_effects_remain_sequential(kind, paired_imprecise):
    source = _tensor("source", 8)
    params = [source]
    if kind == "raw_pointer":
        destination_ptr = ir.Var("destination_ptr", ir.PtrType("int32", scope="shared"))
        source_ptr = ir.Var("source_ptr", ir.PtrType("int32", scope="global"))
        params += [destination_ptr, source_ptr]
        operation = ir.Call("tl.ptx_cp_async", [destination_ptr, source_ptr, ir.const(1)])
    elif kind == "opaque_with_tensor":
        operation = tirx.call_extern("unknown_effect", tirx.tensor_data_ptr(source), ty=ir.PrimType("void"))
    else:
        operation = tirx.call_extern("unknown_global_state", ty=ir.PrimType("void"))
    statements = [ir.Evaluate(operation)]
    if paired_imprecise:
        offset = ir.Var("offset", "int64")
        params.append(offset)
        statements.append(ir.TensorStore(source, [offset], ir.const(7)))
    function = _function(params, [], statements)
    _accesses(function, unknown_address=True)
    _assert_sequential(function)
