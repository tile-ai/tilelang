"""ThreadSync preserves finite-width hazards and proves unit-axis swizzles."""

import pytest
import tilelang
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target


@pytest.fixture(params=["shared", "shared.dyn"])
def scope(request):
    return request.param


def _tensor(name, size, dtype, scope="global"):
    return ir.Var(name, tirx.TensorType((size,), dtype, scope=scope, layout="default"))


def _allocate(tensor):
    return ir.Bind(tensor, ir.Call("tirx.alloc_tensor", [], ty_args=[tensor.ty]))


def _nodes(value, kind):
    found = []
    tvm_ffi.structural_walk(value, (kind, found.append))
    return found


def _barriers(value):
    return [
        call
        for call in _nodes(value, ir.Call)
        if isinstance(call.op, ir.Op) and call.op.name in {"tirx.gpu_storage_sync", "tl.storage_sync"}
    ]


def _apply(function, scope):
    assert tirx.analysis.check_well_formed(function)
    serialized = ir.save_json(function)
    try:
        result = tilelang.transform.ThreadSync(scope)(ir.IRModule({"main": function}))["main"]
    finally:
        assert ir.save_json(function) == serialized
    assert tirx.analysis.check_well_formed(result)
    return result


def _grid(variables, body, y=1, z=1):
    dimensions = [4, 4, 128, y, z]
    tags = ["blockIdx.x", "blockIdx.y", "threadIdx.x", "threadIdx.y", "threadIdx.z"]
    for tag, extent, variable in reversed(list(zip(tags, dimensions, variables))):
        body = ir.RegionStmt("tirx.launch_thread", [ir.StringImm(tag), ir.const(extent)], [variable], {}, body)
    return body


def _variables():
    return [ir.Var(name, "int32") for name in ("block_0", "block_1", "tx", "ty", "tz")]


def _function(params, allocations, body):
    return tirx.Function(params, [*allocations, body]).with_attr("target", Target("cuda"))


def _swizzle(thread, row):
    return thread // 16 * 1024 + row * 16 + (thread % 16 // 8 + row % 8 // 4) % 2 * 8 + thread % 8


@pytest.mark.parametrize("byte_view", [False, True])
def test_injective_width48_swizzle_with_unit_thread_axes(scope, byte_view):
    addresses = [_swizzle(thread, row) for thread in range(48) for row in range(64)]
    assert sorted(addresses) == list(range(3072))
    shared = _tensor("xs", 3072, "bfloat16", scope)
    allocations = [_allocate(shared)]
    if byte_view:
        arena = _tensor("arena", 6144, "uint8", scope)
        pointer = ir.Call(
            "tirx.ptr_byte_offset", [tirx.tensor_data_ptr(arena), ir.const(0, "int64")], ty=ir.PtrType(shared.ty.dtype, shared.ty.scope)
        )
        allocations = [_allocate(arena), ir.Bind(shared, ir.Call("tirx.decl_tensor", [pointer], ty_args=[shared.ty]))]
    variables = _variables()
    tx, row = variables[2], ir.Var("p0_3", "int32")
    loop = ir.For(
        row,
        ir.const(0),
        ir.const(64),
        ir.const(1),
        ir.ForKind.UNROLLED,
        [ir.TensorStore(shared, [_swizzle(tx, row)], ir.const(1, "bfloat16"))],
    )
    function = _function([], allocations, _grid(variables, [ir.If(tx < 48, loop, None)]))
    result = _apply(function, scope)
    assert not _barriers(result)
    assert len(_nodes(result, ir.For)) == 1
    assert len(_nodes(result, ir.TensorStore)) == 1


def test_width48_swizzle_keeps_cross_thread_read_barrier_outside_partial_branch(scope):
    shared = _tensor("xs", 3072, "bfloat16", scope)
    output = _tensor("output", 128, "bfloat16")
    variables = _variables()
    tx, row = variables[2], ir.Var("row", "int32")
    owned = [{_swizzle(thread, r) for r in range(64)} for thread in range(48)]
    for thread in range(48):
        assert _swizzle((thread + 1) % 48, 0) in owned[(thread + 1) % 48]
        assert _swizzle((thread + 1) % 48, 0) not in owned[thread]
    loop = ir.For(
        row,
        ir.const(0),
        ir.const(64),
        ir.const(1),
        ir.ForKind.UNROLLED,
        [ir.TensorStore(shared, [_swizzle(tx, row)], ir.const(1, "bfloat16"))],
    )
    reader = ir.TensorStore(output, [tx], ir.TensorLoad(shared, [_swizzle((tx + 1) % 48, ir.const(0))]))
    function = _function([output], [_allocate(shared)], _grid(variables, [ir.If(tx < 48, loop, None), ir.If(tx < 48, reader, None)]))
    result = _apply(function, scope)
    assert len(_barriers(result)) == 1
    branches = _nodes(result, ir.If)
    assert len(branches) == 2 and all(not _barriers(branch) for branch in branches)


@pytest.mark.parametrize("y,z", [(1, 1), (2, 1), (1, 2), (2, 2)])
@pytest.mark.parametrize("separate_planes", [False, True])
def test_nonunit_axes_remain_part_of_thread_identity(scope, y, z, separate_planes):
    variables = _variables()
    tx, ty, tz = variables[2:]
    shared = _tensor("shared", 128 * y * z, "int32", scope)
    index = tx + 128 * (ty + y * tz) if separate_planes else tx
    body = [ir.TensorStore(shared, [index], ir.const(1)), ir.TensorStore(shared, [index], ir.const(2))]
    function = _function([], [_allocate(shared)], _grid(variables, body, y, z))
    result = _apply(function, scope)
    expected = int(not separate_planes and y * z > 1)
    assert len(_barriers(result)) == expected
    assert len(_nodes(result, ir.TensorStore)) == 2


@pytest.mark.parametrize("y,z", [(2, 1), (1, 2)])
def test_real_cross_axis_conflict_still_rejects_partial_warp_barrier(scope, y, z):
    variables = _variables()
    tx = variables[2]
    shared = _tensor("shared", 128, "int32", scope)
    body = [ir.TensorStore(shared, [tx], ir.const(1)), ir.TensorStore(shared, [tx], ir.const(2))]
    function = _function([], [_allocate(shared)], _grid(variables, [ir.If(tx < 48, body, None)], y, z))
    with pytest.raises(RuntimeError, match="not warp-uniform"):
        _apply(function, scope)


@pytest.mark.parametrize("kind", ["int16_ramp", "uint32_product", "uint64_product"])
def test_wrapped_arithmetic_keeps_a_real_cross_thread_raw(scope, kind):
    tx = ir.Var("tx", "int32")
    if kind == "int16_ramp":

        def wrapped(value):
            return (value + 32768) % 65536 - 32768

        writes = [{wrapped(thread + lane * 32767) % 251 for lane in range(3)} for thread in range(2)]
        assert writes == [{0, 137, 249}, {1, 113, 250}]
        reads = [248, 249]
        # Native Ramp uses 32/64-bit arithmetic. Narrow each resulting lane
        # explicitly to retain the original int16 wrapping address oracle.
        base = ir.prim.Cast("int32", ir.prim.Cast("int16", tx))
        vector = ir.prim.Cast("int16x3", ir.prim.Ramp(base, ir.const(32767, "int32"), 3))
        indices = ir.prim.Cast("int64x3", vector) % ir.prim.Broadcast(ir.const(251, "int64"), 3)
        reader = ir.prim.Cast("int64", tx) + 248
    else:
        bits = 32 if kind == "uint32_product" else 64
        dtype = f"uint{bits}"
        base, factor = (1 << bits) - 12, (1 << bits) - 5
        writes = [{((base + thread * 4 + lane) * factor) % (1 << bits) for lane in range(3)} for thread in range(2)]
        assert writes == [{50, 55, 60}, {30, 35, 40}]
        assert base * factor > (1 << 63) - 1
        reads = [60, 50]
        vector = ir.prim.Ramp(ir.prim.Cast(dtype, tx * 4) + ir.const(base, dtype), ir.const(1, dtype), 3)
        indices = ir.prim.Cast("int64x3", vector * ir.prim.Broadcast(ir.const(factor, dtype), 3))
        reader = ir.prim.Cast("int64", 60 - tx * 10)
    assert writes[0].isdisjoint(writes[1])
    assert reads[1] in writes[0] and reads[1] not in writes[1]
    shared = _tensor("shared", 256, "int32", scope)
    output = _tensor("output", 2, "int32")
    body = [
        ir.TensorStore(shared, [indices], ir.prim.Broadcast(ir.const(1), 3)),
        ir.TensorStore(output, [tx], ir.TensorLoad(shared, [reader])),
    ]
    launch = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(2)], [tx], {}, body)
    result = _apply(_function([output], [_allocate(shared)], launch), scope)
    assert len(_barriers(result)) == 1
    sequence = result.body[-1].body
    assert isinstance(sequence[0], ir.TensorStore)
    assert _barriers(sequence[1])
    assert isinstance(sequence[2], ir.TensorStore)


@pytest.mark.parametrize("dtype", ["int16", "uint16"])
def test_proven_in_range_lane_cast_keeps_thread_private_accesses(scope, dtype):
    tx = ir.Var("tx", "int32")
    writes = [{4 * thread + lane for lane in range(3)} for thread in range(2)]
    assert writes[0].isdisjoint(writes[1])
    assert all(4 * thread + 2 in writes[thread] for thread in range(2))
    shared, output = _tensor("shared", 8, "int32", scope), _tensor("output", 2, "int32")
    narrow = ir.prim.Cast(f"{dtype}x3", ir.prim.Ramp(tx * 4, 1, 3))
    index = ir.prim.Cast("int64x3", narrow)
    body = [
        ir.TensorStore(shared, [index], ir.prim.Broadcast(ir.const(1), 3)),
        ir.TensorStore(output, [tx], ir.TensorLoad(shared, [tx * 4 + 2])),
    ]
    launch = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(2)], [tx], {}, body)
    result = _apply(_function([output], [_allocate(shared)], launch), scope)
    assert not _barriers(result)


@pytest.mark.parametrize("cross_thread", [False, True])
def test_uint63_lane_cast_respects_thread_ownership(scope, cross_thread):
    tx = ir.Var("tx", "int32")
    writes = [{4 * thread + lane for lane in range(3)} for thread in range(2)]
    reads = [4 * (1 - thread if cross_thread else thread) + 2 for thread in range(2)]
    assert writes[0].isdisjoint(writes[1])
    for thread in range(2):
        owner = 1 - thread if cross_thread else thread
        assert reads[thread] in writes[owner]
        assert reads[thread] not in writes[1 - owner]
    shared = _tensor("shared", 8, "int32", scope)
    output = _tensor("output", 2, "int32")
    base = ir.prim.Cast("int64", tx) * ir.const(4, "int64")
    ramp = ir.prim.Ramp(base, ir.const(1, "int64"), 3)
    # The cast is a real unsigned 63-bit IR value; no 63-bit tensor storage
    # or device codegen is needed to check the barrier analysis contract.
    index = ir.prim.Cast("int64x3", ir.prim.Cast("uint63x3", ramp))
    reader = (1 - tx if cross_thread else tx) * 4 + 2
    body = [
        ir.TensorStore(shared, [index], ir.prim.Broadcast(ir.const(1), 3)),
        ir.TensorStore(output, [tx], ir.TensorLoad(shared, [reader])),
    ]
    launch = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(2)], [tx], {}, body)
    result = _apply(_function([output], [_allocate(shared)], launch), scope)
    assert len(_barriers(result)) == int(cross_thread)
