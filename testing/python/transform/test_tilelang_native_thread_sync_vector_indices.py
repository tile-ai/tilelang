"""ThreadSync reasons about every physical vector lane with scalar constraints."""

import numpy as np
import pytest
import tvm_tirx
from tvm_tirx import ir, runtime, tirx
from tvm_tirx.ir import prim

import tilelang
from tilelang.testing.ir import collect_calls, collect_nodes
from tilelang.testing.packed import compile_packed_function


def _cast(value, dtype):
    return prim.Cast(ir.PrimType(dtype), value)


def _ramp(base, stride, lanes, dtype="int32"):
    if isinstance(base, int):
        base = ir.const(base, dtype)
    if ir.PrimType(dtype).dtype.bits < 32:
        # Native Ramp accepts 32/64-bit elements. Calculate these bounded
        # fixture lanes in int32 and narrow each result: modulo 2**bits gives
        # exactly the original fixed-width lane oracle, including signed wrap.
        wide = prim.Ramp(_cast(base, "int32"), ir.const(stride, "int32"), lanes)
        return _cast(wide, f"{dtype}x{lanes}")
    return prim.Ramp(base, ir.const(stride, dtype), lanes)


def _case(write_index, read_index, threads=64, size=2048):
    tx = ir.Var("tx", ir.PrimType("int32"))
    output = ir.Var("output", tirx.TensorType((threads * 2,), "int32", layout="default"))
    shared = ir.Var("shared", tirx.TensorType((size,), "int32", scope="shared", layout="default"))
    write_index, read_index = write_index(tx), read_index(tx)
    value = prim.Broadcast(ir.const(1, "int32"), write_index.ty.dtype.lanes)
    output_index = _ramp(tx * 2, 1, 2) if read_index.ty.dtype.lanes == 2 else tx
    read_value = ir.TensorLoad(shared, [read_index])
    body = [
        ir.Bind(shared, ir.Call("tirx.alloc_tensor", [], ty_args=[shared.ty])),
        ir.TensorStore(shared, [write_index], value),
    ]
    if read_index.ty.dtype.lanes == 1 or read_index.ty.dtype.lanes == 2:
        body.append(ir.TensorStore(output, [output_index], read_value))
    else:
        body.append(ir.Evaluate(read_value))
    launch = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(threads)], [tx], {}, body)
    function = tirx.Function([output], launch, ret_type=ir.PrimType("void")).with_attr(
        "target", tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_90"})
    )
    assert tirx.analysis.check_well_formed(function)
    return function


def _lower(function):
    result = tilelang.transform.ThreadSync("shared")(ir.IRModule({"main": function}))["main"]
    assert tirx.analysis.check_well_formed(result)
    return result


def _barriers(function):
    return collect_calls(function, op="tirx.gpu_storage_sync")


@pytest.mark.parametrize(
    "read_partition, expected",
    [("different_threads", 1), ("same_thread", 0), ("disjoint", 0)],
)
def test_mixed_lane_counts_use_scalar_access_domains(read_partition, expected):
    def read(tx):
        base = tx * (2 if read_partition == "different_threads" else 8)
        if read_partition == "disjoint":
            base = base + 1024
        return _cast(_ramp(base, 1, 2), "int64x2")

    function = _case(lambda tx: _cast(_ramp(tx * 8, 1, 8), "int64x8"), read)
    lowered = _lower(function)
    barriers = _barriers(lowered)
    assert len(barriers) == expected
    if expected:
        sequence = lowered.body[0].body
        writer = next(i for i, statement in enumerate(sequence) if isinstance(statement, ir.TensorStore))
        barrier = next(i for i, statement in enumerate(sequence) if isinstance(statement, ir.Evaluate))
        reader = max(i for i, statement in enumerate(sequence) if isinstance(statement, ir.TensorStore))
        assert writer < barrier < reader


@pytest.mark.parametrize("cross_thread, expected", [(False, 0), (True, 1)])
def test_descending_vector_lanes_preserve_thread_ownership(cross_thread, expected):
    function = _case(
        lambda tx: _cast(_ramp(tx * 8 + 7, -1, 8), "int64x8"),
        lambda tx: _cast(_ramp(((tx + 1) % 64 if cross_thread else tx) * 8, 1, 2), "int64x2"),
    )
    assert len(_barriers(_lower(function))) == expected


@pytest.mark.parametrize("kind", ["signed_ramp", "unsigned_ramp", "narrowing_cast"])
def test_wrapping_lane_values_cannot_hide_a_cross_thread_read(kind):
    # Both writer sets are disjoint across two threads. The reader in thread 1
    # touches a lane written only by thread 0. Evaluating just cast/ramp
    # endpoints would miss the high middle lane or the wrapped final lane.
    if kind == "signed_ramp":

        def write(tx):
            return _cast(_ramp(_cast(tx, "int8"), 127, 3, "int8"), "int64x3") + prim.Broadcast(ir.const(128, "int64"), 3)

        def read(tx):
            return _cast(tx, "int64") + 254

        sets = [{128, 255, 126}, {129, 0, 127}]
        reads = [254, 255]
    elif kind == "unsigned_ramp":

        def write(tx):
            return _cast(_ramp(_cast(tx + 250, "uint8"), 4, 3, "uint8"), "int64x3")

        def read(tx):
            return _cast(tx, "int64") + 1

        sets = [{250, 254, 2}, {251, 255, 3}]
        reads = [1, 2]
    else:

        def write(tx):
            return _cast(_cast(_ramp(_cast(tx + 250, "int16"), 4, 3, "int16"), "uint8x3"), "int64x3")

        def read(tx):
            return _cast(tx, "int64") + 1

        sets = [{250, 254, 2}, {251, 255, 3}]
        reads = [1, 2]
    assert sets[0].isdisjoint(sets[1])
    assert reads[1] in sets[0] and reads[1] not in sets[1]
    function = _case(write, read, threads=2, size=256)
    assert len(_barriers(_lower(function))) == 1


def test_unknown_vector_addresses_keep_a_possible_conflict():
    tx = ir.Var("tx", ir.PrimType("int32"))
    unknown = ir.Var("indices", ir.PrimType("int32x2"))
    shared = ir.Var("shared", tirx.TensorType((256,), "int32", scope="shared", layout="default"))
    output = ir.Var("output", tirx.TensorType((128,), "int32", layout="default"))
    launch = ir.RegionStmt(
        "tirx.launch_thread",
        [ir.StringImm("threadIdx.x"), ir.const(64)],
        [tx],
        {},
        [
            ir.Bind(shared, ir.Call("tirx.alloc_tensor", [], ty_args=[shared.ty])),
            ir.TensorStore(shared, [_ramp(tx * 4, 1, 4)], prim.Broadcast(ir.const(1), 4)),
            ir.TensorStore(output, [_ramp(tx * 2, 1, 2)], ir.TensorLoad(shared, [unknown])),
        ],
    )
    function = tirx.Function([output, unknown], launch, ret_type=ir.PrimType("void")).with_attr(
        "target", tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_90"})
    )
    assert len(_barriers(_lower(function))) == 1


def test_subbyte_vector_lanes_keep_shared_word_conflicts():
    tx = ir.Var("tx", ir.PrimType("int32"))
    shared = ir.Var("shared", tirx.TensorType((256,), "int4", scope="shared", layout="default"))
    launch = ir.RegionStmt(
        "tirx.launch_thread",
        [ir.StringImm("threadIdx.x"), ir.const(32)],
        [tx],
        {},
        [
            ir.Bind(shared, ir.Call("tirx.alloc_tensor", [], ty_args=[shared.ty])),
            ir.TensorStore(shared, [_cast(_ramp(tx * 8, 1, 2), "int64x2")], prim.Broadcast(ir.const(1, "int4"), 2)),
            ir.TensorStore(shared, [_cast(_ramp(((tx + 1) % 32) * 8 + 2, 1, 2), "int64x2")], prim.Broadcast(ir.const(2, "int4"), 2)),
        ],
    )
    function = tirx.Function([], launch, ret_type=ir.PrimType("void")).with_attr(
        "target", tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_90"})
    )
    lowered = _lower(function)
    assert len(_barriers(lowered)) == 1
    assert len(collect_nodes(lowered, ir.TensorStore)) == 2


@pytest.mark.parametrize("dtype,threads", [("int8", 8), ("uint8", 16), ("int16", 64), ("uint16", 64), ("uint32", 64)])
def test_nonwrapping_narrow_lanes_preserve_same_thread_elision(dtype, threads):
    def index(tx, lanes):
        return _cast(_ramp(_cast(tx * 8, dtype), 1, lanes, dtype), f"int64x{lanes}")

    function = _case(lambda tx: index(tx, 8), lambda tx: index(tx, 2), threads=threads)
    assert len(_barriers(_lower(function))) == 0


@pytest.mark.parametrize("dtype", ["int16", "uint16", "uint32"])
def test_wider_integer_wrap_boundaries_keep_possible_conflicts(dtype):
    bits = int(dtype.lstrip("uint"))
    if dtype.startswith("uint"):
        base = (1 << bits) - 6
        stride = 4
        expected = [{250, 254, 2}, {251, 255, 3}]

        def reader(tx):
            return _cast(tx, "int64") + 1

        def writer(tx):
            value = _cast(_ramp(_cast(tx, dtype) + ir.const(base, dtype), stride, 3, dtype), "int64x3")
            return value % prim.Broadcast(ir.const(256, "int64"), 3)
    else:
        stride = (1 << (bits - 1)) - 1
        expected = [{0, 137, 249}, {1, 113, 250}]

        def reader(tx):
            return _cast(tx, "int64") + 248

        def writer(tx):
            value = _cast(_ramp(_cast(tx, dtype), stride, 3, dtype), "int64x3")
            return value % prim.Broadcast(ir.const(251, "int64"), 3)

    # Reader thread 1 uses element 2 (unsigned) or 249 (signed), written by
    # thread 0 and absent from its own write set. The modulo just bounds the
    # tensor allocation; wrap already occurs in the fixed-width vector value
    # (the final lane cast for narrow types, or the uint32 Ramp itself).
    read_index = 2 if dtype.startswith("uint") else 249
    assert expected[0].isdisjoint(expected[1])
    assert read_index in expected[0] and read_index not in expected[1]
    assert len(_barriers(_lower(_case(writer, reader, threads=2, size=256)))) == 1


@pytest.mark.parametrize(
    "dtype,base,stride",
    [("int8", 0, 127), ("uint8", 250, 4), ("int16", 0, 32767), ("uint16", 65530, 4), ("uint32", (1 << 32) - 6, 4)],
)
def test_native_llvm_lane_values_match_fixed_width_oracle(dtype, base, stride):
    # Execute the original vector expression, not the analysis scalarization.
    # This checks that the wrap examples describe real native IR semantics.
    thread = ir.Var("thread", ir.PrimType("int32"))
    output = ir.Var("output", tirx.TensorType((3,), "int64", layout="default"))
    base_expr = _cast(thread, dtype) + ir.const(base, dtype)
    vector = _cast(_ramp(base_expr, stride, 3, dtype), "int64x3")
    function = tirx.Function([output, thread], [ir.TensorStore(output, [_ramp(0, 1, 3)], vector)]).with_attr("global_symbol", "main")
    executable = compile_packed_function(function, c_codegen="compute")
    bits = int(dtype.lstrip("uint"))
    for thread_value in (0, 1):
        expected = []
        for lane in range(3):
            value = (base + thread_value + lane * stride) % (1 << bits)
            if dtype.startswith("int") and value >= (1 << (bits - 1)):
                value -= 1 << bits
            expected.append(value)
        actual = runtime.tensor(np.zeros(3, dtype="int64"))
        executable(actual, thread_value)
        np.testing.assert_array_equal(actual.numpy(), expected)
