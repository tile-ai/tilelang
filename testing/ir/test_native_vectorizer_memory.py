"""Vectorized native tensor accesses retain one lane axis and exact values."""

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx

from test_native_reducer_contracts import execute, function, integer, objects, tensor


def vectorize(value):
    before = ir.save_json(value)
    result = tvm_ffi.get_global_func("tl.transform.VectorizeLoop")(True)(ir.IRModule({"main": value}))["main"]
    assert ir.save_json(value) == before
    assert tirx.analysis.check_well_formed(result)
    assert not tirx.analysis.undefined_vars(result)
    return result


def execute_vector(value, arrays):
    from tilelang.testing.packed import compile_packed_function

    # The TileLang compute emitter owns vector C types and lane operations.
    # Generic native C remains a valid scalar ABI path in the fallback tests.
    value = value.with_attr("target", tvm_tirx.target.Target("c"))
    compiled = compile_packed_function(value, target="c", c_codegen="compute")
    tensors = [tvm_tirx.runtime.tensor(array) for array in arrays]
    compiled(*tensors)
    return [tensor.numpy() for tensor in tensors]


@pytest.mark.parametrize("load_rank", [1, 2])
@pytest.mark.parametrize("dtype", ["float32", "int32", "float32x2"])
def test_uniform_leading_index_keeps_vectorized_memory(load_rank, dtype):
    a = tensor("A", [8] if load_rank == 1 else [2, 4], dtype, "global")
    b = tensor("B", [2, 4], dtype, "global")
    block, lane = ir.Var("block", "int32"), ir.Var("lane", "int32")
    flat = block * 2 + lane
    indices = [flat // 4, flat % 4]
    load = ir.TensorLoad(a, [flat] if load_rank == 1 else indices)
    body = ir.TensorStore(b, indices, load)
    inner = ir.For(lane, integer(0), integer(2), integer(1), ir.ForKind.VECTORIZED, [body])
    outer = ir.For(block, integer(0), integer(4), integer(1), ir.ForKind.DEFAULT, [inner])
    result = vectorize(function([a, b], [outer]).with_attr("tirx.noalias", True))

    # The row is uniform over each pair. It must stay scalar, while the last
    # index and value retain the two lanes of the eliminated inner loop.
    assert len(objects(result, ir.For)) == 1
    stores = objects(result, ir.TensorStore)
    assert len(stores) == 1
    store = stores[0]
    assert store.indices[0].ty.dtype.lanes == 1
    assert store.indices[1].ty.dtype.lanes == 2
    assert store.value.ty.dtype.lanes == 2 * ir.PrimType(dtype).dtype.lanes
    for value in objects(result, ir.TensorLoad):
        assert all(index.ty.dtype.lanes == 1 for index in value.indices[:-1])
        assert value.indices[-1].ty.dtype.lanes == 2

    if dtype != "float32x2":
        source = np.arange(8, dtype=dtype).reshape((8,) if load_rank == 1 else (2, 4))
        actual = execute_vector(result, [source, np.zeros((2, 4), dtype=dtype)])[-1]
        np.testing.assert_array_equal(actual, source.reshape(2, 4))


@pytest.mark.parametrize("load_rank", [1, 2])
def test_varying_leading_index_preserves_diagonal_access(load_rank):
    a = tensor("A", [4] if load_rank == 1 else [4, 4], "int32", "global")
    b = tensor("B", [4, 4], "int32", "global")
    lane = ir.Var("lane", "int32")
    load = ir.TensorLoad(a, [lane] if load_rank == 1 else [lane, lane])
    loop = ir.For(
        lane,
        integer(0),
        integer(4),
        integer(1),
        ir.ForKind.VECTORIZED,
        [ir.TensorStore(b, [lane, lane], load)],
    )
    result = vectorize(function([a, b], [loop]))
    (remaining,) = objects(result, ir.For)
    assert remaining.kind == ir.ForKind.DEFAULT
    source = np.arange(4 if load_rank == 1 else 16, dtype="int32").reshape((4,) if load_rank == 1 else (4, 4))
    expected = np.zeros((4, 4), dtype="int32")
    np.fill_diagonal(expected, source if load_rank == 1 else np.diag(source))
    actual = execute(result, [source, np.zeros_like(expected)], f"vector-diagonal-{load_rank}")[-1]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("load_rank", [1, 2])
@pytest.mark.parametrize("dtype", ["float32", "int32", "float32x2"])
def test_original_memory_without_noalias_keeps_scalar_fallback(load_rank, dtype):
    a = tensor("A", [8] if load_rank == 1 else [2, 4], dtype, "global")
    b = tensor("B", [2, 4], dtype, "global")
    block, lane = ir.Var("block", "int32"), ir.Var("lane", "int32")
    flat = block * 2 + lane
    indices = [flat // 4, flat % 4]
    load = ir.TensorLoad(a, [flat] if load_rank == 1 else indices)
    body = ir.TensorStore(b, indices, load)
    inner = ir.For(lane, integer(0), integer(2), integer(1), ir.ForKind.VECTORIZED, [body])
    outer = ir.For(block, integer(0), integer(4), integer(1), ir.ForKind.DEFAULT, [inner])
    value = function([a, b], [outer])
    assert "tirx.noalias" not in value.attrs
    result = vectorize(value)
    assert len(objects(result, ir.For)) == 2
    assert all(loop.kind == ir.ForKind.DEFAULT for loop in objects(result, ir.For))
    if dtype != "float32x2":
        source = np.arange(8, dtype=dtype).reshape((8,) if load_rank == 1 else (2, 4))
        actual = execute(result, [source, np.zeros((2, 4), dtype=dtype)], f"scalar-memory-{load_rank}-{dtype}")[-1]
        np.testing.assert_array_equal(actual, source.reshape(2, 4))


@pytest.mark.parametrize("load_rank", [1, 2])
def test_overlapping_parameter_storage_retains_scalar_iteration_order(load_rank):
    from tilelang.testing.packed import compile_packed_function

    a = ir.Var("A", tirx.TensorType([8] if load_rank == 1 else [2, 4], "int32", scope="global", align=4))
    b = ir.Var("B", tirx.TensorType([2, 4], "int32", scope="global", align=4))
    block, lane = ir.Var("block", "int32"), ir.Var("lane", "int32")
    flat = block * 2 + lane
    indices = [flat // 4, flat % 4]
    load = ir.TensorLoad(a, [flat] if load_rank == 1 else indices)
    inner = ir.For(lane, integer(0), integer(2), integer(1), ir.ForKind.VECTORIZED, [ir.TensorStore(b, indices, load)])
    outer = ir.For(block, integer(0), integer(4), integer(1), ir.ForKind.DEFAULT, [inner])
    result = vectorize(function([a, b], [outer]))
    assert len(objects(result, ir.For)) == 2
    compiled = compile_packed_function(result, target="c")
    storage = np.arange(9, dtype="int32")
    source = storage[:-1].reshape((8,) if load_rank == 1 else (2, 4))
    destination = storage[1:].reshape(2, 4)
    compiled(tvm_ffi.from_dlpack(source), tvm_ffi.from_dlpack(destination))
    np.testing.assert_array_equal(storage, np.zeros(9, dtype="int32"))


@pytest.mark.parametrize("dtype,begin,end", [("uint8", 255, 1), ("uint8", 0, 2), ("int16", 0, 2), ("int32", 4, 0), ("int64", 3, 1)])
def test_unsupported_or_empty_loop_does_not_gain_vector_execution(dtype, begin, end):
    output = tensor("output", [4], "int32", "global")
    lane = ir.Var("lane", dtype)
    widened = ir.prim.Cast("int32", lane)
    loop = ir.For(
        lane,
        integer(begin, dtype),
        integer(end, dtype),
        integer(1, dtype),
        ir.ForKind.VECTORIZED,
        [ir.TensorStore(output, [widened], widened + 7)],
    )
    result = vectorize(function([output], [loop]))
    assert all(value.kind == ir.ForKind.DEFAULT for value in objects(result, ir.For))
    expected = np.full(4, -999, dtype="int32")
    for value in range(begin, end):
        expected[value] = value + 7
    actual = execute(result, [np.full(4, -999, dtype="int32")], f"scalar-header-{dtype}-{begin}-{end}")[0]
    np.testing.assert_array_equal(actual, expected)
