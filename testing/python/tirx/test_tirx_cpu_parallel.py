import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, tirx
from tvm_tirx.ir import prim
from tvm_tirx.ir.transform import PassContext
from tvm_tirx.target import Target

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def grid(variable, extent, body, axis=0, begin=0, step=1):
    return ir.For(
        variable,
        begin,
        extent,
        step,
        ir.ForKind.DEFAULT,
        body,
        annotations={"tl.cpu_grid_dim": ir.const(axis), "tl.cpu_num_threads": ir.const(4)},
    )


def lower(body, parameters=(), target="c", attributes=None, config=None):
    function = tirx.Function(list(parameters), body).with_attr("global_symbol", "parallel_grid")
    for key, value in (attributes or {}).items():
        function = function.with_attr(key, value)
    module = tirx.transform.BindTarget(Target(target))(ir.IRModule({"main": function}))
    with PassContext(config=config or {}):
        module = tvm_ffi.get_global_func("tl.cpu.transform.MaterializeCPUParallelGrid")()(module)
    assert tirx.analysis.check_well_formed(module)
    ir.assert_structural_equal(module, ir.load_json(ir.save_json(module)))
    return module


def parallel_loops(module):
    loops = []
    stmt_functor.post_order_visit(module, lambda node: loops.append(node) if isinstance(node, ir.For) else None)
    assert all("tl.cpu_grid_dim" not in loop.annotations for loop in loops)
    return [loop for loop in loops if loop.kind == ir.ForKind.PARALLEL]


@pytest.mark.parametrize("target,expected", [("c", 2), ("llvm", 1)])
def test_multidimensional_disjoint_grid_executes(target, expected):
    output = TensorView.create("output", [4, 8], "int32")
    row, column = ir.Var("row", "int32"), ir.Var("column", "int32")
    module = lower(grid(row, 4, grid(column, 8, output.store([row, column], row * 10 + column), axis=1)), [output.tensor], target)
    assert len(parallel_loops(module)) == expected
    executable = tirx.compile(module, target=target)
    actual = runtime.empty((4, 8), "int32")
    if target == "c":
        executable.jit(options=["-fopenmp"]).main(actual)
    else:
        executable(actual)
    np.testing.assert_array_equal(actual.numpy(), np.arange(4)[:, None] * 10 + np.arange(8))


@pytest.mark.parametrize("pattern", ["neighbor", "same", "cast_overflow", "cast_cancellation"])
def test_colliding_accesses_stay_serial(pattern):
    storage = TensorView.create("storage", [512], "int32")
    index = ir.Var("index", "int32")
    position = {
        "neighbor": index,
        "same": ir.const(0),
        "cast_overflow": prim.Cast("uint8", index),
        "cast_cancellation": prim.Cast("int32", prim.Cast("int16", index)) - index,
    }[pattern]
    value = storage.load(index + 1) if pattern == "neighbor" else index
    module = lower(grid(index, 511, storage.store(position, value)), [storage.tensor])
    assert not parallel_loops(module)


@pytest.mark.parametrize(
    "attributes,expected",
    [
        ({}, True),
        ({"tirx.noalias": False}, False),
        ({"tl.non_restrict_param_indices": [ir.const(0)]}, True),
        ({"tl.non_restrict_param_indices": [ir.const(0), ir.const(1)]}, False),
    ],
)
def test_parameter_alias_contract(attributes, expected):
    source = TensorView.create("source", [16], "int32")
    output = TensorView.create("output", [16], "int32")
    index = ir.Var("index", "int32")
    module = lower(grid(index, 16, output.store(index, source.load(index))), [source.tensor, output.tensor], attributes=attributes)
    assert bool(parallel_loops(module)) == expected


@pytest.mark.parametrize("tile_stride,expected", [(128, True), (64, False)])
def test_flattened_vector_tiles_prove_grid_independence(tile_stride, expected):
    output = TensorView.create("output", [512 * 512], "int32")
    column, row, element = (ir.Var(name, "int32") for name in ("column", "row", "element"))
    offset = row * 65536 + element // 32 * 512 + column * tile_stride + element % 32 * 4
    store = output.store(prim.Ramp(offset, 1, 4), prim.Broadcast(ir.const(1), 4))
    body = grid(column, 4, grid(row, 4, ir.For(element, 0, 4096, 1, ir.ForKind.DEFAULT, store), axis=1))
    module = lower(body, [output.tensor])
    assert bool(parallel_loops(module)) == expected
    if expected:
        executable = tirx.compile(module, target="c")
        actual = runtime.tensor(np.zeros((512 * 512,), dtype="int32"))
        executable.jit(options=["-fopenmp"]).main(actual)
        np.testing.assert_array_equal(actual.numpy(), np.ones((512 * 512,), dtype="int32"))


@pytest.mark.parametrize("reset", ["every", "first"])
def test_iteration_private_scratch_and_carried_storage(reset):
    scratch = TensorView.create("scratch", [2], "int32", scope="local")
    output = TensorView.create("output", [4], "int32")
    index, element = ir.Var("index", "int32"), ir.Var("element", "int32")
    initialize = ir.For(element, 0, 2, 1, ir.ForKind.DEFAULT, scratch.store(element, 1))
    if reset == "first":
        initialize = ir.If(index == 0, initialize, None)
    module = lower(
        [scratch.allocate(), grid(index, 4, [initialize, output.store(index, scratch.load(0)), scratch.store(0, scratch.load(0) + 1)])],
        [output.tensor],
    )
    assert bool(parallel_loops(module)) == (reset == "every")
    executable = tirx.compile(module, target="c")
    actual = runtime.empty((4,), "int32")
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), [1, 1, 1, 1] if reset == "every" else [1, 2, 3, 4])


@pytest.mark.parametrize("stride,expected", [(1, False), (2, True), (4, True)])
def test_vector_access_lanes_participate_in_independence_proof(stride, expected):
    output = TensorView.create("output", [64], "int32")
    index = ir.Var("index", "int32")
    body = output.store(prim.Ramp(index * stride, 1, 2), prim.Broadcast(index, 2))
    module = lower(grid(index, 16, body), [output.tensor])
    assert bool(parallel_loops(module)) == expected


@pytest.mark.parametrize("minimum,expected", [(0, True), (16, True), (17, False)])
def test_minimum_trip_count(minimum, expected):
    output = TensorView.create("output", [16], "int32")
    index = ir.Var("index", "int32")
    module = lower(grid(index, 16, output.store(index, index)), [output.tensor], config={"tl.cpu_parallel_min_trip": minimum})
    assert bool(parallel_loops(module)) == expected


@pytest.mark.parametrize("pure", [True, False])
def test_scalar_external_side_effects(pure):
    output = TensorView.create("output", [16], "int32")
    index = ir.Var("index", "int32")
    call = (tirx.call_pure_extern if pure else tirx.call_extern)("scalar_helper", index, ty="int32")
    module = lower(grid(index, 16, output.store(index, call)), [output.tensor])
    assert bool(parallel_loops(module)) == pure


def test_overflowing_unsigned_address_math_stays_serial():
    output = TensorView.create("output", [1 << 32], "int32")
    index = ir.Var("index", "int32")
    address = prim.Cast("uint32", index) * ir.const(1 << 31, "uint32")
    module = lower(grid(index, 4, output.store(address, index)), [output.tensor])
    assert not parallel_loops(module)


def test_native_c_grid_uses_requested_openmp_threads():
    output = TensorView.create("output", [64], "int32")
    index = ir.Var("index", "int32")
    thread_number = tirx.call_pure_extern("omp_get_thread_num", ty="int32")
    module = lower(grid(index, 64, output.store(index, thread_number)), [output.tensor])
    assert len(parallel_loops(module)) == 1
    module = module.with_attr("codegen.preamble", "#include <omp.h>\n")
    executable = tirx.compile(module, target="c")
    actual = runtime.empty((64,), "int32")
    executable.jit(options=["-fopenmp"]).main(actual)
    np.testing.assert_array_equal(np.unique(actual.numpy()), np.arange(4))
