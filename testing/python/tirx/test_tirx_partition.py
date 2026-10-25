import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, sym, tirx
from tvm_tirx.target import Target

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView


def native(name, *arguments):
    return tvm_ffi.get_global_func(f"tl.{name}")(*arguments)


def parallel_loop(tensor, shape, *, increment=False):
    indices = [ir.Var(f"axis_{axis}", "int32") for axis in range(len(shape))]
    value = ir.const(0, "int32")
    for index, extent in zip(indices, shape):
        value = value * extent + index
    if increment:
        value = tensor.load(indices) + 1
    body = tensor.store(indices, value)
    for index, extent in reversed(list(zip(indices, shape))):
        body = ir.For(index, 0, extent, 1, ir.ForKind.PARALLEL, body)
    return body


def partition(loop, layout, thread, thread_range, padding=False):
    analyzer = sym.Analyzer()
    analyzer.bind(thread, thread_range)
    return native("PartitionLoop", loop, thread, analyzer, layout, padding)


@pytest.mark.parametrize("shape,width,thread_begin", [((8, 16), 1, 0), ((8, 16), 4, 0), ((4, 8), 2, 16), ((7, 13), 1, 0), ((4,), 1, 0)])
def test_partition_covers_logical_space_once(native_layouts, shape, width, thread_begin):
    output = TensorView.create("output", shape, "int32")
    original = parallel_loop(output, shape, increment=True)
    thread_range = ir.Range.from_min_extent(thread_begin, 32)
    layout = native("PlanLoopPartition", original, thread_range, width)
    thread = ir.Var("thread", "int32")
    lowered = partition(original, layout, thread, thread_range, padding=True)
    assert original.kind == ir.ForKind.PARALLEL
    assert lowered.kind == ir.ForKind.DEFAULT
    body = ir.For(thread, thread_begin, thread_begin + 32, 1, ir.ForKind.DEFAULT, lowered)
    function = tirx.Function([output.tensor], body, ret_type=ir.TupleType([])).with_attr("global_symbol", "partition_once")
    assert tirx.analysis.check_well_formed(function)
    ir.assert_structural_equal(function, ir.load_json(ir.save_json(function)))
    executable = tirx.compile(function, target="c")
    actual = runtime.tensor(np.zeros(shape, dtype="int32"))
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), np.ones(shape, dtype="int32"))


def test_partition_replica_side_effects_execute_once(native_layouts):
    output = TensorView.create("output", [8], "int32")
    index = ir.Var("index", "int32")
    marker = ir.RegionStmt("tl.parallel_multiplicity", [], [], None, output.store(index, output.load(index) + 1))
    original = ir.For(index, 0, 8, 1, ir.ForKind.PARALLEL, marker)
    layout = native_layouts.Fragment([8], forward_fn=lambda index, replica: (index + replica * 8, [0]), replicate=4)
    thread_range = ir.Range(0, 32)
    thread = ir.Var("thread", "int32")
    lowered = partition(original, layout, thread, thread_range)
    remaining = []
    stmt_functor.post_order_visit(
        lowered,
        lambda node: remaining.append(node) if isinstance(node, ir.RegionStmt) and node.op.name == "tl.parallel_multiplicity" else None,
    )
    assert not remaining
    body = ir.For(thread, 0, 32, 1, ir.ForKind.DEFAULT, lowered)
    function = tirx.Function([output.tensor], body, ret_type=ir.TupleType([])).with_attr("global_symbol", "replica_effect")
    executable = tirx.compile(function, target="c")
    actual = runtime.tensor(np.zeros(8, dtype="int32"))
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), np.ones(8, dtype="int32"))


def test_partition_marks_constant_serial_loops_for_unrolling(native_layouts):
    output = TensorView.create("output", [8], "int32")
    index = ir.Var("index", "int32")
    original = ir.For(index, 2, 8, 1, ir.ForKind.DEFAULT, output.store(index, 1))
    unrolled = native("PragmaUnrollLoop", original)
    assert unrolled.kind == ir.ForKind.UNROLLED
    assert original.kind == ir.ForKind.DEFAULT
    assert unrolled.body.same_as(original.body)
    extent = ir.Var("extent", "int32")
    dynamic = ir.For(index, 0, extent, 1, ir.ForKind.DEFAULT, output.store(index, 1))
    assert native("PragmaUnrollLoop", dynamic).same_as(dynamic)


def test_partition_rejects_invalid_iteration_domains(native_layouts):
    output = TensorView.create("output", [8], "int32")
    index = ir.Var("index", "int32")
    thread_range = ir.Range(0, 32)
    for begin, step in [(1, 1), (0, 2)]:
        loop = ir.For(index, begin, 8, step, ir.ForKind.PARALLEL, output.store(index, 1))
        with pytest.raises(ValueError, match="zero-based, unit-step"):
            native("PlanLoopPartition", loop, thread_range, 1)
    loop = parallel_loop(output, [8])
    with pytest.raises(ValueError, match="Thread range extent"):
        native("PlanLoopPartition", loop, ir.Range(0, ir.Var("threads", "int32")), 1)
    with pytest.raises(ValueError):
        native("PlanLoopPartition", loop, thread_range, 0)


@pytest.mark.gpu
def test_native_partitioned_parallel_loop_executes_on_cuda(native_layouts):
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("A CUDA device is required")
    shape = (8, 16)
    output = TensorView.create("output", shape, "int32")
    original = parallel_loop(output, shape)
    thread_range = ir.Range(0, 32)
    layout = native("PlanLoopPartition", original, thread_range, 4)
    thread = ir.Var("thread", "int32")
    lowered = partition(original, layout, thread, thread_range)
    body = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(32)], [thread], None, lowered)
    function = tirx.Function([output.tensor], body, ret_type=ir.TupleType([])).with_attr("global_symbol", "partition_cuda")
    version = device.compute_version.replace(".", "")
    target = Target({"kind": "cuda", "arch": f"sm_{version}", "backend_config": {"cuda": {"compiler": "nvcc"}}}, host="c")
    executable = tirx.compile(function, target=target)
    actual = runtime.empty(shape, "int32", device=device)
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), np.arange(128, dtype="int32").reshape(shape))
