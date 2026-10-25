import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, tirx
from tvm_tirx.script.ir_builder import IRBuilder
from tvm_tirx.tirx.script import ir_builder as T

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView

pytestmark = pytest.mark.usefixtures("native_layouts")


def native(name, *arguments):
    return tvm_ffi.get_global_func("tl." + name)(*arguments)


def function(body, parameters=()):
    return tirx.Function(list(parameters), body, ret_type=ir.TupleType([])).with_attr("global_symbol", "frame_test")


def nodes_of(root, cls):
    nodes = []
    stmt_functor.post_order_visit(root, lambda node: nodes.append(node) if isinstance(node, cls) else None)
    return nodes


@pytest.mark.parametrize("shape", [(3,), (3, 5), (2, 3, 5)])
def test_parallel_frame_uses_native_loop_domains(shape):
    with IRBuilder() as builder:
        frame = native("Parallel", [ir.const(extent) for extent in shape], {"coalesced_width": 4})
        with frame:
            T.evaluate(sum(frame.vars))
    body = builder.get()
    loops = nodes_of(body, ir.For)
    assert len(loops) == len(shape)
    outer = loops[-1]
    assert outer.annotations["coalesced_width"] == 4
    assert all(not loop.annotations for loop in loops[:-1])
    assert all(loop.kind == ir.ForKind.PARALLEL for loop in loops)
    assert tirx.analysis.check_well_formed(function(body))


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_pipelined_frame_preserves_nonzero_begin_and_index_type(dtype):
    output = TensorView.create("output", [16], "int32")
    with IRBuilder() as builder:
        frame = native("Pipelined", ir.const(3), ir.const(11, dtype), 0, [], [], {})
        with frame as index:
            T.emit_(output.store([index], ir.const(7)))
    body = builder.get()
    loop = nodes_of(body, ir.For)[0]
    assert loop.var.ty == ir.PrimType(dtype)
    assert int(loop.begin) == 3 and int(loop.end) == 11
    executable = tirx.compile(function(body, [output.tensor]), target="c")
    actual = runtime.tensor(np.zeros(16, dtype="int32"))
    executable(actual)
    expected = np.zeros(16, dtype="int32")
    expected[3:11] = 7
    np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize("shape,workers,group", [((3, 17), 4, 4), ((1, 2), 9, 8), ((7,), 2, 3), ((2, 3, 13), 5, 4)])
def test_persistent_frame_covers_each_coordinate_once(shape, workers, group):
    output = TensorView.create("output", shape, "int32")
    worker = ir.Var("worker", "int32")
    with IRBuilder() as builder:
        frame = native("Persistent", [ir.const(extent) for extent in shape], ir.const(workers), worker, ir.const(group), 0, {})
        with frame:
            T.emit_(output.store(list(frame.vars), output.load(list(frame.vars)) + 1))
    body = ir.For(worker, 0, workers, 1, ir.ForKind.DEFAULT, builder.get())
    primitive = function(body, [output.tensor])
    assert tirx.analysis.check_well_formed(primitive)
    ir.assert_structural_equal(primitive, ir.load_json(ir.save_json(primitive)))
    executable = tirx.compile(primitive, target="c")
    actual = runtime.tensor(np.zeros(shape, dtype="int32"))
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), np.ones(shape, dtype="int32"))


def make_launch(grid=(3,), threads=None, use_thread=False, attributes=None):
    with IRBuilder() as builder:
        frame = native(
            "KernelLaunch",
            [ir.const(extent) for extent in grid],
            None if threads is None else [ir.const(extent) for extent in threads],
            attributes or {},
        )
        with frame:
            T.evaluate(frame.thread_vars[0] if use_thread else ir.const(0))
    return function(builder.get())


@pytest.mark.parametrize("grid", [(), (3,), (2, 3, 4), (0,)])
@pytest.mark.parametrize("gpu", [False, True])
def test_kernel_launch_materializes_native_bindings(grid, gpu):
    primitive = make_launch(grid, (64,))
    module = ir.IRModule({"main": primitive})
    if gpu and grid == (0,):
        with pytest.raises(ValueError, match="Kernel grid extents must be positive"):
            native("transform.MaterializeKernelLaunch", gpu, gpu, None, None, None)(module)
        return
    result = native("transform.MaterializeKernelLaunch", gpu, gpu, None, None, None)(module)
    assert tirx.analysis.check_well_formed(result)
    ir.assert_structural_equal(result, ir.load_json(ir.save_json(result)))
    assert not any(call.op.name == "tl.launch_thread_idx" for call in nodes_of(result, ir.Call))
    regions = nodes_of(result, ir.RegionStmt)
    assert not any(region.op.name == "tl.launch_grid" for region in regions)
    if grid == (0,):
        assert not regions
    elif gpu:
        launch = [region for region in regions if region.op.name == "tirx.launch_thread"]
        assert len(launch) == len(grid) + 3
        assert any(str(region.args[0].value) == "threadIdx.x" and int(region.args[1]) == 64 for region in launch)
    else:
        assert len(nodes_of(result, ir.For)) == len(grid)


def test_kernel_launch_default_threads_and_target_diagnostics():
    module = ir.IRModule({"main": make_launch(use_thread=True)})
    result = native("transform.MaterializeKernelLaunch", True, True, [ir.const(128)], None, None)(module)
    assert tirx.analysis.check_well_formed(result)
    with pytest.raises(ValueError, match="no default thread-block size"):
        native("transform.MaterializeKernelLaunch", True, True, None, None, None)(module)
    with pytest.raises(ValueError, match="no SIMT thread bindings"):
        native("transform.MaterializeKernelLaunch", False, False, None, None, None)(module)
    module = ir.IRModule({"main": make_launch(attributes={"cluster_dims": [2, 1, 1]})})
    with pytest.raises(ValueError, match="cluster_dims.*not supported"):
        native("transform.MaterializeKernelLaunch", False, False, None, ["cluster_dims"], None)(module)


def test_warp_specialization_frame_emits_a_bound_native_conditional():
    thread = ir.Var("thread", "int32")
    with IRBuilder() as builder, native("WarpSpecialize", [ir.const(0), ir.const(2)], thread, 32):
        T.evaluate(thread)
    primitive = function(builder.get(), [thread])
    assert tirx.analysis.check_well_formed(primitive)
    conditionals = nodes_of(primitive, ir.If)
    assert len(conditionals) == 1
    for lane in (0, 31, 32, 63, 64, 95, 96):
        condition = stmt_functor.substitute(conditionals[0].condition, {thread: ir.const(lane)})
        from tvm_tirx import sym

        assert bool(int(sym.Analyzer().simplify(condition))) == (lane < 32 or 64 <= lane < 96)
