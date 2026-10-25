import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, tirx
from tvm_tirx.target import Target

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def apply(body, parameters, *, plan=False, target=None):
    function = tirx.Function(list(parameters), body, ret_type=ir.TupleType([]))
    function = function.with_attr("target", target or Target("c")).with_attr("global_symbol", "pipeline_kernel")
    module = ir.IRModule({"main": function})
    if plan:
        module = tvm_ffi.get_global_func("tl.transform.PipelinePlanning")()(module)
    module = tvm_ffi.get_global_func("tl.transform.InjectSoftwarePipeline")()(module)
    assert tirx.analysis.check_well_formed(module)
    ir.assert_structural_equal(module, ir.load_json(ir.save_json(module)))
    return module


def annotations(stages, order=None, *, planned=True):
    prefix = "software_pipeline" if planned else "tl_pipeline"
    return {f"{prefix}_stage": list(stages), f"{prefix}_order": list(range(len(stages))) if order is None else list(order)}


@pytest.mark.parametrize("offset", [1, 32])
@pytest.mark.parametrize("materialized", [False, True])
@pytest.mark.parametrize("stages,order", [([1, 0], [0, 1]), ([0, 0], [1, 0])])
def test_pipeline_reordering_checks_independent_thread_footprints(offset, materialized, stages, order):
    shared = TensorView.create("shared", [64], "int32", scope="shared")
    output = TensorView.create("output", [64], "int32")
    thread = ir.Var("thread", "int32")
    iteration = ir.Var("iteration", "int32")
    loop = ir.For(
        iteration,
        0,
        2,
        1,
        ir.ForKind.DEFAULT,
        [shared.store(thread, iteration), output.store(iteration * 32 + thread, shared.load(thread + offset))],
        annotations=annotations(stages, order),
    )
    body = ir.SeqStmt([shared.allocate(), shared.store(thread + 32, 0), loop])
    parameters = [output.tensor]
    if materialized:
        body = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(32)], [thread], None, body)
    else:
        parameters.append(thread)
    if offset == 1 or not materialized:
        with pytest.raises(ValueError, match="reverses.*dependency"):
            apply(body, parameters)
    else:
        apply(body, parameters)


@pytest.mark.parametrize("extent", [0, 1, 2, 5, 17])
@pytest.mark.parametrize("planned", [False, True])
def test_pipeline_ring_executes_with_short_and_empty_loops(extent, planned):
    source = TensorView.create("source", [max(1, extent)], "int32")
    output = TensorView.create("output", [max(1, extent)], "int32")
    temporary = TensorView.create("temporary", [1], "int32", scope="local")
    index = ir.Var("index", "int32")
    body = ir.For(
        index,
        0,
        extent,
        1,
        ir.ForKind.DEFAULT,
        [temporary.allocate(), temporary.store(0, source.load(index)), output.store(index, temporary.load(0) * 3 + 1)],
        annotations=annotations([0, 2], planned=planned),
    )
    module = apply(body, [source.tensor, output.tensor], plan=not planned)
    allocations = []
    stmt_functor.post_order_visit(
        module, lambda node: allocations.append(node) if isinstance(node, ir.Bind) and isinstance(node.var.ty, tirx.TensorType) else None
    )
    assert any(list(map(int, allocation.var.ty.shape)) == [3, 1] for allocation in allocations)
    executable = tirx.compile(module, target="c")
    values = np.arange(max(1, extent), dtype="int32")
    actual = runtime.tensor(np.full(max(1, extent), -99, dtype="int32"))
    executable(runtime.tensor(values), actual)
    expected = values * 3 + 1 if extent else np.array([-99], dtype="int32")
    np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize("stages,order", [([1, 0], [0, 1]), ([0, 0], [1, 0])])
def test_pipeline_reorders_disjoint_slices_without_versioning_initialized_state(stages, order):
    temporary = TensorView.create("temporary", [2], "int32", scope="local")
    output = TensorView.create("output", [9], "int32")
    iteration = ir.Var("iteration", "int32")
    loop = ir.For(
        iteration,
        0,
        8,
        1,
        ir.ForKind.DEFAULT,
        [temporary.store(0, iteration), output.store(iteration, temporary.load(1))],
        annotations=annotations(stages, order),
    )
    module = apply(
        [temporary.allocate(), temporary.store(1, 73), loop, output.store(8, temporary.load(0))],
        [output.tensor],
    )
    actual = runtime.empty((9,), "int32")
    tirx.compile(module, target="c")(actual)
    np.testing.assert_array_equal(actual.numpy(), [73] * 8 + [7])


@pytest.mark.parametrize("begin,end,step", [(2, 19, 3), (18, 1, -3)])
def test_pipeline_preserves_nonunit_and_descending_iteration(begin, end, step):
    output = TensorView.create("output", [20], "int32")
    temporary = TensorView.create("temporary", [1], "int32", scope="local")
    index = ir.Var("index", "int32")
    body = ir.For(
        index,
        begin,
        end,
        step,
        ir.ForKind.DEFAULT,
        [temporary.allocate(), temporary.store(0, index + 100), output.store(index, temporary.load(0))],
        annotations=annotations([0, 3]),
    )
    module = apply(body, [output.tensor])
    actual = runtime.tensor(np.full(20, -1, dtype="int32"))
    tirx.compile(module, target="c")(actual)
    expected = np.full(20, -1, dtype="int32")
    for index in range(begin, end, step):
        expected[index] = index + 100
    np.testing.assert_array_equal(actual.numpy(), expected)


def test_scalar_memory_reads_are_versioned_without_replay():
    source = TensorView.create("source", [19], "int32")
    output = TensorView.create("output", [19], "int32")
    index = ir.Var("index", "int32")
    value = ir.Var("value", "int32")
    derived = ir.Var("derived", "int32")
    body = ir.For(
        index,
        0,
        19,
        1,
        ir.ForKind.DEFAULT,
        [ir.Bind(value, source.load(index)), ir.Bind(derived, value * 2 + index), output.store(index, derived + value)],
        annotations=annotations([0, 2]),
    )
    module = apply(body, [source.tensor, output.tensor])
    values = np.arange(19, dtype="int32")
    actual = runtime.empty((19,), "int32")
    tirx.compile(module, target="c")(runtime.tensor(values), actual)
    np.testing.assert_array_equal(actual.numpy(), values * 4)


@pytest.mark.parametrize("no_alias", [False, True])
def test_auto_pipeline_replays_only_readonly_non_aliasing_indices(no_alias):
    source = TensorView.create("source", [7], "int32")
    indices = TensorView.create("indices", [7], "int32")
    temporary = TensorView.create("temporary", [1], "int32", scope="local")
    output = TensorView.create("output", [7], "int32")
    iteration = ir.Var("iteration", "int32")
    offset = ir.Var("offset", "int32")
    loop = ir.For(
        iteration,
        0,
        7,
        1,
        ir.ForKind.DEFAULT,
        [ir.Bind(offset, indices.load(iteration)), temporary.store(0, source.load(offset)), output.store(iteration, temporary.load(0))],
        annotations={"num_stages": 3},
    )
    function = tirx.Function([source.tensor, indices.tensor, output.tensor], [temporary.allocate(), loop], ret_type=ir.TupleType([]))
    function = function.with_attr({"target": Target("c"), "global_symbol": "pipeline_kernel", "tirx.noalias": no_alias})
    planned = tvm_ffi.get_global_func("tl.transform.PipelinePlanning")()(ir.IRModule({"main": function}))
    planned_loop = planned["main"].body.seq[1]
    assert list(map(int, planned_loop.annotations["software_pipeline_stage"])) == ([0, 2] if no_alias else [0, 0, 2])
    if no_alias:
        assert list(map(int, planned_loop.annotations["software_pipeline_replayable_scalar_binds"])) == [1, 0, 0]
    else:
        assert "software_pipeline_replayable_scalar_binds" not in planned_loop.annotations
    lowered = tvm_ffi.get_global_func("tl.transform.InjectSoftwarePipeline")()(planned)
    lowered = tvm_ffi.get_global_func("tl.transform.LowerTileOp")()(lowered)
    assert tirx.analysis.check_well_formed(lowered)
    values = np.arange(7, dtype="int32") * 11
    positions = np.arange(6, -1, -1, dtype="int32")
    actual = runtime.empty((7,), "int32")
    tirx.compile(lowered, target="c")(runtime.tensor(values), runtime.tensor(positions), actual)
    np.testing.assert_array_equal(actual.numpy(), values[positions])


def test_scalar_side_effects_execute_once_per_iteration(tilelang_builders):
    counter = TensorView.create("counter", [1], "int32")
    output = TensorView.create("output", [13], "int32")
    index = ir.Var("index", "int32")
    value = ir.Var("value", "int32")
    call = ir.Call("tirx.call_extern", [ir.StringImm("next_value"), counter.data], ty="int32")
    loop = ir.For(
        index, 0, 13, 1, ir.ForKind.DEFAULT, [ir.Bind(value, call), output.store(index, value + value)], annotations=annotations([0, 2])
    )
    imported = '#ifdef __cplusplus\nextern "C"\n#endif\nint32_t next_value(int32_t *counter) { return (*counter)++; }\n'
    body = ir.RegionStmt("tirx.scope", [], [], {"pragma_import_c": imported}, loop)
    module = apply(body, [counter.tensor, output.tensor])
    actual = runtime.empty((13,), "int32")
    count = runtime.tensor(np.array([0], dtype="int32"))
    tirx.compile(module, target="c")(count, actual)
    np.testing.assert_array_equal(actual.numpy(), np.arange(13, dtype="int32") * 2)
    np.testing.assert_array_equal(count.numpy(), [13])


@pytest.mark.parametrize("stages,order", [([1, 0], [0, 1]), ([0, 0], [1, 0]), ([0, 1], [0, 0])])
def test_invalid_dependency_orders_are_rejected(stages, order):
    temporary = TensorView.create("temporary", [1], "int32", scope="local")
    output = TensorView.create("output", [8], "int32")
    index = ir.Var("index", "int32")
    loop = ir.For(
        index,
        0,
        8,
        1,
        ir.ForKind.DEFAULT,
        [temporary.allocate(), temporary.store(0, index), output.store(index, temporary.load(0))],
        annotations=annotations(stages, order),
    )
    with pytest.raises(ValueError, match="dependency|unique"):
        apply(loop, [output.tensor])


def test_iteration_carried_storage_cannot_be_silently_multiversioned():
    accumulator = TensorView.create("accumulator", [1], "int32", scope="local")
    output = TensorView.create("output", [8], "int32")
    index = ir.Var("index", "int32")
    loop = ir.For(
        index,
        0,
        8,
        1,
        ir.ForKind.DEFAULT,
        [accumulator.store(0, accumulator.load(0) + 1), output.store(index, accumulator.load(0))],
        annotations=annotations([0, 2]),
    )
    with pytest.raises(ValueError, match="iteration-carried"):
        apply([accumulator.allocate(), accumulator.store(0, 0), loop], [output.tensor])


def test_auto_planner_pipelines_copy_and_consumer():
    source = TensorView.create("source", [32, 32], "float32")
    temporary = TensorView.create("temporary", [32], "float32", scope="shared")
    output = TensorView.create("output", [32, 32], "float32")
    index = ir.Var("index", "int32")
    copy_in = ir.Evaluate(
        ir.Call("tl.tileop.copy", [source.region([ir.Range.from_min_extent(index, 1), ir.Range(0, 32)]), temporary.tensor], attrs={})
    )
    copy_out = ir.Evaluate(
        ir.Call("tl.tileop.copy", [temporary.tensor, output.region([ir.Range.from_min_extent(index, 1), ir.Range(0, 32)])], attrs={})
    )
    loop = ir.For(index, 0, 32, 1, ir.ForKind.DEFAULT, [copy_in, copy_out], annotations={"num_stages": 3})
    function = tirx.Function([source.tensor, output.tensor], [temporary.allocate(), loop], ret_type=ir.TupleType([])).with_attr(
        "target", Target({"kind": "cuda", "arch": "sm_80"})
    )
    module = tvm_ffi.get_global_func("tl.transform.PipelinePlanning")()(ir.IRModule({"main": function}))
    assert list(map(int, module["main"].body.seq[1].annotations["software_pipeline_stage"])) == [0, 2]
    assert tirx.analysis.check_well_formed(module)


def test_nested_pipeline_annotations_are_consumed():
    output = TensorView.create("output", [3, 7], "int32")
    temporary = TensorView.create("temporary", [1], "int32", scope="local")
    outer = ir.Var("outer", "int32")
    inner = ir.Var("inner", "int32")
    inner_loop = ir.For(
        inner,
        0,
        7,
        1,
        ir.ForKind.DEFAULT,
        [temporary.allocate(), temporary.store(0, outer * 7 + inner), output.store([outer, inner], temporary.load(0))],
        annotations=annotations([0, 2]),
    )
    loop = ir.For(outer, 0, 3, 1, ir.ForKind.DEFAULT, inner_loop, annotations=annotations([0]))
    module = apply(loop, [output.tensor])
    loops = []
    stmt_functor.post_order_visit(module, lambda node: loops.append(node) if isinstance(node, ir.For) else None)
    assert all("software_pipeline_stage" not in loop.annotations for loop in loops)
    actual = runtime.empty((3, 7), "int32")
    tirx.compile(module, target="c")(actual)
    np.testing.assert_array_equal(actual.numpy(), np.arange(21, dtype="int32").reshape(3, 7))


def test_pipeline_preserves_physical_strides_and_buffer_alignment():
    output = TensorView.create("output", [11], "int32")
    temporary = TensorView.create("temporary", [1], "int32", scope="local", strides=[3], elem_offset=2, data_alignment=128)
    index = ir.Var("index", "int32")
    loop = ir.For(
        index,
        0,
        11,
        1,
        ir.ForKind.DEFAULT,
        [temporary.allocate(), temporary.store(0, index), output.store(index, temporary.load(0))],
        annotations=annotations([0, 2]),
    )
    module = apply(loop, [output.tensor])
    actual = runtime.empty((11,), "int32")
    tirx.compile(module, target="c")(actual)
    np.testing.assert_array_equal(actual.numpy(), np.arange(11, dtype="int32"))


@pytest.mark.parametrize("extent", [0, 2, 17])
def test_shared_copy_pipeline_executes_on_cuda(extent, tilelang_builders):
    from pathlib import Path
    from tvm_tirx.ir.transform import PassContext

    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("CUDA device unavailable")
    root = Path(__file__).resolve().parents[3]
    target = Target(
        {
            "kind": "cuda",
            "arch": f"sm_{device.compute_version.replace('.', '')}",
            "backend_config": {"cuda": {"compiler": "nvcc", "nvcc": [f"-I{root / 'src'}", f"-I{root / '3rdparty/cutlass/include'}"]}},
        },
        host="c",
    )
    source = TensorView.create("source", [max(1, extent), 128], "float32")
    output = TensorView.create("output", [max(1, extent), 128], "float32")
    temporary = TensorView.create("temporary", [128], "float32", scope="shared")
    index = ir.Var("index", "int32")
    thread = ir.Var("thread", "int32")
    copy_in = ir.Evaluate(
        ir.Call(
            "tl.tileop.copy",
            [source.region([ir.Range.from_min_extent(index, 1), ir.Range(0, 128)]), temporary.tensor],
            attrs={"disable_tma": True},
        )
    )
    loop = ir.For(
        index,
        0,
        extent,
        1,
        ir.ForKind.DEFAULT,
        [copy_in, output.store([index, thread], temporary.load(thread) * 2 + 3)],
        annotations={"num_stages": 3},
    )
    launch = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(128)], [thread], None, [temporary.allocate(), loop])
    with target, PassContext():
        module = apply(launch, [source.tensor, output.tensor], plan=True, target=target)
        for name in ("LayoutInference", "LowerTileOp", "LowerAccessPtr"):
            module = tvm_ffi.get_global_func(f"tl.transform.{name}")()(module)
        module = tvm_ffi.get_global_func("tl.transform.ThreadSync")("shared")(module)
    values = np.arange(max(1, extent) * 128, dtype="float32").reshape(max(1, extent), 128)
    actual = runtime.tensor(np.full(values.shape, -1, dtype="float32"), device=device)
    tirx.compile(module, target=target)(runtime.tensor(values, device=device), actual)
    expected = values * 2 + 3 if extent else np.full(values.shape, -1, dtype="float32")
    np.testing.assert_array_equal(actual.numpy(), expected)
