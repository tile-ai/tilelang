import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, sym, tirx
from tvm_tirx.target import Target

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView
from tilelang.transform import LegalizeVectorizedLoop

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def native(name, *arguments):
    return tvm_ffi.get_global_func(f"tl.{name}")(*arguments)


def cuda_target(arch="sm_90"):
    return Target({"kind": "cuda", "arch": arch})


def plan(loop, layouts=None, target=None):
    with target or cuda_target():
        return native("GetVectorizeSize", loop, sym.Analyzer(), layouts or {})


def rewrite(loop, width=-1, layouts=None):
    with cuda_target():
        return native("VectorizeLoop", loop, sym.Analyzer(), layouts or {}, width)


def loop_nodes(body):
    result = []
    stmt_functor.post_order_visit(body, lambda node: result.append(node) if isinstance(node, ir.For) else None)
    return result


@pytest.mark.parametrize("dtype", ["float16", "float32", "int8", "int16"])
def test_metal_vectorization_respects_native_four_lane_types(dtype):
    source = TensorView.create("source", [32], dtype)
    output = TensorView.create("output", [32], dtype)
    index = ir.Var("index", "int32")
    loop = ir.For(index, 0, 32, 1, ir.ForKind.VECTORIZED, output.store(index, source.load(index)))
    assert plan(loop, target=Target("metal")) == 4


@pytest.mark.parametrize(
    "offset,width,allow_broadcast,expected",
    [
        (lambda index: index, 4, False, True),
        (lambda index: index + 1, 4, False, False),
        (lambda index: index + 4, 4, False, True),
        (lambda index: index * 2, 4, False, False),
        (lambda index: index // 4, 4, True, True),
        (lambda index: index // 4, 4, False, False),
        (lambda index: index // 4 * 6 + index % 4, 4, False, False),
        (lambda index: index // 4 * 6 + index % 4, 2, False, True),
        (lambda index: index // 4 * 6 + index % 4, 4, True, True),
        (lambda index: index ^ 1, 2, False, False),
        (lambda index: index // 16 * 16 + index % 16, 4, False, True),
        (lambda index: index % 8, 4, False, True),
    ],
)
def test_native_index_vector_legality(offset, width, allow_broadcast, expected):
    index = ir.Var("index", "int32")
    result = native("IndicesCanVectorize", offset(index), index, ir.const(16), width, sym.Analyzer(), allow_broadcast, ir.const(0))
    assert result == expected


@pytest.mark.parametrize("minimum,expected", [(0, True), (2, False), (4, True)])
def test_invariance_uses_iteration_ordinal(minimum, expected):
    index = ir.Var("index", "int32")
    assert native("IsExprInvariantInVectorBoundary", index // 4, index, ir.const(4), 4, sym.Analyzer(), ir.const(minimum)) == expected


@pytest.mark.parametrize("row_width,expected", [(128, True), (4, False)])
def test_mixed_width_pipeline_predicate_vector_groups(row_width, expected):
    partition = ir.Var("partition", "int32")
    thread = ir.Var("thread", "int32")
    clock = ir.Var("clock", "int32")
    offset = ir.Var("offset", "int64")
    length = ir.Var("length", "int64")
    analyzer = sym.Analyzer()
    analyzer.bind(thread, ir.Range(0, 256))
    flat = (partition // 8 * 2048 + thread * 8 + partition % 8) % 16384
    row = (flat // row_width).astype("int32").astype("int64")
    index = clock.astype("int64") * 128 + offset + row
    predicate = (index < length) & (index >= 0)
    assert native("IsExprInvariantInVectorBoundary", predicate, partition, ir.const(64), 8, analyzer, ir.const(0)) == expected


@pytest.mark.parametrize("extent,expected", [(16, 4), (12, 4), (6, 2), (7, 1)])
def test_vector_planner_obeys_dtype_and_loop_extent(extent, expected):
    source = TensorView.create("source", [extent], "float32")
    destination = TensorView.create("destination", [extent], "float32")
    index = ir.Var("index", "int32")
    body = destination.store(index, source.load(index))
    loop = ir.For(index, 0, extent, 1, ir.ForKind.PARALLEL, body)
    assert plan(loop) == expected
    lowered = rewrite(loop)
    assert loop.kind == ir.ForKind.PARALLEL
    vector_loops = [node for node in loop_nodes(lowered) if node.kind == ir.ForKind.VECTORIZED]
    assert [int(node.end - node.begin) for node in vector_loops] == ([] if expected == 1 else [expected])


def test_pure_lane_assumptions_do_not_prevent_vector_planning():
    source = TensorView.create("source", [16], "float32")
    destination = TensorView.create("destination", [16], "float32")
    index = ir.Var("index", "int32")
    extent = ir.Var("extent", "int32")
    body = [ir.Evaluate(ir.Call("prim.assume", [index < extent])), destination.store(index, source.load(index))]
    loop = ir.For(index, 0, 16, 1, ir.ForKind.PARALLEL, body)
    assert plan(loop) == 4


@pytest.mark.parametrize("offset,expected", [(0, 4), (1, 1), (2, 2), (4, 4)])
def test_native_tensor_origin_constrains_vector_alignment(offset, expected):
    source = TensorView.create("source", [16], "float32", elem_offset=offset)
    destination = TensorView.create("destination", [16], "float32")
    index = ir.Var("index", "int32")
    loop = ir.For(index, 0, 16, 1, ir.ForKind.DEFAULT, destination.store(index, source.load(index)))
    assert plan(loop) == expected


def test_native_strides_and_layout_map_are_not_double_applied(native_layouts):
    source = TensorView.create("source", [4, 4], "float32", strides=[12, 2])
    destination = TensorView.create("destination", [4, 4], "float32")
    index = ir.Var("index", "int32")
    loop = ir.For(index, 0, 4, 1, ir.ForKind.DEFAULT, destination.store([0, index], source.load([0, index])))
    assert plan(loop) == 1
    layout = native_layouts.Layout([4, 4], lambda row, column: (row, column))
    with pytest.raises(RuntimeError, match="already owns"):
        plan(loop, {source.tensor: layout})


@pytest.mark.parametrize("index_fn", [lambda index: ir.const(0), lambda index: index // 4])
def test_reduction_axis_stores_remain_scalar(index_fn):
    source = TensorView.create("source", [16], "float32")
    accumulator = TensorView.create("accumulator", [4], "float32", scope="local")
    index = ir.Var("index", "int32")
    destination = index_fn(index)
    loop = ir.For(index, 0, 16, 1, ir.ForKind.PARALLEL, accumulator.store(destination, accumulator.load(destination) + source.load(index)))
    assert plan(loop) == 1


@pytest.mark.parametrize("minimum,width", [(0, 4), (1, 1), (2, 2), (4, 4)])
def test_nonzero_minimum_rewrite_preserves_indices(minimum, width):
    source = TensorView.create("source", [32], "int32")
    output = TensorView.create("output", [32], "int32")
    index = ir.Var("index", "int32")
    loop = ir.For(index, minimum, minimum + 8, 1, ir.ForKind.PARALLEL, output.store(index, source.load(index)))
    assert plan(loop) == width
    rewritten = rewrite(loop)
    function = tirx.Function([source.tensor, output.tensor], rewritten, ret_type=ir.TupleType([]))
    assert tirx.analysis.check_well_formed(function)
    ir.assert_structural_equal(function, ir.load_json(ir.save_json(function)))


@pytest.mark.parametrize("step", [2, -1])
def test_nonunit_loops_are_not_split(step):
    output = TensorView.create("output", [16], "int32")
    index = ir.Var("index", "int32")
    loop = ir.For(index, 0 if step > 0 else 15, 16 if step > 0 else -1, step, ir.ForKind.DEFAULT, output.store(index, index))
    assert plan(loop) == 1
    with pytest.raises(RuntimeError, match="unit loop step"):
        rewrite(loop, 4)


@pytest.mark.parametrize("scope,expected", [("global", 4), ("shared", 1), ("local", 1)])
def test_native_atomic_width_policy(scope, expected):
    assert native("GetMaxAtomicVectorSize", ir.PrimType("float32"), scope, cuda_target()) == expected
    assert native("GetMaxAtomicVectorSize", ir.PrimType("float32"), scope, cuda_target("sm_80")) == 1
    assert native("GetMaxAtomicVectorSize", ir.PrimType("float16"), scope, cuda_target()) == 2


@pytest.mark.parametrize("broadcast,width", [(False, 4), (True, 1)])
def test_typed_atomic_pointer_legality_survives_lowering(broadcast, width):
    destination = TensorView.create("destination", [16], "float32")
    index = ir.Var("index", "int32")
    pointer = native("MakeAccessPtrFromBufferLoad", destination.load(index // 4 if broadcast else index), 3)
    for address in [pointer, native("LowerAccessPtrCall", pointer)]:
        call = ir.Call("tl.atomic_add_elem_op", [address, ir.const(1.0, "float32"), 0], ty="float32")
        loop = ir.For(index, 0, 16, 1, ir.ForKind.DEFAULT, ir.Evaluate(call))
        assert plan(loop) == width


def test_stateful_call_keeps_per_lane_arguments():
    index = ir.Var("index", "int32")
    loop = ir.For(index, 0, 16, 1, ir.ForKind.DEFAULT, ir.Evaluate(ir.Call("tl.rng_init", [index, 0, 0, 0], ty="int32")))
    assert plan(loop) == 1


@pytest.mark.parametrize("use_attr", [False, True])
def test_vector_loop_legalization_pass_uses_native_target(use_attr):
    source = TensorView.create("source", [16], "float32")
    output = TensorView.create("output", [16], "float32")
    index = ir.Var("index", "int32")
    loop = ir.For(index, 0, 16, 1, ir.ForKind.VECTORIZED, output.store(index, source.load(index)))
    function = tirx.Function([source.tensor, output.tensor], loop, ret_type=ir.TupleType([]))
    if use_attr:
        function = function.with_attr("target", cuda_target())
        lowered = LegalizeVectorizedLoop()(ir.IRModule({"main": function}))["main"]
    else:
        with cuda_target():
            lowered = LegalizeVectorizedLoop()(ir.IRModule({"main": function}))["main"]
    assert [int(node.end - node.begin) for node in loop_nodes(lowered.body) if node.kind == ir.ForKind.VECTORIZED] == [4]
    assert tirx.analysis.check_well_formed(lowered)


@pytest.mark.parametrize("access", ["load", "store", "condition", "let"])
def test_large_integer_coefficients_do_not_overflow_the_planner(access):
    source = TensorView.create("source", [2000000000], "float32")
    output = TensorView.create("output", [2000000000], "float32")
    index = ir.Var("index", "int32")
    scaled = (index - index // 4 * 4) * 600000000
    if access == "load":
        body = output.store(index, source.load(scaled))
    elif access == "store":
        body = output.store(scaled, source.load(index))
    elif access == "condition":
        body = ir.If(scaled != 0, output.store(index, source.load(index)), None)
    else:
        bound = ir.Var("bound", "int32")
        body = output.store(index, source.load(ir.prim.Let(bound, scaled, bound - scaled)))
    loop = ir.For(index, 0, 8, 1, ir.ForKind.VECTORIZED, body)
    assert plan(loop) in (1, 2, 4, 8)


@pytest.mark.gpu
def test_planned_native_vector_loop_executes_on_cuda():
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("A CUDA device is required")
    source = TensorView.create("source", [32], "float32")
    output = TensorView.create("output", [32], "float32")
    index = ir.Var("index", "int32")
    loop = ir.For(index, 4, 28, 1, ir.ForKind.PARALLEL, output.store(index, source.load(index) * 2 + 1))
    lowered = rewrite(loop)
    thread = ir.Var("thread", "int32")
    lowered = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(1)], [thread], None, lowered)
    function = tirx.Function([source.tensor, output.tensor], lowered, ret_type=ir.TupleType([])).with_attr("global_symbol", "vector_copy")
    target = Target(
        {"kind": "cuda", "arch": f"sm_{device.compute_version.replace('.', '')}", "backend_config": {"cuda": {"compiler": "nvcc"}}},
        host="c",
    )
    executable = tirx.compile(function, target=target)
    values = np.arange(32, dtype="float32")
    actual = runtime.tensor(np.full(32, -1, dtype="float32"), device=device)
    executable(runtime.tensor(values, device=device), actual)
    expected = np.full(32, -1, dtype="float32")
    expected[4:28] = values[4:28] * 2 + 1
    np.testing.assert_array_equal(actual.numpy(), expected)
