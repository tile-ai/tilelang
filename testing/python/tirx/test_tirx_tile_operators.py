import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, tirx
from tvm_tirx.target import Target

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView


pytestmark = pytest.mark.usefixtures("native_layouts")


def native(name, *arguments):
    return tvm_ffi.get_global_func(f"tl.{name}")(*arguments)


def parse(name, arguments):
    return native("ParseOperator", ir.Call(f"tl.tileop.{name}", arguments), {})


def infer(operation, target, threads, layouts=None, strict_layouts=None, level=0, bindings=None):
    with target:
        return native("testing.InferOperator", operation, target, threads, layouts or {}, {}, strict_layouts or {}, level, bindings or {})


def lower(operation, target, thread=0, threads=None, layouts=None):
    threads = threads or ir.Range(0, 1)
    thread = ir.const(thread, "int32") if isinstance(thread, int) else thread
    with target:
        return native("testing.LowerOperator", operation, target, threads, thread, layouts or {}, {})


def function(body, tensors, name="native_tile_operator"):
    result = tirx.Function([tensor.tensor for tensor in tensors], body, ret_type=ir.TupleType([])).with_attr("global_symbol", name)
    assert tirx.analysis.check_well_formed(result)
    ir.assert_structural_equal(result, ir.load_json(ir.save_json(result)))
    return result


def tensor_region(tensor, begins, extents):
    return tensor.region([ir.Range.from_min_extent(begin, extent) for begin, extent in zip(begins, extents)])


@pytest.mark.parametrize("shape,begins,extents", [((8, 16), (0, 0), (8, 16)), ((7, 13), (2, 3), (3, 7)), ((9,), (3,), (0,)), ((), (), ())])
def test_fill_cpu_preserves_region_and_values(shape, begins, extents):
    output = TensorView.create("output", shape, "float32")
    region = tensor_region(output, begins, extents)
    operation = parse("fill", [region, ir.const(-3.25, "float32")])
    read_regions, write_regions = native("testing.GetOperatorAccessRegions", operation)
    assert not read_regions
    assert len(write_regions) == 1
    ir.assert_structural_equal(write_regions[0], region)
    cloned = native("testing.CloneOperator", operation)
    ir.assert_structural_equal(native("testing.GetOperatorAccessRegions", cloned)[1][0], region)
    body = lower(operation, Target("c"))
    executable = tirx.compile(function(body, [output]), target="c")
    expected = np.full(shape, 11.0, dtype="float32")
    actual = runtime.tensor(expected.copy())
    executable(actual)
    expected[tuple(slice(begin, begin + extent) for begin, extent in zip(begins, extents))] = -3.25
    np.testing.assert_array_equal(actual.numpy(), expected)


def test_fill_dynamic_shape_cpu():
    extent = ir.Var("extent", "int32")
    output = TensorView.create("output", [extent], "int32")
    operation = parse("fill", [output.tensor, ir.const(5, "int32")])
    executable = tirx.compile(function(lower(operation, Target("c")), [output]), target="c")
    for size in (1, 7, 32):
        actual = runtime.tensor(np.zeros(size, dtype="int32"))
        executable(actual)
        np.testing.assert_array_equal(actual.numpy(), np.full(size, 5, dtype="int32"))


def test_fill_casts_value_to_destination_type():
    output = TensorView.create("output", [16], "float32")
    operation = parse("fill", [output.tensor, ir.const(7, "int32")])
    executable = tirx.compile(function(lower(operation, Target("c")), [output]), target="c")
    actual = runtime.empty((16,), "float32")
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), np.full(16, 7, dtype="float32"))


@pytest.mark.parametrize("begin,extent", [(-1, 2), (6, 3)])
def test_fill_rejects_out_of_bounds_region(begin, extent):
    output = TensorView.create("output", [8], "int32")
    with pytest.raises(ValueError, match="Fill region"):
        parse("fill", [tensor_region(output, [begin], [extent]), ir.const(1, "int32")])


@pytest.mark.parametrize("shape,begins,extents", [((8, 16), (0, 0), (8, 16)), ((7, 13), (2, 3), (3, 7)), ((3,), (0,), (3,))])
def test_gpu_fill_partition_is_complete_in_cpu_simulation(shape, begins, extents):
    output = TensorView.create("output", shape, "int32")
    operation = parse("fill", [tensor_region(output, begins, extents), ir.const(23, "int32")])
    thread = ir.Var("thread", "int32")
    body = lower(operation, Target({"kind": "cuda", "arch": "sm_90"}), thread, ir.Range(0, 32))
    body = ir.For(thread, 0, 32, 1, ir.ForKind.DEFAULT, body)
    module = ir.IRModule({"main": function(body, [output])})
    module = tirx.transform.VectorizeLoop(False)(module)
    executable = tirx.compile(module, target="c")
    expected = np.zeros(shape, dtype="int32")
    actual = runtime.tensor(expected.copy())
    executable(actual)
    expected[tuple(slice(begin, begin + extent) for begin, extent in zip(begins, extents))] = 23
    np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(8, 16), (7, 13)])
def test_fill_executes_on_cuda(shape):
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("A CUDA device is required")
    output = TensorView.create("output", shape, "float32")
    operation = parse("fill", [output.tensor, ir.const(-0.0, "float32")])
    thread = ir.Var("thread", "int32")
    architecture = f"sm_{device.compute_version.replace('.', '')}"
    target = Target({"kind": "cuda", "arch": architecture, "backend_config": {"cuda": {"compiler": "nvcc"}}}, host="c")
    body = lower(operation, target, thread, ir.Range(0, 32))
    body = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(32)], [thread], None, body)
    executable = tirx.compile(function(body, [output]), target=target)
    actual = runtime.empty(shape, "float32", device=device)
    executable(actual)
    expected = np.full(shape, -0.0, dtype="float32")
    np.testing.assert_array_equal(actual.numpy().view("uint32"), expected.view("uint32"))


@pytest.mark.parametrize("value,bulk", [(0.0, True), (-0.0, False), (2.0, False)])
def test_blackwell_bulk_fill_preserves_bit_pattern(value, bulk):
    temporary = TensorView.create("temporary", [128], "float32", scope="shared", data_alignment=16)
    operation = parse("fill", [temporary.tensor, ir.const(value, "float32")])
    thread = ir.Var("thread", "int32")
    body = lower(operation, Target({"kind": "cuda", "arch": "sm_100"}), thread, ir.Range(0, 128))
    calls = []
    stmt_functor.post_order_visit(
        body, lambda node: calls.append(node) if isinstance(node, ir.Call) and node.op.name == "tl.ptx_st_bulk_shared" else None
    )
    assert bool(calls) is bulk
    if bulk:
        pointer = calls[0].args[0]
        assert pointer.ty == ir.PtrType("float32", "shared")
        assert int(pointer.args[2]) == 2
        assert int(calls[0].args[1]) == 512


def test_parallel_layout_follows_fragment_ownership(native_layouts):
    output = TensorView.create("output", [128], "float32", scope="local.fragment")
    index = ir.Var("index", "int32")
    loop = ir.For(index, 0, 128, 1, ir.ForKind.PARALLEL, output.store(index, ir.const(1.0, "float32")))
    operation = native("testing.MakeParallel", loop)
    fragment = native_layouts.Fragment(
        [128], forward_thread_fn=lambda coordinate: coordinate % 32, forward_index_fn=lambda coordinate: coordinate // 32
    )
    assert not infer(operation, Target({"kind": "cuda", "arch": "sm_90"}), ir.Range(0, 32), {output.tensor: fragment})
    layout, predicate, padding = native("testing.ParallelState", operation, ir.Var("thread", "int32"))
    assert not padding
    for coordinate in range(128):
        assert int(layout.map_forward_thread([coordinate])[0]) == coordinate % 32
        assert int(layout(coordinate)[0]) == coordinate // 32
    assert predicate is None


def test_parallel_layout_rejects_conflicting_fragment_ownership(native_layouts):
    source = TensorView.create("source", [32], "float32", scope="local.fragment")
    output = TensorView.create("output", [32], "float32", scope="local.fragment")
    index = ir.Var("index", "int32")
    loop = ir.For(index, 0, 32, 1, ir.ForKind.PARALLEL, output.store(index, source.load(index)))
    operation = native("testing.MakeParallel", loop)
    source_layout = native_layouts.Fragment([32], forward_thread_fn=lambda coordinate: coordinate)
    output_layout = native_layouts.Fragment([32], forward_thread_fn=lambda coordinate: (coordinate % 8) * 4 + coordinate // 8)
    with pytest.raises(RuntimeError, match="conflict"):
        infer(
            operation,
            Target({"kind": "cuda", "arch": "sm_90"}),
            ir.Range(0, 32),
            {source.tensor: source_layout, output.tensor: output_layout},
        )


def test_reducer_induced_layout_preserves_collective_width(native_layouts):
    row = ir.Var("row", "int32")
    column = ir.Var("column", "int32")
    layout = native_layouts.Fragment(
        [2, 32], forward_thread_fn=lambda row, column: row * 32 + column, forward_index_fn=lambda row, column: 0
    )
    analysis = native("testing.AnalyzeReducerSite", layout, [row, column], [row], ir.const(1.0, "float32"), [2], ir.Range(0, 64))
    assert analysis["eligible"], analysis["reason"]
    assert analysis["combine_size"] == 32
    assert [list(step) for step in analysis["steps"]] == [[32, 1]]


@pytest.mark.parametrize(
    "operation,dtype,expected",
    [
        ("sum", "int32", 0),
        ("max", "int32", -(2**31)),
        ("min", "uint64", 2**64 - 1),
        ("bitand", "uint32", 2**32 - 1),
        ("bitor", "int32", 0),
        ("bitxor", "int32", 0),
    ],
)
def test_reducer_identity_uses_native_integer_width(operation, dtype, expected):
    identity = native("testing.ReducerIdentity", operation, ir.PrimType(dtype))
    assert identity.ty == ir.PrimType(dtype)
    assert int(identity) == expected


def test_reducer_ops_keep_typed_access_regions():
    reducer = TensorView.create("reducer", [8], "float32", scope="local.reducer")
    output = TensorView.create("output", [8], "float32", scope="local.fragment")
    init = parse("reducer_init", [reducer.tensor, ir.const(2.0, "float32")])
    reads, writes = native("testing.GetOperatorAccessRegions", init)
    assert not reads and writes[0].source.same_as(reducer.tensor)
    finalize = parse("finalize_reducer_v2", [reducer.tensor, output.tensor])
    reads, writes = native("testing.GetOperatorAccessRegions", finalize)
    assert reads[0].source.same_as(reducer.tensor)
    assert writes[0].source.same_as(reducer.tensor)
    assert writes[1].source.same_as(output.tensor)
    with pytest.raises(RuntimeError, match="materialized"):
        lower(init, Target("c"))


def test_reducer_rejects_effectful_contributions(native_layouts):
    index = ir.Var("index", "int32")
    layout = native_layouts.Fragment([32], forward_thread_fn=lambda coordinate: coordinate)
    value = tirx.call_extern("side_effect", ty="float32")
    result = native("testing.AnalyzeReducerSite", layout, [index], [ir.const(0)], value, [1], ir.Range(0, 32))
    assert not result["eligible"]
    assert "replica-safe" in result["reason"]
