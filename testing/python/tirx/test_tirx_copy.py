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


def parse(name, arguments, annotations=None, block_annotations=None):
    return native("ParseOperator", ir.Call(f"tl.tileop.{name}", arguments, attrs=annotations or {}), block_annotations or {})


def lower(operation, target, thread=0, threads=1, normal=False):
    thread = ir.const(thread) if isinstance(thread, int) else thread
    with target:
        if normal:
            return native("testing.LowerNormalCopy", operation, target, ir.Range(0, threads), thread, {})
        return native("testing.LowerOperator", operation, target, ir.Range(0, threads), thread, {}, {})


def function(body, tensors):
    result = tirx.Function([tensor.tensor for tensor in tensors], body, ret_type=ir.TupleType([])).with_attr(
        "global_symbol", "native_copy_test"
    )
    assert tirx.analysis.check_well_formed(result)
    ir.assert_structural_equal(result, ir.load_json(ir.save_json(result)))
    return result


def region(tensor, begins, extents):
    return tensor.region([ir.Range.from_min_extent(begin, extent) for begin, extent in zip(begins, extents)])


def nodes(body, node_type):
    result = []
    stmt_functor.post_order_visit(body, lambda node: result.append(node) if isinstance(node, node_type) else None)
    return result


@pytest.mark.parametrize("shape", [(8, 16), (7, 13), (), (0, 3)])
@pytest.mark.parametrize("dtype", ["float32", "int32"])
def test_copy_cpu(shape, dtype):
    source = TensorView.create("source", shape, dtype)
    output = TensorView.create("output", shape, dtype)
    operation = parse("copy", [source.tensor, output.tensor])
    executable = tirx.compile(function(lower(operation, Target("c")), [source, output]), target="c")
    expected = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    actual = runtime.tensor(np.full(shape, -1, dtype=dtype))
    executable(runtime.tensor(expected), actual)
    np.testing.assert_array_equal(actual.numpy(), expected)


def test_copy_dynamic_shape_cpu():
    extent = ir.Var("extent", "int32")
    source = TensorView.create("source", [extent], "float32")
    output = TensorView.create("output", [extent], "float32")
    executable = tirx.compile(function(lower(parse("copy", [source.tensor, output.tensor]), Target("c")), [source, output]), target="c")
    for size in (1, 7, 32):
        expected = np.arange(size, dtype="float32")
        actual = runtime.empty((size,), "float32")
        executable(runtime.tensor(expected), actual)
        np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize("source_begin,destination_begin,extent", [(-2, 0, 8), (5, 1, 8), (0, -2, 8), (0, 6, 8), (2, 3, 3), (0, 0, 0)])
def test_copy_slices_guard_both_tensors(source_begin, destination_begin, extent):
    source = TensorView.create("source", [8], "float32")
    output = TensorView.create("output", [8], "int32")
    operation = parse(
        "copy",
        [region(source, [source_begin], [extent]), region(output, [destination_begin], [extent])],
        block_annotations={"safe_value_map": {source.tensor: ir.const(-5.0, "float32")}},
    )
    executable = tirx.compile(function(lower(operation, Target("c")), [source, output]), target="c")
    source_values = np.arange(8, dtype="float32") + 0.5
    expected = np.full((8,), -9, dtype="int32")
    for offset in range(extent):
        if 0 <= destination_begin + offset < 8:
            expected[destination_begin + offset] = source_values[source_begin + offset] if 0 <= source_begin + offset < 8 else -5
    actual = runtime.tensor(np.full((8,), -9, dtype="int32"))
    executable(runtime.tensor(source_values), actual)
    np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize("source_begin,destination_begin", [(-1, 0), (2, 0), (0, -1), (0, 2), (1, 1)])
def test_copy_unit_dimensions_are_bounds_checked(source_begin, destination_begin):
    source = TensorView.create("source", [2, 8], "float32")
    output = TensorView.create("output", [2, 8], "float32")
    operation = parse("copy", [region(source, [source_begin, 0], [1, 8]), region(output, [destination_begin, 0], [1, 8])])
    executable = tirx.compile(function(lower(operation, Target("c")), [source, output]), target="c")
    source_values = np.arange(16, dtype="float32").reshape(2, 8)
    expected = np.full((2, 8), -1, dtype="float32")
    if 0 <= destination_begin < 2:
        expected[destination_begin] = source_values[source_begin] if 0 <= source_begin < 2 else 0
    actual = runtime.tensor(np.full((2, 8), -1, dtype="float32"))
    executable(runtime.tensor(source_values), actual)
    np.testing.assert_array_equal(actual.numpy(), expected)


def test_copy_larger_local_region_zero_fills_beyond_requested_source():
    source = TensorView.create("source", [16], "float32")
    output = TensorView.create("output", [8], "float32", scope="local")
    operation = parse("copy", [region(source, [2], [5]), output.tensor])
    executable = tirx.compile(function(lower(operation, Target("c")), [source, output]), target="c")
    source_values = np.arange(16, dtype="float32")
    actual = runtime.empty((8,), "float32")
    executable(runtime.tensor(source_values), actual)
    np.testing.assert_array_equal(actual.numpy(), np.array([2, 3, 4, 5, 6, 0, 0, 0], dtype="float32"))


@pytest.mark.parametrize("name,flag", [("copy", None), ("async_copy", "is_async_copy"), ("tma_copy", "is_tma_copy")])
def test_copy_aliases_keep_metadata_and_accesses(name, flag):
    source = TensorView.create("source", [16], "float32")
    output = TensorView.create("output", [16], "float32")
    operation = parse(name, [source.tensor, output.tensor], {"coalesced_width": ir.const(4), "dst_block": ir.const(2)})
    if flag:
        assert int(operation.annotations[flag]) == 1
    assert int(operation.dst_block) == 2
    cloned = native("testing.CloneOperator", operation)
    for current in (operation, cloned):
        reads, writes = native("testing.GetOperatorAccessRegions", current)
        assert reads[0].source.same_as(source.tensor)
        assert writes[0].source.same_as(output.tensor)
        assert int(current.annotations["coalesced_width"]) == 4


@pytest.mark.parametrize("shape", [(8, 16), (7, 13), ()])
def test_normal_copy_partition_in_cpu_simulation(shape):
    source = TensorView.create("source", shape, "float32")
    output = TensorView.create("output", shape, "float32")
    operation = parse("copy", [source.tensor, output.tensor])
    thread = ir.Var("thread", "int32")
    body = lower(operation, Target({"kind": "cuda", "arch": "sm_90"}), thread, 32, normal=True)
    body = ir.For(thread, 0, 32, 1, ir.ForKind.DEFAULT, body)
    module = tirx.transform.VectorizeLoop(False)(ir.IRModule({"main": function(body, [source, output])}))
    executable = tirx.compile(module, target="c")
    expected = np.arange(np.prod(shape), dtype="float32").reshape(shape)
    actual = runtime.empty(shape, "float32")
    executable(runtime.tensor(expected), actual)
    np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(8, 16), (7, 13), ()])
def test_normal_copy_executes_on_cuda(shape):
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("A CUDA device is required")
    source = TensorView.create("source", shape, "float32")
    output = TensorView.create("output", shape, "float32")
    operation = parse("copy", [source.tensor, output.tensor])
    thread = ir.Var("thread", "int32")
    target = Target(
        {"kind": "cuda", "arch": f"sm_{device.compute_version.replace('.', '')}", "backend_config": {"cuda": {"compiler": "nvcc"}}},
        host="c",
    )
    body = lower(operation, target, thread, 32, normal=True)
    body = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(32)], [thread], None, body)
    executable = tirx.compile(function(body, [source, output]), target=target)
    expected = np.arange(np.prod(shape), dtype="float32").reshape(shape)
    actual = runtime.empty(shape, "float32", device=device)
    executable(runtime.tensor(expected, device=device), actual)
    np.testing.assert_array_equal(actual.numpy(), expected)


def im2col_expected(source, shape, position_step, channel_step, kernel, stride, dilation, padding):
    batch, height, width, channels = source.shape
    out_h = max(0, (height + 2 * padding - (kernel - 1) * dilation - 1) // stride + 1)
    out_w = max(0, (width + 2 * padding - (kernel - 1) * dilation - 1) // stride + 1)
    expected = np.zeros(shape, dtype=source.dtype)
    for row, column in np.ndindex(shape):
        position = position_step * shape[0] + row
        channel = channel_step * shape[1] + column
        if position < 0 or position >= batch * out_h * out_w or channel < 0 or channel >= channels * kernel * kernel:
            continue
        batch_index, pixel = divmod(position, out_h * out_w)
        out_row, out_column = divmod(pixel, out_w)
        source_row = out_row * stride + (channel // (kernel * channels)) * dilation - padding
        source_column = out_column * stride + (channel // channels % kernel) * dilation - padding
        if 0 <= source_row < height and 0 <= source_column < width:
            expected[row, column] = source[batch_index, source_row, source_column, channel % channels]
    return expected


@pytest.mark.parametrize(
    "parameters", [(0, 0, 3, 1, 1, 1), (1, 1, 3, 2, 1, 0), (0, 0, 2, 1, 2, 1), (-1, -1, 2, 1, 1, 0), (8, 8, 3, 1, 1, 1), (0, 0, 8, 1, 1, 0)]
)
def test_im2col_cpu_guards_batch_channel_and_padding(parameters):
    source = TensorView.create("source", [2, 4, 5, 3], "float32")
    output = TensorView.create("output", [32, 16], "float32")
    operation = parse("im2col", [source.tensor, output.tensor, *(ir.const(value) for value in parameters)])
    executable = tirx.compile(function(lower(operation, Target("c")), [source, output]), target="c")
    source_values = np.arange(120, dtype="float32").reshape(2, 4, 5, 3)
    actual = runtime.empty((32, 16), "float32")
    executable(runtime.tensor(source_values), actual)
    np.testing.assert_array_equal(actual.numpy(), im2col_expected(source_values, (32, 16), *parameters))


def test_im2col_honors_source_and_destination_regions():
    source = TensorView.create("source", [2, 6, 7, 3], "float32")
    output = TensorView.create("output", [1, 20, 18], "float32")
    parameters = (0, 0, 2, 1, 1, 0)
    operation = parse(
        "c2d_im2col",
        [region(source, [1, 1, 2, 1], [1, 4, 4, 2]), region(output, [0, 2, 3], [1, 16, 12]), *(ir.const(value) for value in parameters)],
    )
    executable = tirx.compile(function(lower(operation, Target("c")), [source, output]), target="c")
    source_values = np.arange(252, dtype="float32").reshape(2, 6, 7, 3)
    expected = np.full((1, 20, 18), -1, dtype="float32")
    expected[0, 2:18, 3:15] = im2col_expected(source_values[1:2, 1:5, 2:6, 1:3], (16, 12), *parameters)
    actual = runtime.tensor(np.full((1, 20, 18), -1, dtype="float32"))
    executable(runtime.tensor(source_values), actual)
    np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize("index,value", [(4, 0), (5, -1), (6, 0), (7, -1)])
def test_im2col_rejects_invalid_parameters(index, value):
    source = TensorView.create("source", [1, 4, 4, 1], "float32")
    output = TensorView.create("output", [16, 4], "float32")
    arguments = [source.tensor, output.tensor, *(ir.const(value) for value in (0, 0, 2, 1, 1, 0))]
    arguments[index] = ir.const(value)
    with pytest.raises(ValueError, match="Im2Col"):
        parse("im2col", arguments)


@pytest.mark.parametrize("shape,dtype", [((8, 16), "int32"), ((2**16, 2**16), "int64")])
def test_parallel_fuser_widens_coordinates(shape, dtype):
    output = TensorView.create("output", shape, "float32")
    row, column = ir.Var("row", "int32"), ir.Var("column", "int32")
    body = output.store([row, column], ir.const(1.0, "float32"))
    body = ir.For(row, 0, shape[0], 1, ir.ForKind.PARALLEL, ir.For(column, 0, shape[1], 1, ir.ForKind.PARALLEL, body))
    fused = native("testing.FuseParallelLoops", body)
    assert len(nodes(fused, ir.For)) == 1
    assert fused.var.ty == ir.PrimType(dtype)
    assert int(fused.end) == np.prod(shape)
    function(fused, [output])


@pytest.mark.parametrize(
    "inner_extent,begin,step,annotations,scope",
    [
        (7, 0, 1, {}, "global"),
        (8, 1, 1, {}, "global"),
        (8, 0, 2, {}, "global"),
        (8, 0, 1, {"inner_marker": ir.const(3)}, "global"),
        (8, 0, 1, {}, "local.fragment"),
    ],
)
def test_parallel_fuser_preserves_nonfusible_nests(inner_extent, begin, step, annotations, scope):
    output = TensorView.create("output", [8, 8], "float32", scope=scope)
    row, column = ir.Var("row", "int32"), ir.Var("column", "int32")
    body = output.store([row, column], ir.const(1.0, "float32"))
    body = ir.For(
        row, 0, 8, 1, ir.ForKind.PARALLEL, ir.For(column, begin, inner_extent, step, ir.ForKind.PARALLEL, body, annotations=annotations)
    )
    ir.assert_structural_equal(native("testing.FuseParallelLoops", body), body)


@pytest.mark.parametrize("begin,end,step", [(1, 8, 2), (7, -1, -2)])
def test_parallel_inference_preserves_serial_steps(begin, end, step):
    output = TensorView.create("output", [32, 8], "float32")
    row, column = ir.Var("row", "int32"), ir.Var("column", "int32")
    body = output.store([row, column], ir.const(1.0, "float32"))
    body = ir.For(row, 0, 32, 1, ir.ForKind.PARALLEL, ir.For(column, begin, end, step, ir.ForKind.DEFAULT, body))
    operation = native("testing.MakeParallel", body)
    target = Target({"kind": "cuda", "arch": "sm_90"})
    with target:
        for level in (0, 1, 2):
            native("testing.InferOperator", operation, target, ir.Range(0, 32), {}, {}, {}, level, {})
    assert int(nodes(body, ir.For)[0].step) == step


@pytest.mark.parametrize("shape", [(8, 16), (7, 13), (2, 3, 5), (1, 8), (1, 1), (0, 3)])
def test_transpose_cpu(shape):
    destination_shape = (*shape[:-2], shape[-1], shape[-2])
    source = TensorView.create("source", shape, "float32")
    output = TensorView.create("output", destination_shape, "int32")
    operation = parse("transpose", [source.tensor, output.tensor])
    executable = tirx.compile(function(lower(operation, Target("c")), [source, output]), target="c")
    source_values = np.arange(np.prod(shape), dtype="float32").reshape(shape) + 0.25
    actual = runtime.empty(destination_shape, "int32")
    executable(runtime.tensor(source_values), actual)
    np.testing.assert_array_equal(actual.numpy(), source_values.swapaxes(-1, -2).astype("int32"))


@pytest.mark.parametrize("source_shape,destination_shape", [((8,), (8,)), ((2, 3), (2, 3)), ((2, 3), (1, 3, 2))])
def test_transpose_rejects_shape_mismatches(source_shape, destination_shape):
    source = TensorView.create("source", source_shape, "float32")
    output = TensorView.create("output", destination_shape, "float32")
    with pytest.raises(ValueError, match="Transpose"):
        parse("transpose", [source.tensor, output.tensor])


def test_transpose_regions_keep_offsets_and_guards():
    source = TensorView.create("source", [4, 6], "float32")
    output = TensorView.create("output", [8, 7], "float32")
    operation = parse("transpose", [region(source, [-1, 2], [4, 3]), region(output, [3, 5], [3, 4])])
    executable = tirx.compile(function(lower(operation, Target("c")), [source, output]), target="c")
    source_values = np.arange(24, dtype="float32").reshape(4, 6)
    expected = np.full((8, 7), -1, dtype="float32")
    expected[3:6, 5] = 0
    expected[3:6, 6] = source_values[0, 2:5]
    actual = runtime.tensor(np.full((8, 7), -1, dtype="float32"))
    executable(runtime.tensor(source_values), actual)
    np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "name,source_shape,output_shape,arguments",
    [("transpose", (7, 13), (13, 7), ()), ("transpose", (1, 1), (1, 1), ()), ("im2col", (2, 4, 5, 3), (32, 16), (1, 1, 3, 1, 1, 1))],
)
def test_data_movement_operators_execute_on_cuda(name, source_shape, output_shape, arguments):
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("A CUDA device is required")
    source = TensorView.create("source", source_shape, "float32")
    output = TensorView.create("output", output_shape, "float32")
    operation = parse(name, [source.tensor, output.tensor, *(ir.const(value) for value in arguments)])
    thread = ir.Var("thread", "int32")
    target = Target(
        {"kind": "cuda", "arch": f"sm_{device.compute_version.replace('.', '')}", "backend_config": {"cuda": {"compiler": "nvcc"}}},
        host="c",
    )
    body = lower(operation, target, thread, 32)
    body = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(32)], [thread], None, body)
    executable = tirx.compile(function(body, [source, output]), target=target)
    source_values = np.arange(np.prod(source_shape), dtype="float32").reshape(source_shape)
    expected = source_values.T if name == "transpose" else im2col_expected(source_values, output_shape, *arguments)
    actual = runtime.empty(output_shape, "float32", device=device)
    executable(runtime.tensor(source_values, device=device), actual)
    np.testing.assert_array_equal(actual.numpy(), expected)


def test_fused_loop_preserves_call_attribute_bindings():
    output = TensorView.create("output", [8, 16], "float32")
    row, column = ir.Var("row", "int32"), ir.Var("column", "int32")
    access = region(output, [row, column], [1, 1])
    call = ir.Call("tl.tileop.fill", [access, ir.const(2.0, "float32")], attrs={"coordinates": [row, column]})
    body = ir.For(row, 0, 8, 1, ir.ForKind.PARALLEL, ir.For(column, 0, 16, 1, ir.ForKind.PARALLEL, ir.Evaluate(call)))
    fused = native("testing.FuseParallelLoops", body)
    function(fused, [output])
    new_call = nodes(fused, ir.Call)[0]
    for actual, domain in zip(new_call.attrs["coordinates"], new_call.args[0].region):
        ir.assert_structural_equal(actual, domain.begin)
