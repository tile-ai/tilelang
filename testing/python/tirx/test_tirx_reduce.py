import numpy as np
from pathlib import Path
import pytest
import tvm_ffi
from tvm_tirx import runtime, tirx
from tilelang._compat import ir
from tvm_tirx.target import Target

from tilelang._compat.tensor import TensorView


pytestmark = pytest.mark.usefixtures("native_layouts")


def native(name, *arguments):
    return tvm_ffi.get_global_func(f"tl.{name}")(*arguments)


def operation(source, destination, kind, dimension, clear=True, annotations=None):
    call = ir.Call(
        "tl.tileop.reduce", [source, destination, ir.StringImm(kind), ir.const(dimension), ir.const(clear)], attrs=annotations or {}
    )
    return native("ParseOperator", call, {})


def cpu_executable(op, source, output, *, compile_tilelang):
    target = Target("c")
    with target:
        body = native("testing.LowerOperator", op, target, ir.Range(0, 1), ir.const(0), {}, {})
    function = tirx.Function([source.tensor, output.tensor], body, ret_type=ir.TupleType([])).with_attr(
        "global_symbol", "native_reduce_test"
    )
    assert ir.check_well_formed(function)
    ir.assert_structural_equal(function, ir.load_json(ir.save_json(function)))
    return compile_tilelang(function, target=target)


def expected_reduce(values, kind, axis, keepdims=False):
    if kind in ("abssum", "absmax"):
        values = np.abs(values)
    if kind in ("sum", "abssum"):
        return values.sum(axis=axis, dtype=values.dtype, keepdims=keepdims)
    if kind in ("max", "absmax"):
        return values.max(axis=axis, keepdims=keepdims)
    if kind == "min":
        return values.min(axis=axis, keepdims=keepdims)
    combine = {"bitand": np.bitwise_and, "bitor": np.bitwise_or, "bitxor": np.bitwise_xor}[kind]
    return combine.reduce(values, axis=axis, keepdims=keepdims)


@pytest.mark.parametrize(
    "kind,dtype",
    [
        (kind, dtype)
        for dtype in ("int32", "uint64", "float32")
        for kind in ("sum", "abssum", "max", "min", "absmax", "bitand", "bitor", "bitxor")
        if dtype != "float32" or not kind.startswith("bit")
    ],
)
@pytest.mark.parametrize("clear", [True, False])
def test_cpu_reduce_values_and_accumulation(kind, dtype, clear, *, compile_tilelang):
    source = TensorView.create("source", [3, 5], dtype, scope="local")
    output = TensorView.create("output", [3], dtype, scope="local")
    op = operation(source.tensor, output.tensor, kind, 1, clear)
    executable = cpu_executable(op, source, output, compile_tilelang=compile_tilelang)
    offset = 0 if dtype == "uint64" else 8
    values = (np.arange(15).reshape(3, 5) - offset).astype(dtype)
    expected = expected_reduce(values, kind, 1)
    initial = np.full((3,), 7, dtype=dtype)
    if not clear:
        combine_kind = {"abssum": "sum", "absmax": "max"}.get(kind, kind)
        expected = expected_reduce(np.stack((initial, expected), axis=1), combine_kind, 1)
    actual = runtime.tensor(initial)
    executable(runtime.tensor(values), actual)
    np.testing.assert_array_equal(actual.numpy(), expected)
    reads, writes = native("testing.GetOperatorAccessRegions", op)
    assert len(reads) == (1 if clear else 2)
    assert len(writes) == 1
    cloned = native("testing.CloneOperator", op)
    assert cloned.src.same_as(source.tensor)
    assert cloned.dst.same_as(output.tensor)


@pytest.mark.parametrize(
    "shape,axis,keepdims",
    [((5,), 0, False), ((5,), 0, True), ((3, 5), 0, False), ((3, 5), 0, True), ((3, 5), 1, True), ((2, 3, 5), 1, False)],
)
def test_cpu_reduce_axes_and_keepdims(shape, axis, keepdims, *, compile_tilelang):
    destination_shape = list(shape)
    if keepdims:
        destination_shape[axis] = 1
    else:
        destination_shape.pop(axis)
    source = TensorView.create("source", shape, "float32", scope="local")
    output = TensorView.create("output", destination_shape, "float32", scope="local")
    executable = cpu_executable(operation(source.tensor, output.tensor, "sum", axis), source, output, compile_tilelang=compile_tilelang)
    values = np.arange(np.prod(shape), dtype="float32").reshape(shape)
    actual = runtime.empty(destination_shape, "float32")
    executable(runtime.tensor(values), actual)
    np.testing.assert_array_equal(actual.numpy(), expected_reduce(values, "sum", axis, keepdims))


def test_cpu_reduce_dynamic_extent(*, compile_tilelang):
    extent = ir.Var("extent", "int32")
    source = TensorView.create("source", [3, extent], "float32", scope="local")
    output = TensorView.create("output", [3], "float32", scope="local")
    executable = cpu_executable(operation(source.tensor, output.tensor, "sum", 1), source, output, compile_tilelang=compile_tilelang)
    for size in (0, 1, 7, 32):
        values = np.arange(3 * size, dtype="float32").reshape(3, size)
        actual = runtime.empty((3,), "float32")
        executable(runtime.tensor(values), actual)
        np.testing.assert_array_equal(actual.numpy(), values.sum(axis=1))


@pytest.mark.parametrize(
    "kind,dtype,expected",
    [
        ("sum", "float32", 0),
        ("abssum", "int32", 0),
        ("absmax", "float32", 0),
        ("max", "float32", -np.inf),
        ("min", "int32", 2**31 - 1),
        ("bitand", "uint64", 2**64 - 1),
    ],
)
def test_empty_reduce_preserves_identity(kind, dtype, expected, *, compile_tilelang):
    source = TensorView.create("source", [3, 0], dtype, scope="local")
    output = TensorView.create("output", [3], dtype, scope="local")
    executable = cpu_executable(operation(source.tensor, output.tensor, kind, 1), source, output, compile_tilelang=compile_tilelang)
    actual = runtime.empty((3,), dtype)
    executable(runtime.tensor(np.empty((3, 0), dtype=dtype)), actual)
    np.testing.assert_array_equal(actual.numpy(), np.full((3,), expected, dtype=dtype))


def test_reduce_preserves_sliced_tensor_regions(*, compile_tilelang):
    source = TensorView.create("source", [5, 8], "float32", scope="local")
    output = TensorView.create("output", [6], "float32", scope="local")
    source_region = source.region([ir.Range(1, 4), ir.Range(2, 6)])
    destination_region = output.region([ir.Range(2, 5)])
    executable = cpu_executable(operation(source_region, destination_region, "sum", 1), source, output, compile_tilelang=compile_tilelang)
    values = np.arange(40, dtype="float32").reshape(5, 8)
    expected = np.full((6,), -1, dtype="float32")
    expected[2:5] = values[1:4, 2:6].sum(axis=1)
    actual = runtime.tensor(np.full((6,), -1, dtype="float32"))
    executable(runtime.tensor(values), actual)
    np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.gpu
@pytest.mark.parametrize("clear", [True, False])
@pytest.mark.parametrize("begin,end", [(1, 5), (2, 7), (3, 3)])
def test_cuda_local_reduce_preserves_sliced_regions(clear, begin, end, *, compile_tilelang):
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("A CUDA device is required")
    root = Path(__file__).resolve().parents[3]
    target = Target(
        {
            "kind": "cuda",
            "arch": f"sm_{device.compute_version.replace('.', '')}",
            "backend_config": {
                "cuda": {"compiler": "nvcc", "nvcc": [f"-I{root / 'src'}", f"-I{root / '3rdparty' / 'cutlass' / 'include'}"]}
            },
        },
        host="c",
    )
    source = TensorView.create("source", [5, 8], "float32")
    output = TensorView.create("output", [6], "float32")
    local_source = TensorView.create("local_source", [5, 8], "float32", scope="local")
    local_output = TensorView.create("local_output", [6], "float32", scope="local")
    source_region = local_source.region([ir.Range(1, 4), ir.Range(begin, end)])
    destination_region = local_output.region([ir.Range(2, 5)])
    op = operation(source_region, destination_region, "sum", 1, clear)
    row, column, output_index, stored_index = [ir.Var(name, "int32") for name in ("row", "column", "output_index", "stored_index")]
    load = ir.For(
        row,
        0,
        5,
        1,
        ir.ForKind.DEFAULT,
        ir.For(column, 0, 8, 1, ir.ForKind.DEFAULT, local_source.store([row, column], source.load([row, column]))),
    )
    initialize = ir.For(output_index, 0, 6, 1, ir.ForKind.DEFAULT, local_output.store(output_index, ir.const(-1.0, "float32")))
    thread = ir.Var("thread", "int32")
    with target:
        reduced = native("testing.LowerReduction", op, target, ir.Range(0, 1), thread, {}, {})
    store = ir.For(stored_index, 0, 6, 1, ir.ForKind.DEFAULT, output.store(stored_index, local_output.load(stored_index)))
    body = ir.SeqStmt([local_source.allocate(), local_output.allocate(), load, initialize, reduced, store])
    body = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(1)], [thread], None, body)
    function = tirx.Function([source.tensor, output.tensor], body, ret_type=ir.TupleType([])).with_attr(
        "global_symbol", "local_slice_reduce_test"
    )
    executable = compile_tilelang(function, target=target)
    values = np.arange(40, dtype="float32").reshape(5, 8)
    expected = np.full((6,), -1, dtype="float32")
    expected[2:5] = values[1:4, begin:end].sum(axis=1) - (0 if clear else 1)
    actual = runtime.empty((6,), "float32", device=device)
    executable(runtime.tensor(values, device=device), actual)
    np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize("dimension", [-1, 2])
def test_reduce_rejects_invalid_axis(dimension):
    source = TensorView.create("source", [3, 5], "float32", scope="local")
    output = TensorView.create("output", [3], "float32", scope="local")
    with pytest.raises(ValueError, match="dimension"):
        operation(source.tensor, output.tensor, "sum", dimension)


@pytest.mark.parametrize("annotations", [{"batch": 0}, {"batch": 2**32}, {"nan_propagate": "yes"}])
def test_reduce_rejects_invalid_annotations(annotations):
    source = TensorView.create("source", [3, 5], "float32", scope="local")
    output = TensorView.create("output", [3], "float32", scope="local")
    with pytest.raises(ValueError, match="ReduceOp"):
        operation(source.tensor, output.tensor, "sum", 1, annotations=annotations)


@pytest.mark.parametrize("shape", [(3,), ()])
def test_reduce_layout_preserves_collective_replication(shape, native_layouts):
    source_shape = [3, 32] if shape else [32]
    source = TensorView.create("source", source_shape, "float32", scope="local.fragment")
    output = TensorView.create("output", shape, "float32", scope="local.fragment")
    op = operation(source.tensor, output.tensor, "sum", len(source_shape) - 1)
    layout = native_layouts.Fragment(source_shape, forward_thread_fn=(lambda row, column: column) if shape else (lambda column: column))
    target = Target({"kind": "cuda", "arch": "sm_90"})
    with target:
        inferred = native("testing.InferOperator", op, target, ir.Range(0, 32), {source.tensor: layout}, {}, {}, 0, {})
    result = inferred[output.tensor]
    assert list(map(int, result.get_input_shape())) == list(shape)
    assert int(result.replicate_size) == 32


@pytest.mark.parametrize(
    "kind,dtype,nan,name",
    [
        ("sum", "float32", False, "tl.add2"),
        ("max", "float16", True, "tl.max2_nan"),
        ("min", "bfloat16", True, "tl.min2_nan"),
        ("absmax", "float16", False, "tl.max2"),
    ],
)
def test_packed_reduce_uses_typed_dialect_ops(kind, dtype, nan, name):
    source = TensorView.create("source", [4], dtype, scope="local")
    output = TensorView.create("output", [], dtype, scope="local")
    op = operation(source.tensor, output.tensor, kind, 0, annotations={"nan_propagate": nan})
    accumulator = ir.Var("accumulator", f"{dtype}x2")
    contribution = ir.Var("contribution", f"{dtype}x2")
    result = native("testing.ReduceCombine", op, 2, accumulator, contribution)
    assert result.ty == accumulator.ty
    assert result.op.name == name
    assert native("testing.ReduceCodegen", op, 2).endswith({"float32": "_f32x2", "float16": "_fp16x2", "bfloat16": "_bf16x2"}[dtype])


def cuda_reduce_function(kind, rows, columns, threads, batch, clear, native_layouts, *, compile_tilelang):
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("A CUDA device is required")
    root = Path(__file__).resolve().parents[3]
    options = [
        f"-I{root / 'src'}",
        f"-I{root / '3rdparty' / 'cutlass' / 'include'}",
        "-include",
        str(root / "src" / "tl_templates" / "cuda" / "reduce.h"),
    ]
    target = Target(
        {
            "kind": "cuda",
            "arch": f"sm_{device.compute_version.replace('.', '')}",
            "backend_config": {"cuda": {"compiler": "nvcc", "nvcc": options}},
        },
        host="c",
    )
    source = TensorView.create("source", [rows, columns], "float32")
    output = TensorView.create("output", [rows], "float32")
    logical_source = TensorView.create("logical_source", [rows, columns], "float32", scope="local.fragment")
    logical_output = TensorView.create("logical_output", [rows], "float32", scope="local.fragment")
    op = operation(logical_source.tensor, logical_output.tensor, kind, 1, clear, {"batch": batch})
    source_layout = native_layouts.Fragment(
        [rows, columns],
        forward_thread_fn=lambda row, column: column % threads,
        forward_index_fn=lambda row, column: [row, column // threads],
    )
    with target:
        inferred = native("testing.InferOperator", op, target, ir.Range(0, threads), {logical_source.tensor: source_layout}, {}, {}, 0, {})
    destination_layout = inferred[logical_output.tensor]
    physical_source = TensorView.create("register_source", source_layout.get_output_shape(), "float32", scope="local")
    physical_output = TensorView.create("register_output", destination_layout.get_output_shape(), "float32", scope="local")
    thread = ir.Var("thread", "int32")
    row = ir.Var("row", "int32")
    slot = ir.Var("slot", "int32")
    load = physical_source.store([row, slot], source.load([row, thread + slot * threads]))
    load = ir.For(row, 0, rows, 1, ir.ForKind.DEFAULT, ir.For(slot, 0, columns // threads, 1, ir.ForKind.DEFAULT, load))
    initialize_row = ir.Var("initialize_row", "int32")
    initialize = ir.For(initialize_row, 0, rows, 1, ir.ForKind.DEFAULT, physical_output.store(initialize_row, ir.const(7.0, "float32")))
    layouts = {logical_source.tensor: source_layout, logical_output.tensor: destination_layout}
    remap = {logical_source.tensor: physical_source.tensor, logical_output.tensor: physical_output.tensor}
    with target:
        reduced = native("testing.LowerReduction", op, target, ir.Range(0, threads), thread, layouts, remap)
    stored_row = ir.Var("stored_row", "int32")
    stored = ir.For(stored_row, 0, rows, 1, ir.ForKind.DEFAULT, output.store(stored_row, physical_output.load(stored_row)))
    stored = ir.If(thread == 0, stored, None)
    body = ir.SeqStmt([physical_source.allocate(), physical_output.allocate(), load, initialize, reduced, stored])
    body = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(threads)], [thread], None, body)
    function = tirx.Function([source.tensor, output.tensor], body, ret_type=ir.TupleType([])).with_attr(
        "global_symbol", "native_cuda_reduce"
    )
    assert ir.check_well_formed(function)
    ir.assert_structural_equal(function, ir.load_json(ir.save_json(function)))
    return compile_tilelang(function, target=target), device


@pytest.mark.gpu
@pytest.mark.parametrize(
    "kind,threads,batch,clear",
    [
        ("sum", 32, 1, True),
        ("sum", 32, 1, False),
        ("sum", 64, 1, True),
        ("sum", 64, 2, True),
        ("sum", 64, 4, False),
        ("max", 32, 1, True),
        ("min", 64, 1, False),
        ("abssum", 32, 2, True),
        ("absmax", 64, 2, False),
    ],
)
def test_fragment_reduction_executes_on_cuda(kind, threads, batch, clear, native_layouts, *, compile_tilelang):
    rows, columns = 4, 128
    executable, device = cuda_reduce_function(kind, rows, columns, threads, batch, clear, native_layouts, compile_tilelang=compile_tilelang)
    values = (np.arange(rows * columns).reshape(rows, columns) % 19 - 9).astype("float32")
    expected = expected_reduce(values, kind, 1)
    if not clear:
        combine_kind = {"abssum": "sum", "absmax": "max"}.get(kind, kind)
        expected = expected_reduce(np.stack((np.full(rows, 7, dtype="float32"), expected), axis=1), combine_kind, 1)
    actual = runtime.empty((rows,), "float32", device=device)
    executable(runtime.tensor(values, device=device), actual)
    np.testing.assert_array_equal(actual.numpy(), expected)
