import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, tirx
from tvm_tirx.target import Target

from tilelang._compat.tensor import TensorView

pytestmark = pytest.mark.usefixtures("native_layouts", "native_operator_helpers")


def atomic_operator(name, source, destination, annotations=None):
    call = ir.Call(f"tl.tileop.atomic{name}", [source, destination], attrs=annotations or {})
    return tvm_ffi.get_global_func("tl.ParseOperator")(call, {})


def combine(name, destination, value):
    return {"add": np.add, "max": np.maximum, "min": np.minimum}[name](destination, value)


@pytest.mark.parametrize("name", ["add", "max", "min"])
@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "int32"])
def test_cpu_atomic_regions_and_scalar_values(name, scalar, dtype):
    source = TensorView.create("source", [5, 8], dtype)
    destination = TensorView.create("destination", [6, 10], dtype)
    source_operand = ir.const(3, dtype) if scalar else source.region([ir.Range(1, 4), ir.Range(2, 7)])
    op = atomic_operator(name, source_operand, destination.region([ir.Range(2, 5), ir.Range(4, 9)]), {"memory_order": 5, "use_tma": False})
    clone = tvm_ffi.get_global_func("tl.testing.CloneOperator")(op)
    get_regions = tvm_ffi.get_global_func("tl.testing.GetOperatorAccessRegions")
    ir.assert_structural_equal(get_regions(op), get_regions(clone))
    target = Target("c")
    with target:
        body = tvm_ffi.get_global_func("tl.testing.LowerOperator")(op, target, ir.Range(0, 1), ir.const(0), {}, {})
    function = tirx.Function([source.tensor, destination.tensor], body, ret_type=ir.TupleType([])).with_attr(
        "global_symbol", "cpu_atomic_test"
    )
    assert tirx.analysis.check_well_formed(function)
    executable = tirx.compile(function, target=target)
    values = (np.arange(40).reshape(5, 8) % 13 - 7).astype(dtype)
    initial = (np.arange(60).reshape(6, 10) % 11 - 5).astype(dtype)
    expected = initial.copy()
    expected[2:5, 4:9] = combine(name, expected[2:5, 4:9], 3 if scalar else values[1:4, 2:7])
    actual = runtime.tensor(initial)
    executable(runtime.tensor(values), actual)
    np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize("name", ["add", "max", "min"])
def test_cpu_atomic_bounds_include_unit_axes(name):
    source = TensorView.create("source", [4, 8], "float32")
    output = TensorView.create("output", [4], "float32")
    op = atomic_operator(name, source.region([ir.Range(7, 8), ir.Range(0, 4)]), output.tensor)
    target = Target("c")
    with target:
        body = tvm_ffi.get_global_func("tl.testing.LowerOperator")(op, target, ir.Range(0, 1), ir.const(0), {}, {})
    function = tirx.Function([source.tensor, output.tensor], body, ret_type=ir.TupleType([])).with_attr(
        "global_symbol", "atomic_bounds_test"
    )
    executable = tirx.compile(function, target=target)
    expected = np.arange(4, dtype="float32")
    actual = runtime.tensor(expected)
    executable(runtime.tensor(np.ones((4, 8), dtype="float32")), actual)
    np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize("annotations", [{"memory_order": -1}, {"memory_order": 6}, {"memory_order": "relaxed"}, {"use_tma": 2}])
def test_atomic_rejects_invalid_annotations(annotations):
    tensor = TensorView.create("tensor", [4], "float32")
    with pytest.raises(ValueError, match="Atomic"):
        atomic_operator("add", ir.const(1.0), tensor.tensor, annotations)


@pytest.mark.gpu
@pytest.mark.parametrize("name", ["add", "max", "min"])
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_atomic_tile_runs_with_multiple_cuda_blocks(name, dtype, compile_tilelang):
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("A CUDA device is required")
    target = Target(
        {
            "kind": "cuda",
            "arch": f"sm_{device.compute_version.replace('.', '')}",
            "backend_config": {"cuda": {"compiler": "nvcc"}},
        },
        host="c",
    )
    source = TensorView.create("source", [256], dtype)
    output = TensorView.create("output", [256], dtype)
    op = atomic_operator(name, source.tensor, output.tensor, {"memory_order": 4})
    thread = ir.Var("thread", "int32")
    block = ir.Var("block", "int32")
    with target:
        body = tvm_ffi.get_global_func("tl.testing.LowerOperator")(op, target, ir.Range(0, 64), thread, {}, {})
    body = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(64)], [thread], None, body)
    body = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("blockIdx.x"), ir.const(7)], [block], None, body)
    function = tirx.Function([source.tensor, output.tensor], body, ret_type=ir.TupleType([])).with_attr("global_symbol", "gpu_atomic_test")
    module = ir.IRModule({"main": function})
    with target:
        module = tvm_ffi.get_global_func("tl.transform.VectorizeLoop")(True)(module)
        module = tvm_ffi.get_global_func("tl.transform.LowerAccessPtr")()(module)
    assert tirx.analysis.check_well_formed(module)
    executable = compile_tilelang(module, target=target)
    numpy_dtype = pytest.importorskip("ml_dtypes").bfloat16 if dtype == "bfloat16" else dtype
    values = (np.arange(256) % 19 - 9).astype(numpy_dtype)
    initial = (np.arange(256) % 11 - 5).astype(numpy_dtype)
    expected = initial.astype("float32")
    for _ in range(7):
        expected = combine(name, expected, values.astype("float32"))
    actual = runtime.tensor(initial, device=device)
    executable(runtime.tensor(values, device=device), actual)
    np.testing.assert_array_equal(actual.numpy().astype("float32"), expected)
