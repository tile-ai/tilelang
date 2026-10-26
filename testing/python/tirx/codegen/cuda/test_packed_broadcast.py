import numpy as np
import pytest

from tvm_tirx import ir, runtime, tirx
from tvm_tirx.ir import prim
from tvm_tirx.target import Target


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("lanes", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("operation", ["min", "max"])
def test_fp8_extrema_promote_comparisons_without_changing_storage(dtype, lanes, operation):
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("CUDA device unavailable")
    target = Target(
        {
            "kind": "cuda",
            "arch": f"sm_{device.compute_version.replace('.', '')}",
            "backend_config": {"cuda": {"compiler": "nvcc"}},
        },
        host="c",
    )
    left = ir.Var("left", tirx.TensorType([lanes], "float32"))
    right = ir.Var("right", tirx.TensorType([lanes], "float32"))
    output = ir.Var("output", tirx.TensorType([lanes], "float32"))
    indices = [ir.const(0) if lanes == 1 else prim.Ramp(0, 1, lanes)]
    vector_type = dtype if lanes == 1 else f"{dtype}x{lanes}"
    result_type = "float32" if lanes == 1 else f"float32x{lanes}"
    left_value = prim.Cast(vector_type, ir.TensorLoad(left, indices))
    right_value = prim.Cast(vector_type, ir.TensorLoad(right, indices))
    value = (prim.Min if operation == "min" else prim.Max)(left_value, right_value)
    body = ir.RegionStmt(
        "tirx.launch_thread",
        [ir.StringImm("threadIdx.x"), ir.const(1)],
        [ir.Var("thread", "int32")],
        None,
        ir.TensorStore(output, indices, prim.Cast(result_type, value)),
    )
    function = tirx.Function([left, right, output], body).with_attr("global_symbol", "fp8_extrema")
    executable = tirx.compile(function, target=target)
    left_array = np.linspace(-3.5, 3.5, lanes, dtype="float32")
    right_array = np.linspace(2.5, -2.5, lanes, dtype="float32")
    storage_dtype = getattr(pytest.importorskip("ml_dtypes"), dtype)
    reference = np.minimum if operation == "min" else np.maximum
    expected = reference(
        left_array.astype(storage_dtype).astype("float32"),
        right_array.astype(storage_dtype).astype("float32"),
    )
    actual = runtime.empty((lanes,), "float32", device=device)
    executable(
        runtime.tensor(left_array, device=device),
        runtime.tensor(right_array, device=device),
        actual,
    )
    np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "dtype,value,lanes",
    [(dtype, value, lanes) for dtype, value in [("float32", -1.25), ("int32", -19), ("uint32", 37)] for lanes in (6, 8)]
    + [(dtype, value, lanes) for dtype, value in [("int8", -19), ("uint8", 197)] for lanes in (4, 8, 16, 32)]
    + [("float32", -1.25, 16)],
)
def test_packed_broadcast_uses_logical_lanes(dtype, value, lanes):
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("CUDA device unavailable")
    target = Target(
        {
            "kind": "cuda",
            "arch": f"sm_{device.compute_version.replace('.', '')}",
            "backend_config": {"cuda": {"compiler": "nvcc"}},
        },
        host="c",
    )
    output = ir.Var("output", tirx.TensorType([lanes], dtype))
    scalar = ir.Var("scalar", "int32" if dtype in ("int8", "uint8") else dtype)
    thread = ir.Var("thread", "int32")
    store = ir.TensorStore(output, [prim.Ramp(0, 1, lanes)], prim.Broadcast(prim.Cast(dtype, scalar), lanes))
    body = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(1)], [thread], None, store)
    function = tirx.Function([output, scalar], body, ret_type=ir.TupleType([])).with_attr("global_symbol", "packed_broadcast")
    executable = tirx.compile(function, target=target)
    actual = runtime.empty((lanes,), dtype, device=device)
    executable(actual, value)
    np.testing.assert_array_equal(actual.numpy(), np.full(lanes, value, dtype=dtype))


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["int8", "uint8"])
@pytest.mark.parametrize("operation", ["copy", "reverse", "add", "cast"])
def test_wide_byte_vectors(dtype, operation):
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("CUDA device unavailable")
    target = Target(
        {
            "kind": "cuda",
            "arch": f"sm_{device.compute_version.replace('.', '')}",
            "backend_config": {"cuda": {"compiler": "nvcc"}},
        },
        host="c",
    )
    input_dtype = ("uint8" if dtype == "int8" else "int8") if operation == "cast" else dtype
    source = ir.Var("source", tirx.TensorType([32], input_dtype))
    output = ir.Var("output", tirx.TensorType([32], dtype))
    ramp = prim.Ramp(0, 1, 32)
    value = ir.TensorLoad(source, [ramp])
    if operation == "reverse":
        value = prim.Shuffle([value], list(reversed(range(32))))
    elif operation == "add":
        value += prim.Broadcast(ir.const(3, dtype), 32)
    elif operation == "cast":
        value = prim.Cast(f"{dtype}x32", value)
    body = ir.RegionStmt(
        "tirx.launch_thread",
        [ir.StringImm("threadIdx.x"), ir.const(1)],
        [ir.Var("thread", "int32")],
        None,
        ir.TensorStore(output, [ramp], value),
    )
    function = tirx.Function([source, output], body).with_attr("global_symbol", "wide_bytes")
    executable = tirx.compile(function, target=target)
    values = (np.arange(32) * 11 - 173).astype(input_dtype)
    actual = runtime.empty((32,), dtype, device=device)
    executable(runtime.tensor(values, device=device), actual)
    expected = values.astype(dtype)
    if operation == "reverse":
        expected = expected[::-1]
    elif operation == "add":
        expected += 3
    np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "dtype,lanes",
    [(dtype, lanes) for dtype in ["float4_e2m1fn", "float8_e4m3fn", "float8_e5m2", "float16", "bfloat16"] for lanes in [2, 4, 8, 16]]
    + [("float4_e2m1fn", 32), ("float4_e2m1fn", 64)]
    + [(dtype, 32) for dtype in ["float8_e4m3fn", "float8_e5m2", "float8_e8m0fnu"]],
)
@pytest.mark.parametrize("operation", ["reverse", "broadcast"])
def test_packed_float_lanes_preserve_storage_bits(dtype, lanes, operation):
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("CUDA device unavailable")
    target = Target(
        {
            "kind": "cuda",
            "arch": f"sm_{device.compute_version.replace('.', '')}",
            "backend_config": {"cuda": {"compiler": "nvcc"}},
        },
        host="c",
    )
    bits = ir.PrimType(dtype).dtype.bits
    size = lanes * bits // 8
    source = ir.Var("source", tirx.TensorType([size], "uint8"))
    output = ir.Var("output", tirx.TensorType([size], "uint8"))
    logical_type = tirx.TensorType([lanes], dtype)
    logical_source = ir.Var("logical_source", logical_type)
    logical_output = ir.Var("logical_output", logical_type)
    declarations = [
        ir.Bind(
            logical,
            ir.Call(
                "tirx.decl_tensor",
                [ir.Call("tirx.reinterpret", [physical.data], ty=ir.PtrType(ir.PrimType(dtype)))],
                ty_args=[logical_type],
            ),
        )
        for logical, physical in ((logical_source, source), (logical_output, output))
    ]
    ramp = prim.Ramp(0, 1, lanes)
    value = ir.TensorLoad(logical_source, [ramp])
    if operation == "reverse":
        value = prim.Shuffle([value], list(reversed(range(lanes))))
    else:
        value = prim.Broadcast(ir.TensorLoad(logical_source, [ir.const(0)]), lanes)
    body = ir.RegionStmt(
        "tirx.launch_thread",
        [ir.StringImm("threadIdx.x"), ir.const(1)],
        [ir.Var("thread", "int32")],
        None,
        ir.SeqStmt([*declarations, ir.TensorStore(logical_output, [ramp], value)]),
    )
    function = tirx.Function([source, output], body).with_attr("global_symbol", "packed_float")
    executable = tirx.compile(function, target=target)
    values = np.arange(size, dtype="uint8") + 0x12
    source_array = runtime.tensor(values, device=device)
    actual = runtime.empty((size,), "uint8", device=device)
    executable(source_array, actual)
    if operation == "reverse":
        expected = values[::-1]
        if bits == 16:
            expected = values.reshape(lanes, 2)[::-1].reshape(-1)
        elif bits == 4:
            expected = ((expected & 15) << 4) | (expected >> 4)
    else:
        if bits == 16:
            expected = np.tile(values[:2], lanes)
        else:
            expected = np.full(size, int(values[0]) if bits == 8 else int(values[0] & 15) * 17, dtype="uint8")
    np.testing.assert_array_equal(actual.numpy(), expected)
