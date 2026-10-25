import numpy as np
import pytest
from tvm_tirx import ir, runtime, tirx
from tvm_tirx.ir import prim
from tvm_tirx.target import Target

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def lower(name, arguments, dtype, backend):
    expression = ir.Call(f"prim.{name}", arguments, ty=dtype)
    return ir.Op.get_attr_map(f"{backend}.FLowerIntrinsic")[expression.op](expression)


@pytest.mark.parametrize("bits", [8, 16, 32, 64])
@pytest.mark.parametrize("signed", [False, True])
def test_cuda_popcount_preserves_input_width(bits, signed):
    dtype = f"{'int' if signed else 'uint'}{bits}"
    argument = ir.Var("argument", dtype)
    result = lower("popcount", [argument], dtype, "cuda")
    calls = []
    stmt_functor.post_order_visit(result, lambda node: calls.append(node) if isinstance(node, ir.Call) else None)
    external = next(call for call in calls if call.op.name == "tirx.call_pure_extern")
    assert external.args[0].value == ("__popcll" if bits == 64 else "__popc")
    assert external.args[1].ty == f"uint{max(bits, 32)}"
    if bits < 32:
        assert external.args[1].value.ty == f"uint{bits}"
    assert result.ty == dtype


@pytest.mark.parametrize(
    "name,dtype,symbol",
    [
        ("exp", "float32", "__expf"),
        ("exp", "float64", "exp"),
        ("exp", "float16", "hexp"),
        ("round", "float32", "nearbyintf"),
        ("round", "float16", "hrint"),
        ("fabs", "bfloat16", "__habs"),
        ("sqrt", "float32", "sqrtf"),
        ("popcount", "uint32", "__popc"),
        ("popcount", "uint64", "__popcll"),
        ("clz", "int64", "__clzll"),
    ],
)
def test_hip_native_math_rules(name, dtype, symbol):
    argument = ir.Var("argument", dtype)
    result = lower(name, [argument], dtype, "hip")
    assert result.op.name == "tirx.call_pure_extern"
    assert result.args[0].value == symbol
    assert result.args[1].same_as(argument)
    result.validate()


def test_hip_vector_math_scalarizes_without_legacy_codegen():
    argument = ir.Var("argument", "float32x4")
    result = lower("sin", [argument], "float32x4", "hip")
    assert isinstance(result, prim.Shuffle)
    assert result.ty == "float32x4"
    for lane, scalar in enumerate(result.vectors):
        assert scalar.args[0].value == "__sinf"
        assert scalar.ty == "float32"
        assert int(scalar.args[1].indices[0]) == lane


@pytest.mark.parametrize("suffix", ["", "_up", "_down"])
def test_hip_shuffle_keeps_typed_value_and_width(suffix):
    expression = ir.Call(f"tirx.gpu_warp_shuffle{suffix}", [ir.const(0xFFFFFFFF, "uint32"), ir.Var("value", "float32"), 1, 32, 64])
    result = ir.Op.get_attr_map("hip.FLowerIntrinsic")[expression.op](expression)
    assert result.op.name == f"tl.hip.__shfl{suffix}_sync"
    assert result.ty == "float32"
    assert len(result.args) == 4
    assert ir.Op.get_attr_map("hip.need_warp_shuffle")[result.op]
    result.validate()


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"])
def test_cuda_popcount_numerical(dtype):
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("A CUDA device is required")
    bits = np.dtype(dtype).itemsize * 8
    mask = (1 << bits) - 1
    raw = [0, 1, 3, 7, mask, mask - 1, 1 << (bits - 1)]
    values = np.array(raw, dtype=f"uint{bits}").view(dtype)
    source = TensorView.create("source", values.shape, dtype)
    output = TensorView.create("output", values.shape, "int32")
    index = ir.Var("index", "int32")
    value = ir.Call("prim.popcount", [source.load(index)], ty=dtype)
    body = ir.RegionStmt(
        "tirx.launch_thread",
        [ir.StringImm("threadIdx.x"), ir.const(len(values))],
        [index],
        None,
        output.store(index, prim.Cast("int32", value)),
    )
    function = tirx.Function([source.tensor, output.tensor], body, ret_type=ir.TupleType([])).with_attr("global_symbol", "native_popcount")
    target = Target(
        {"kind": "cuda", "arch": f"sm_{device.compute_version.replace('.', '')}", "backend_config": {"cuda": {"compiler": "nvcc"}}},
        host="c",
    )
    executable = tirx.compile(function, target=target)
    actual = runtime.empty(values.shape, "int32", device=device)
    executable(runtime.tensor(values, device=device), actual)
    np.testing.assert_array_equal(actual.numpy(), [value.bit_count() for value in raw])


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cuda_isfinite_numerical(dtype):
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("A CUDA device is required")
    values = np.array([0, -0.0, 1, -3, np.inf, -np.inf, np.nan], dtype=dtype)
    source = TensorView.create("source", values.shape, dtype)
    output = TensorView.create("output", values.shape, "int32")
    index = ir.Var("index", "int32")
    value = ir.Call("prim.isfinite", [source.load(index)], ty="bool")
    body = ir.RegionStmt(
        "tirx.launch_thread",
        [ir.StringImm("threadIdx.x"), ir.const(len(values))],
        [index],
        None,
        output.store(index, prim.Cast("int32", value)),
    )
    function = tirx.Function([source.tensor, output.tensor], body, ret_type=ir.TupleType([])).with_attr("global_symbol", "native_isfinite")
    target = Target(
        {"kind": "cuda", "arch": f"sm_{device.compute_version.replace('.', '')}", "backend_config": {"cuda": {"compiler": "nvcc"}}},
        host="c",
    )
    executable = tirx.compile(function, target=target)
    actual = runtime.empty(values.shape, "int32", device=device)
    executable(runtime.tensor(values, device=device), actual)
    np.testing.assert_array_equal(actual.numpy(), np.isfinite(values))
