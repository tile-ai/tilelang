from pathlib import Path

import numpy as np
import pytest
from tvm_tirx import ir, runtime, tirx
from tvm_tirx.backend.cuda.codegen.registry import get_codegen
from tvm_tirx.ir import prim
from tvm_tirx.target import Target

from tilelang._compat.tensor import TensorView

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


@pytest.mark.parametrize("width", [2, 4])
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("returns_value", [False, True])
@pytest.mark.parametrize("order", [None, 0, 2, 3, 4])
def test_vector_atomic_lowering_preserves_default_and_explicit_order(width, dtype, returns_value, order):
    destination = ir.Var("destination", ir.PtrType(dtype, scope="global"))
    source = ir.Var("source", ir.PtrType(dtype, scope="local"))
    operands = [destination, source]
    if order is not None:
        operands.append(ir.const(order))
    suffix = "ret_elem_op" if returns_value else "elem_op"
    call, tags = get_codegen(f"tl.atomic_addx{width}_{suffix}")(operands)
    assert call.op.name == "tirx.cuda.func_call"
    assert int(call.args[-2]) == (0 if order is None else order)
    assert call.args[1].same_as(destination)
    assert call.args[2].same_as(source)
    assert call.ty == ir.PrimType(f"{dtype}x{width}" if returns_value else "void")
    assert not tags


@pytest.mark.parametrize("reverse", [False, True])
def test_packed_mul_add_fuses_without_scalarizing(reverse):
    left = ir.Var("left", "float32x2")
    right = ir.Var("right", "float32x2")
    addend = ir.Var("addend", "float32x2")
    product = ir.Call("tl.mul2", [left, right], ty="float32x2")
    operands = [addend, product] if reverse else [product, addend]
    call, tags = get_codegen("tl.add2")(operands)
    assert call.op.name == "tirx.cuda.func_call"
    assert str(call.args[0].value) == "tl::fma2"
    assert call.ty == "float32x2"
    assert all(actual.same_as(expected) for actual, expected in zip(call.args[1:4], [left, right, addend]))
    assert not tags
    assert get_codegen("tirx.tl.add2") is None


@pytest.mark.parametrize(
    "name,dtype,arity",
    [
        ("cluster_arrive", None, 0),
        ("cluster_arrive_relaxed", None, 0),
        ("cluster_wait", None, 0),
        ("cluster_sync", None, 0),
        ("block_rank_in_cluster", "int32", 0),
        ("clc_try_cancel", None, 2),
        ("clc_try_cancel_multicast", None, 2),
        ("clc_is_canceled", "int32", 1),
        ("clc_get_first_ctaid_x", "uint32", 1),
        ("clc_get_first_ctaid_y", "uint32", 1),
        ("clc_get_first_ctaid_z", "uint32", 1),
    ],
)
def test_native_cluster_helpers_preserve_result_types(name, dtype, arity):
    operands = [ir.Var(f"pointer_{index}", ir.PtrType("void", scope="shared")) for index in range(arity)]
    call, tags = get_codegen(f"tl.{name}")(operands)
    assert call.op.name == "tirx.cuda.func_call"
    assert call.args[0].value == f"tl::{name}"
    assert call.ty == ir.PrimType(dtype or "void")
    assert "tl_templates/cuda/cluster.h" in call.args[-1].value
    assert not tags
    with pytest.raises(ValueError, match="operands"):
        get_codegen(f"tl.{name}")(operands + [ir.const(0)])


def test_native_remote_barrier_arrival_has_an_active_predicate():
    barrier = TensorView.create("barrier", [1], "uint64", scope="shared")
    barrier_value = ir.TensorLoad(barrier.tensor, [0])
    call, tags = get_codegen("tl.ptx_arrive_cluster_barrier")([barrier_value, ir.const(1)])
    assert call.args[0].value == "tl::mbarrier_arrive"
    assert call.args[1].same_as(barrier_value)
    assert int(call.args[2]) == 1
    assert int(call.args[3]) == 1
    assert call.args[3].ty == ir.PrimType("uint32")
    assert call.ty == ir.PrimType("void")
    assert not tags


def test_deferred_async_copy_arrival_preserves_the_barrier_lvalue():
    barrier = TensorView.create("barrier", [2], "uint64", scope="shared")
    value = barrier.load(1)
    call, tags = get_codegen("tl.ptx_cp_async_barrier_noinc")([value], ir.DictAttrs({}))
    assert call.args[0].value == "tl::mbarrier_cp_async_arrive_noinc"
    assert call.args[1].same_as(value)
    assert "tl_templates/cuda/barrier.h" in call.args[-1].value
    assert not tags
    assert call.ty == ir.PrimType("void")


@pytest.mark.parametrize("dtype", ["float32", "float32x4", "int32x2"])
def test_packed_codegen_rejects_incompatible_types(dtype):
    value = ir.Var("value", dtype)
    with pytest.raises(TypeError, match="packed floating-point"):
        get_codegen("tl.add2")([value, value])


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("name", ["add2", "sub2", "mul2", "fma2", "max2", "min2", "abs2", "max2_nan", "min2_nan"])
def test_native_packed_cuda_helper_abi(dtype, name):
    if dtype == "float32" and name.endswith("_nan"):
        pytest.skip("NaN-propagating helpers have fp16/bf16 signatures")
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
    sources = [TensorView.create(f"source_{index}", [2], "float32") for index in range(3)]
    output = TensorView.create("output", [2], "float32")
    indices = [prim.Ramp(0, 1, 2)]
    arguments = [prim.Cast(f"{dtype}x2", source.load(indices)) for source in sources]
    arity = 1 if name == "abs2" else 3 if name == "fma2" else 2
    value = ir.Call(f"tl.{name}", arguments[:arity], ty=f"{dtype}x2")
    body = output.store(indices, prim.Cast("float32x2", value))
    thread = ir.Var("thread", "int32")
    body = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(1)], [thread], None, body)
    function = tirx.Function([source.tensor for source in sources] + [output.tensor], body, ret_type=ir.TupleType([])).with_attr(
        "global_symbol", "packed_helper_test"
    )
    executable = tirx.compile(function, target=target)
    left = np.array([-2.0, 3.0], dtype="float32")
    right = np.array([4.0, -1.0], dtype="float32")
    addend = np.array([1.5, 2.5], dtype="float32")
    expected = {
        "add2": left + right,
        "sub2": left - right,
        "mul2": left * right,
        "fma2": left * right + addend,
        "max2": np.maximum(left, right),
        "min2": np.minimum(left, right),
        "abs2": np.abs(left),
        "max2_nan": np.maximum(left, right),
        "min2_nan": np.minimum(left, right),
    }[name]
    if name.endswith("_nan"):
        left[0] = np.nan
        expected[0] = np.nan
    actual = runtime.empty((2,), "float32", device=device)
    executable(*(runtime.tensor(value, device=device) for value in (left, right, addend)), actual)
    np.testing.assert_array_equal(actual.numpy(), expected)
