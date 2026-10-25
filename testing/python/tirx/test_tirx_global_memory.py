from pathlib import Path

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, tirx
from tvm_tirx.ir import prim
from tvm_tirx.ir.transform import PassContext
from tvm_tirx.target import Target

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def cuda_target():
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("CUDA device unavailable")
    root = Path(__file__).resolve().parents[3]
    return Target(
        {
            "kind": "cuda",
            "arch": f"sm_{device.compute_version.replace('.', '')}",
            "backend_config": {
                "cuda": {"compiler": "nvcc", "nvcc": [f"-I{root / 'src'}", f"-I{root / '3rdparty' / 'cutlass' / 'include'}"]}
            },
        },
        host="c",
    )


def lower(body, parameters, target, unguarded=False, predicated=True):
    function = tirx.Function(parameters, body, ret_type=ir.TupleType([]))
    module = ir.IRModule({"main": function.with_attr("target", target).with_attr("global_symbol", "global_access_test")})
    with PassContext(config={"tl.enable_lower_ldgstg": unguarded, "tl.enable_lower_ldgstg_predicated": predicated}):
        result = tvm_ffi.get_global_func("tl.cuda.transform.LowerLDGSTG")()(module)
    assert tirx.analysis.check_well_formed(result)
    ir.assert_structural_equal(result, ir.load_json(ir.save_json(result)))
    return result


def collect(root, kind=ir.Call):
    result = []
    stmt_functor.post_order_visit(root, lambda node: result.append(node) if isinstance(node, kind) else None)
    return result


def named_calls(root, name):
    return [call for call in collect(root) if isinstance(call.op, ir.Op) and call.op.name == name]


@pytest.mark.parametrize("lanes", [1, 2, 4, 8])
@pytest.mark.parametrize("enclosing", [False, True])
@pytest.mark.parametrize("unguarded", [False, True])
def test_predicated_global_access_preserves_nested_guards(lanes, enclosing, unguarded, tilelang_builders):
    target = cuda_target()
    source = TensorView.create("source", [128], "float32")
    output = TensorView.create("output", [128], "float32")
    outer = ir.Var("outer", "int32")
    inner = ir.Var("inner", "int32")
    thread = ir.Var("thread", "int32")
    index = thread if lanes == 1 else prim.Ramp(thread * lanes, 1, lanes)
    zero = ir.const(0.0, "float32") if lanes == 1 else prim.Broadcast(ir.const(0.0, "float32"), lanes)
    load_index = index if not enclosing else index + (1 - outer) * 4096
    value = prim.if_then_else(inner > 0, source.load(load_index), zero)
    body = output.store(index, value)
    if enclosing:
        body = ir.If(outer > 0, body, None)
    body = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(128 // lanes)], [thread], None, body)
    module = lower(body, [source.tensor, output.tensor, outer, inner], target, unguarded)
    assert len(named_calls(module, f"tl.ldg{32 * lanes}")) == 1
    if enclosing:
        assert len(named_calls(module, f"tl.stg{32 * lanes}")) == 1
        assert not collect(module, ir.If)
    executable = tirx.compile(module, target=target)
    values = np.arange(128, dtype="float32") * 0.25
    for outer_value, inner_value in ((0, 0), (0, 1), (1, 0), (1, 1)):
        actual = runtime.tensor(np.full(128, -7, dtype="float32"), device=runtime.cuda(0))
        executable(runtime.tensor(values, device=runtime.cuda(0)), actual, outer_value, inner_value)
        expected = (
            np.full(128, -7, dtype="float32")
            if enclosing and not outer_value
            else (values if inner_value else np.zeros(128, dtype="float32"))
        )
        np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize(
    "scope,dtype,reason", [("shared", "float32", "shared"), ("local", "float32", "local"), ("global", "float16", "unsupported width")]
)
def test_predication_keeps_guards_for_non_speculatable_loads(scope, dtype, reason):
    source = TensorView.create("source", [1], dtype, scope=scope)
    output = TensorView.create("output", [1], "float32")
    condition = ir.Var("condition", "bool")
    body = ir.If(condition, output.store(0, prim.Cast("float32", source.load(0))), None)
    parameters = [output.tensor, condition]
    if scope == "global":
        parameters.append(source.tensor)
    else:
        body = [source.allocate(), body]
    module = lower(body, parameters, Target({"kind": "cuda", "arch": "sm_90"}))
    assert len(collect(module, ir.If)) == 1, reason
    assert not named_calls(module, "tl.stg32")


def test_predicate_with_atomic_side_effect_is_evaluated_once(tilelang_builders):
    target = cuda_target()
    output = TensorView.create("output", [1], "int32")
    counter = TensorView.create("counter", [1], "int32")
    condition = ir.Call("tl.atomic_add_ret_elem_op", [counter.pointer_at(0), 1, 0], ty="int32") == 0
    thread = ir.Var("thread", "int32")
    body = ir.RegionStmt(
        "tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(1)], [thread], None, ir.If(condition, output.store(0, 37), None)
    )
    module = lower(body, [output.tensor, counter.tensor], target)
    assert len(named_calls(module, "tl.atomic_add_ret_elem_op")) == 1
    executable = tirx.compile(module, target=target)
    actual = runtime.tensor(np.zeros(1, dtype="int32"), device=runtime.cuda(0))
    calls = runtime.tensor(np.zeros(1, dtype="int32"), device=runtime.cuda(0))
    executable(actual, calls)
    np.testing.assert_array_equal(actual.numpy(), [37])
    np.testing.assert_array_equal(calls.numpy(), [1])


def test_unguarded_memory_pass_preserves_address_operands():
    storage = TensorView.create("storage", [1], "int32")
    call = ir.Evaluate(ir.Call("tl.atomic_add_elem_op", [storage.pointer_at(0), 1, 0]))
    module = lower(call, [storage.tensor], Target({"kind": "cuda", "arch": "sm_90"}), True)
    assert not named_calls(module, "tl.ldg32")
    assert len(named_calls(module, "tl.atomic_add_elem_op")) == 1


def test_predication_does_not_hoist_trapping_integer_arithmetic():
    denominator = ir.Var("denominator", "int32")
    output = TensorView.create("output", [1], "int32")
    body = ir.If(denominator != 0, output.store(0, 9 // denominator), None)
    module = lower(body, [output.tensor, denominator], Target({"kind": "cuda", "arch": "sm_90"}))
    assert len(collect(module, ir.If)) == 1
    assert not named_calls(module, "tl.stg32")
