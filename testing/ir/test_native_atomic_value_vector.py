"""Packed atomics consume typed vector values without replaying scalar Bind values."""

import pytest
import tvm_ffi
import tilelang
from tilelang.testing.ir import collect_nodes
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target


def tensor(name, dtype):
    return ir.Var(name, tirx.TensorType((32,), dtype, layout="default", data_alignment=64))


def transform(dtype, bound, *, opaque=False, order=0):
    width = 4 if dtype == "float32" else 2
    source, destination = tensor("A", "float32"), tensor("B", dtype)
    lane = ir.Var("lane", "int32")
    value = ir.Var("contribution", "float32")
    producer = tirx.call_extern("next_value", ty="float32") if opaque else source[lane]
    body = [ir.Bind(value, producer)] if bound else []
    pointer = ir.Call("tl.access_ptr", [destination[lane], ir.const(1), ir.const(3)])
    body.append(ir.Evaluate(ir.Call("tl.atomic_add_elem_op", [pointer, value if bound else producer, ir.const(order)])))
    loop = ir.For(lane, 0, width, 1, ir.ForKind.VECTORIZED, body)
    function = tirx.Function([source, destination], loop).with_attrs(
        {"target": Target({"kind": "cuda", "arch": "sm_90"}), "tirx.noalias": True}
    )
    before = ir.save_json(function)
    result = tilelang.transform.VectorizeLoop()(ir.IRModule({"main": function}))["main"]
    assert ir.save_json(function) == before
    assert tirx.analysis.check_well_formed(result)
    assert not tirx.analysis.undefined_vars(result)
    for call in collect_nodes(result, ir.Call):
        call.validate()
    return width, result


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("order", [0, 2, 3, 4])
def test_bound_contribution_keeps_vector_definition_and_order(dtype, order):
    width, result = transform(dtype, True, order=order)
    assert not collect_nodes(result, ir.For)
    (binding,) = collect_nodes(result, ir.Bind)
    assert binding.var.ty == ir.PrimType(f"float32x{width}")
    atoms = [call for call in collect_nodes(result, ir.Call) if call.op.name.startswith("tl.atomic_add")]
    (atomic,) = atoms
    assert atomic.op.name == f"tl.atomic_addx{width}_elem_op"
    assert atomic.args[1].ty == ir.PrimType(f"{dtype}x{width}")
    assert int(atomic.args[2].value) == order
    used = []
    tvm_ffi.structural_walk(atomic.args[1], (ir.Var, used.append))
    assert any(var.same_as(binding.var) for var in used)
    assert not collect_nodes(atomic.args[1], ir.TensorLoad)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_mixed_load_converts_numeric_lanes(dtype):
    width, result = transform(dtype, False)
    assert not collect_nodes(result, ir.For)
    (atomic,) = [call for call in collect_nodes(result, ir.Call) if call.op.name.startswith("tl.atomic_add")]
    assert atomic.op.name == "tl.atomic_addx2_elem_op"
    assert isinstance(atomic.args[1], ir.prim.Cast)
    assert atomic.args[1].ty == ir.PrimType(dtype + "x2")
    assert atomic.args[1].value.ty == ir.PrimType("float32x2")


def test_opaque_bound_contribution_retains_each_iteration():
    width, result = transform("float32", True, opaque=True)
    (loop,) = collect_nodes(result, ir.For)
    assert loop.kind == ir.ForKind.DEFAULT
    assert int(loop.end - loop.begin) == width
    calls = collect_nodes(result, ir.Call)
    assert sum(call.op.name == "tirx.call_extern" for call in calls) == 1
    assert sum(call.op.name == "tl.atomic_add_elem_op" for call in calls) == 1
    assert not any(call.op.name.startswith("tl.atomic_addx") for call in calls)


@pytest.mark.parametrize("value_dtype", ["float32", "float32x2", "float64x4", "int32x4"])
def test_packed_value_rejects_wrong_dtype_or_width(value_dtype):
    destination = ir.Var("B", ir.PtrType(ir.PrimType("float32")))
    value = ir.Var("value", ir.PrimType(value_dtype))
    with pytest.raises(TypeError, match="destination element type and lane count"):
        ir.Call("tl.atomic_addx4_elem_op", [destination, value]).validate()


def test_return_previous_does_not_silently_acquire_new_value_abi():
    destination = ir.Var("B", ir.PtrType(ir.PrimType("float16")))
    value = ir.Var("value", ir.PrimType("float16x4"))
    with pytest.raises(TypeError, match="typed source pointer"):
        ir.Call("tl.atomic_addx4_ret_elem_op", [destination, value]).validate()
