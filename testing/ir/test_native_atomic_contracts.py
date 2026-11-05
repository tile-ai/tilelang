"""Native atomic types are derived from typed addresses and remain effectful."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx


def pointer(dtype="float32", name="destination"):
    return ir.Var(name, ir.PtrType(ir.PrimType(dtype), "global"))


@pytest.mark.parametrize("name", ["add", "max", "min"])
@pytest.mark.parametrize("dtype", ["int32", "float32", "float64"])
@pytest.mark.parametrize("return_prev", [False, True])
def test_scalar_result_comes_from_destination(name, dtype, return_prev):
    dst = pointer(dtype)
    value = ir.PrimType(dtype)(2)
    spelling = f"tl.atomic_{name}{'_ret' if return_prev else ''}_elem_op"
    result = ir.PrimType(dtype if return_prev else "void")
    call = ir.Call(spelling, [dst, value])
    call.validate()
    assert tvm_ffi.structural_equal(call.ty, result)
    module = ir.IRModule({"main": tirx.Function([dst], [ir.Evaluate(call)], ir.NoneType())})
    lowered = tirx.transform.RemoveNoOp()(module)["main"]
    assert len(lowered.body) == 1
    assert lowered.body[0].value.op.name == spelling


@pytest.mark.parametrize("width", [2, 4])
@pytest.mark.parametrize("return_prev", [False, True])
def test_vector_atomic_pointer_pair_and_return_lanes(width, return_prev):
    dst, src = pointer(), pointer(name="source")
    name = f"tl.atomic_addx{width}{'_ret' if return_prev else ''}_elem_op"
    call = ir.Call(name, [dst, src])
    call.validate()
    assert tvm_ffi.structural_equal(call.ty, ir.PrimType(f"float32x{width}" if return_prev else "void"))


@pytest.mark.parametrize("order", [0, 1, 2, 5])
def test_load_result_and_valid_orders(order):
    call = ir.Call("tl.atomic_load_elem_op", [pointer("int64"), ir.IntImm("int32", order)])
    call.validate()
    assert tvm_ffi.structural_equal(call.ty, ir.PrimType("int64"))


@pytest.mark.parametrize("order", [0, 3, 5])
def test_store_result_and_valid_orders(order):
    call = ir.Call("tl.atomic_store_elem_op", [pointer("int64"), ir.IntImm("int32", 1), ir.IntImm("int32", order)])
    call.validate()
    assert tvm_ffi.structural_equal(call.ty, ir.PrimType("void"))


def test_annotation_tuple_keeps_dynamic_expression_and_result_type():
    dst = pointer("int64")
    value = ir.Var("contribution", "int32")
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({"debug_value": value})
    call = ir.Call(
        "tl.atomic_add_ret_elem_op", [dst, value, payload], attrs={"operand_count": 2, "schema_version": 1, "annotation_schema": schema}
    )
    call.validate()
    assert tvm_ffi.structural_equal(call.ty, ir.PrimType("int64"))
    with pytest.raises((ValueError, IndexError)):
        ir.Call("tl.atomic_add_ret_elem_op", [dst, value, ir.Tuple([])], attrs=call.attrs).validate()


@pytest.mark.parametrize(
    "case",
    [
        "wrong_result",
        "nonreturn_value",
        "scalar_address",
        "void_pointer",
        "vector_value",
        "vector_source",
        "order_range",
        "order_dynamic",
        "load_release",
        "store_acquire",
        "float_or",
        "few",
        "many",
    ],
)
def test_invalid_contract_is_rejected(case):
    name = "tl.atomic_add_ret_elem_op"
    operands = [pointer(), ir.FloatImm("float32", 1)]
    ty_args = []
    if case == "wrong_result":
        ty_args = [ir.PrimType("int64")]
    elif case == "nonreturn_value":
        name, ty_args = "tl.atomic_add_elem_op", [ir.PrimType("float32")]
    elif case == "scalar_address":
        operands[0] = ir.IntImm("int32", 0)
    elif case == "void_pointer":
        operands[0] = pointer("void")
    elif case == "vector_value":
        operands[1] = ir.prim.Broadcast(operands[1], 2)
    elif case == "vector_source":
        name, operands[1] = "tl.atomic_addx2_ret_elem_op", pointer("int32", "source")
    elif case == "order_range":
        operands.append(ir.IntImm("int32", 6))
    elif case == "order_dynamic":
        operands.append(ir.Var("order", "int32"))
    elif case == "load_release":
        name, operands = "tl.atomic_load_elem_op", [pointer(), ir.IntImm("int32", 3)]
    elif case == "store_acquire":
        name = "tl.atomic_store_elem_op"
        operands.append(ir.IntImm("int32", 2))
    elif case == "float_or":
        name = "tl.atomic_or_elem_op"
    elif case == "few":
        operands.pop()
    else:
        operands.extend([ir.IntImm("int32", 0), ir.IntImm("int32", 0)])
    with pytest.raises((TypeError, ValueError)):
        ir.Call(name, operands, ty_args=ty_args).validate()
