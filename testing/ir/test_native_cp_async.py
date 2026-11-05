"""cp.async is an effectful void expression with an explicit element contract."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx


def args():
    destination = ir.Var("destination", ir.PtrType(ir.PrimType("float32"), "shared"))
    source = ir.Var("source", ir.PtrType(ir.PrimType("float32"), "global"))
    return [destination, source, ir.IntImm("int32", 2)]


@pytest.mark.parametrize("predicated", [False, True])
def test_inferred_void_result_and_element_count(predicated):
    operands = args()
    if predicated:
        operands.append(ir.Var("predicate", "bool"))
    call = ir.Call("tl.ptx_cp_async", operands)
    call.validate()
    assert tvm_ffi.structural_equal(call.ty, ir.PrimType("void"))
    assert int(call.args[2]) == 2


def test_annotation_payload_is_validated_and_keeps_void_result():
    operands = args() + [ir.Var("predicate", "bool")]
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({"pipeline": 2, "mask": operands[-1]})
    attributes = {"schema_version": 1, "operand_count": len(operands), "annotation_schema": schema}
    call = ir.Call("tl.ptx_cp_async", [*operands, payload], attrs=attributes)
    call.validate()
    assert tvm_ffi.structural_equal(call.ty, ir.PrimType("void"))
    invalid = ir.Call("tl.ptx_cp_async", [*operands, ir.Tuple([])], attrs=attributes)
    with pytest.raises((ValueError, IndexError)):
        invalid.validate()


def test_async_copy_side_effect_survives_remove_no_op():
    operands = args()
    call = ir.Call("tl.ptx_cp_async", operands)
    function = tirx.Function(operands[:2], [ir.Evaluate(call)], ir.NoneType())
    lowered = tirx.transform.RemoveNoOp()(ir.IRModule({"copy": function}))["copy"]
    assert len(lowered.body) == 1
    statement = lowered.body[0]
    assert isinstance(statement, ir.Evaluate)
    assert statement.value.op.name == "tl.ptx_cp_async"
    assert tvm_ffi.structural_equal(statement.value, call)


@pytest.mark.parametrize("case", ["few", "many", "scalar_pointer", "float_count", "negative_count", "integer_predicate"])
def test_invalid_native_operand_contract(case):
    operands = args()
    if case == "few":
        operands.pop()
    elif case == "many":
        operands.extend([ir.IntImm("bool", True), ir.IntImm("int32", 7)])
    elif case == "scalar_pointer":
        operands[0] = ir.IntImm("int32", 0)
    elif case == "float_count":
        operands[2] = ir.FloatImm("float32", 2.0)
    elif case == "negative_count":
        operands[2] = ir.IntImm("int32", -1)
    else:
        operands.append(ir.IntImm("int32", 1))
    with pytest.raises((TypeError, ValueError)):
        ir.Call("tl.ptx_cp_async", operands).validate()
