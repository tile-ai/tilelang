"""Metal intrinsics retain named storage across native tensor layout lowering."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

import tilelang  # noqa: F401


_OPERATIONS = (
    "tl.cooperative_tensor_fill",
    "tl.cooperative_tensor_load",
    "tl.cooperative_tensor_store",
    "tl.cooperative_tensor_multiply_accumulate",
)


def _fragment(kind, scope="metal.cooperative_tensor"):
    if kind == "pointer_var":
        return ir.Var("fragment", ir.PtrType("float32", scope=scope))
    tensor = ir.Var("fragment", tirx.TensorType((512,), "float32", scope=scope))
    if kind == "tensor_var":
        return tensor
    return ir.Call("tirx.tensor_data_ptr", [tensor])


def _intrinsic(name, fragment):
    index = ir.IntImm("int32", 0)

    def imm(value):
        return ir.IntImm("int32", value)

    if name == "tl.cooperative_tensor_fill":
        operands = [fragment, index, ir.FloatImm("float32", 0), imm(16), imm(32)]
    elif name in ("tl.cooperative_tensor_load", "tl.cooperative_tensor_store"):
        external = ir.Var("external", ir.PtrType("float32"))
        operands = [fragment, index, external, imm(32), imm(16), imm(32), index, imm(16), imm(32), imm(16), index]
    else:
        operands = [fragment, index, fragment, index, fragment, index, fragment, index, imm(16), imm(32), imm(16), index, index]
    return ir.Call(name, operands, ty=ir.PrimType("int32"))


@pytest.mark.parametrize("operation", _OPERATIONS)
@pytest.mark.parametrize("kind", ["tensor_var", "pointer_var", "tensor_data_ptr"])
def test_named_fragment_survives_call_validation_and_serialization(operation, kind):
    call = _intrinsic(operation, _fragment(kind))
    call.validate()
    restored = ir.load_json(ir.save_json(call))
    restored.validate()
    assert tvm_ffi.structural_equal(call, restored, map_free_vars=True)


@pytest.mark.parametrize("operation", _OPERATIONS)
@pytest.mark.parametrize("kind", ["tensor_var", "pointer_var", "tensor_data_ptr"])
def test_fragment_rejects_other_storage_scopes(operation, kind):
    with pytest.raises(TypeError, match="metal.cooperative_tensor"):
        _intrinsic(operation, _fragment(kind, "global")).validate()


@pytest.mark.parametrize("operation", _OPERATIONS)
@pytest.mark.parametrize("kind", ["offset", "external_call", "nonvariable_tensor"])
def test_fragment_rejects_arbitrary_pointer_expressions(operation, kind):
    fragment = _fragment("tensor_data_ptr")
    if kind == "offset":
        operand = ir.Call("tirx.ptr_byte_offset", [fragment, ir.IntImm("int64", 4)])
    elif kind == "external_call":
        operand = tirx.call_extern("foreign_fragment", ty=fragment.ty)
    else:
        allocation = ir.Call("tirx.alloc_tensor", [], ty_args=[fragment.args[0].ty])
        # Explicit result type allows construction of this provisional bad
        # operand; the fragment validator must still reject its lost identity.
        operand = ir.Call("tirx.tensor_data_ptr", [allocation], ty=fragment.ty)
    with pytest.raises(TypeError, match="Metal fragment operand"):
        _intrinsic(operation, operand).validate()
