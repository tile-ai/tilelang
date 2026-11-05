"""PTO pointers preserve allocation identity, memory space and physical offsets."""

import ast
from dataclasses import dataclass

import pytest
import tvm_tirx as tvm

from testing.ascend.codegen.test_native_codegen_pto import _alloc, _decl, _mad, _ptr, _source, _tensor

I = tvm.ir
SPACES = {
    "shared.dyn": "ub",
    "shared.l1": "mat",
    "shared.l1.dyn": "mat",
    "shared.l0a": "left",
    "shared.l0a.dyn": "left",
    "shared.l0b": "right",
    "shared.l0b.dyn": "right",
    "shared.l0c": "acc",
    "shared.l0c.dyn": "acc",
}


@dataclass(frozen=True)
class PointerType:
    element_bytes: int
    space: str


@dataclass(frozen=True)
class Pointer:
    address: int
    ty: PointerType


def _address_oracle(source, bindings=None):
    """Interpret just authored pointer arithmetic, independently of C++ helpers.

    This checks generated-source address semantics; it is not a PTO device run.
    Unknown expressions fail, so the oracle cannot silently ignore a new cast.
    """
    env = dict(bindings or {})
    sizes = {"i8": 1, "u8": 1, "ui8": 1, "i16": 2, "u16": 2, "bf16": 2, "f16": 2, "i32": 4, "u32": 4, "f32": 4, "i64": 8, "int64": 8}

    def expression(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "pto":
            return sizes[node.attr]
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            args = [expression(arg) for arg in node.args]
            if name == "pto.ptr":
                return PointerType(*args)
            if name == "pto.const":
                return args[0]
            if name == "pto.castptr":
                value, ty = args
                return Pointer(value.address if isinstance(value, Pointer) else value, ty)
            if name == "pto.addptr":
                value, index = args
                return Pointer(value.address + index * value.ty.element_bytes, value.ty)
        raise AssertionError("unhandled generated pointer expression: " + ast.dump(node))

    (function,) = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)]
    for statement in function.body:
        if isinstance(statement, ast.Assign):
            assert len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name)
            env[statement.targets[0].id] = expression(statement.value)
    return env


@pytest.mark.parametrize("scope,space", list(SPACES.items()))
@pytest.mark.parametrize("index", [0, 16])
def test_non_cube_allocation_has_a_typed_address(scope, space, index, tmp_path):
    tensor = _tensor("storage", "float32", scope, (256,))
    pointer = _ptr("result", "float32", scope)
    source = _source([], [_alloc(tensor), I.Bind(pointer, tvm.tirx.address_of(tensor[index]))], tmp_path)
    values = _address_oracle(source)
    assert values["storage"] == Pointer(0, PointerType(4, space))
    assert values["result"] == Pointer(index * 4, PointerType(4, space))
    assert "pto.load(" not in source


@pytest.mark.parametrize("scope,space", [("global", "gm"), ("shared.dyn", "ub"), ("shared.l1", "mat"), ("shared.l0c", "acc")])
@pytest.mark.parametrize("index", [0, 7])
def test_declared_reinterpret_view_retains_nonzero_byte_base(scope, space, index, tmp_path):
    raw = _ptr("raw", "uint8", scope)
    dtype = I.PtrType(I.PrimType("float32"), scope)
    offset = tvm.tirx.ptr_byte_offset(raw, I.const(64, "int64"), ty=raw.ty)
    tensor = _tensor("view", "float32", scope)
    pointer = _ptr("result", "float32", scope)
    source = _source(
        [raw],
        [_decl(tensor, I.Call("tirx.reinterpret", [offset], ty=dtype)), I.Bind(pointer, tvm.tirx.address_of(tensor[index]))],
        tmp_path,
    )
    values = _address_oracle(source, {"raw": Pointer(4096, PointerType(1, space))})
    assert values["view"] == Pointer(4160, PointerType(4, space))
    assert values["result"] == Pointer(4160 + index * 4, PointerType(4, space))


@pytest.mark.parametrize("index", [0, 16])
def test_cube_integer_allocation_casts_before_element_offset(index, tmp_path):
    a = _tensor("a", "float32", "shared.l0a", (256,))
    b = _tensor("b", "float32", "shared.l0b", (256,))
    c = _tensor("c", "float32", "shared.l0c", (256,))
    pointer = _ptr("result", "float32", "shared.l0a")
    source = _source([], [_alloc(a), _alloc(b), _alloc(c), I.Bind(pointer, tvm.tirx.address_of(a[index])), _mad(a, b, c)], tmp_path)
    values = _address_oracle(source)
    assert values["a"] == 0
    assert values["result"] == Pointer(index * 4, PointerType(4, "left"))


@pytest.mark.parametrize("scope", ["global", "shared.l1"])
def test_scalar_fp4_odd_pointer_offset_remains_rejected(scope, tmp_path):
    raw = _ptr("raw", "float4_e2m1fn", scope)
    tensor = _tensor("view", "float4_e2m1fn", scope)
    pointer = _ptr("result", "float4_e2m1fn", scope)
    with pytest.raises(RuntimeError, match="offset must be aligned to packed storage"):
        _source([raw], [_decl(tensor, raw), I.Bind(pointer, tvm.tirx.address_of(tensor[1]))], tmp_path)


def test_register_tensor_has_no_invented_memory_address_space(tmp_path):
    tensor = _tensor("reg", "int32", "local.var", (1,))
    pointer = _ptr("result", "int32", "local.var")
    with pytest.raises(ValueError, match="unsupported address space: local.var"):
        _source([], [_alloc(tensor), I.Bind(pointer, tvm.tirx.address_of(tensor[0]))], tmp_path)
