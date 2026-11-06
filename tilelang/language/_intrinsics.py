"""Typed DSL calls, ABI pointer access, and synchronization construction."""

from __future__ import annotations

import tvm_ffi
from tvm_tirx import ir
from tvm_tirx import tirx as _tirx
from tvm_tirx.ir import prim as _prim

from ._construction import as_expr, current_location, dtype_of
from ._annotation_codec import encode_annotations, is_static_annotation


def call_intrin(dtype, func_name, *args, annotations=None, span=None):
    """Construct a typed call; tile op effects and validation are registered natively."""
    if isinstance(func_name, ir.Op):
        name = func_name.name
    else:
        name = func_name
    operands = [as_expr(value) for value in args]
    if name.startswith("tl.tileop."):
        schema, payload = encode_annotations(annotations)
        attrs = {"schema_version": 1, "operand_count": len(operands), "annotation_schema": schema}
        return ir.Call(name, [*operands, payload], attrs=attrs, loc=span or current_location())
    if annotations and is_static_annotation(annotations):
        attrs = dict(annotations)
        reserved = {"schema_version", "operand_count", "annotation_schema", "tl.source_annotation_keys"}
        if reserved.intersection(attrs):
            raise ValueError(f"Reserved intrinsic annotation key: {sorted(reserved.intersection(attrs))}")
        # Preserve the source language's annotation-insensitive shape comparison
        # separately from native attributes, without an extra Call operand.
        attrs["tl.source_annotation_keys"] = sorted(attrs)
    elif annotations:
        schema, payload = encode_annotations(annotations)
        attrs = {"schema_version": 1, "operand_count": len(operands), "annotation_schema": schema}
        operands.append(payload)
    else:
        attrs = None
    ty = ir.PtrType(ir.PrimType("void")) if str(dtype) == "handle" else ir.PrimType(dtype)
    op = func_name if isinstance(func_name, ir.Op) else ir.Op.get(name)
    # A return annotation is a type argument only when the registered native
    # signature accepts one. External/LLVM calls require an explicit Call.ty
    # and zero type arguments; fixed-result intrinsics keep their own result.
    type_args = [ty] if op.ty_args_info or op.var_ty_args_info is not None else []
    value = ir.Call(op, operands, attrs=attrs, ty_args=type_args, loc=span or current_location())
    if isinstance(value.ty, ir.MissingType):
        # External calls and other operators without a registered result rule
        # need the public dtype as Call.ty, not merely as a type argument.
        # Retain inferred native types when the operator owns that contract.
        value = ir.Call(op, operands, attrs=attrs, ty_args=type_args, ty=ty, loc=span or current_location())
    return value


def tensor_descriptor_ptr(tensor, *, loc=None):
    """Borrow a matched input's original descriptor through TileLang's binder.

    This frontend operation is resolved before native layout/flattening passes.
    Ordinary pointer parameters and ``decl_tensor`` views do not need it.
    """
    return ir.Call("tl.tensor_descriptor_ptr", [as_expr(tensor)], loc=loc or current_location())


def tvm_struct_get(arr, index, field, dtype):
    """Read a runtime ABI field using the existing public field numbering.

    The DLTensor and TVMFFIAny field indices are unchanged. A source handle
    matched to a tensor still denotes its original input descriptor; the
    TileLang binds that handle at function entry and resolves it in its own ABI.
    """
    from .dtypes import get_tvm_dtype

    ty = ir.PtrType(ir.PrimType("void")) if str(dtype) == "handle" else ir.PrimType(get_tvm_dtype(dtype))
    return _tirx.op.abi_field_get(as_expr(arr), as_expr(index), as_expr(field), ty=ty, loc=current_location())


def tvm_struct_set(arr, index, field, value):
    """Write a runtime ABI field and retain the native update-state effect."""
    return _tirx.op.abi_field_set(as_expr(arr), as_expr(index), as_expr(field), as_expr(value), loc=current_location())


def tvm_storage_sync(storage_scope, *barrier_values):
    """Keep the public barrier spelling while constructing its native contract.

    Whole-block barriers belong to TIRx. Named barriers retain their ID and
    participant count in TileLang's operation through synchronization planning.
    CUDA consumes them as named barriers; the final WebGPU codegen boundary
    preserves its established whole-workgroup fallback.
    """
    if barrier_values:
        return ir.Call(
            "tl.storage_sync",
            [as_expr(storage_scope), *[as_expr(value) for value in barrier_values]],
            loc=current_location(),
        )
    return _tirx.op.gpu_storage_sync(storage_scope, loc=current_location())


def call_extern(dtype, func_name, *args, span=None):
    """Preserve TileLang's dtype-first external-call spelling on typed native calls."""
    ty = ir.PtrType(ir.PrimType("void")) if str(dtype) == "handle" else ir.PrimType(dtype)
    return _tirx.call_extern(func_name, *[as_expr(value) for value in args], ty=ty, loc=span or current_location())


def call_pure_extern(dtype, func_name, *args, span=None):
    """Construct a native external call with an explicitly pure effect contract."""
    ty = ir.PtrType(ir.PrimType("void")) if str(dtype) == "handle" else ir.PrimType(dtype)
    return _tirx.call_pure_extern(func_name, *[as_expr(value) for value in args], ty=ty, loc=span or current_location())


def tvm_access_ptr(ptype, data, offset, extent, rw_mask):
    """Preserve a raw pointer access interval using the native pointer contract.

    ``offset`` and ``extent`` count elements of ``ptype``, including when an
    explicit dtype differs from the input pointer's element type. The access
    keeps the input storage scope and never guesses a tensor's logical layout
    from its raw pointer. Tensor-coordinate accesses use ``T.access_ptr``.
    """
    data = as_expr(data)
    if not isinstance(data.ty, ir.PtrType):
        raise TypeError("tvm_access_ptr requires a typed pointer expression")
    if not isinstance(ptype, ir.DataTypeImm):
        raise TypeError("tvm_access_ptr ptype must be T.type_annotation(dtype)")
    element_type = ir.PrimType(ptype.value)
    pointer_type = ir.PtrType(element_type, data.ty.scope)
    if not tvm_ffi.structural_equal(data.ty, pointer_type):
        data = _tirx.reinterpret(data, ty=pointer_type, loc=current_location())
    mask = as_expr(rw_mask)
    if not isinstance(mask, ir.IntImm) or not 0 <= int(mask.value) <= 3:
        raise ValueError("tvm_access_ptr rw_mask must be a constant in [0, 3]")
    result = ir.Call("tl.ptr_access", [data, as_expr(offset), as_expr(extent), mask], loc=current_location())
    result.validate()
    return result


def isfinite(value, span=None, *, loc=None):
    """Keep floating finiteness visible for backend intrinsic selection.

    Native prim.isfinite expands during construction, before TileLang can
    select CUDA's isfinite instruction. The registered TileLang operation
    keeps exactly the same scalar/vector boolean type and has a portable
    lowering for targets without a dedicated intrinsic.
    """
    value = as_expr(value)
    location = loc if loc is not None else span if span is not None else current_location()
    dtype = dtype_of(value)
    if dtype.type_code in (tvm_ffi.dtype("float32").type_code, tvm_ffi.dtype("bfloat16").type_code):
        return ir.Call("tl.isfinite", [value], loc=location)
    return _prim.isfinite(value, loc=location)
