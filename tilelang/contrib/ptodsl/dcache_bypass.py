"""PTODSL helpers for scalar GM accesses that bypass the data cache."""

from __future__ import annotations

from ptodsl import pto
from ptoas.mlir.dialects import arith
from ptoas.mlir.ir import IntegerType

from .common import (
    as_logical_bool,
    coerce_i1,
    scalar_bitcast,
    scalar_cast,
    wrap_surface_value,
)


_INTEGER_PAYLOAD_DTYPES = {
    1: pto.i8,
    2: pto.i16,
    4: pto.i32,
    8: pto.i64,
}


def _logical_bool_to_i8(value):
    logical_value = coerce_i1(value, context="PTO GM dcache bypass bool store")
    return wrap_surface_value(arith.ExtUIOp(pto.i8.resolve(), logical_value).result)


def _payload_info(logical_dtype):
    """Map a logical scalar dtype to its physical GM payload representation.

    PTOAS ld_dev/st_dev operate on integer-width payloads. Floating-point
    values retain their bits via bitcast, while a TileLang bool is logical i1
    in expressions but byte-backed i8 in GM.
    """
    logical_type = logical_dtype.resolve()
    if IntegerType.isinstance(logical_type):
        integer_type = IntegerType(logical_type)
        width = integer_type.width
        if width == 1:
            # TileLang bool buffers are byte-backed although scalar predicates
            # use i1 in expressions.
            return pto.i8, "bool"
        if width not in (8, 16, 32, 64):
            raise TypeError(f"PTO GM dcache bypass supports only 1/2/4/8-byte integer elements, got {logical_type}")
        payload_dtype = _INTEGER_PAYLOAD_DTYPES[width // 8]
        # Signed/unsigned annotations do not change the stored bits. Use a
        # same-width signless payload and restore the authored signedness at
        # the scalar boundary when required.
        adaptation = "integer" if integer_type.is_signless else "integer_cast"
        return payload_dtype, adaptation

    bytewidth = pto.bytewidth(logical_dtype)
    try:
        return _INTEGER_PAYLOAD_DTYPES[bytewidth], "bitcast"
    except KeyError as exc:
        raise TypeError(f"PTO GM dcache bypass supports only 1/2/4/8-byte scalar elements, got {logical_type}") from exc


def read_gm_bypass_dcache(ptr, offset, logical_dtype):
    """Load one logical scalar through the AICore GM dcache-bypass path."""
    payload_dtype, adaptation = _payload_info(logical_dtype)
    payload_ptr = ptr
    if pto.const_expr(adaptation in ("bitcast", "integer_cast")):
        payload_ptr = pto.castptr(ptr, pto.ptr(payload_dtype, "gm"))

    # bypass_l1=True selects PTOAS pto.ld_dev rather than the normal scalar
    # load path, preserving the Ascend ReadGmByPassDCache semantics.
    value = pto.load_scalar(payload_ptr, offset, bypass_l1=True)
    if pto.const_expr(adaptation == "bool"):
        return as_logical_bool(value)
    if pto.const_expr(adaptation == "bitcast"):
        return scalar_bitcast(value, logical_dtype)
    if pto.const_expr(adaptation == "integer_cast"):
        # This is a same-width signedness adaptation, not a numeric conversion.
        return scalar_cast(value, logical_dtype, context="PTO GM dcache bypass integer load")
    return value


def write_gm_bypass_dcache(ptr, offset, value, logical_dtype):
    """Store one logical scalar through the AICore GM dcache-bypass path."""
    payload_dtype, adaptation = _payload_info(logical_dtype)
    payload_ptr = ptr
    if pto.const_expr(adaptation in ("bitcast", "integer_cast")):
        payload_ptr = pto.castptr(ptr, pto.ptr(payload_dtype, "gm"))
    if pto.const_expr(adaptation == "bitcast"):
        value = scalar_bitcast(value, payload_dtype)
    elif pto.const_expr(adaptation == "integer_cast"):
        # Normalize signed/unsigned annotations without changing the payload
        # bits before issuing the physical integer store.
        value = scalar_cast(value, payload_dtype, context="PTO GM dcache bypass integer store")
    elif pto.const_expr(adaptation == "bool"):
        value = _logical_bool_to_i8(value)

    # bypass_l1=True selects PTOAS pto.st_dev; the wrapper has already adapted
    # logical values to the physical integer payload width above.
    pto.store_scalar(payload_ptr, offset, value, bypass_l1=True)
