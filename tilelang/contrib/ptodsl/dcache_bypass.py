"""PTODSL helpers for scalar GM accesses that bypass the data cache."""

from __future__ import annotations

from typing import Literal

from ptodsl import pto
from ptodsl._types import _DType
from ptoas.mlir.ir import IntegerType

from .common import (
    as_logical_bool,
    coerce_i1,
)


_INTEGER_PAYLOAD_DTYPES = {
    1: pto.i8,
    2: pto.i16,
    4: pto.i32,
    8: pto.i64,
}


def _logical_bool_to_i8(value):
    logical_value = coerce_i1(value, context="PTO GM dcache bypass bool store")
    return pto.cast(logical_value, pto.i8)


def _payload_info(logical_dtype: _DType) -> tuple[_DType, Literal["bool", "integer", "integer_cast", "bitcast"]]:
    """Return the integer payload dtype and logical adaptation kind."""
    logical_type = logical_dtype.resolve()
    if IntegerType.isinstance(logical_type):
        integer_type = IntegerType(logical_type)
        width = integer_type.width
        if width == 1:
            # TileLang bool buffers are byte-backed even though scalar bools
            # use i1 in expressions.
            return pto.i8, "bool"
        if width not in (8, 16, 32, 64):
            raise TypeError(f"PTO GM dcache bypass supports only 1/2/4/8-byte integer elements, got {logical_type}")
        payload_dtype = _INTEGER_PAYLOAD_DTYPES[width // 8]
        # pto.ld_dev/st_dev require signless integer payloads. Preserve an
        # authored signed/unsigned logical type around the access explicitly.
        adaptation = "integer" if integer_type.is_signless else "integer_cast"
        return payload_dtype, adaptation

    bytewidth = pto.bytewidth(logical_dtype)
    try:
        return _INTEGER_PAYLOAD_DTYPES[bytewidth], "bitcast"
    except KeyError as exc:
        raise TypeError(f"PTO GM dcache bypass supports only 1/2/4/8-byte scalar elements, got {logical_type}") from exc


def read_gm_bypass_dcache(ptr, offset, logical_dtype):
    """Load one logical scalar from GM through the cache-bypass pipeline."""
    payload_dtype, adaptation = _payload_info(logical_dtype)
    payload_ptr = ptr
    if pto.const_expr(adaptation in ("bitcast", "integer_cast")):
        payload_ptr = pto.castptr(ptr, pto.ptr(payload_dtype, "gm"))

    value = pto.ld_dev(payload_ptr, offset)
    if pto.const_expr(adaptation == "bool"):
        return as_logical_bool(value)
    if pto.const_expr(adaptation == "bitcast"):
        return pto.bitcast(value, logical_dtype)
    if pto.const_expr(adaptation == "integer_cast"):
        return pto.cast(value, logical_dtype)
    return value


def write_gm_bypass_dcache(ptr, offset, value, logical_dtype):
    """Store one logical scalar to GM through the cache-bypass pipeline."""
    payload_dtype, adaptation = _payload_info(logical_dtype)
    payload_ptr = ptr
    if pto.const_expr(adaptation in ("bitcast", "integer_cast")):
        payload_ptr = pto.castptr(ptr, pto.ptr(payload_dtype, "gm"))
    if pto.const_expr(adaptation == "bitcast"):
        value = pto.bitcast(value, payload_dtype)
    elif pto.const_expr(adaptation == "integer_cast"):
        value = pto.cast(value, payload_dtype)
    elif pto.const_expr(adaptation == "bool"):
        value = _logical_bool_to_i8(value)

    pto.st_dev(payload_ptr, offset, value)
