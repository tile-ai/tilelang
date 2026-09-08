"""Ascend MTE DMA copy primitives and their pad-value / ND->NZ helpers."""

from __future__ import annotations

from tvm import tirx


def _to_int32(v):
    return tirx.IntImm("int32", v) if isinstance(v, int) else v


_PAD_VALUE_SUPPORTED_DTYPES = frozenset(
    {
        "int8",
        "uint8",
        "int16",
        "uint16",
        "float16",
        "bfloat16",
        "int32",
        "uint32",
        "float32",
    }
)


def ascend_set_copy_pad_value(value, dtype=None) -> None:
    """Set the padding fill value for subsequent Ascend padded MTE copies.

    Generates: asc_set_copy_pad_val(value_bits);

    This call moves no data; it configures the stateful hardware pad value
    consumed by a following padded GM -> UB/L1 copy. Set it immediately
    before each padded copy rather than relying on a single set persisting
    across unrelated copies.

    Parameters
    ----------
    value : int, float, or PrimExpr
        Padding fill value.
    dtype : str, optional
        Copy element dtype (one of int8/uint8/int16/uint16/float16/bfloat16/
        int32/uint32/float32). Required when ``value`` is a Python literal;
        for a typed PrimExpr it defaults to ``value.dtype``.

    Example
    -------
    >>> T.ascend_set_copy_pad_value(0.0, dtype="float32")
    """
    if dtype is None:
        inferred = getattr(value, "dtype", None)
        if isinstance(inferred, str):
            dtype = inferred
    if dtype is None:
        raise ValueError('ascend_set_copy_pad_value requires dtype for Python literals; use dtype="float32" or pass a typed PrimExpr.')
    if dtype not in _PAD_VALUE_SUPPORTED_DTYPES:
        raise ValueError(
            f"ascend_set_copy_pad_value does not support dtype {dtype!r}; supported dtypes are {sorted(_PAD_VALUE_SUPPORTED_DTYPES)}."
        )
    value = tirx.const(value, dtype) if isinstance(value, (int, float)) else tirx.Cast(dtype, value)
    return tirx.call_intrin("void", tirx.op.Op.get("tl.ascend_set_copy_pad_value"), value)


def ascend_nd2nz_post_copy(dst, src, rows, cols, full_rows, dst_dtype: str) -> None:
    """Ascend ND->NZ post-copy from UB scratch to L1 with NZ pointer correction."""
    from tvm.tirx import BufferLoad
    from tilelang.language.builtin import access_ptr

    if isinstance(dst, BufferLoad):
        dst = access_ptr(dst, "w")
    if isinstance(src, BufferLoad):
        src = access_ptr(src, "r")
    args = [dst, src, _to_int32(rows), _to_int32(cols), _to_int32(full_rows), dst_dtype]
    return tirx.call_intrin("void", tirx.op.Op.get("tl.ascend_nd2nz_post_copy"), *args)


__all__ = [
    "ascend_set_copy_pad_value",
    "ascend_nd2nz_post_copy",
]
