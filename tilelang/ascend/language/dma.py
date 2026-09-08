"""Ascend MTE DMA copy primitives and their pad-value / ND->NZ helpers."""

from __future__ import annotations

from tvm import tirx


def _to_int32(v):
    return tirx.IntImm("int32", v) if isinstance(v, int) else v


def ascend_copy_gm_to_ubuf(
    dst,
    src,
    sid,
    n_burst,
    burst_len,
    left_padding,
    right_padding,
    data_select,
    l2_cache_ctl,
    burst_src_stride,
    burst_dst_stride,
) -> None:
    """Insert an Ascend DMA copy from GM to UBuf.

    Generates: copy_gm_to_ubuf_align_v2(dst, src, sid, nBurst, burstLen,
    leftPadding, rightPadding, dataSelect, l2CacheCtl, burstSrcStride,
    burstDstStride);

    Parameters
    ----------
    dst : PrimExpr
        Destination UBuf pointer (tvm_access_ptr or similar).
    src : PrimExpr
        Source GM pointer (tvm_access_ptr or similar).
    sid : int or PrimExpr
        Sub-block ID (usually 0).
    n_burst : int or PrimExpr
        Number of bursts.
    burst_len : int or PrimExpr
        Length of each burst in bytes.
    left_padding : int or PrimExpr
        Left padding count.
    right_padding : int or PrimExpr
        Right padding count.
    data_select : bool, int, or PrimExpr
        v2 data select / constant padding control.
    l2_cache_ctl : int or PrimExpr
        L2 cache control.
    burst_src_stride : int or PrimExpr
        Source stride between burst starts in bytes.
    burst_dst_stride : int or PrimExpr
        Destination stride between burst starts in bytes.
    """
    args = [dst, src] + [
        _to_int32(a)
        for a in [
            sid,
            n_burst,
            burst_len,
            left_padding,
            right_padding,
            data_select,
            l2_cache_ctl,
            burst_src_stride,
            burst_dst_stride,
        ]
    ]
    return tirx.call_intrin("void", tirx.op.Op.get("tl.ascend_copy_gm_to_ubuf"), *args)


def ascend_copy_ubuf_to_gm(dst, src, sid, burst_num, burst_len, l2_cache_ctl, burst_dst_stride, burst_src_stride) -> None:
    """Insert an Ascend DMA copy from UBuf to GM.

    Generates: copy_ubuf_to_gm_align_v2(dst, src, sid, burst_num, burst_len, l2_cache_ctl, burst_dst_stride, burst_src_stride);

    Parameters
    ----------
    dst : PrimExpr
        Destination GM pointer.
    src : PrimExpr
        Source UBuf pointer.
    sid : int or PrimExpr
        Sub-block ID (usually 0).
    burst_num : int or PrimExpr
        Number of bursts.
    burst_len : int or PrimExpr
        Length of each burst in bytes.
    l2_cache_ctl : int or PrimExpr
        L2 cache control.
    burst_dst_stride : int or PrimExpr
        Destination stride between bursts.
    burst_src_stride : int or PrimExpr
        Source stride between bursts.
    """
    args = [dst, src] + [
        _to_int32(a)
        for a in [
            sid,
            burst_num,
            burst_len,
            l2_cache_ctl,
            burst_dst_stride,
            burst_src_stride,
        ]
    ]
    return tirx.call_intrin("void", tirx.op.Op.get("tl.ascend_copy_ubuf_to_gm"), *args)


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

    Generates: AscendC::SetPadValue<T>(value);

    This call moves no data; it configures the stateful hardware pad value
    consumed by a following padded GM -> UB/L1 copy (e.g. a
    ``ascend_copy_gm_to_ubuf`` with ``data_select == 1``). Set it immediately
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
