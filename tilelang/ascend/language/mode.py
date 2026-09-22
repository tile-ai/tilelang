"""Ascend stateful hardware mode controls for Cube and GM stores."""

from __future__ import annotations

from tvm import tirx


def set_mmad_direction(direction: str):
    """Select whether Cube generates results along M or N first.

    ``direction`` is ``"m"`` or ``"n"``. The setting applies to subsequent
    GEMMs on the AIC, including block-scaled GEMMs. It changes the traversal
    within a MAD, independently of the kernel's outer tile-loop order.
    """
    if direction not in ("m", "n"):
        raise ValueError(f"mmad direction must be 'm' or 'n', got {direction!r}")
    return tirx.call_intrin(
        "void",
        tirx.op.Op.get("tl.ascend_set_mmad_direction"),
        tirx.StringImm(direction),
    )


def set_hf32_mode(mode=None):
    """Set HF32 mode for Ascend Cube fp32 matmul.

    Controls whether fp32 inputs to the Cube unit are truncated to
    HF32 (19-bit mantissa) before multiplication, trading precision for
    throughput (~2x).  Ignored for non-fp32 input types.

    Generates: AscendC::SetHF32Mode(HF32Mode::DISABLE/ENABLE) and
               AscendC::SetHF32TransMode(HF32TransMode::NEAREST_ZERO/NEAREST_EVEN).

    Parameters
    ----------
    mode : None | "nearest_zero" | "nearest_even"
        - None (default): disable HF32, full fp32 precision.
        - "nearest_zero": enable HF32, round toward zero.
        - "nearest_even": enable HF32, round to nearest even.

    Example
    -------
    >>> T.set_hf32_mode("nearest_even")   # enable HF32
    >>> T.set_hf32_mode(None)             # restore full fp32
    """
    _HF32_MODE_INT = {None: 0, "nearest_zero": 1, "nearest_even": 2}
    if mode not in _HF32_MODE_INT:
        raise ValueError(f"hf32 mode must be None, 'nearest_zero', or 'nearest_even', got {mode!r}")
    return tirx.call_intrin(
        "void",
        tirx.op.Op.get("tl.ascend_set_hf32_mode"),
        tirx.IntImm("int32", _HF32_MODE_INT[mode]),
    )


_ATOMIC_OPS = ("add", "max", "min")
# Accumulate types the dav_3510 store-mode atomic supports (CANN
# kernel_operator_set_atomic_impl.h: SupportType<float, half, int16_t, int32_t,
# int8_t, bfloat16_t>).
_ATOMIC_DTYPES = ("float32", "float", "float16", "bfloat16", "int8", "int16", "int32")


def set_atomic(op="add", dtype="float32"):
    """Arm an Ascend hardware store-mode atomic op.

    Call once before the stores it should affect. With the flag armed, an
    ordinary L0C->GM (``T.copy``) or UB->GM (``T.dual_copy``) store reduces into
    GM (``D op= tile``) in hardware instead of overwriting — no vector loop.
    Clear with :func:`set_atomic_none` to restore plain stores.

    Generates ``AscendC::SetAtomic{Add,Max,Min}<T>()``.

    Parameters
    ----------
    op : "add" | "max" | "min"
        The store-mode reduction.
    dtype : str
        Accumulate type of the GM destination. One of float32/float16/bfloat16/
        int8/int16/int32 (the dav_3510-supported set).

    Example
    -------
    >>> T.set_atomic("add", "float32")   # arm
    >>> # ... stores that should accumulate into GM ...
    >>> T.set_atomic_none()              # restore
    """
    if op not in _ATOMIC_OPS:
        raise ValueError(f"set_atomic op must be one of {_ATOMIC_OPS}, got {op!r}")
    if dtype not in _ATOMIC_DTYPES:
        raise ValueError(f"set_atomic dtype must be one of {_ATOMIC_DTYPES}, got {dtype!r}")
    # Pass a typed zero so codegen reads its dtype to instantiate SetAtomic*<T>().
    return tirx.call_intrin(
        "void",
        tirx.op.Op.get("tl.ascend_set_atomic"),
        tirx.StringImm(op),
        tirx.const(0, dtype),
    )


def set_atomic_none():
    """Clear the Ascend store-mode atomic flag, restoring plain (overwrite) stores.

    Call once to undo :func:`set_atomic`.
    """
    return tirx.call_intrin("void", tirx.op.Op.get("tl.ascend_set_atomic_none"))


__all__ = [
    "set_mmad_direction",
    "set_hf32_mode",
    "set_atomic",
    "set_atomic_none",
]
