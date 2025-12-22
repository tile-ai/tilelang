"""Common math intrinsics exposed on the TileLang language surface."""

from tvm import tir


def _validate_rounding_mode(rounding_mode):
    """Validate that the rounding mode is one of the supported IEEE modes"""
    valid_modes = {"rn", "rz", "ru", "rd"}
    if isinstance(rounding_mode, str) and rounding_mode in valid_modes:
        return
    raise ValueError(f"Invalid rounding mode '{rounding_mode}'. Must be one of: {valid_modes}")


def __log(x):
    """Calculate log(x) with fast math

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    x = tir.convert(x)
    return tir.call_intrin(x.dtype, tir.op.Op.get("tl.__log"), x)


def __log2(x):
    """Calculate log2(x) with fast math

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    x = tir.convert(x)
    return tir.call_intrin(x.dtype, tir.op.Op.get("tl.__log2"), x)


def __log10(x):
    """Calculate log10(x) with fast math

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    x = tir.convert(x)
    return tir.call_intrin(x.dtype, tir.op.Op.get("tl.__log10"), x)


def __tan(x):
    """Calculate tan(x) with fast math

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    x = tir.convert(x)
    return tir.call_intrin(x.dtype, tir.op.Op.get("tl.__tan"), x)


def __cos(x):
    """Calculate cos(x) with fast math

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    x = tir.convert(x)
    return tir.call_intrin(x.dtype, tir.op.Op.get("tl.__cos"), x)


def __sin(x):
    """Calculate sin(x) with fast math

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    x = tir.convert(x)
    return tir.call_intrin(x.dtype, tir.op.Op.get("tl.__sin"), x)


def __exp10(x):
    """Calculate 10**x with fast math

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    x = tir.convert(x)
    return tir.call_intrin(x.dtype, tir.op.Op.get("tl.__exp10"), x)


def __exp(x):
    """Calculate 2**x with fast math

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    x = tir.convert(x)
    return tir.call_intrin(x.dtype, tir.op.Op.get("tl.__exp"), x)


# IEEE-compliant operations
def ieee_add(x, y, rounding_mode="rn"):
    """IEEE-compliant addition with specified rounding mode

    Parameters
    ----------
    x : PrimExpr
        First operand.
    y : PrimExpr
        Second operand.
    rounding_mode : str, optional
        Rounding mode: 'rn' (round to nearest), 'rz' (round toward zero),
        'ru' (round toward positive infinity), 'rd' (round toward negative infinity).
        Default is 'rn'.

    Returns
    -------
    result : PrimExpr
        The result.
    """
    _validate_rounding_mode(rounding_mode)
    x = tir.convert(x)
    y = tir.convert(y)
    rounding_mode = tir.convert(rounding_mode)
    return tir.call_intrin(x.dtype, tir.op.Op.get("tl.ieee_add"), x, y, rounding_mode)


def ieee_sub(x, y, rounding_mode="rn"):
    """IEEE-compliant subtraction with specified rounding mode

    Parameters
    ----------
    x : PrimExpr
        First operand.
    y : PrimExpr
        Second operand.
    rounding_mode : str, optional
        Rounding mode: 'rn', 'rz', 'ru', 'rd'. Default is 'rn'.

    Returns
    -------
    result : PrimExpr
        The result.
    """
    _validate_rounding_mode(rounding_mode)
    x = tir.convert(x)
    y = tir.convert(y)
    rounding_mode = tir.convert(rounding_mode)
    return tir.call_intrin(x.dtype, tir.op.Op.get("tl.ieee_sub"), x, y, rounding_mode)


def ieee_mul(x, y, rounding_mode="rn"):
    """IEEE-compliant multiplication with specified rounding mode

    Parameters
    ----------
    x : PrimExpr
        First operand.
    y : PrimExpr
        Second operand.
    rounding_mode : str, optional
        Rounding mode: 'rn', 'rz', 'ru', 'rd'. Default is 'rn'.

    Returns
    -------
    result : PrimExpr
        The result.
    """
    _validate_rounding_mode(rounding_mode)
    x = tir.convert(x)
    y = tir.convert(y)
    rounding_mode = tir.convert(rounding_mode)
    return tir.call_intrin(x.dtype, tir.op.Op.get("tl.ieee_mul"), x, y, rounding_mode)


def ieee_fmaf(x, y, z, rounding_mode="rn"):
    """IEEE-compliant fused multiply-add with specified rounding mode

    Parameters
    ----------
    x : PrimExpr
        First operand.
    y : PrimExpr
        Second operand.
    z : PrimExpr
        Third operand (addend).
    rounding_mode : str, optional
        Rounding mode: 'rn', 'rz', 'ru', 'rd'. Default is 'rn'.

    Returns
    -------
    result : PrimExpr
        The result of x * y + z.
    """
    _validate_rounding_mode(rounding_mode)
    x = tir.convert(x)
    y = tir.convert(y)
    z = tir.convert(z)
    rounding_mode = tir.convert(rounding_mode)
    return tir.call_intrin(x.dtype, tir.op.Op.get("tl.ieee_fmaf"), x, y, z, rounding_mode)


def ieee_frcp(x, rounding_mode="rn"):
    """IEEE-compliant reciprocal with specified rounding mode

    Parameters
    ----------
    x : PrimExpr
        Input operand.
    rounding_mode : str, optional
        Rounding mode: 'rn', 'rz', 'ru', 'rd'. Default is 'rn'.

    Returns
    -------
    result : PrimExpr
        The result of 1/x.
    """
    _validate_rounding_mode(rounding_mode)
    x = tir.convert(x)
    rounding_mode = tir.convert(rounding_mode)
    return tir.call_intrin(x.dtype, tir.op.Op.get("tl.ieee_frcp"), x, rounding_mode)


def ieee_fsqrt(x, rounding_mode="rn"):
    """IEEE-compliant square root with specified rounding mode

    Parameters
    ----------
    x : PrimExpr
        Input operand.
    rounding_mode : str, optional
        Rounding mode: 'rn', 'rz', 'ru', 'rd'. Default is 'rn'.

    Returns
    -------
    result : PrimExpr
        The result of sqrt(x).
    """
    _validate_rounding_mode(rounding_mode)
    x = tir.convert(x)
    rounding_mode = tir.convert(rounding_mode)
    return tir.call_intrin(x.dtype, tir.op.Op.get("tl.ieee_fsqrt"), x, rounding_mode)


def ieee_frsqrt(x):
    """IEEE-compliant reciprocal square root (round to nearest only)

    Parameters
    ----------
    x : PrimExpr
        Input operand.

    Returns
    -------
    result : PrimExpr
        The result of 1/sqrt(x).
    """
    x = tir.convert(x)
    return tir.call_intrin(x.dtype, tir.op.Op.get("tl.ieee_frsqrt"), x)


def ieee_fdiv(x, y, rounding_mode="rn"):
    """IEEE-compliant division with specified rounding mode

    Parameters
    ----------
    x : PrimExpr
        Dividend.
    y : PrimExpr
        Divisor.
    rounding_mode : str, optional
        Rounding mode: 'rn', 'rz', 'ru', 'rd'. Default is 'rn'.

    Returns
    -------
    result : PrimExpr
        The result of x/y.
    """
    _validate_rounding_mode(rounding_mode)
    x = tir.convert(x)
    y = tir.convert(y)
    rounding_mode = tir.convert(rounding_mode)
    return tir.call_intrin(x.dtype, tir.op.Op.get("tl.ieee_fdiv"), x, y, rounding_mode)


__all__ = [
    "__log",  # noqa: F401
    "__log2",  # noqa: F401
    "__log10",  # noqa: F401
    "__tan",  # noqa: F401
    "__cos",  # noqa: F401
    "__sin",  # noqa: F401
    "__exp10",  # noqa: F401
    "__exp",  # noqa: F401
    "ieee_add",  # noqa: F401
    "ieee_sub",  # noqa: F401
    "ieee_mul",  # noqa: F401
    "ieee_fmaf",  # noqa: F401
    "ieee_frcp",  # noqa: F401
    "ieee_fsqrt",  # noqa: F401
    "ieee_frsqrt",  # noqa: F401
    "ieee_fdiv",  # noqa: F401
]
