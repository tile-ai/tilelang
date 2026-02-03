"""Fast math operations exposed on the TileLang language surface."""

from tvm import tir
from tvm.tir import PrimExpr


def __log(x: PrimExpr) -> PrimExpr:
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


def __log2(x: PrimExpr) -> PrimExpr:
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


def __log10(x: PrimExpr) -> PrimExpr:
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


def __tan(x: PrimExpr) -> PrimExpr:
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


def __cos(x: PrimExpr) -> PrimExpr:
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


def __sin(x: PrimExpr) -> PrimExpr:
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


def __exp10(x: PrimExpr) -> PrimExpr:
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


def __exp(x: PrimExpr) -> PrimExpr:
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


__all__ = [
    "__log",  # noqa: F401
    "__log2",  # noqa: F401
    "__log10",  # noqa: F401
    "__tan",  # noqa: F401
    "__cos",  # noqa: F401
    "__sin",  # noqa: F401
    "__exp10",  # noqa: F401
    "__exp",  # noqa: F401
]
