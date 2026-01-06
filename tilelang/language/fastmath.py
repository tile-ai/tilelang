"""Fast math operations exposed on the TileLang language surface."""

from tvm import tir


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
