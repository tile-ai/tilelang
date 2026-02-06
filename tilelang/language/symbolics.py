"""Symbolic variable helpers exposed on the TileLang language surface."""

from __future__ import annotations
import re
from tvm import tir

from tilelang.utils import deprecated

__all__ = ["dynamic", "symbolic"]


def dynamic(name: str, dtype: str = "int32") -> tuple[tir.Var, ...] | tir.Var:
    """
    Create a TIR dynamic symbolic variable.

    Parameters:
        name (str): Identifier for the variable in generated TIR.
        dtype (str): Data type string for the variable (e.g., "int32"). Defaults to "int32".

    Returns:
        tir.Var: A TIR variable with the given name and dtype for use in TIR/TensorIR kernels.
    """
    if "," in name:
        names = re.split(r"\s*,\s*", name)
        return tuple(tir.Var(n, dtype) for n in names)
    if " " in name:
        names = re.split(r"\s+", name)
        return tuple(tir.Var(n, dtype) for n in names)
    return tir.Var(name, dtype)


@deprecated("T.symbolic(...)", "T.dynamic(...)", "v0.1.9")
def symbolic(name: str, dtype: str = "int32"):
    """Deprecated alias for `T.dynamic`."""
    return dynamic(name, dtype)
