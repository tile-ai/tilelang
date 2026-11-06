"""Symbolic variable helpers exposed on the TileLang language surface."""

from __future__ import annotations

from tvm_tirx import ir as _ir
import re


from tilelang._typing import DType
from tilelang.utils import deprecated

__all__ = ["dynamic", "symbolic"]


def dynamic(name: str, dtype: DType = "int32") -> tuple[_ir.Var, ...] | _ir.Var:
    """
    Create a TIR dynamic symbolic variable.

    Parameters:
        name (str): Identifier for the variable in generated TIR.
        dtype (str): Data type string for the variable (e.g., "int32"). Defaults to "int32".

    Returns:
        tirx.Var: A TIR variable with the given name and dtype for use in TIR/TensorIR kernels.
    """
    if "," in name:
        names = re.split(r"\s*,\s*", name)
        return tuple(_ir.Var(n, dtype) for n in names)
    if " " in name:
        names = re.split(r"\s+", name)
        return tuple(_ir.Var(n, dtype) for n in names)
    return _ir.Var(name, dtype)


@deprecated("T.symbolic(...)", "T.dynamic(...)", "v0.1.15")
def symbolic(name: str, dtype: DType = "int32") -> tuple[_ir.Var, ...] | _ir.Var:
    """Deprecated alias for `T.dynamic`."""
    return dynamic(name, dtype)
