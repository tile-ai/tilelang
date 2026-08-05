"""Shared utilities for the TileLang graph compiler."""

import torch

from tvm import relax as _relax, tirx as _tir


# Reuse existing dtype maps from tilelang.language.dtypes
def torch_dtype_to_tvm(dtype: torch.dtype) -> str:
    """Convert a PyTorch dtype to a TVM dtype string."""
    from tilelang.language.dtypes import _TORCH_DTYPE_TO_STR

    if dtype not in _TORCH_DTYPE_TO_STR:
        raise ValueError(f"Unsupported torch dtype: {dtype}")
    return _TORCH_DTYPE_TO_STR[dtype]


def tvm_dtype_to_torch(dtype_str: str) -> torch.dtype:
    """Convert a TVM dtype string to a PyTorch dtype."""
    from tilelang.language import dtype

    return dtype(dtype_str).as_torch()


def validate_cuda_tensors(values, expected_device: torch.device | None = None) -> torch.device:
    """Validate graph inputs and return their canonical CUDA device."""
    tensors = [value for value in values if isinstance(value, torch.Tensor)]
    if not tensors:
        raise ValueError("TileLang graph compilation requires at least one CUDA tensor")

    first_device = tensors[0].device
    if first_device.type != "cuda":
        raise ValueError("TileLang graph compilation currently requires contiguous CUDA tensors")

    device_index = first_device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    device = torch.device("cuda", device_index)

    if expected_device is not None and device != expected_device:
        raise ValueError(f"TileLang graph runner was compiled for {expected_device}, but received {device}")

    for tensor in tensors:
        if tensor.device != device:
            raise ValueError("TileLang graph compilation requires all CUDA tensors to be on the same device")
        if tensor.layout != torch.strided or not tensor.is_contiguous():
            raise ValueError("TileLang graph compilation currently requires contiguous CUDA tensors")

    return device


# ---------------------------------------------------------------------------
# Relax expression remapping
# ---------------------------------------------------------------------------


def remap_expr(expr, env):
    """Replace Var references in a Relax expression using *env*.

    Recursively walks Call, Tuple, TupleGetItem, If, and ShapeExpr nodes.
    Preserves ``sinfo_args`` and ``attrs`` (required for ``call_tir``).

    Parameters
    ----------
    expr : relax.Expr
        The expression to remap.
    env : dict[relax.Var, relax.Var]
        Mapping from old vars to new vars.

    Returns
    -------
    relax.Expr
        The expression with vars substituted.
    """
    if not env:
        return expr
    if isinstance(expr, _relax.Var):
        return env.get(expr, expr)
    if isinstance(expr, _relax.Call):
        new_op = env.get(expr.op, expr.op) if isinstance(expr.op, _relax.Var) else expr.op
        new_args = [remap_expr(a, env) for a in expr.args]
        return _relax.Call(new_op, new_args, expr.attrs, expr.sinfo_args, expr.span)
    if isinstance(expr, _relax.Tuple):
        return _relax.Tuple([remap_expr(f, env) for f in expr.fields], expr.span)
    if isinstance(expr, _relax.TupleGetItem):
        return _relax.TupleGetItem(remap_expr(expr.tuple_value, env), expr.index, expr.span)
    if isinstance(expr, _relax.ShapeExpr):
        return expr
    if isinstance(expr, _relax.If):
        return _relax.If(remap_expr(expr.cond, env), remap_expr(expr.true_branch, env), remap_expr(expr.false_branch, env), expr.span)
    return expr


# ---------------------------------------------------------------------------
# Struct info helpers
# ---------------------------------------------------------------------------


def get_static_shape(var) -> list[int] | None:
    """Extract static integer shape from a Relax Var's struct info.

    Returns ``None`` if the var has no TensorStructInfo, no shape,
    or any symbolic (non-IntImm) dimension.
    """
    si = var.struct_info_ if hasattr(var, "struct_info_") else None
    if not isinstance(si, _relax.TensorStructInfo) or si.shape is None:
        return None
    values = si.shape.values
    if not all(isinstance(s, _tir.IntImm) for s in values):
        return None
    return [int(s) for s in values]
