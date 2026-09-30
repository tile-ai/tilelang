"""Architecture-aware orientation selection for plain TileIR GEMMs.

Hopper's warp-group MMA path is substantially faster when a sufficiently wide
dimension is presented as the hardware M dimension.  For a skinny logical
``M x N`` output, the algebraically equivalent form

    ``A @ B + C == (B.T @ A.T + C.T).T``

exposes ``N`` as that dimension without changing program semantics.  This pass
records the decision on :class:`Gemm`; emission remains a mechanical lowering
of the explicit ``swap_ab`` attribute.
"""

from __future__ import annotations

import re
from numbers import Integral
from typing import Any

from tilelang.tileir.ir.ops import Gemm
from tilelang.tileir.ir.value import Block
from tilelang.tileir.passes.base import PassContext, walk_block

__all__ = ["gemm_orientation_pass"]


_ARCH_RE = re.compile(r"^sm_(\d+)[a-z]*$")
_FP8_INPUT_DTYPES = frozenset({"float8_e4m3fn", "float8_e5m2"})
_SUPPORTED_INPUT_DTYPES = frozenset({"float16", "bfloat16", *_FP8_INPUT_DTYPES})


def _is_hopper(arch: Any) -> bool:
    if not isinstance(arch, str):
        return False
    match = _ARCH_RE.fullmatch(arch.strip())
    return match is not None and int(match.group(1)) == 90


def _static_matrix_shape(value: Any) -> tuple[int, int] | None:
    value_type = getattr(value, "type", None)
    shape = getattr(value_type, "shape", ())
    if len(shape) != 2 or not all(isinstance(dim, Integral) for dim in shape):
        return None
    return int(shape[0]), int(shape[1])


def _should_swap(op: Gemm, arch: Any) -> bool:
    if not _is_hopper(arch):
        return False
    if getattr(getattr(op.lhs, "type", None), "dtype", None) is None:
        return False
    lhs_dtype = op.lhs.type.dtype.name
    rhs_dtype = op.rhs.type.dtype.name
    if lhs_dtype not in _SUPPORTED_INPUT_DTYPES or rhs_dtype not in _SUPPORTED_INPUT_DTYPES:
        return False

    acc_shape = _static_matrix_shape(op.acc)
    if acc_shape is None:
        return False
    m, n = acc_shape

    # Keep the policy narrow: the extra permutes regress larger non-FP8 tiles.
    # FP8 GEMMs remain profitable up to M=64; other supported inputs stop at 32.
    max_swapped_m = 64 if lhs_dtype in _FP8_INPUT_DTYPES and rhs_dtype in _FP8_INPUT_DTYPES else 32
    return 0 < m <= max_swapped_m and n >= 64 and n >= 2 * m


def gemm_orientation_pass(root: Block, ctx: PassContext) -> None:
    """Mark profitable Hopper skinny-M :class:`Gemm` ops as ``swap_ab``."""
    arch = ctx.results.get("target_arch")
    visited = 0
    swapped = 0

    def _visit(op: Any) -> None:
        nonlocal visited, swapped
        if not isinstance(op, Gemm):
            return
        visited += 1
        if not op.swap_ab and _should_swap(op, arch):
            op.swap_ab = True
        if op.swap_ab:
            swapped += 1

    walk_block(root, _visit)
    ctx.results["gemm_orientation"] = {
        "target_arch": arch,
        "visited": visited,
        "swapped": swapped,
    }
