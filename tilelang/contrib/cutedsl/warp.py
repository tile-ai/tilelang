# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""
Warp-level primitives for CuTeDSL backend.
Re-exports from cutlass.cute.arch with TileLang naming conventions.
"""

__all__ = ['__activemask', '__shfl_down_sync', '__shfl_sync']

from cutlass._mlir.dialects import llvm, nvvm
from cutlass.base_dsl.typing import Uint32, Int32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.arch import shuffle_sync, shuffle_sync_down


FULL_MASK = 0xFFFFFFFF
WARP_SIZE = 32


@dsl_user_op
def __activemask(*, loc=None, ip=None) -> Uint32:
    """
    Returns a 32-bit integer mask of all currently active threads in the calling warp.
    
    PTX: activemask.b32 %mask;
    """
    result = llvm.inline_asm(
        T.i32(),
        [],
        "activemask.b32 $0;",
        "=r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Uint32(result)


def __shfl_down_sync(mask, val, delta, width=32):
    """
    Shuffle down within warp.
    
    Uses CuTeDSL's shuffle_sync_down with proper mask_and_clamp calculation.
    For shfl.down: mask_and_clamp = (width - 1) | 0x1f00
    """
    mask_and_clamp = ((width - 1) & 0x1f) | 0x1f00
    return shuffle_sync_down(val, offset=delta, mask=mask, mask_and_clamp=mask_and_clamp)


def __shfl_sync(mask, val, srcLane, width=32):
    """
    Broadcast from a specific lane within warp.
    
    Uses CuTeDSL's shuffle_sync (idx mode) with proper mask_and_clamp.
    For shfl.idx: mask_and_clamp = (width - 1)
    """
    mask_and_clamp = (width - 1) & 0x1f
    return shuffle_sync(val, offset=srcLane, mask=mask, mask_and_clamp=mask_and_clamp)
