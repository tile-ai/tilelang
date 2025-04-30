# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from tvm.script.ir_builder.tir.frame import TIRFrame
from tvm._ffi import register_object
from tilelang import _ffi_api
from .kernel import get_thread_bindings, get_thread_extents


@register_object("tl.WarpSpecializeFrame")
class WarpSpecializeFrame(TIRFrame):
    """
    WarpSpecializeFrame is a custom TIRFrame that manages warp group indices
    and handles the entry and exit of the kernel launch scope.
    """


def WarpSpecialize(warp_group_idx: int,):
    """Tools to construct a warp group frame.

    Parameters
    ----------
    warp_group_idx : int
        A integer representing warp group index
        Or a list of integers representing blockDim.(x|y|z)
        if the value is -1, we skip the threadIdx.x binding.

    Returns
    -------
    res : Tuple[frame.LaunchThreadFrame]
        The result LaunchThreadFrame.
    """
    id_x, id_y, id_z = get_thread_bindings()
    ex_x, ex_y, _ = get_thread_extents()
    tid = id_z * (ex_y * ex_x) + id_y * ex_x + id_x
    # only available for nvidia gpus.
    warp_group_size = 128

    return _ffi_api.WarpSpecialize(warp_group_idx, tid, warp_group_size)


# Alias for WarpSpecialize for more concise usage
ws = WarpSpecialize
