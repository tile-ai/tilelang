"""The language interface for tl programs."""

from tvm.script.ir_builder.tir.frame import TIRFrame
from tvm.ffi import register_object
from tilelang import _ffi_api
from .kernel import get_thread_bindings, get_thread_extents


@register_object("tl.WarpSpecializeFrame")
class WarpSpecializeFrame(TIRFrame):
    """
    WarpSpecializeFrame is a custom TIRFrame that manages warp group indices
    and handles the entry and exit of the kernel launch scope.
    """


def WarpSpecialize(*warp_group_idx):
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
    Examples:
        >>> T.ws(0) -> if tx < 128
        >>> T.ws(1) -> if tx >= 128 and tx < 256
        >>> T.ws(0, 1) -> if tx < 128 or (tx >= 128 and tx < 256)
    """
    id_x, id_y, id_z = get_thread_bindings()
    ex_x, ex_y, ex_z = get_thread_extents()
    tid = id_x
    if ex_y > 1:
        tid = id_y * ex_x + tid
    if ex_z > 1:
        tid = id_z * (ex_y * ex_x) + tid

    # only available for nvidia gpus.
    warp_group_size = 128

    warp_group_ids: list[int] = []
    for warp_group_id in warp_group_idx:
        warp_group_ids.append(warp_group_id)

    assert len(warp_group_ids) > 0, "warp_group_idx must be non-empty"

    return _ffi_api.WarpSpecialize(warp_group_ids, tid, warp_group_size)


# Alias for WarpSpecialize for more concise usage
ws = WarpSpecialize
