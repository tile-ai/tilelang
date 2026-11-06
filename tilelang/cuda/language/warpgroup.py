"""The language interface for tl programs."""

from tilelang.language import _annotation_codec
from tilelang.language import frame as _loop_frame
from tvm_tirx.ir import prim as _prim
from tvm_tirx.tirx.script import ir_builder as _builder


from tilelang.language.kernel import get_thread_bindings, get_thread_extents

__all__ = ["WarpSpecialize", "WarpSpecializeFrame", "ws"]


class WarpSpecializeFrame(_loop_frame.LoopFrame):
    """
    WarpSpecializeFrame is a custom TIRFrame that manages warp group indices
    and handles the entry and exit of the kernel launch scope.
    """

    def __init__(self, warp_group_ids, thread_idx, warp_group_size=128):
        groups = sorted(int(group) for group in warp_group_ids)
        if not groups:
            raise ValueError("warp_group_idx must be non-empty")
        merged = []
        for group in groups:
            if merged and group == merged[-1][1]:
                merged[-1] = (merged[-1][0], group + 1)
            else:
                merged.append((group, group + 1))
        condition = None
        for start, end in merged:
            bounded = _prim.And(
                thread_idx >= start * warp_group_size,
                thread_idx < end * warp_group_size,
            )
            condition = bounded if condition is None else _prim.Or(condition, bounded)
        self.frames = [
            _builder.if_(condition),
            _builder.then_(),
            _annotation_codec.attr(0, "warp_specialize", 1),
        ]
        self.vars = []
        self.annotations = None
        self._stack = None

    def __enter__(self):
        super().__enter__()
        return None


def WarpSpecialize(*warp_group_idx) -> WarpSpecializeFrame:
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

    return WarpSpecializeFrame(warp_group_ids, tid, warp_group_size)


# Alias for WarpSpecialize for more concise usage
ws = WarpSpecialize
