"""Ascend NPU mixed-kernel (AIC + AIV) launch frame."""

from __future__ import annotations

from tilelang import _ffi_api
from tilelang.jit.exceptions import JITNoBuilderError
from tvm import tirx

__all__ = ["MixedKernel"]


def MixedKernel(
    *blocks: int | tirx.PrimExpr,
    sids: int = 2,
    prelude: str | None = None,
):
    """Construct an Ascend NPU Mixed-kernel launch frame with sub-kernel ID binding.

    Parameters
    ----------
    blocks : int
        Number of blocks in the 1-D grid (blockIdx.x extent).
    sids : int
        Number of active AIV sub-cores (1 or 2). Default is 2. On dav-3510,
        mixed kernels always launch the physical ``__mix__(1, 2)`` group;
        ``sids=1`` restricts the vector body to sub-core 0. Binds as ``sid``
        via ``get_subblockid()`` in generated code.
    prelude : str, optional
        Import C code injected before the generated kernel.

    Returns
    -------
    res : KernelLaunchFrame
        The resulting frame providing ``(bx, sid)`` variable bindings.

    Examples
    --------
    .. code-block:: python

        with T.MixedKernel(NUM_BLOCKS, sids=2) as (bx, sid):
            with T.Cube():
                # AIC code using bx
                ...
            with T.Vector():
                # AIV code using both bx and sid
                ...
    """
    from tilelang.language.eager.builder import Builder
    from tilelang.ascend.target import check_ascend_availability

    if Builder.current() is None:
        raise JITNoBuilderError("T.MixedKernel() can only be used inside @tilelang.jit or @T.prim_func context. No Builder is available.")

    if not check_ascend_availability():
        raise RuntimeError("T.MixedKernel() requires an Ascend NPU environment (torch.npu.is_available() must return True).")

    if len(blocks) != 1:
        raise ValueError(f"T.MixedKernel() only supports 1-D block grid. Got {len(blocks)}-D: {blocks}")

    if sids not in (1, 2):
        raise ValueError(f"T.MixedKernel() sids must be 1 or 2. Got {sids}")

    attrs: dict = {}
    attrs["tilelang.is_npu_kernel_frame"] = True

    if prelude is not None:
        attrs["pragma_import_c"] = prelude

    return _ffi_api.MixedKernelLaunch(blocks, sids, attrs)
