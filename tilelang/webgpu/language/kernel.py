"""WebGPU dialect of ``T.Kernel``: the common launch plus WebGPU launch annotations."""

from __future__ import annotations

from tvm import tirx

from tilelang.language.kernel import KernelLaunchFrame, launch_kernel

__all__ = ["Kernel"]


def Kernel(
    *blocks: int | tirx.PrimExpr,
    threads: int | list[int] | tuple[int, ...] | None = None,
) -> KernelLaunchFrame:
    """Construct a kernel launch frame for WebGPU: a grid of workgroups.

    Code inside the launch operates at the workgroup level: ``T.Parallel``,
    ``T.copy`` and friends are mapped onto invocations by the compiler.
    ``T.get_thread_binding()`` exposes the invocation index for thread-level
    code. ``threads`` is recorded at trace time and materialized by the WebGPU
    pipeline once the target is known.

    Parameters
    ----------
    *blocks : int | PrimExpr
        Grid extent along each axis (1-3 dimensions). The launch yields one
        workgroup index per axis.
    threads : int | list[int] | tuple[int, ...], optional
        Invocations per workgroup: a count or up to three per-dimension
        extents. Defaults to 128 when omitted.

    Examples
    --------
    .. code-block:: python

        with T.Kernel(T.ceildiv(N, 128), threads=128) as bx:
            ...
    """
    return launch_kernel(blocks, threads=threads)
