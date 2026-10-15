"""Metal dialect of ``T.Kernel``: the common launch plus Metal launch annotations."""

from __future__ import annotations

from tvm import tirx

from tilelang.language.kernel import KernelLaunchFrame, kernel_launch_factory, launch_kernel

__all__ = ["Kernel"]


@kernel_launch_factory
def Kernel(
    *blocks: int | tirx.PrimExpr,
    threads: int | list[int] | tuple[int, ...] | None = None,
    prelude: str | None = None,
) -> KernelLaunchFrame:
    """Construct a kernel launch frame for Metal: a grid of threadgroups.

    Code inside the launch operates at the threadgroup level: ``T.Parallel``,
    ``T.copy`` and friends are mapped onto threads by the compiler.
    ``T.get_thread_binding()`` exposes the thread index for thread-level code.
    The keyword arguments are recorded at trace time and materialized by the
    Metal pipeline once the target is known.

    Parameters
    ----------
    *blocks : int | PrimExpr
        Grid extent along each axis (1-3 dimensions). The launch yields one
        threadgroup index per axis.
    threads : int | list[int] | tuple[int, ...], optional
        Threads per threadgroup: a count or up to three per-dimension extents.
        Defaults to 128 when omitted.
    prelude : str, optional
        Source injected before the generated kernel, e.g. ``#include`` lines or
        helper functions.

    Examples
    --------
    .. code-block:: python

        with T.Kernel(T.ceildiv(N, 128), threads=128) as bx:
            ...
    """
    return launch_kernel(blocks, threads=threads, prelude=prelude)
