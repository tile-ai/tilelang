"""CUDA dialect of ``T.Kernel``: the common launch plus CUDA launch annotations."""

from __future__ import annotations

from tvm import tirx

from tilelang.language.kernel import KernelLaunchFrame, launch_kernel

__all__ = ["Kernel"]


def Kernel(
    *blocks: int | tirx.PrimExpr,
    threads: int | list[int] | tuple[int, ...] | None = None,
    prelude: str | None = None,
    cluster_dims: int | tuple[int, int, int] | list[int] | None = None,
) -> KernelLaunchFrame:
    """Construct a kernel launch frame for CUDA: a grid of thread blocks.

    Code inside the launch operates at the block level: ``T.Parallel``,
    ``T.copy`` and friends are mapped onto threads by the compiler.
    ``T.get_thread_binding()`` exposes ``threadIdx`` for thread-level code.
    The keyword arguments are recorded at trace time and materialized by the
    CUDA pipeline once the target is known.

    Parameters
    ----------
    *blocks : int | PrimExpr
        Grid extent along each axis (1-3 dimensions, ``gridDim.(x|y|z)``). The
        launch yields one block index per axis (``blockIdx.(x|y|z)``).
    threads : int | list[int] | tuple[int, ...], optional
        Threads per block: a count for ``blockDim.x`` or up to three
        per-dimension extents for ``blockDim.(x|y|z)``. Defaults to 128 when
        omitted.
    prelude : str, optional
        CUDA source injected before the generated kernel, e.g. ``#include``
        lines or helper functions.
    cluster_dims : int | tuple[int, int, int] | list[int], optional
        Thread block cluster shape (SM90+). ``2`` or ``(2, 1, 1)`` launches
        2-CTA clusters via ``cudaLaunchKernelEx``. ``T.ClusterKernel`` is the
        same launch with a required ``cluster_dims``.

    Examples
    --------
    .. code-block:: python

        with T.Kernel(T.ceildiv(N, 128), threads=128) as bx:
            ...

        with T.Kernel(grid_x, grid_y, threads=(64, 2)) as (bx, by):
            tx, ty = T.get_thread_bindings()
            ...
    """
    return launch_kernel(blocks, threads=threads, prelude=prelude, cluster_dims=cluster_dims)
