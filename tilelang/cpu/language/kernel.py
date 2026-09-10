"""CPU dialect of ``T.Kernel``: the common launch plus CPU launch annotations."""

from __future__ import annotations

from tvm import tirx

from tilelang.language.kernel import KernelLaunchFrame, launch_kernel

__all__ = ["Kernel"]


def Kernel(
    *blocks: int | tirx.PrimExpr,
    prelude: str | None = None,
) -> KernelLaunchFrame:
    """Construct a kernel launch frame for CPU: a grid of tile programs.

    The grid becomes the outer loop nest of the generated function and each
    tile program runs as a plain serial body; ``T.Parallel`` loops are lowered
    to serial loops. There are no SIMT threads, so this dialect has no
    ``threads`` and ``T.get_thread_binding()`` is rejected at compile time.

    Parameters
    ----------
    *blocks : int | PrimExpr
        Grid extent along each axis (1-3 dimensions). The launch yields one
        program index per axis.
    prelude : str, optional
        C source injected before the generated kernel, e.g. ``#include`` lines
        or helper functions.

    Examples
    --------
    .. code-block:: python

        with T.Kernel(T.ceildiv(N, 128)) as bx:
            for i in T.Parallel(128):
                ...
    """
    return launch_kernel(blocks, prelude=prelude)
