"""CPU dialect of ``T.Kernel``: the common launch plus CPU launch annotations."""

from __future__ import annotations

from tvm import tirx

from tilelang.language.kernel import KernelLaunchFrame, kernel_launch_factory, launch_kernel

__all__ = ["Kernel"]


@kernel_launch_factory
def Kernel(
    *blocks: int | tirx.PrimExpr,
    prelude: str | None = None,
    cpu_num_threads: int | None = None,
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
    cpu_num_threads : int, optional
        OpenMP thread count for the grid parallel region when the
        ``tl.cpu_parallel`` pass config is enabled: emitted as the
        ``num_threads(n)`` clause. ``None`` (default) omits the clause and
        lets the OpenMP runtime pick the thread count (e.g. from
        ``OMP_NUM_THREADS``). Only the ``c`` target consumes it — on the
        ``llvm`` target it has no effect, since that path lowers to
        ``TVMBackendParallelLaunch`` and uses TVM's own thread pool.

    Examples
    --------
    .. code-block:: python

        with T.Kernel(T.ceildiv(N, 128)) as bx:
            for i in T.Parallel(128):
                ...
    """
    annotations = {}
    if cpu_num_threads is not None:
        if isinstance(cpu_num_threads, bool) or not isinstance(cpu_num_threads, int) or cpu_num_threads <= 0:
            raise ValueError(f"cpu_num_threads must be a positive integer, got {cpu_num_threads}")
        annotations["tl.cpu_num_threads"] = tirx.IntImm("int32", cpu_num_threads)
    return launch_kernel(blocks, prelude=prelude, **annotations)
