"""CPU-specific transformation frontends."""

from .. import _ffi_api


def LowerCPUAtomics():
    """Lower tl.atomic_*_elem_op intrinsics to serial read-modify-write.

    CPU targets only: the pass rewrites scalar-path atomic intrinsics into
    plain BufferLoad/BufferStore RMW so that both CPU codegens (`c` and
    `llvm`) can consume them. Tile-region atomics are lowered separately by
    the cpu.AtomicAdd/cpu.AtomicReduce impls inside LowerTileOp.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LowerCPUAtomics()  # type: ignore


def MarkCPUAtomics():
    """Tag functions that call any ``tl.atomic*`` op with the
    ``tl.cpu_had_atomics`` attribute.

    Runs before LowerTileOp (both atomic forms are lowered to plain
    read-modify-write afterwards, where the tail pass could no longer see
    them). MaterializeCPUParallelGrid keeps such kernels serial: a parallel
    grid would turn the RMW into a data race.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.MarkCPUAtomics()  # type: ignore


def MaterializeCPUParallelGrid():
    """Convert the annotated CPU grid loop nest to OpenMP parallel loops.

    CPU targets only; runs at the tail of the CPU pipeline (only inserted
    when the ``tl.cpu_parallel`` pass config is enabled). Consumes the
    ``tl.cpu_grid_dim`` annotations added by MaterializeKernelLaunch,
    converts the grid loops to ForKind::kParallel (all dims on the ``c``
    target for collapse(n); the first non-unit dim on ``llvm``), and sinks
    function-scope allocations whose uses all lie inside the parallel
    region into the parallel loop body for per-worker privacy.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.MaterializeCPUParallelGrid()  # type: ignore
