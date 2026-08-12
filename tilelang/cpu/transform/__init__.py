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
