# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""
Grid-level synchronization for CuTeDSL backend.

Implements the grid barrier of CUDA cooperative groups (``this_grid().sync()``)
on the per-launch grid workspace that the driver provides to cooperative
launches (cuLaunchCooperativeKernel). Its address is read from the
``%envreg1`` (high half) and ``%envreg2`` (low half) special registers, as in
``cooperative_groups::details::get_grid_workspace``, so concurrent launches of the same
kernel never share barrier state.

The barrier:
1. __syncthreads() within each block
2. Thread 0 of each block adds its arrival to the workspace barrier word with
   release semantics; block 0 adds the value that flips bit 31 once all
   blocks have arrived
3. Thread 0 spins with acquire loads until bit 31 differs from its arrival
4. __syncthreads() releases the rest of the block
"""

__all__ = ["sync_grid"]

from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
import cutlass.cute as cute


def sync_grid():
    """Synchronize all thread blocks in a grid.

    NOTE: This requires the kernel to be launched with cuLaunchCooperativeKernel,
    which guarantees all thread blocks are resident simultaneously and provides
    the grid workspace. The CuTeDSL wrapper handles this automatically when the
    kernel uses sync_grid().
    """
    cute.arch.sync_threads()
    _grid_sync_barrier()
    cute.arch.sync_threads()


@dsl_user_op
def _grid_sync_barrier(*, loc=None, ip=None) -> None:
    """Thread 0 of each block arrives at and waits on the grid barrier.

    Mirrors ``cooperative_groups::details::sync_grids``: the barrier word is
    ``grid_workspace::barrier`` (offset 4 of the driver workspace). Each block
    adds 1, block 0 adds ``0x80000000 - (nblocks - 1)``, so the sum of one
    round flips bit 31 and leaves the low bits unchanged for the next round.
    The release/acquire at GPU scope, combined with the surrounding block
    barriers, orders the global writes of the whole block.
    """
    llvm.inline_asm(
        None,
        [],
        """
{
    .reg .u32 %r_tid;
    .reg .u32 %r_tmp;
    .reg .u32 %r_ctaid;
    .reg .u32 %r_nblocks;
    .reg .u32 %r_lo;
    .reg .u32 %r_hi;
    .reg .u32 %r_add;
    .reg .u32 %r_old;
    .reg .u32 %r_cur;
    .reg .u64 %rd_bar;
    .reg .pred %p_not_master;
    .reg .pred %p_gpu_master;
    .reg .pred %p_wait;

    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_tmp, %tid.y;
    or.b32 %r_tid, %r_tid, %r_tmp;
    mov.u32 %r_tmp, %tid.z;
    or.b32 %r_tid, %r_tid, %r_tmp;
    setp.ne.u32 %p_not_master, %r_tid, 0;
    @%p_not_master bra GRID_SYNC_DONE;

    mov.u32 %r_hi, %envreg1;
    mov.u32 %r_lo, %envreg2;
    mov.b64 %rd_bar, {%r_lo, %r_hi};
    // No workspace: the kernel was not launched cooperatively.
    setp.eq.u64 %p_wait, %rd_bar, 0;
    @%p_wait trap;
    add.u64 %rd_bar, %rd_bar, 4;

    mov.u32 %r_nblocks, %nctaid.x;
    mov.u32 %r_tmp, %nctaid.y;
    mul.lo.u32 %r_nblocks, %r_nblocks, %r_tmp;
    mov.u32 %r_tmp, %nctaid.z;
    mul.lo.u32 %r_nblocks, %r_nblocks, %r_tmp;

    mov.u32 %r_ctaid, %ctaid.x;
    mov.u32 %r_tmp, %ctaid.y;
    or.b32 %r_ctaid, %r_ctaid, %r_tmp;
    mov.u32 %r_tmp, %ctaid.z;
    or.b32 %r_ctaid, %r_ctaid, %r_tmp;
    setp.eq.u32 %p_gpu_master, %r_ctaid, 0;
    sub.u32 %r_add, 0x80000001, %r_nblocks;
    selp.u32 %r_add, %r_add, 1, %p_gpu_master;

    atom.add.release.gpu.u32 %r_old, [%rd_bar], %r_add;

GRID_SYNC_SPIN:
    ld.acquire.gpu.u32 %r_cur, [%rd_bar];
    xor.b32 %r_cur, %r_cur, %r_old;
    and.b32 %r_cur, %r_cur, 0x80000000;
    setp.eq.u32 %p_wait, %r_cur, 0;
    @%p_wait bra GRID_SYNC_SPIN;

GRID_SYNC_DONE:
}
""",
        "~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
