# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""
Grid-level synchronization for CuTeDSL backend.

Implements a software grid barrier using atomic operations on a device-global
counter declared via llvm.mlir.global. Requires cooperative kernel launch
(cuLaunchCooperativeKernel) to guarantee all thread blocks are resident.

The barrier:
1. __syncthreads() within each block
2. All threads execute a device-scope fence for prior global writes
3. Thread 0 atomically increments global counter and spin-waits until release
4. All threads wait for thread 0 and execute a device-scope fence before return
"""

__all__ = ["sync_grid"]

from cutlass._mlir import ir as mlir_ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
import cutlass.cute as cute


def _find_gpu_module_from_ip():
    """Find gpu.module by walking up from current insertion point.

    Hierarchy: block -> KernelOp -> gpu.module -> builtin.module
    Uses bounded for-loop to avoid CuTeDSL DSL preprocessor issues with while/break.
    """
    current_ip = mlir_ir.InsertionPoint.current
    block = current_ip.block
    op = block.owner
    for _ in range(10):
        if op is None:
            return None
        if hasattr(op, "name") and op.name == "gpu.module":
            return op
        op = op.parent
    return None


def _ensure_global_counter(loc=None):
    """Declare the global counter variable at gpu.module level if not already done.

    Creates:
    - @__grid_sync_ctr = internal global i32 0, addrspace(1)
    - @__grid_sync_gen = internal global i32 0, addrspace(1)
    """
    gpu_mod = _find_gpu_module_from_ip()
    if gpu_mod is None:
        raise RuntimeError("Cannot find gpu.module in MLIR hierarchy for global variable declaration")

    # Check if the globals already exist.
    module_body = gpu_mod.regions[0].blocks[0]
    found_ctr = False
    found_gen = False
    for op in module_body.operations:
        if op.name == "llvm.mlir.global":
            sym = op.attributes.get("sym_name")
            if sym is not None:
                if str(sym) == '"__grid_sync_ctr"':
                    found_ctr = True
                elif str(sym) == '"__grid_sync_gen"':
                    found_gen = True
    if found_ctr and found_gen:
        return

    # Insert at the beginning of the module body
    ctx = mlir_ir.Context.current
    linkage_attr = mlir_ir.Attribute.parse("#llvm.linkage<internal>", ctx)

    with mlir_ir.InsertionPoint.at_block_begin(module_body):
        if not found_gen:
            llvm.GlobalOp(
                T.i32(),
                "__grid_sync_gen",
                linkage_attr,
                value=mlir_ir.IntegerAttr.get(T.i32(), 0),
                addr_space=1,
                loc=loc,
            )
        if not found_ctr:
            llvm.GlobalOp(
                T.i32(),
                "__grid_sync_ctr",
                linkage_attr,
                value=mlir_ir.IntegerAttr.get(T.i32(), 0),
                addr_space=1,
                loc=loc,
            )


def sync_grid():
    """Synchronize all thread blocks in a grid.

    NOTE: This requires the kernel to be launched with cuLaunchCooperativeKernel
    to guarantee all blocks are resident simultaneously. The CuTeDSL wrapper
    handles this automatically when the kernel uses sync_grid().
    """
    cute.arch.sync_threads()
    _grid_sync_barrier()
    cute.arch.sync_threads()


@dsl_user_op
def _grid_sync_barrier(*, loc=None, ip=None) -> None:
    """Software grid barrier using inline PTX.

    Declares a module-level global counter via llvm.mlir.global, then uses
    inline PTX for the barrier protocol (atomic increment + spin-wait + reset).
    Only thread 0 per block updates the global barrier state. All threads
    participate in device-scope fences around the thread-0 publish/wait path so
    global memory protected by the grid barrier is ordered for the whole block.
    """
    _ensure_global_counter(loc=loc)

    # Get addresses of the global barrier state (ptr in address space 1 = global memory)
    counter_ptr = llvm.mlir_addressof(
        llvm.PointerType.get(1),
        "__grid_sync_ctr",
        loc=loc,
        ip=ip,
    )
    generation_ptr = llvm.mlir_addressof(
        llvm.PointerType.get(1),
        "__grid_sync_gen",
        loc=loc,
        ip=ip,
    )

    # All barrier logic in one inline PTX block:
    # - Thread 0 check, grid dim computation, atomic increment, sense wait
    # - The last arriving block resets the counter and advances generation.
    #   Other blocks wait for generation to change, so they cannot miss release
    #   if the counter is reset before they observe the final arrival count.
    llvm.inline_asm(
        None,
        [counter_ptr, generation_ptr],
        """
{
    .reg .s32 %r_tid;
    .reg .s32 %r_nctaid_x;
    .reg .s32 %r_nctaid_y;
    .reg .s32 %r_nctaid_z;
    .reg .s32 %r_num_blocks;
    .reg .s32 %r_arrived;
    .reg .s32 %r_generation;
    .reg .s32 %r_new_generation;
    .reg .pred %p_is_thread0;
    .reg .pred %p_is_last;
    .reg .pred %p_done;

    // Check if this is thread 0
    mov.u32 %r_tid, %tid.x;
    // Order prior global writes from every thread before thread 0 publishes
    // this block's arrival.
    membar.gl;
    setp.ne.s32 %p_is_thread0, %r_tid, 0;
    @%p_is_thread0 bra GRID_SYNC_BLOCK_WAIT;

    // Compute total number of blocks
    mov.u32 %r_nctaid_x, %nctaid.x;
    mov.u32 %r_nctaid_y, %nctaid.y;
    mov.u32 %r_nctaid_z, %nctaid.z;
    mul.lo.s32 %r_num_blocks, %r_nctaid_x, %r_nctaid_y;
    mul.lo.s32 %r_num_blocks, %r_num_blocks, %r_nctaid_z;

    // Capture the current barrier generation before publishing arrival.
    ld.acquire.gpu.global.s32 %r_generation, [$1];

    // Atomic increment with GPU scope. atom.add returns the old value.
    atom.add.release.gpu.s32 %r_arrived, [$0], 1;
    add.s32 %r_arrived, %r_arrived, 1;

    // Last block releases the barrier by resetting the counter and advancing
    // generation. Non-last blocks wait for generation to change.
    setp.eq.s32 %p_is_last, %r_arrived, %r_num_blocks;
    @!%p_is_last bra GRID_SYNC_SPIN;

    st.release.gpu.global.s32 [$0], 0;
    add.s32 %r_new_generation, %r_generation, 1;
    st.release.gpu.global.s32 [$1], %r_new_generation;
    bra GRID_SYNC_BLOCK_WAIT;

GRID_SYNC_SPIN:
    ld.acquire.gpu.global.s32 %r_new_generation, [$1];
    setp.ne.s32 %p_done, %r_new_generation, %r_generation;
    @!%p_done bra GRID_SYNC_SPIN;

GRID_SYNC_BLOCK_WAIT:
    // Keep non-zero threads in the block from returning before thread 0 has
    // observed the grid release, then acquire global writes before user code
    // resumes after sync_grid().
    bar.sync 0;
    membar.gl;

GRID_SYNC_DONE:
}
""",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
