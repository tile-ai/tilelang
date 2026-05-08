# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""
Grid-level synchronization for CuTeDSL backend.

Implements a software grid barrier using atomic operations on a device-global
counter declared via llvm.mlir.global. Requires cooperative kernel launch
(cuLaunchCooperativeKernel) to guarantee all thread blocks are resident.

The barrier:
1. __syncthreads() within each block
2. Thread 0 atomically increments global counter, spin-waits until all blocks arrive
3. Thread 0 resets counter
4. __syncthreads() within each block
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

    Creates: @__grid_sync_ctr = internal global i32 0, addrspace(1)
    """
    gpu_mod = _find_gpu_module_from_ip()
    if gpu_mod is None:
        raise RuntimeError("Cannot find gpu.module in MLIR hierarchy for global variable declaration")

    # Check if the global already exists
    module_body = gpu_mod.regions[0].blocks[0]
    for op in module_body.operations:
        if op.name == "llvm.mlir.global":
            sym = op.attributes.get("sym_name")
            if sym is not None and str(sym) == '"__grid_sync_ctr"':
                return

    # Insert at the beginning of the module body
    ctx = mlir_ir.Context.current
    linkage_attr = mlir_ir.Attribute.parse("#llvm.linkage<internal>", ctx)

    with mlir_ir.InsertionPoint.at_block_begin(module_body):
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
    Only thread 0 per block participates.
    """
    _ensure_global_counter(loc=loc)

    # Get address of the global counter (ptr in address space 1 = global memory)
    counter_ptr = llvm.mlir_addressof(
        llvm.PointerType.get(1),
        "__grid_sync_ctr",
        loc=loc,
        ip=ip,
    )

    # All barrier logic in one inline PTX block:
    # - Thread 0 check, grid dim computation, atomic increment, spin-wait, reset
    llvm.inline_asm(
        None,
        [counter_ptr],
        """
{
    .reg .s32 %r_tid;
    .reg .s32 %r_nctaid_x;
    .reg .s32 %r_nctaid_y;
    .reg .s32 %r_nctaid_z;
    .reg .s32 %r_num_blocks;
    .reg .s32 %r_arrived;
    .reg .pred %p_is_thread0;
    .reg .pred %p_done;

    // Check if this is thread 0
    mov.u32 %r_tid, %tid.x;
    setp.ne.s32 %p_is_thread0, %r_tid, 0;
    @%p_is_thread0 bra GRID_SYNC_DONE;

    // Compute total number of blocks
    mov.u32 %r_nctaid_x, %nctaid.x;
    mov.u32 %r_nctaid_y, %nctaid.y;
    mov.u32 %r_nctaid_z, %nctaid.z;
    mul.lo.s32 %r_num_blocks, %r_nctaid_x, %r_nctaid_y;
    mul.lo.s32 %r_num_blocks, %r_num_blocks, %r_nctaid_z;

    // Atomic increment with GPU scope
    atom.add.release.gpu.s32 %r_arrived, [$0], 1;

    // Spin until all blocks arrive (acquire GPU scope)
GRID_SYNC_SPIN:
    ld.acquire.gpu.global.s32 %r_arrived, [$0];
    setp.ge.s32 %p_done, %r_arrived, %r_num_blocks;
    @!%p_done bra GRID_SYNC_SPIN;

    // Reset counter with GPU scope
    st.release.gpu.global.s32 [$0], 0;

GRID_SYNC_DONE:
}
""",
        "l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
