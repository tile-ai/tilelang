import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import nvvm
from cutlass.cutlass_dsl import T

# re-export cutlass.cute.arch functions first
from cutlass.cute.arch import sync_threads  # noqa: F401
from cutlass.cute.arch import alloc_smem, get_dyn_smem  # noqa: F401
from cutlass.cute.arch import warpgroup_reg_alloc, warpgroup_reg_dealloc  # noqa: F401

from cutlass.cute import make_tensor, make_rmem_tensor, recast_ptr  # noqa: F401
from cutlass.cute.typing import Numeric

from cutlass.base_dsl.typing import as_numeric, Int32, Uint16, Uint32  # noqa: F401
from cutlass._mlir.dialects import llvm, arith  # noqa: F401
from cutlass._mlir import ir as mlir_ir
from cutlass.cutlass_dsl import dsl_user_op

# Import our custom implementations (will override if names conflict)
from .mbar import *
from .cpasync import *
from .gemm_V1 import *
from .reduce import *
from .ldsm import *
from .math import *
from .threadblock_swizzle import *

# Forward nvvm enums
from cutlass._mlir.dialects.nvvm import (
    MemOrderKind,
    MemScopeKind,
    AtomicOpKind,
)

BYTES_PER_TENSORMAP = 128
BYTES_PER_POINTER = 8


def make_filled_tensor(shape, value):
    t = cute.make_rmem_tensor(shape, type(value))
    t.fill(value)
    return t


def make_tensor_at_offset(ptr: cute.Pointer, offset, shape, div_by=1):
    if div_by != 1:
        offset = cute.assume(cutlass.as_numeric(offset), divby=div_by)
    return cute.make_tensor(ptr + offset, shape)


def shuffle_elect(thread_extent):
    # thread_extent is the number of threads of a warpgroup
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    if thread_extent == 0:
        return warp_idx == 0
    else:
        return (warp_idx % (thread_extent // 32)) == 0


def sync_thread_partial(barrier_id=None, thread_count=None):
    bar_sync_ptx(barrier_id, thread_count)


# Packing functions
def pack_half2(x, y):
    """
    Pack two half-precision (fp16) values into a single 32-bit value.
    Corresponds to CUDA's __pack_half2 intrinsic.

    This packs two fp16 values into a single int32 by treating the fp16 bits
    as raw data and concatenating them.
    """

    @dsl_user_op
    def pack_half2_impl(x_val, y_val, *, loc=None, ip=None):
        # Cast fp16 to uint16 (bitcast)
        x_ir = x_val.ir_value(loc=loc, ip=ip) if hasattr(x_val, "ir_value") else x_val
        y_ir = y_val.ir_value(loc=loc, ip=ip) if hasattr(y_val, "ir_value") else y_val

        # Bitcast fp16 to i16
        i16_type = mlir_ir.IntegerType.get_signless(16)
        x_i16 = llvm.bitcast(i16_type, x_ir, loc=loc, ip=ip)
        y_i16 = llvm.bitcast(i16_type, y_ir, loc=loc, ip=ip)

        packed_xy = llvm.inline_asm(
            Int32.mlir_type,
            [x_i16, y_i16],
            "mov.b32 $0, {$1, $2};",
            "=r,h,h",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

        return Int32(packed_xy)

    return pack_half2_impl(x, y)


def AtomicAdd(ptr: cute.Pointer, value: Numeric, *, loc=None, ip=None):
    if ptr.dtype == cutlass.Float32:
        ret = nvvm.atomicrmw(
            T.f32(),
            AtomicOpKind.FADD,
            ptr.llvm_ptr,
            ptr.dtype(value).ir_value(loc=loc, ip=ip),
            mem_order=MemOrderKind.RELAXED,
            syncscope=MemScopeKind.GPU,
            loc=loc,
            ip=ip,
        )
    elif ptr.dtype == cutlass.Int32:
        ret = nvvm.atomicrmw(
            T.i32(),
            AtomicOpKind.ADD,
            ptr.llvm_ptr,
            ptr.dtype(value).ir_value(loc=loc, ip=ip),
            mem_order=MemOrderKind.RELAXED,
            syncscope=MemScopeKind.GPU,
            loc=loc,
            ip=ip,
        )
    else:
        raise ValueError(f"Unsupported dtype: {ptr.dtype}")
    return ptr.dtype(ret)
