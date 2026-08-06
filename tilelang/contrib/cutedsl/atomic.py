"""
Atomic operations for CuTeDSL backend.

This module provides implementations of atomic operations using NVVM and LLVM dialects.
"""

__all__ = [
    "AtomicAdd",
    "AtomicAddRet",
    "AtomicAddx2",
    "AtomicAddx4",
    "AtomicMax",
    "AtomicMaxRet",
    "AtomicMin",
    "AtomicMinRet",
    "AtomicLoad",
    "AtomicStore",
]

import cutlass
from cutlass import cute
from cutlass._mlir.extras import types as T
from cutlass._mlir.dialects import llvm
from cutlass.experimental import primitives as prims

# Type alias for numeric values
Numeric = cutlass.Float32 | cutlass.Float16 | cutlass.Int32 | cutlass.Int64 | int | float


def _atomic_ptr(ptr):
    return ptr.llvm_ptr if hasattr(ptr, "llvm_ptr") else ptr


def _fence_sc_gpu(*, loc=None, ip=None) -> None:
    prims.inline_ptx("fence.sc.gpu;", loc=loc, ip=ip)


# =============================================================================
# AtomicAdd - Scalar atomic addition
# =============================================================================


def AtomicAdd(ptr: cute.Pointer, value: Numeric, *, loc=None, ip=None):
    """Perform atomic addition on a pointer.

    Supports float16, float32, int32, and int64 types.
    Returns the old value before addition (atomicrmw semantics).
    """
    if ptr.dtype == cutlass.Float32:
        ret = prims.atomicrmw(
            prims.AtomicOp.ADD,
            _atomic_ptr(ptr),
            ptr.dtype(value),
            mem_order=prims.MemOrder.RELAXED,
            syncscope=prims.MemScope.GPU,
            loc=loc,
            ip=ip,
        )
        return ptr.dtype(ret)
    elif ptr.dtype == cutlass.Float16:
        # For float16, use inline PTX: atom.add.noftz.f16 (no .global qualifier)
        # LLVM inline_asm doesn't support f16 directly, so we bitcast to i16
        # The PTX syntax is: atom.add.noftz.f16 result, [ptr], value
        val_ir = cutlass.Float16(value).ir_value(loc=loc, ip=ip)
        # Bitcast f16 -> i16 for inline asm compatibility
        val_i16 = llvm.bitcast(T.i16(), val_ir, loc=loc, ip=ip)
        res_i16 = llvm.inline_asm(
            T.i16(),
            [ptr.llvm_ptr, val_i16],
            "atom.add.noftz.f16 $0, [$1], $2;",
            "=h,l,h",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
        # Bitcast i16 -> f16 for return
        result = llvm.bitcast(T.f16(), res_i16, loc=loc, ip=ip)
        return cutlass.Float16(result)
    elif ptr.dtype == cutlass.Int32 or ptr.dtype == cutlass.Int64:
        ret = prims.atomicrmw(
            prims.AtomicOp.ADD,
            _atomic_ptr(ptr),
            ptr.dtype(value),
            mem_order=prims.MemOrder.RELAXED,
            syncscope=prims.MemScope.GPU,
            loc=loc,
            ip=ip,
        )
        return ptr.dtype(ret)
    else:
        raise ValueError(f"Unsupported dtype for AtomicAdd: {ptr.dtype}")


def AtomicAddRet(ptr: cute.Pointer, value: Numeric, *, loc=None, ip=None):
    """Perform atomic addition and return the previous value.

    This is the same as AtomicAdd since prims.atomicrmw always returns old value.
    """
    return AtomicAdd(ptr, value, loc=loc, ip=ip)


# =============================================================================
# AtomicAddx2/x4 - Vectorized atomic addition
# =============================================================================


def _load_from_src(src_values, count):
    """Load elements from src_values, handling both TensorSSA and _Pointer types."""
    if isinstance(src_values, cute.Pointer):
        # Create a tensor from pointer and load elements
        src_tensor = cute.make_tensor(src_values, cute.make_layout((count,)))
        return [src_tensor[i] for i in range(count)]
    return [src_values[i] for i in range(count)]


def AtomicAddx2(dst_ptr: cute.Pointer, src_values, *, loc=None, ip=None):
    """Vectorized atomic add for 2 consecutive elements.

    Uses PTX atom.add.v2.f32 for float32 or atom.add.noftz.v2.f16 for float16.

    Args:
        dst_ptr: Pointer to destination (2 consecutive elements)
        src_values: Source values - can be TensorSSA (loaded tensor) or Pointer
    """
    vals = _load_from_src(src_values, 2)
    val0 = vals[0]
    val1 = vals[1]

    if dst_ptr.dtype == cutlass.Float16:
        # fp16: use atom.add.noftz.v2.f16 with i16 bitcast (LLVM asm doesn't support f16)
        val0_ir = cutlass.Float16(val0).ir_value(loc=loc, ip=ip)
        val1_ir = cutlass.Float16(val1).ir_value(loc=loc, ip=ip)
        val0_i16 = llvm.bitcast(T.i16(), val0_ir, loc=loc, ip=ip)
        val1_i16 = llvm.bitcast(T.i16(), val1_ir, loc=loc, ip=ip)
        res_type = llvm.StructType.get_literal([T.i16()] * 2)
        llvm.inline_asm(
            res_type,
            [dst_ptr.llvm_ptr, val0_i16, val1_i16],
            "atom.add.noftz.v2.f16 {$0,$1}, [$2], {$3,$4};",
            "=h,=h,l,h,h",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    elif dst_ptr.dtype == cutlass.BFloat16:
        # bf16: use atom.add.noftz.v2.bf16 with i16 bitcast
        val0_ir = cutlass.BFloat16(val0).ir_value(loc=loc, ip=ip)
        val1_ir = cutlass.BFloat16(val1).ir_value(loc=loc, ip=ip)
        val0_i16 = llvm.bitcast(T.i16(), val0_ir, loc=loc, ip=ip)
        val1_i16 = llvm.bitcast(T.i16(), val1_ir, loc=loc, ip=ip)
        res_type = llvm.StructType.get_literal([T.i16()] * 2)
        llvm.inline_asm(
            res_type,
            [dst_ptr.llvm_ptr, val0_i16, val1_i16],
            "atom.add.noftz.v2.bf16 {$0,$1}, [$2], {$3,$4};",
            "=h,=h,l,h,h",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    else:
        # float32 (default): use atom.add.v2.f32
        res_type = llvm.StructType.get_literal([T.f32()] * 2)
        llvm.inline_asm(
            res_type,
            [dst_ptr.llvm_ptr, cutlass.Float32(val0).ir_value(loc=loc, ip=ip), cutlass.Float32(val1).ir_value(loc=loc, ip=ip)],
            "atom.add.v2.f32 {$0,$1}, [$2], {$3,$4};",
            "=f,=f,l,f,f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )


def AtomicAddx4(dst_ptr: cute.Pointer, src_values, *, loc=None, ip=None):
    """Vectorized atomic add for 4 consecutive float32 elements.

    Uses PTX atom.global.add.v4.f32 for true vectorized atomic operation on SM90+.

    Args:
        dst_ptr: Pointer to destination (4 consecutive float32 elements)
        src_values: Source values - can be TensorSSA (loaded tensor) or Pointer
    """
    vals = _load_from_src(src_values, 4)
    val0 = vals[0]
    val1 = vals[1]
    val2 = vals[2]
    val3 = vals[3]

    # Use inline PTX for vectorized atomic add
    res_type = llvm.StructType.get_literal([T.f32()] * 4)
    llvm.inline_asm(
        res_type,
        [
            dst_ptr.llvm_ptr,
            cutlass.Float32(val0).ir_value(loc=loc, ip=ip),
            cutlass.Float32(val1).ir_value(loc=loc, ip=ip),
            cutlass.Float32(val2).ir_value(loc=loc, ip=ip),
            cutlass.Float32(val3).ir_value(loc=loc, ip=ip),
        ],
        "atom.global.add.v4.f32 {$0,$1,$2,$3}, [$4], {$5,$6,$7,$8};",
        "=f,=f,=f,=f,l,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# =============================================================================
# AtomicMax - Scalar atomic maximum
# =============================================================================


def AtomicMax(ptr: cute.Pointer, value: Numeric, *, loc=None, ip=None):
    """Perform atomic maximum operation.

    For integers, uses prims.atomicrmw with MAX.
    For floats, uses CAS loop since PTX doesn't have atomic max for float32.
    """
    if ptr.dtype == cutlass.Int32:
        ret = prims.atomicrmw(
            prims.AtomicOp.MAX,
            _atomic_ptr(ptr),
            ptr.dtype(value),
            mem_order=prims.MemOrder.RELAXED,
            syncscope=prims.MemScope.GPU,
            loc=loc,
            ip=ip,
        )
        return ptr.dtype(ret)
    elif ptr.dtype == cutlass.Float32:
        # For float32, use atomicCAS loop via inline PTX
        # This implements: atomicMax for float using compare-and-swap
        # PTX doesn't have atom.max.f32, so we use atom.cas loop
        val_ir = cutlass.Float32(value).ir_value(loc=loc, ip=ip)
        # Use inline PTX with a CAS loop for float max
        # The PTX instruction is: atom.global.cas.b32
        # We load, compare with max, then CAS until success
        # NOTE: The retry comparison uses integer (b32) domain instead of
        # floating-point to avoid infinite loops when values are NaN
        # (IEEE 754: NaN != NaN is always true in float domain).
        result = llvm.inline_asm(
            T.f32(),
            [ptr.llvm_ptr, val_ir],
            """
            {
                .reg .pred p;
                .reg .f32 expected, new_val;
                .reg .b32 expected_bits, new_bits, result_bits;
                ld.f32 expected, [$1];
            retry:
                max.f32 new_val, expected, $2;
                mov.b32 expected_bits, expected;
                mov.b32 new_bits, new_val;
                atom.cas.b32 result_bits, [$1], expected_bits, new_bits;
                setp.ne.b32 p, result_bits, expected_bits;
                mov.b32 expected, result_bits;
                @p bra retry;
                mov.f32 $0, expected;
            }
            """,
            "=f,l,f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
        return cutlass.Float32(result)
    elif ptr.dtype == cutlass.Int64:
        ret = prims.atomicrmw(
            prims.AtomicOp.MAX,
            _atomic_ptr(ptr),
            ptr.dtype(value),
            mem_order=prims.MemOrder.RELAXED,
            syncscope=prims.MemScope.GPU,
            loc=loc,
            ip=ip,
        )
        return ptr.dtype(ret)
    else:
        raise ValueError(f"Unsupported dtype for AtomicMax: {ptr.dtype}")


def AtomicMaxRet(ptr: cute.Pointer, value: Numeric, *, loc=None, ip=None):
    """Perform atomic maximum and return the previous value."""
    return AtomicMax(ptr, value, loc=loc, ip=ip)


# =============================================================================
# AtomicMin - Scalar atomic minimum
# =============================================================================


def AtomicMin(ptr: cute.Pointer, value: Numeric, *, loc=None, ip=None):
    """Perform atomic minimum operation.

    For integers, uses prims.atomicrmw with MIN.
    For floats, uses CAS loop since PTX doesn't have atomic min for float32.
    """
    if ptr.dtype == cutlass.Int32:
        ret = prims.atomicrmw(
            prims.AtomicOp.MIN,
            _atomic_ptr(ptr),
            ptr.dtype(value),
            mem_order=prims.MemOrder.RELAXED,
            syncscope=prims.MemScope.GPU,
            loc=loc,
            ip=ip,
        )
        return ptr.dtype(ret)
    elif ptr.dtype == cutlass.Float32:
        # For float32, use atomicCAS loop via inline PTX
        # PTX doesn't have atom.min.f32, so we use atom.cas loop
        # NOTE: The retry comparison uses integer (b32) domain instead of
        # floating-point to avoid infinite loops when values are NaN.
        val_ir = cutlass.Float32(value).ir_value(loc=loc, ip=ip)
        result = llvm.inline_asm(
            T.f32(),
            [ptr.llvm_ptr, val_ir],
            """
            {
                .reg .pred p;
                .reg .f32 expected, new_val;
                .reg .b32 expected_bits, new_bits, result_bits;
                ld.f32 expected, [$1];
            retry:
                min.f32 new_val, expected, $2;
                mov.b32 expected_bits, expected;
                mov.b32 new_bits, new_val;
                atom.cas.b32 result_bits, [$1], expected_bits, new_bits;
                setp.ne.b32 p, result_bits, expected_bits;
                mov.b32 expected, result_bits;
                @p bra retry;
                mov.f32 $0, expected;
            }
            """,
            "=f,l,f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
        return cutlass.Float32(result)
    elif ptr.dtype == cutlass.Int64:
        ret = prims.atomicrmw(
            prims.AtomicOp.MIN,
            _atomic_ptr(ptr),
            ptr.dtype(value),
            mem_order=prims.MemOrder.RELAXED,
            syncscope=prims.MemScope.GPU,
            loc=loc,
            ip=ip,
        )
        return ptr.dtype(ret)
    else:
        raise ValueError(f"Unsupported dtype for AtomicMin: {ptr.dtype}")


def AtomicMinRet(ptr: cute.Pointer, value: Numeric, *, loc=None, ip=None):
    """Perform atomic minimum and return the previous value."""
    return AtomicMin(ptr, value, loc=loc, ip=ip)


# =============================================================================
# AtomicLoad - Atomic load with memory ordering
# =============================================================================


def _get_load_ordering(memory_order: int) -> prims.MemOrder:
    """Get primitive memory ordering for load.

    TileLang memory order:
        0: relaxed -> relaxed.gpu
        1: consume -> acquire.gpu (consume deprecated)
        2: acquire -> acquire.gpu
        3: release -> acquire.gpu (invalid for load)
        4: acq_rel -> acquire.gpu (load part)
        5: seq_cst -> fence.sc.gpu + relaxed.gpu
    """
    if memory_order == 5:
        return prims.MemOrder.RELAXED
    elif memory_order in (1, 2, 3, 4):
        return prims.MemOrder.ACQUIRE
    else:  # 0 or default
        return prims.MemOrder.RELAXED


def AtomicLoad(ptr: cute.Pointer, memory_order: int, *, loc=None, ip=None):
    """Perform atomic load with specified memory ordering.

    Args:
        ptr: Pointer to load from
        memory_order: TileLang memory order ID (0=relaxed, 2=acquire, 5=seq_cst, etc.)

    Returns:
        The loaded value

    NVVM mapping:
        relaxed: relaxed.gpu
        acquire: acquire.gpu
        seq_cst: fence.sc.gpu + relaxed.gpu
    """
    if memory_order == 5:
        _fence_sc_gpu(loc=loc, ip=ip)

    if ptr.dtype == cutlass.Int32:
        result = prims.load_ext(
            _atomic_ptr(ptr),
            dtype=cutlass.Int32,
            order=_get_load_ordering(memory_order),
            scope=prims.MemScope.GPU,
            loc=loc,
            ip=ip,
        )
        return cutlass.Int32(result)
    elif ptr.dtype == cutlass.Float32:
        result = prims.load_ext(
            _atomic_ptr(ptr),
            dtype=cutlass.Float32,
            order=_get_load_ordering(memory_order),
            scope=prims.MemScope.GPU,
            loc=loc,
            ip=ip,
        )
        return cutlass.Float32(result)
    elif ptr.dtype == cutlass.Int64:
        result = prims.load_ext(
            _atomic_ptr(ptr),
            dtype=cutlass.Int64,
            order=_get_load_ordering(memory_order),
            scope=prims.MemScope.GPU,
            loc=loc,
            ip=ip,
        )
        return cutlass.Int64(result)
    else:
        raise ValueError(f"Unsupported dtype for AtomicLoad: {ptr.dtype}")


# =============================================================================
# AtomicStore - Atomic store with memory ordering
# =============================================================================


def _get_store_ordering(memory_order: int) -> prims.MemOrder:
    """Get primitive memory ordering for store.

    TileLang memory order:
        0: relaxed -> relaxed.gpu
        1: consume -> release.gpu (invalid for store)
        2: acquire -> release.gpu (invalid for store)
        3: release -> release.gpu
        4: acq_rel -> release.gpu (store part)
        5: seq_cst -> fence.sc.gpu + relaxed.gpu
    """
    if memory_order == 5:
        return prims.MemOrder.RELAXED
    elif memory_order in (1, 2, 3, 4):
        return prims.MemOrder.RELEASE
    else:  # 0 or default
        return prims.MemOrder.RELAXED


def AtomicStore(ptr: cute.Pointer, value: Numeric, memory_order: int, *, loc=None, ip=None):
    """Perform atomic store with specified memory ordering.

    Args:
        ptr: Pointer to store to
        value: Value to store
        memory_order: TileLang memory order ID (0=relaxed, 3=release, 5=seq_cst, etc.)

    NVVM mapping:
        relaxed: relaxed.gpu
        release: release.gpu
        seq_cst: fence.sc.gpu + relaxed.gpu
    """
    if memory_order == 5:
        _fence_sc_gpu(loc=loc, ip=ip)

    if ptr.dtype == cutlass.Int32:
        prims.store_ext(
            cutlass.Int32(value).ir_value(loc=loc, ip=ip),
            _atomic_ptr(ptr),
            order=_get_store_ordering(memory_order),
            scope=prims.MemScope.GPU,
            loc=loc,
            ip=ip,
        )
    elif ptr.dtype == cutlass.Float32:
        prims.store_ext(
            cutlass.Float32(value).ir_value(loc=loc, ip=ip),
            _atomic_ptr(ptr),
            order=_get_store_ordering(memory_order),
            scope=prims.MemScope.GPU,
            loc=loc,
            ip=ip,
        )
    elif ptr.dtype == cutlass.Int64:
        prims.store_ext(
            cutlass.Int64(value).ir_value(loc=loc, ip=ip),
            _atomic_ptr(ptr),
            order=_get_store_ordering(memory_order),
            scope=prims.MemScope.GPU,
            loc=loc,
            ip=ip,
        )
    else:
        raise ValueError(f"Unsupported dtype for AtomicStore: {ptr.dtype}")
