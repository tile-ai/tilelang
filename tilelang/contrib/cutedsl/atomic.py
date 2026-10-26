"""
Atomic operations for CuTeDSL backend.

This module provides implementations of atomic operations using NVVM and LLVM dialects.
"""

__all__ = [
    "AtomicAdd",
    "AtomicAddRet",
    "AtomicAddx2",
    "AtomicAddx2Ret",
    "AtomicAddx4",
    "AtomicAddx4Ret",
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
from cutlass.cutlass_dsl import CuTeDSL
from cutlass.experimental import primitives as prims

from .utils import bitcast

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


def _arch_at_least(sm):
    arch = CuTeDSL._get_dsl().envar.arch
    return int(arch[3:].rstrip("af")) >= sm


# Bodies of the packed pair atomics, mirroring CUDA's ``atomicAdd(half2 *)`` and
# ``atomicAdd(__nv_bfloat162 *)``: the two 16-bit lanes travel as one b32 word.
_PACKED2_ATOM = """
{
    .reg .b32 v, r;
    mov.b32 v, {$3,$4};
    atom.add.noftz.%s r, [$2], v;
    mov.b32 {$0,$1}, r;
}
"""
# Below sm_90 there is no bf16x2 atomic add; cuda_bf16.hpp falls back to a
# 32-bit CAS loop around ``__hadd2``, which itself is ``fma.rn.bf16x2`` with 1.0.
_BF16X2_CAS = """
{
    .reg .b32 v, one, old, assumed, sum;
    .reg .pred p;
    mov.b32 v, {$3,$4};
    mov.b32 one, 0x3f803f80;
    ld.b32 old, [$2];
retry:
    mov.b32 assumed, old;
    fma.rn.bf16x2 sum, v, one, assumed;
    atom.cas.b32 old, [$2], assumed, sum;
    setp.ne.b32 p, old, assumed;
    @p bra retry;
    mov.b32 {$0,$1}, old;
}
"""


def _generic_addr(ptr: cute.Pointer, *, loc=None, ip=None):
    """Address operand for a generic-addressing ``atom``: shared pointers are
    converted (``cvta``), global ones are already valid generic addresses."""
    if ptr.memspace == cutlass.AddressSpace.smem:
        return llvm.addrspacecast(llvm.PointerType.get(), ptr.llvm_ptr, loc=loc, ip=ip)
    return ptr.llvm_ptr


def _atomic_add_packed2_ret(dst_ptr: cute.Pointer, v0, v1, dtype, *, loc=None, ip=None):
    """Atomic add of two adjacent 16-bit lanes; returns the previous lanes as i16 bits.

    Uses generic addressing like CUDA's ``atomicAdd(half2 *)``, so ``dst_ptr``
    may point to global or shared memory.
    """
    is_f16 = dtype == cutlass.Float16
    cast = cutlass.Float16 if is_f16 else cutlass.BFloat16
    if is_f16:
        asm = _PACKED2_ATOM % "f16x2"
    elif _arch_at_least(90):
        asm = _PACKED2_ATOM % "bf16x2"
    else:
        asm = _BF16X2_CAS
    v0_i16 = llvm.bitcast(T.i16(), cast(v0).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    v1_i16 = llvm.bitcast(T.i16(), cast(v1).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    res_type = llvm.StructType.get_literal([T.i16()] * 2)
    prev = llvm.inline_asm(
        res_type,
        [_generic_addr(dst_ptr, loc=loc, ip=ip), v0_i16, v1_i16],
        asm,
        "=h,=h,l,h,h",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        llvm.extractvalue(T.i16(), prev, [0], loc=loc, ip=ip),
        llvm.extractvalue(T.i16(), prev, [1], loc=loc, ip=ip),
    )


def _f32_vector_atomic(dst_ptr: cute.Pointer, vals, *, loc=None, ip=None):
    """``atom.global.add.v{2,4}.f32`` (sm_90+, global memory only); returns the old lanes."""
    n = len(vals)
    outs = ",".join(f"${i}" for i in range(n))
    ins = ",".join(f"${i + n + 1}" for i in range(n))
    prev = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * n),
        [dst_ptr.llvm_ptr] + [cutlass.Float32(v).ir_value(loc=loc, ip=ip) for v in vals],
        f"atom.global.add.v{n}.f32 {{{outs}}}, [${n}], {{{ins}}};",
        ",".join(["=f"] * n + ["l"] + ["f"] * n),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return [llvm.extractvalue(T.f32(), prev, [i], loc=loc, ip=ip) for i in range(n)]


def _v4_16bit_atomic(dst_ptr: cute.Pointer, vals, dtype, *, loc=None, ip=None):
    """``atom.global.v4.{f16,bf16}.add.noftz`` (sm_90+, global memory only)."""
    cast = cutlass.Float16 if dtype == cutlass.Float16 else cutlass.BFloat16
    kind = "f16" if dtype == cutlass.Float16 else "bf16"
    bits = [llvm.bitcast(T.i16(), cast(v).ir_value(loc=loc, ip=ip), loc=loc, ip=ip) for v in vals]
    llvm.inline_asm(
        llvm.StructType.get_literal([T.i16()] * 4),
        [dst_ptr.llvm_ptr] + bits,
        f"atom.global.v4.{kind}.add.noftz {{$0,$1,$2,$3}}, [$4], {{$5,$6,$7,$8}};",
        "=h,=h,=h,=h,l,h,h,h,h",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def AtomicAddx2(dst_ptr: cute.Pointer, src_values, *, loc=None, ip=None):
    """Vectorized atomic add for 2 consecutive elements.

    Follows ``AtomicAddx2`` in ``src/tl_templates/cuda/atomic.h``: float16 and
    bfloat16 use one packed ``f16x2`` / ``bf16x2`` atomic (a CAS loop for bf16
    below sm_90), which also works on shared memory; float32 uses
    ``atom.global.add.v2.f32`` on sm_90+ and two scalar atomics below.

    Args:
        dst_ptr: Pointer to destination (2 consecutive elements)
        src_values: Source values - can be TensorSSA (loaded tensor) or Pointer
    """
    vals = _load_from_src(src_values, 2)
    if dst_ptr.dtype in (cutlass.Float16, cutlass.BFloat16):
        _atomic_add_packed2_ret(dst_ptr, vals[0], vals[1], dst_ptr.dtype, loc=loc, ip=ip)
    elif _arch_at_least(90):
        _f32_vector_atomic(dst_ptr, vals, loc=loc, ip=ip)
    else:
        for i in range(2):
            AtomicAdd(dst_ptr + i, cutlass.Float32(vals[i]), loc=loc, ip=ip)


def AtomicAddx2Ret(dst_ptr: cute.Pointer, src_values, *, loc=None, ip=None):
    """Vectorized atomic add for 2 consecutive elements, returning the previous values.

    Follows ``AtomicAddx2Ret`` in ``src/tl_templates/cuda/atomic.h``:

    * float16/bfloat16: one packed ``f16x2`` / ``bf16x2`` atomic (a CAS loop for
      bf16 below sm_90) with generic addressing, so shared memory works too.
    * float32: one ``atom.global.add.v2.f32`` on sm_90+, two scalar atomics below.

    Args:
        dst_ptr: Pointer to destination (2 consecutive elements)
        src_values: Source values - can be TensorSSA (loaded tensor) or Pointer
    """
    vals = _load_from_src(src_values, 2)
    if dst_ptr.dtype in (cutlass.Float16, cutlass.BFloat16):
        lo, hi = _atomic_add_packed2_ret(dst_ptr, vals[0], vals[1], dst_ptr.dtype, loc=loc, ip=ip)
        prev_bits = cute.make_rmem_tensor((2,), cutlass.Int16)
        prev_bits[0] = cutlass.Int16(lo)
        prev_bits[1] = cutlass.Int16(hi)
        target = "float16" if dst_ptr.dtype == cutlass.Float16 else "bfloat16"
        return bitcast(prev_bits.load(), target)

    if dst_ptr.dtype != cutlass.Float32:
        raise NotImplementedError(f"AtomicAddx2Ret does not support a {dst_ptr.dtype} destination")

    out = cute.make_rmem_tensor((2,), cutlass.Float32)
    if _arch_at_least(90):
        prev = _f32_vector_atomic(dst_ptr, vals, loc=loc, ip=ip)
        for i in range(2):
            out[i] = cutlass.Float32(prev[i])
    else:
        for i in range(2):
            out[i] = AtomicAdd(dst_ptr + i, cutlass.Float32(vals[i]), loc=loc, ip=ip)
    return out.load()


def AtomicAddx4(dst_ptr: cute.Pointer, src_values, *, loc=None, ip=None):
    """Vectorized atomic add for 4 consecutive elements.

    Follows ``AtomicAddx4`` in ``src/tl_templates/cuda/atomic.h``. On sm_90+ it
    is one ``atom.global`` vector atomic (``v4.f32`` or ``v4.{f16,bf16}``);
    below that, float32 becomes four scalar atomics and 16-bit types two packed
    pair atomics.

    Args:
        dst_ptr: Pointer to destination (4 consecutive elements)
        src_values: Source values - can be TensorSSA (loaded tensor) or Pointer
    """
    vals = _load_from_src(src_values, 4)
    is_16bit = dst_ptr.dtype in (cutlass.Float16, cutlass.BFloat16)
    if _arch_at_least(90):
        if is_16bit:
            _v4_16bit_atomic(dst_ptr, vals, dst_ptr.dtype, loc=loc, ip=ip)
        else:
            _f32_vector_atomic(dst_ptr, vals, loc=loc, ip=ip)
    elif is_16bit:
        for base in (0, 2):
            _atomic_add_packed2_ret(dst_ptr + base, vals[base], vals[base + 1], dst_ptr.dtype, loc=loc, ip=ip)
    else:
        for i in range(4):
            AtomicAdd(dst_ptr + i, cutlass.Float32(vals[i]), loc=loc, ip=ip)


def AtomicAddx4Ret(dst_ptr: cute.Pointer, src_values, *, loc=None, ip=None):
    """Vectorized atomic add for 4 consecutive elements, returning the previous values.

    Follows ``AtomicAddx4Ret`` in ``src/tl_templates/cuda/atomic.h``:

    * float32: one ``atom.global.add.v4.f32`` on sm_90+, four scalar atomics below.
    * float16/bfloat16: no x4 atomic exists, so two packed pair atomics
      (``f16x2`` / ``bf16x2``; a CAS loop for bf16 below sm_90). The 8-byte
      destination is never written as 16 bytes.

    Args:
        dst_ptr: Pointer to destination (4 consecutive elements)
        src_values: Source values - can be TensorSSA (loaded tensor) or Pointer
    """
    if dst_ptr.dtype in (cutlass.Float16, cutlass.BFloat16):
        vals = _load_from_src(src_values, 4)
        prev_bits = cute.make_rmem_tensor((4,), cutlass.Int16)
        for base in (0, 2):
            lo, hi = _atomic_add_packed2_ret(dst_ptr + base, vals[base], vals[base + 1], dst_ptr.dtype, loc=loc, ip=ip)
            prev_bits[base] = cutlass.Int16(lo)
            prev_bits[base + 1] = cutlass.Int16(hi)
        target = "float16" if dst_ptr.dtype == cutlass.Float16 else "bfloat16"
        return bitcast(prev_bits.load(), target)

    if dst_ptr.dtype != cutlass.Float32:
        raise NotImplementedError(f"AtomicAddx4Ret does not support a {dst_ptr.dtype} destination")

    vals = _load_from_src(src_values, 4)
    out = cute.make_rmem_tensor((4,), cutlass.Float32)
    if _arch_at_least(90):
        prev = _f32_vector_atomic(dst_ptr, vals, loc=loc, ip=ip)
        for i in range(4):
            out[i] = cutlass.Float32(prev[i])
    else:
        for i in range(4):
            out[i] = AtomicAdd(dst_ptr + i, cutlass.Float32(vals[i]), loc=loc, ip=ip)
    return out.load()


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
