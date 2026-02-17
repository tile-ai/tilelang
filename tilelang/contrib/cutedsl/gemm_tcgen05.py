"""
tcgen05 (SM100/Blackwell) MMA support for CuTeDSL backend.

Provides:
  - Tcgen05SmemDescriptor: 64-bit SMEM descriptor for tcgen05 MMA
  - initialize_tcgen05_descriptor: bitfield packing matching common.h layout
  - tcgen05mma_ss / tcgen05mma_ws_ss / tcgen05mma_ts: MMA PTX inline asm
  - tcgen05_mma_arrive: mbarrier arrive for MMA commit
  - tmem_allocate / tmem_deallocate: TMEM allocation/deallocation
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass._mlir import ir
from cutlass.cutlass_dsl import Constexpr, dsl_user_op


# ──────────────────────────────────────────────────────────────────────
# Tcgen05 SMEM Descriptor
# ──────────────────────────────────────────────────────────────────────

class Tcgen05SmemDescriptor:
    """64-bit shared-memory descriptor for tcgen05 MMA (Blackwell).

    Mirrors tl::Tcgen05SMemDescriptor from common.h.
    Stored as two Int32 registers; recast to Int64 for the PTX operand.
    """

    def __init__(self, desc_64: cute.Int64 = None):
        self.desc = cute.make_rmem_tensor((2,), dtype=cutlass.Int32)
        self.desc_i64 = cute.make_tensor(
            cute.recast_ptr(self.desc.iterator, dtype=cute.Int64), (1,))
        if desc_64 is not None:
            self.desc_i64[0] = desc_64

    def __add__(self, offset):
        """Add byte offset.  Like C++ operator+, shifts offset >> 4."""
        res = cute.make_rmem_tensor((2,), dtype=cutlass.Int32)
        res_i64 = cute.make_tensor(
            cute.recast_ptr(res.iterator, dtype=cute.Int64), (1,))
        # Address is in 16-byte units: add (offset >> 4)
        res[0] = self.desc[0] + (offset >> 4)
        res[1] = self.desc[1]
        return Tcgen05SmemDescriptor(res_i64[0])


# ──────────────────────────────────────────────────────────────────────
# Descriptor initialization
# ──────────────────────────────────────────────────────────────────────

def initialize_tcgen05_descriptor(desc, start_address, leading_byte_offset,
                                  stride_byte_offset, base_offset,
                                  leading_abs, swizzle_mode):
    """Pack the tcgen05 SMEM descriptor bitfields.

    Matches the C++ ``initialize_tcgen05_descriptor`` in common.h:
      Low 32 bits (reg32_[0]):
        [0:14)   start_address >> 4
        [16:30)  leading_byte_offset  (already >>4 from TIR)
      High 32 bits (reg32_[1]):
        [0:14)   stride_byte_offset   (already >>4 from TIR)
        [14:16)  version = 1
        [17:20)  base_offset & 0x7
        [20:21)  lbo_mode (leading_is_absolute ? 1 : 0)
        [29:32)  layout_type (swizzle_mode & 0x7)
    """
    ptr_val = start_address.toint() >> 4
    desc.desc[0] = (cutlass.Int32(ptr_val)
                    | cutlass.Int32(cutlass.Int32(leading_byte_offset) << 16))
    desc.desc[1] = (cutlass.Int32(stride_byte_offset)
                    | cutlass.Int32(1 << 14)                       # version = 1
                    | cutlass.Int32(cutlass.Int32(base_offset & 0x7) << 17)
                    | cutlass.Int32(cutlass.Int32(leading_abs) << 20)
                    | cutlass.Int32(cutlass.Int32(swizzle_mode & 0x7) << 29))


# ──────────────────────────────────────────────────────────────────────
# PTX kind mapping  (TIR dtype string  ->  PTX kind suffix)
# ──────────────────────────────────────────────────────────────────────

_TCGEN05_KIND_MAP = {
    "fp16": "f16", "bf16": "f16",
    "float16": "f16", "bfloat16": "f16",
    "tf32": "tf32", "float32": "tf32",
    "s8": "i8", "u8": "i8",
    "int8": "i8", "uint8": "i8",
    "e4m3": "f8f6f4", "e5m2": "f8f6f4",
    "float8_e4m3": "f8f6f4", "float8_e4m3fn": "f8f6f4",
    "float8_e5m2": "f8f6f4",
}


def _kind_for(dtype_str):
    kind = _TCGEN05_KIND_MAP.get(dtype_str)
    if kind is None:
        raise ValueError(f"tcgen05mma: unsupported dtype '{dtype_str}'")
    return kind


def _ir(val, loc=None, ip=None):
    """Extract MLIR IR value from a CuTeDSL value."""
    return val.ir_value(loc=loc, ip=ip) if hasattr(val, 'ir_value') else val


# ──────────────────────────────────────────────────────────────────────
# tcgen05mma_ss  —  both A and B from SMEM descriptors (non-WS)
# ──────────────────────────────────────────────────────────────────────

@cute.jit
def tcgen05mma_ss(kind_dtype: str,
                  desc_a: Tcgen05SmemDescriptor,
                  desc_b: Tcgen05SmemDescriptor,
                  tmem_c: int,
                  desc_val: int,
                  scale_out: int,
                  mask0: int, mask1: int, mask2: int, mask3: int):
    """tcgen05.mma.cta_group::1.kind::{kind} [tmem_c], desc_a, desc_b, desc_val, {masks}, p;

    Guarded by elect_one_sync — only one thread in the warp issues the MMA.
    The TIR codegen also wraps calls in ``if (threadIdx.x >> 5) == 0``
    which selects warp 0.
    """
    kind = _kind_for(kind_dtype)

    # elect.sync selects one thread in the warp to issue the MMA.
    # The @q predicate goes on the MMA instruction itself (not the block scope).
    asm_str = (
        "{\n"
        ".reg .pred p;\n"
        ".reg .pred q;\n"
        "elect.sync _|q, 0xFFFFFFFF;\n"
        "setp.ne.b32 p, $4, 0;\n"
        f"@q tcgen05.mma.cta_group::1.kind::{kind} "
        "[$0], $1, $2, $3, {$5, $6, $7, $8}, p;\n"
        "}"
    )

    @dsl_user_op
    def _do_mma(c_val, da_val, db_val, dv_val, sc_val,
                m0_val, m1_val, m2_val, m3_val, *, loc=None, ip=None):
        llvm.inline_asm(
            None,
            [_ir(c_val, loc, ip), _ir(da_val, loc, ip),
             _ir(db_val, loc, ip), _ir(dv_val, loc, ip),
             _ir(sc_val, loc, ip), _ir(m0_val, loc, ip),
             _ir(m1_val, loc, ip), _ir(m2_val, loc, ip),
             _ir(m3_val, loc, ip)],
            asm_str,
            "r,l,l,r,r,r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    _do_mma(
        cutlass.Int32(tmem_c),
        desc_a.desc_i64[0],
        desc_b.desc_i64[0],
        cutlass.Int32(desc_val),
        cutlass.Int32(scale_out),
        cutlass.Int32(mask0),
        cutlass.Int32(mask1),
        cutlass.Int32(mask2),
        cutlass.Int32(mask3),
    )


# ──────────────────────────────────────────────────────────────────────
# tcgen05mma_ws_ss  —  warp-specialized variant
# ──────────────────────────────────────────────────────────────────────

@cute.jit
def tcgen05mma_ws_ss(kind_dtype: str,
                     desc_a: Tcgen05SmemDescriptor,
                     desc_b: Tcgen05SmemDescriptor,
                     tmem_c: int,
                     desc_val: int,
                     scale_out: int):
    """tcgen05.mma.ws.cta_group::1.kind::{kind} [tmem_c], desc_a, desc_b, desc_val, p, 0;"""
    kind = _kind_for(kind_dtype)

    asm_str = (
        "{\n"
        ".reg .pred p;\n"
        ".reg .pred q;\n"
        "elect.sync _|q, 0xFFFFFFFF;\n"
        "setp.ne.b32 p, $4, 0;\n"
        f"@q tcgen05.mma.ws.cta_group::1.kind::{kind} "
        "[$0], $1, $2, $3, p, 0;\n"
        "}"
    )

    @dsl_user_op
    def _do_mma_ws(c_val, da_val, db_val, dv_val, sc_val,
                   *, loc=None, ip=None):
        llvm.inline_asm(
            None,
            [_ir(c_val, loc, ip), _ir(da_val, loc, ip),
             _ir(db_val, loc, ip), _ir(dv_val, loc, ip),
             _ir(sc_val, loc, ip)],
            asm_str,
            "r,l,l,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    _do_mma_ws(
        cutlass.Int32(tmem_c),
        desc_a.desc_i64[0],
        desc_b.desc_i64[0],
        cutlass.Int32(desc_val),
        cutlass.Int32(scale_out),
    )


# ──────────────────────────────────────────────────────────────────────
# tcgen05mma_ts  —  A from TMEM, B from SMEM descriptor
# ──────────────────────────────────────────────────────────────────────

@cute.jit
def tcgen05mma_ts(kind_dtype: str,
                  tmem_a: int,
                  desc_b: Tcgen05SmemDescriptor,
                  tmem_c: int,
                  desc_val: int,
                  scale_out: int,
                  mask0: int, mask1: int, mask2: int, mask3: int):
    """tcgen05.mma.cta_group::1.kind::{kind} [tmem_c], [tmem_a], desc_b, desc_val, {masks}, p;"""
    kind = _kind_for(kind_dtype)

    # A is [$1] (indirect via TMEM address), not $1 (direct descriptor)
    asm_str = (
        "{\n"
        ".reg .pred p;\n"
        ".reg .pred q;\n"
        "elect.sync _|q, 0xFFFFFFFF;\n"
        "setp.ne.b32 p, $4, 0;\n"
        f"@q tcgen05.mma.cta_group::1.kind::{kind} "
        "[$0], [$1], $2, $3, {$5, $6, $7, $8}, p;\n"
        "}"
    )

    @dsl_user_op
    def _do_mma_ts(c_val, a_val, db_val, dv_val, sc_val,
                   m0_val, m1_val, m2_val, m3_val,
                   *, loc=None, ip=None):
        llvm.inline_asm(
            None,
            [_ir(c_val, loc, ip), _ir(a_val, loc, ip),
             _ir(db_val, loc, ip), _ir(dv_val, loc, ip),
             _ir(sc_val, loc, ip), _ir(m0_val, loc, ip),
             _ir(m1_val, loc, ip), _ir(m2_val, loc, ip),
             _ir(m3_val, loc, ip)],
            asm_str,
            "r,r,l,r,r,r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    _do_mma_ts(
        cutlass.Int32(tmem_c),
        cutlass.Int32(tmem_a),
        desc_b.desc_i64[0],
        cutlass.Int32(desc_val),
        cutlass.Int32(scale_out),
        cutlass.Int32(mask0),
        cutlass.Int32(mask1),
        cutlass.Int32(mask2),
        cutlass.Int32(mask3),
    )


# ──────────────────────────────────────────────────────────────────────
# tcgen05_mma_arrive  —  mbarrier arrive for MMA commit
# ──────────────────────────────────────────────────────────────────────

@cute.jit
def tcgen05_mma_arrive(mbar_ptr: cute.Pointer):
    """tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [mbar];

    Guarded by elect_one_sync — only one thread in the warp issues the commit.
    """

    @dsl_user_op
    def _do_arrive(bar_val, *, loc=None, ip=None):
        llvm.inline_asm(
            None,
            [_ir(bar_val, loc, ip)],
            "{\n"
            ".reg .pred q;\n"
            "elect.sync _|q, 0xFFFFFFFF;\n"
            "@q tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];\n"
            "}",
            "r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    bar_intptr = cutlass.Int32(mbar_ptr.toint())
    _do_arrive(bar_intptr)


# ──────────────────────────────────────────────────────────────────────
# TMEM allocation / deallocation
# ──────────────────────────────────────────────────────────────────────

@cute.jit
def tmem_allocate(tmem_buffer_ptr: cute.Pointer, num_cols: int):
    """tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [dst], num_cols;

    tmem_buffer_ptr: SMEM pointer that receives the allocated TMEM address.
    num_cols: number of columns to allocate.
    """
    @dsl_user_op
    def _do_alloc(dst_val, ncols_val, *, loc=None, ip=None):
        llvm.inline_asm(
            None,
            [_ir(dst_val, loc, ip), _ir(ncols_val, loc, ip)],
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [$0], $1;",
            "r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    dst_intptr = cutlass.Int32(tmem_buffer_ptr.toint())
    _do_alloc(dst_intptr, cutlass.Int32(num_cols))


@cute.jit
def tmem_deallocate(tmem_ptr: cute.Pointer, num_cols: int):
    """tcgen05.dealloc.cta_group::1.sync.aligned.b32 tmem_addr, num_cols;

    tmem_ptr: SMEM pointer to the uint32 holding the TMEM address.
    num_cols: number of columns to deallocate.
    """
    # Read the TMEM address from the SMEM location
    tmem_addr = cute.make_tensor(tmem_ptr, (1,))[0]

    @dsl_user_op
    def _do_dealloc(tptr_val, ncols_val, *, loc=None, ip=None):
        llvm.inline_asm(
            None,
            [_ir(tptr_val, loc, ip), _ir(ncols_val, loc, ip)],
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 $0, $1;",
            "r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    _do_dealloc(cutlass.Int32(tmem_addr), cutlass.Int32(num_cols))


# ──────────────────────────────────────────────────────────────────────
# TMEM load  —  tcgen05.ld.sync.aligned.32x32b.xN.b32
#
# Uses the same pattern as wgmma_rs: direct llvm.inline_asm calls from
# within @cute.jit context (no @dsl_user_op wrapper).  The helper
# functions below are called at Python / JIT-compile time and emit MLIR
# operations directly into the surrounding @cute.jit function.
# ──────────────────────────────────────────────────────────────────────

# Max segment size for TMEM loads.  LLVM's NVPTX inline asm chokes on very
# large operand counts (e.g. x128 = 129 operands), so we cap at x32 which
# gives 33 operands — well within limits.  For N=128 this produces 4
# sequential x32 loads.
_TMEM_LD_MAX_LOG_N = 3   # 1 << 3 = 8  (keep small to avoid LLVM hangs with many operands)


def _emit_tmem_ld_segment(seg_size, addr_ir):
    """Emit one tcgen05.ld inline asm for a power-of-2 segment.

    Called during @cute.jit compilation — emits MLIR ops directly.
    Returns a list of seg_size cutlass.Int32 values.
    """
    i32_type = ir.IntegerType.get_signless(32)

    if seg_size == 1:
        result = llvm.inline_asm(
            i32_type,
            [addr_ir],
            "tcgen05.ld.sync.aligned.32x32b.x1.b32 $0, [$1];",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
        return [cutlass.Int32(result)]

    # Multi-output: struct of i32s
    out_types = [i32_type] * seg_size
    result_type = llvm.StructType.get_literal(out_types)

    out_regs = ", ".join(f"${i}" for i in range(seg_size))
    src_idx = seg_size  # source addr is the last operand
    asm_str = (
        f"tcgen05.ld.sync.aligned.32x32b.x{seg_size}.b32 "
        f"{{{out_regs}}}, [${src_idx}];"
    )
    constraints = ",".join(["=r"] * seg_size) + ",r"

    result = llvm.inline_asm(
        result_type,
        [addr_ir],
        asm_str,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )

    return [cutlass.Int32(llvm.extractvalue(i32_type, result, [i]))
            for i in range(seg_size)]


def _emit_tmem_ld(n, max_log_n, src_addr, dst_view, offset=0):
    """Recursively split N into power-of-2 segments and emit TMEM loads.

    Called during @cute.jit compilation.
    src_addr: CuTeDSL Int32 (runtime TMEM address).
    dst_view: CuTeDSL tensor view over destination registers.
    offset:   Python int — current element offset (compile-time constant).
    """
    if n <= 0:
        return

    log_n = n.bit_length() - 1
    seg_log = min(log_n, max_log_n)
    seg_size = 1 << seg_log

    # Compute address for this segment
    if offset == 0:
        addr_ir = src_addr.ir_value()
    else:
        addr_ir = (src_addr + cutlass.Int32(offset)).ir_value()

    # Emit inline asm and store results
    results = _emit_tmem_ld_segment(seg_size, addr_ir)
    for j, val in enumerate(results):
        dst_view[offset + j] = val

    # Recurse for remainder
    _emit_tmem_ld(n - seg_size, max_log_n, src_addr, dst_view, offset + seg_size)


def _emit_tmem_fence():
    """Emit tcgen05.wait fence.  Called during @cute.jit compilation."""
    llvm.inline_asm(
        None, [],
        "tcgen05.wait::ld.sync.aligned;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def tcgen05_ld_32dp32bNx(N: Constexpr[int], pack16: Constexpr[bool],
                         tmem_start_col: int,
                         tmem_col_offset: int,
                         dst_ptr: cute.Pointer):
    """Load N uint32 values from TMEM using tcgen05.ld.sync.aligned.32x32b.

    Matches tl::tcgen05_ld_32dp32bNx from copy_sm100.h.
    N: number of 32-bit elements to load (compile-time constant).
    pack16: if True, use 16-bit packing (not implemented yet).
    tmem_start_col: TMEM base column address.
    tmem_col_offset: additional column offset.
    dst_ptr: destination pointer (register memory).
    """
    src_addr = cutlass.Int32(tmem_start_col) + cutlass.Int32(tmem_col_offset)
    dst_view = cute.make_tensor(
        cute.recast_ptr(dst_ptr, dtype=cute.Int32), (N,))
    _emit_tmem_ld(N, _TMEM_LD_MAX_LOG_N, src_addr, dst_view)
    _emit_tmem_fence()


@cute.jit
def tcgen05_ld_32dp64bNx(N: Constexpr[int], pack16: Constexpr[bool],
                         tmem_start_col: int,
                         tmem_col_offset: int,
                         dst_ptr: cute.Pointer):
    """Load N values from TMEM using tcgen05.ld.sync.aligned.32x64b."""
    src_addr = cutlass.Int32(tmem_start_col) + cutlass.Int32(tmem_col_offset)
    dst_view = cute.make_tensor(
        cute.recast_ptr(dst_ptr, dtype=cute.Int32), (N,))
    _emit_tmem_ld(N, _TMEM_LD_MAX_LOG_N, src_addr, dst_view)
    _emit_tmem_fence()


@cute.jit
def tcgen05_ld_32dp128bNx(N: Constexpr[int], pack16: Constexpr[bool],
                          tmem_start_col: int,
                          tmem_col_offset: int,
                          dst_ptr: cute.Pointer):
    """Load N values from TMEM using tcgen05.ld.sync.aligned.32x128b."""
    src_addr = cutlass.Int32(tmem_start_col) + cutlass.Int32(tmem_col_offset)
    dst_view = cute.make_tensor(
        cute.recast_ptr(dst_ptr, dtype=cute.Int32), (N,))
    _emit_tmem_ld(N, min(_TMEM_LD_MAX_LOG_N, 6), src_addr, dst_view)
    _emit_tmem_fence()


@cute.jit
def tcgen05_ld_32dp256bNx(N: Constexpr[int], pack16: Constexpr[bool],
                          tmem_start_col: int,
                          tmem_col_offset: int,
                          dst_ptr: cute.Pointer):
    """Load N values from TMEM using tcgen05.ld.sync.aligned.32x256b."""
    src_addr = cutlass.Int32(tmem_start_col) + cutlass.Int32(tmem_col_offset)
    dst_view = cute.make_tensor(
        cute.recast_ptr(dst_ptr, dtype=cute.Int32), (N,))
    _emit_tmem_ld(N, min(_TMEM_LD_MAX_LOG_N, 6), src_addr, dst_view)
    _emit_tmem_fence()
