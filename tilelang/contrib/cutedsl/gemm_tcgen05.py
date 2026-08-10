"""
tcgen05 (SM100/Blackwell) MMA support for CuTeDSL backend.

Provides:
  - Tcgen05SmemDescriptor: 64-bit SMEM descriptor for tcgen05 MMA
  - initialize_tcgen05_descriptor: bitfield packing matching common.h layout
  - tcgen05mma_ss / tcgen05mma_ws_ss / tcgen05mma_ts: primitive MMA wrappers
  - tcgen05_mma_arrive: mbarrier arrive for MMA commit
  - tmem_allocate / tmem_deallocate: TMEM allocation/deallocation
"""

__all__ = [
    "Tcgen05SmemDescriptor",
    "initialize_tcgen05_descriptor",
    "tcgen05mma_ss",
    "tcgen05mma_ws_ss",
    "tcgen05mma_ts",
    "tcgen05_mma_arrive",
    "tcgen05_before_thread_sync",
    "tcgen05_after_thread_sync",
    "tmem_allocate",
    "tmem_deallocate",
    "tcgen05_ld_32dp32bNx",
    "tcgen05_ld_32dp64bNx",
    "tcgen05_ld_32dp128bNx",
    "tcgen05_ld_32dp256bNx",
    "tcgen05_ld_16dp64bNx",
    "tcgen05_ld_16dp128bNx",
    "tcgen05_ld_16dp256bNx",
    "tcgen05_st_32dp32bNx",
    "tcgen05_st_32dp64bNx",
    "tcgen05_st_32dp128bNx",
    "tcgen05_st_32dp256bNx",
    "tcgen05_st_16dp64bNx",
    "tcgen05_st_16dp128bNx",
    "tcgen05_st_16dp256bNx",
    "tcgen05mma_blockscaled_ss",
    "tcgen05_cp_warpx4",
    "tcgen05_sf_warp_transpose",
]

import cutlass
import cutlass.cute as cute
from cutlass._mlir_helpers.vector import Vector
from cutlass.cutlass_dsl import Constexpr, dsl_user_op
from cutlass.experimental import primitives as prims
from cutlass.experimental.primitives import nvvm_wrapper as _nvvm_prims


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
        self.desc_i64 = cute.make_tensor(cute.recast_ptr(self.desc.iterator, dtype=cute.Int64), (1,))
        if desc_64 is not None:
            self.desc_i64[0] = desc_64

    def __add__(self, offset):
        """Add byte offset.  Like C++ operator+, shifts offset >> 4."""
        res = cute.make_rmem_tensor((2,), dtype=cutlass.Int32)
        res_i64 = cute.make_tensor(cute.recast_ptr(res.iterator, dtype=cute.Int64), (1,))
        # Address is in 16-byte units: add (offset >> 4)
        res[0] = self.desc[0] + (offset >> 4)
        res[1] = self.desc[1]
        return Tcgen05SmemDescriptor(res_i64[0])


# ──────────────────────────────────────────────────────────────────────
# Descriptor initialization
# ──────────────────────────────────────────────────────────────────────


def initialize_tcgen05_descriptor(desc, start_address, leading_byte_offset, stride_byte_offset, base_offset, leading_abs, swizzle_mode):
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
    desc.desc[0] = cutlass.Int32(ptr_val) | cutlass.Int32(cutlass.Int32(leading_byte_offset) << 16)
    desc.desc[1] = (
        cutlass.Int32(stride_byte_offset)
        | cutlass.Int32(1 << 14)  # version = 1
        | cutlass.Int32(cutlass.Int32(base_offset & 0x7) << 17)
        | cutlass.Int32(cutlass.Int32(leading_abs) << 20)
        | cutlass.Int32(cutlass.Int32(swizzle_mode & 0x7) << 29)
    )


# ──────────────────────────────────────────────────────────────────────
# tcgen05 kind mapping  (TIR dtype string  ->  primitive kind enum)
# ──────────────────────────────────────────────────────────────────────

_TCGEN05_KIND_MAP = {
    "fp16": prims.Tcgen05MMAKind.F16,
    "bf16": prims.Tcgen05MMAKind.F16,
    "float16": prims.Tcgen05MMAKind.F16,
    "bfloat16": prims.Tcgen05MMAKind.F16,
    "tf32": prims.Tcgen05MMAKind.TF32,
    "float32": prims.Tcgen05MMAKind.TF32,
    "s8": prims.Tcgen05MMAKind.INT8,
    "u8": prims.Tcgen05MMAKind.INT8,
    "int8": prims.Tcgen05MMAKind.INT8,
    "uint8": prims.Tcgen05MMAKind.INT8,
    "e4m3": prims.Tcgen05MMAKind.F8F6F4,
    "e5m2": prims.Tcgen05MMAKind.F8F6F4,
    "float8_e4m3": prims.Tcgen05MMAKind.F8F6F4,
    "float8_e4m3fn": prims.Tcgen05MMAKind.F8F6F4,
    "float8_e5m2": prims.Tcgen05MMAKind.F8F6F4,
    "e2m1": prims.Tcgen05MMAKind.F8F6F4,
    "float4_e2m1fn": prims.Tcgen05MMAKind.F8F6F4,
    "float4_e2m1_unpacked": prims.Tcgen05MMAKind.F8F6F4,
}


def _kind_for(dtype_str):
    kind = _TCGEN05_KIND_MAP.get(dtype_str)
    if kind is None:
        raise ValueError(f"tcgen05mma: unsupported dtype '{dtype_str}'")
    return kind


def _blockscaled_kind_for(dtype_str):
    if dtype_str not in _TCGEN05_KIND_MAP:
        raise ValueError(f"tcgen05mma blockscaled: unsupported dtype '{dtype_str}'")
    return prims.MMABlockScaleKind.MXF8F6F4


def _desc_value(desc):
    return desc.desc_i64[0] if isinstance(desc, Tcgen05SmemDescriptor) else desc


@dsl_user_op
def _tcgen05_mma_ws(mma_kind, d, a, b, idesc, enable_input_d, *, loc=None, ip=None):
    """Emit tcgen05.mma.ws while working around a CUTLASS 4.7 pre-release kw typo."""
    _nvvm_prims._assert_tensor_mem(d, "tcgen05.mma.ws")
    _nvvm_prims._nvvm.tcgen05_mma_ws(
        _nvvm_prims._TCGEN05_MMA_KIND_TO_DIALECT[mma_kind],
        d,
        a,
        cutlass.Int64(b),
        cutlass.Int32(idesc),
        cutlass.Boolean(enable_input_d),
        col_b_zero_mask=None,
        loc=loc,
        ip=ip,
    )


def _tmem_ptr(tmem_addr):
    return prims.make_tmem_ptr(cutlass.Int32(tmem_addr), cutlass.Int32)


def _write_disable_mask(mask0, mask1, mask2, mask3):
    return Vector.from_elements(
        (
            cutlass.Int32(mask0),
            cutlass.Int32(mask1),
            cutlass.Int32(mask2),
            cutlass.Int32(mask3),
        ),
        cutlass.Int32,
    )


# ──────────────────────────────────────────────────────────────────────
# tcgen05mma_ss  —  both A and B from SMEM descriptors (non-WS)
# ──────────────────────────────────────────────────────────────────────


@cute.jit
def tcgen05mma_ss(
    kind_dtype: str,
    desc_a: Tcgen05SmemDescriptor,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
    mask0: int,
    mask1: int,
    mask2: int,
    mask3: int,
    use_2cta: Constexpr[bool] = False,
):
    """tcgen05.mma.cta_group::{1|2}.kind::{kind} [tmem_c], desc_a, desc_b, desc_val, ...;

    Guarded by elect_one_sync — only one thread in the warp issues the MMA.
    The TIR codegen also wraps calls in ``if (threadIdx.x >> 5) == 0``
    which selects warp 0.
    """
    kind = _kind_for(kind_dtype)
    if use_2cta:
        if prims.elect_sync():
            prims.tcgen05_mma(
                kind,
                prims.CTAGroup.CTA_2,
                _tmem_ptr(tmem_c),
                desc_a.desc_i64[0],
                desc_b.desc_i64[0],
                desc_val,
                scale_out,
            )
    else:
        if prims.elect_sync():
            prims.tcgen05_mma(
                kind,
                prims.CTAGroup.CTA_1,
                _tmem_ptr(tmem_c),
                desc_a.desc_i64[0],
                desc_b.desc_i64[0],
                desc_val,
                scale_out,
                write_disable_mask=_write_disable_mask(mask0, mask1, mask2, mask3),
            )


# ──────────────────────────────────────────────────────────────────────
# tcgen05mma_ws_ss  —  warp-specialized variant
# ──────────────────────────────────────────────────────────────────────


@cute.jit
def tcgen05mma_ws_ss(
    kind_dtype: str, desc_a: Tcgen05SmemDescriptor, desc_b: Tcgen05SmemDescriptor, tmem_c: int, desc_val: int, scale_out: int
):
    """tcgen05.mma.ws.cta_group::1.kind::{kind} [tmem_c], desc_a, desc_b, desc_val, p, 0;"""
    kind = _kind_for(kind_dtype)
    if prims.elect_sync():
        _tcgen05_mma_ws(
            kind,
            _tmem_ptr(tmem_c),
            desc_a.desc_i64[0],
            desc_b.desc_i64[0],
            desc_val,
            scale_out,
        )


# ──────────────────────────────────────────────────────────────────────
# tcgen05mma_ts  —  A from TMEM, B from SMEM descriptor
# ──────────────────────────────────────────────────────────────────────


@cute.jit
def tcgen05mma_ts(
    kind_dtype: str,
    tmem_a: int,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
    mask0: int,
    mask1: int,
    mask2: int,
    mask3: int,
    use_2cta: Constexpr[bool] = False,
):
    """tcgen05.mma.cta_group::{1|2}.kind::{kind} [tmem_c], [tmem_a], desc_b, desc_val, ...;"""
    kind = _kind_for(kind_dtype)
    if use_2cta:
        if prims.elect_sync():
            prims.tcgen05_mma(
                kind,
                prims.CTAGroup.CTA_2,
                _tmem_ptr(tmem_c),
                _tmem_ptr(tmem_a),
                desc_b.desc_i64[0],
                desc_val,
                scale_out,
            )
    else:
        if prims.elect_sync():
            prims.tcgen05_mma(
                kind,
                prims.CTAGroup.CTA_1,
                _tmem_ptr(tmem_c),
                _tmem_ptr(tmem_a),
                desc_b.desc_i64[0],
                desc_val,
                scale_out,
                write_disable_mask=_write_disable_mask(mask0, mask1, mask2, mask3),
            )


# ──────────────────────────────────────────────────────────────────────
# tcgen05_mma_arrive  —  mbarrier arrive for MMA commit
# ──────────────────────────────────────────────────────────────────────


@cute.jit
def tcgen05_mma_arrive(mbar_ptr: cute.Pointer, use_2cta: Constexpr[bool] = False):
    """Commit prior tcgen05 work to an mbarrier."""
    group = prims.CTAGroup.CTA_2 if use_2cta else prims.CTAGroup.CTA_1
    multicast_mask = cutlass.Int16(3) if use_2cta else None
    if prims.elect_sync():
        prims.tcgen05_commit(mbar_ptr, multicast_mask=multicast_mask, group=group)


@dsl_user_op
def tcgen05_before_thread_sync(*, loc=None, ip=None) -> None:
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC, loc=loc, ip=ip)


@dsl_user_op
def tcgen05_after_thread_sync(*, loc=None, ip=None) -> None:
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC, loc=loc, ip=ip)


# ──────────────────────────────────────────────────────────────────────
# TMEM allocation / deallocation
# ──────────────────────────────────────────────────────────────────────


@cute.jit
def tmem_allocate(tmem_buffer_ptr: cute.Pointer, num_cols: int, use_2cta: Constexpr[bool] = False):
    """Allocate TMEM columns for tcgen05 operations.

    tmem_buffer_ptr: SMEM pointer that receives the allocated TMEM address.
    num_cols: number of columns to allocate.
    """
    group = prims.CTAGroup.CTA_2 if use_2cta else prims.CTAGroup.CTA_1
    prims.tcgen05_alloc(tmem_buffer_ptr, cutlass.Int32(num_cols), group=group)


@cute.jit
def tmem_deallocate(tmem_ptr: cute.Pointer, num_cols: int, use_2cta: Constexpr[bool] = False):
    """Deallocate TMEM columns for tcgen05 operations.

    tmem_ptr: SMEM pointer to the uint32 holding the TMEM address.
    num_cols: number of columns to deallocate.
    """
    group = prims.CTAGroup.CTA_2 if use_2cta else prims.CTAGroup.CTA_1
    tmem_addr = cute.make_tensor(tmem_ptr, (1,))[0]
    tmem_addr_ptr = prims.make_tmem_ptr(cutlass.Int32(tmem_addr), cutlass.Int32)
    prims.tcgen05_dealloc(tmem_addr_ptr, cutlass.Int32(num_cols), group=group)


# ──────────────────────────────────────────────────────────────────────
# Block-scaled tcgen05 MMA
# ──────────────────────────────────────────────────────────────────────


@cute.jit
def tcgen05mma_blockscaled_ss(
    kind_dtype: str,
    desc_a,
    desc_b,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
    tmem_sfa: int,
    tmem_sfb: int,
    use_2cta: Constexpr[bool] = False,
):
    """Block-scaled tcgen05.mma SS path with scale factors already in TMEM."""
    group = prims.CTAGroup.CTA_2 if use_2cta else prims.CTAGroup.CTA_1
    d_ptr = prims.make_tmem_ptr(cutlass.Int32(tmem_c), cutlass.Int32)
    sfa_ptr = prims.make_tmem_ptr(cutlass.Int32(tmem_sfa), cutlass.Int32)
    sfb_ptr = prims.make_tmem_ptr(cutlass.Int32(tmem_sfb), cutlass.Int32)
    if prims.elect_sync():
        prims.tcgen05_mma_block_scale(
            _blockscaled_kind_for(kind_dtype),
            group,
            d_ptr,
            _desc_value(desc_a),
            _desc_value(desc_b),
            desc_val,
            enable_input_d=scale_out,
            scale_a=sfa_ptr,
            scale_b=sfb_ptr,
        )


# ──────────────────────────────────────────────────────────────────────
# Scale-factor SMEM to TMEM helpers
# ──────────────────────────────────────────────────────────────────────


@cute.jit
def tcgen05_cp_warpx4(smem_ptr: cute.Pointer, tmem_ptr: cute.Pointer, tmem_col_offset: int, use_2cta: Constexpr[bool] = False):
    """Copy a 32x128b scale-factor tile from SMEM to TMEM with warpx4 multicast."""
    smem_addr = smem_ptr.toint() if hasattr(smem_ptr, "toint") else smem_ptr
    smem_desc = prims.Tcgen05SmemDesc.build(
        start_address=smem_addr,
        leading_byte_offset=0,
        stride_byte_offset=128,
        base_offset=0,
        layout=prims.Tcgen05SmemSwizzle.NONE,
    )
    tmem_addr_tensor = tmem_ptr if isinstance(tmem_ptr, cute.Tensor) else cute.make_tensor(tmem_ptr, (1,))
    tmem_addr = cutlass.Int32(tmem_addr_tensor[0]) + cutlass.Int32(tmem_col_offset)
    tmem_dst = prims.make_tmem_ptr(tmem_addr, cutlass.Int32)
    group = prims.CTAGroup.CTA_2 if use_2cta else prims.CTAGroup.CTA_1
    shape, multicast = prims.S2TCopyMode.S2T_32x128b_WARPX4
    if prims.elect_sync():
        prims.tcgen05_cp(
            shape,
            tmem_dst,
            smem_desc,
            group=group,
            multicast=multicast,
        )


@cute.jit
def tcgen05_sf_warp_transpose(smem_ptr: cute.Pointer):
    """In-place warp transpose for 128 uint32 scale-factor words in SMEM."""
    smem = cute.make_tensor(cute.recast_ptr(smem_ptr, dtype=cutlass.Uint32), (128,))
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx % 32
    lane_quad = lane >> 3
    v0 = smem[(0 ^ lane_quad) * 32 + lane]
    v1 = smem[(1 ^ lane_quad) * 32 + lane]
    v2 = smem[(2 ^ lane_quad) * 32 + lane]
    v3 = smem[(3 ^ lane_quad) * 32 + lane]
    cute.arch.sync_warp()
    smem[lane * 4 + (0 ^ lane_quad)] = v0
    smem[lane * 4 + (1 ^ lane_quad)] = v1
    smem[lane * 4 + (2 ^ lane_quad)] = v2
    smem[lane * 4 + (3 ^ lane_quad)] = v3


# ──────────────────────────────────────────────────────────────────────
# TMEM load helpers.
# ──────────────────────────────────────────────────────────────────────

_TMEM_LD_MAX_LOG_N = 3  # 1 << 3 = 8  (keep small to avoid LLVM hangs with many operands)


def _emit_tmem_ld_segment(shape, seg_x, regs_per_x, addr):
    """Emit one tcgen05.ld wrapper call for a power-of-2 segment.

    Called during @cute.jit compilation — emits MLIR ops directly.
    shape:       TMEM load shape, e.g. "32x32b", "16x64b", "16x128b", "16x256b"
    seg_x:       x-count in the PTX instruction (power of 2)
    regs_per_x:  number of i32 output registers per x-element
    Returns a list of (seg_x * regs_per_x) cutlass.Int32 values.
    """
    total_regs = seg_x * regs_per_x
    tmem_ptr = prims.make_tmem_ptr(cutlass.Int32(addr), cutlass.Int32)
    result = prims.tcgen05_ld(shape, tmem_ptr, num=seg_x)
    return [cutlass.Int32(result[i]) for i in range(total_regs)]


def _emit_tmem_ld(n_x, max_log_n, ptx_type, regs_per_x, cols_per_x, src_addr, dst_view, dst_offset=0, src_col_offset=0):
    """Recursively split x-count into power-of-2 segments and emit TMEM loads.

    Called during @cute.jit compilation.
    n_x:            remaining x-element count to load
    max_log_n:      max log2 of x-count per PTX instruction
    ptx_type:       TMEM load shape, e.g. "32x32b", "16x64b"
    regs_per_x:     i32 output registers per x-element
    cols_per_x:     b32 columns covered per x-element (1/2/4/8 for 32/64/128/256b)
    src_addr:       CuTeDSL Int32 (runtime TMEM base address)
    dst_view:       CuTeDSL tensor view over destination registers
    dst_offset:     Python int — i32 offset into dst_view (compile-time constant)
    src_col_offset: Python int — TMEM column offset from src_addr (compile-time constant)
    """
    if n_x <= 0:
        return

    log_n = n_x.bit_length() - 1
    seg_log = min(log_n, max_log_n)
    seg_x = 1 << seg_log

    if src_col_offset == 0:
        addr = src_addr
    else:
        addr = src_addr + cutlass.Int32(src_col_offset)

    results = _emit_tmem_ld_segment(ptx_type, seg_x, regs_per_x, addr)
    for j, val in enumerate(results):
        dst_view[dst_offset + j] = val

    # Recurse for remainder
    total_regs_emitted = seg_x * regs_per_x
    _emit_tmem_ld(
        n_x - seg_x,
        max_log_n,
        ptx_type,
        regs_per_x,
        cols_per_x,
        src_addr,
        dst_view,
        dst_offset + total_regs_emitted,
        src_col_offset + seg_x * cols_per_x,
    )


def _emit_tmem_fence():
    """Wait for pending tcgen05.ld operations."""
    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)


@cute.jit
def tcgen05_ld_32dp32bNx(N: Constexpr[int], pack16: Constexpr[bool], tmem_start_col: int, tmem_col_offset: int, dst_ptr: cute.Pointer):
    """Load N uint32 values from TMEM using tcgen05.ld.sync.aligned.32x32b.

    Matches tl::tcgen05_ld_32dp32bNx from copy_sm100.h.
    N: number of 32-bit elements to load (x-count, compile-time constant).
    pack16: if True, use 16-bit packing (not implemented yet).
    tmem_start_col: TMEM base column address.
    tmem_col_offset: additional column offset.
    dst_ptr: destination pointer (register memory).
    """
    src_addr = cutlass.Int32(tmem_start_col) + cutlass.Int32(tmem_col_offset)
    dst_view = cute.make_tensor(cute.recast_ptr(dst_ptr, dtype=cute.Int32), (N,))
    _emit_tmem_ld(N, _TMEM_LD_MAX_LOG_N, "32x32b", 1, 1, src_addr, dst_view)
    _emit_tmem_fence()


@cute.jit
def tcgen05_ld_32dp64bNx(N: Constexpr[int], pack16: Constexpr[bool], tmem_start_col: int, tmem_col_offset: int, dst_ptr: cute.Pointer):
    """Load from TMEM using 32dp64b pattern (2x 16x64b for lower/upper 16 rows).

    Matches tl::tmem_ld_32dp64bNx from tcgen_05_ld.h.
    N: x-count for 16x64b instructions. Total output: 2*N i32 regs.
    """
    total_regs = N * 2
    src_addr = cutlass.Int32(tmem_start_col) + cutlass.Int32(tmem_col_offset)
    dst_view = cute.make_tensor(cute.recast_ptr(dst_ptr, dtype=cute.Int32), (total_regs,))
    # Lower 16 rows
    _emit_tmem_ld(N, _TMEM_LD_MAX_LOG_N, "16x64b", 1, 2, src_addr, dst_view, dst_offset=0, src_col_offset=0)
    # Upper 16 rows (TMEM row offset = 16 << 16)
    upper_addr = src_addr + cutlass.Int32(16 << 16)
    _emit_tmem_ld(N, _TMEM_LD_MAX_LOG_N, "16x64b", 1, 2, upper_addr, dst_view, dst_offset=N, src_col_offset=0)
    _emit_tmem_fence()


@cute.jit
def tcgen05_ld_32dp128bNx(N: Constexpr[int], pack16: Constexpr[bool], tmem_start_col: int, tmem_col_offset: int, dst_ptr: cute.Pointer):
    """Load from TMEM using 32dp128b pattern (2x 16x128b for lower/upper 16 rows).

    Matches tl::tmem_ld_32dp128bNx from tcgen_05_ld.h.
    N: x-count for 16x128b instructions. Total output: 4*N i32 regs.
    16x128b.xN produces 2*N i32 regs per half.
    """
    regs_per_half = N * 2
    total_regs = regs_per_half * 2
    src_addr = cutlass.Int32(tmem_start_col) + cutlass.Int32(tmem_col_offset)
    dst_view = cute.make_tensor(cute.recast_ptr(dst_ptr, dtype=cute.Int32), (total_regs,))
    # Lower 16 rows
    _emit_tmem_ld(N, min(_TMEM_LD_MAX_LOG_N, 6), "16x128b", 2, 4, src_addr, dst_view, dst_offset=0, src_col_offset=0)
    # Upper 16 rows (TMEM row offset = 16 << 16)
    upper_addr = src_addr + cutlass.Int32(16 << 16)
    _emit_tmem_ld(N, min(_TMEM_LD_MAX_LOG_N, 6), "16x128b", 2, 4, upper_addr, dst_view, dst_offset=regs_per_half, src_col_offset=0)
    _emit_tmem_fence()


@cute.jit
def tcgen05_ld_32dp256bNx(N: Constexpr[int], pack16: Constexpr[bool], tmem_start_col: int, tmem_col_offset: int, dst_ptr: cute.Pointer):
    """Load from TMEM using 32dp256b pattern (2x 16x256b for lower/upper 16 rows).

    Matches tl::tmem_ld_32dp256bNx from tcgen_05_ld.h.
    N: x-count for 16x256b instructions. Total output: 8*N i32 regs.
    16x256b.xN produces 4*N i32 regs per half.
    """
    regs_per_half = N * 4
    total_regs = regs_per_half * 2
    src_addr = cutlass.Int32(tmem_start_col) + cutlass.Int32(tmem_col_offset)
    dst_view = cute.make_tensor(cute.recast_ptr(dst_ptr, dtype=cute.Int32), (total_regs,))
    # Lower 16 rows
    _emit_tmem_ld(N, min(_TMEM_LD_MAX_LOG_N, 6), "16x256b", 4, 8, src_addr, dst_view, dst_offset=0, src_col_offset=0)
    # Upper 16 rows (TMEM row offset = 16 << 16)
    upper_addr = src_addr + cutlass.Int32(16 << 16)
    _emit_tmem_ld(N, min(_TMEM_LD_MAX_LOG_N, 6), "16x256b", 4, 8, upper_addr, dst_view, dst_offset=regs_per_half, src_col_offset=0)
    _emit_tmem_fence()


# ──────────────────────────────────────────────────────────────────────
# Half-subpartition (16 datapath) TMEM loads.  The 32dp wrappers above
# issue the 16x{64,128,256}b shape twice, once per half; a PTX Layout F
# tile (1SM M=64) occupies only the low 16 datapaths of each
# sub-partition and needs a single issue.  Matches
# tl::tcgen05_ld_16dp{64,128,256}bNx from copy_sm100.h.
# ──────────────────────────────────────────────────────────────────────


@cute.jit
def tcgen05_ld_16dp64bNx(N: Constexpr[int], pack16: Constexpr[bool], tmem_start_col: int, tmem_col_offset: int, dst_ptr: cute.Pointer):
    """Load from TMEM using a single 16x64b issue (low 16 rows only).

    N: x-count for the 16x64b instruction. Total output: N i32 regs.
    """
    src_addr = cutlass.Int32(tmem_start_col) + cutlass.Int32(tmem_col_offset)
    dst_view = cute.make_tensor(cute.recast_ptr(dst_ptr, dtype=cute.Int32), (N,))
    _emit_tmem_ld(N, _TMEM_LD_MAX_LOG_N, "16x64b", 1, 2, src_addr, dst_view)
    _emit_tmem_fence()


@cute.jit
def tcgen05_ld_16dp128bNx(N: Constexpr[int], pack16: Constexpr[bool], tmem_start_col: int, tmem_col_offset: int, dst_ptr: cute.Pointer):
    """Load from TMEM using a single 16x128b issue (low 16 rows only).

    N: x-count for the 16x128b instruction. Total output: 2*N i32 regs.
    """
    src_addr = cutlass.Int32(tmem_start_col) + cutlass.Int32(tmem_col_offset)
    dst_view = cute.make_tensor(cute.recast_ptr(dst_ptr, dtype=cute.Int32), (N * 2,))
    _emit_tmem_ld(N, min(_TMEM_LD_MAX_LOG_N, 6), "16x128b", 2, 4, src_addr, dst_view)
    _emit_tmem_fence()


@cute.jit
def tcgen05_ld_16dp256bNx(N: Constexpr[int], pack16: Constexpr[bool], tmem_start_col: int, tmem_col_offset: int, dst_ptr: cute.Pointer):
    """Load from TMEM using a single 16x256b issue (low 16 rows only).

    N: x-count for the 16x256b instruction. Total output: 4*N i32 regs.
    """
    src_addr = cutlass.Int32(tmem_start_col) + cutlass.Int32(tmem_col_offset)
    dst_view = cute.make_tensor(cute.recast_ptr(dst_ptr, dtype=cute.Int32), (N * 4,))
    _emit_tmem_ld(N, min(_TMEM_LD_MAX_LOG_N, 6), "16x256b", 4, 8, src_addr, dst_view)
    _emit_tmem_fence()


# ──────────────────────────────────────────────────────────────────────
# TMEM stores (tcgen05.st) — register -> TMEM, mirroring the loads.
# ──────────────────────────────────────────────────────────────────────


def _emit_tmem_st_segment(shape, seg_x, regs_per_x, addr, values):
    """Emit one tcgen05.st wrapper call for a power-of-2 segment.

    Called during @cute.jit compilation — emits MLIR ops directly.
    values: list of (seg_x * regs_per_x) Int32 register values to store.
    """
    tmem_ptr = prims.make_tmem_ptr(cutlass.Int32(addr), cutlass.Int32)
    if len(values) == 1:
        prims.tcgen05_st(shape, tmem_ptr, values[0])
    else:
        prims.tcgen05_st(shape, tmem_ptr, Vector.from_elements(tuple(values), cutlass.Int32))


def _emit_tmem_st(n_x, max_log_n, ptx_type, regs_per_x, cols_per_x, dst_addr, src_view, src_offset=0, dst_col_offset=0):
    """Recursively split x-count into power-of-2 segments and emit TMEM stores.

    Mirror image of _emit_tmem_ld: registers are read from src_view and the
    TMEM address is the destination.
    """
    if n_x <= 0:
        return

    log_n = n_x.bit_length() - 1
    seg_log = min(log_n, max_log_n)
    seg_x = 1 << seg_log

    if dst_col_offset == 0:
        addr = dst_addr
    else:
        addr = dst_addr + cutlass.Int32(dst_col_offset)

    total_regs = seg_x * regs_per_x
    values = [cutlass.Int32(src_view[src_offset + j]) for j in range(total_regs)]
    _emit_tmem_st_segment(ptx_type, seg_x, regs_per_x, addr, values)

    _emit_tmem_st(
        n_x - seg_x,
        max_log_n,
        ptx_type,
        regs_per_x,
        cols_per_x,
        dst_addr,
        src_view,
        src_offset + total_regs,
        dst_col_offset + seg_x * cols_per_x,
    )


def _emit_tmem_st_fence():
    """Wait for pending tcgen05.st operations."""
    prims.tcgen05_wait(prims.Tcgen05Wait.STORE)


def _check_no_unpack16(name, unpack16):
    if unpack16:
        raise NotImplementedError(f"{name}: the unpack::16b modifier is not implemented in the cutedsl runtime")


@cute.jit
def tcgen05_st_32dp32bNx(N: Constexpr[int], unpack16: Constexpr[bool], tmem_start_col: int, tmem_col_offset: int, src_ptr: cute.Pointer):
    """Store N uint32 values to TMEM using tcgen05.st.sync.aligned.32x32b.

    Matches tl::tcgen05_st_32dp32bNx from copy_sm100.h.
    """
    _check_no_unpack16("tcgen05_st_32dp32bNx", unpack16)
    dst_addr = cutlass.Int32(tmem_start_col) + cutlass.Int32(tmem_col_offset)
    src_view = cute.make_tensor(cute.recast_ptr(src_ptr, dtype=cute.Int32), (N,))
    _emit_tmem_st(N, _TMEM_LD_MAX_LOG_N, "32x32b", 1, 1, dst_addr, src_view)
    _emit_tmem_st_fence()


@cute.jit
def tcgen05_st_32dp64bNx(N: Constexpr[int], unpack16: Constexpr[bool], tmem_start_col: int, tmem_col_offset: int, src_ptr: cute.Pointer):
    """Store to TMEM using 32dp64b pattern (2x 16x64b for lower/upper 16 rows).

    N: x-count for 16x64b instructions. Total input: 2*N i32 regs, the lower
    half's registers first (the duplicate issue's registers follow all
    repetitions, matching the C++ wrappers).
    """
    _check_no_unpack16("tcgen05_st_32dp64bNx", unpack16)
    dst_addr = cutlass.Int32(tmem_start_col) + cutlass.Int32(tmem_col_offset)
    src_view = cute.make_tensor(cute.recast_ptr(src_ptr, dtype=cute.Int32), (N * 2,))
    _emit_tmem_st(N, _TMEM_LD_MAX_LOG_N, "16x64b", 1, 2, dst_addr, src_view, src_offset=0, dst_col_offset=0)
    upper_addr = dst_addr + cutlass.Int32(16 << 16)
    _emit_tmem_st(N, _TMEM_LD_MAX_LOG_N, "16x64b", 1, 2, upper_addr, src_view, src_offset=N, dst_col_offset=0)
    _emit_tmem_st_fence()


@cute.jit
def tcgen05_st_32dp128bNx(N: Constexpr[int], unpack16: Constexpr[bool], tmem_start_col: int, tmem_col_offset: int, src_ptr: cute.Pointer):
    """Store to TMEM using 32dp128b pattern (2x 16x128b for lower/upper 16 rows).

    N: x-count for 16x128b instructions. Total input: 4*N i32 regs.
    """
    _check_no_unpack16("tcgen05_st_32dp128bNx", unpack16)
    regs_per_half = N * 2
    dst_addr = cutlass.Int32(tmem_start_col) + cutlass.Int32(tmem_col_offset)
    src_view = cute.make_tensor(cute.recast_ptr(src_ptr, dtype=cute.Int32), (regs_per_half * 2,))
    _emit_tmem_st(N, min(_TMEM_LD_MAX_LOG_N, 6), "16x128b", 2, 4, dst_addr, src_view, src_offset=0, dst_col_offset=0)
    upper_addr = dst_addr + cutlass.Int32(16 << 16)
    _emit_tmem_st(N, min(_TMEM_LD_MAX_LOG_N, 6), "16x128b", 2, 4, upper_addr, src_view, src_offset=regs_per_half, dst_col_offset=0)
    _emit_tmem_st_fence()


@cute.jit
def tcgen05_st_32dp256bNx(N: Constexpr[int], unpack16: Constexpr[bool], tmem_start_col: int, tmem_col_offset: int, src_ptr: cute.Pointer):
    """Store to TMEM using 32dp256b pattern (2x 16x256b for lower/upper 16 rows).

    N: x-count for 16x256b instructions. Total input: 8*N i32 regs.
    """
    _check_no_unpack16("tcgen05_st_32dp256bNx", unpack16)
    regs_per_half = N * 4
    dst_addr = cutlass.Int32(tmem_start_col) + cutlass.Int32(tmem_col_offset)
    src_view = cute.make_tensor(cute.recast_ptr(src_ptr, dtype=cute.Int32), (regs_per_half * 2,))
    _emit_tmem_st(N, min(_TMEM_LD_MAX_LOG_N, 6), "16x256b", 4, 8, dst_addr, src_view, src_offset=0, dst_col_offset=0)
    upper_addr = dst_addr + cutlass.Int32(16 << 16)
    _emit_tmem_st(N, min(_TMEM_LD_MAX_LOG_N, 6), "16x256b", 4, 8, upper_addr, src_view, src_offset=regs_per_half, dst_col_offset=0)
    _emit_tmem_st_fence()


@cute.jit
def tcgen05_st_16dp64bNx(N: Constexpr[int], unpack16: Constexpr[bool], tmem_start_col: int, tmem_col_offset: int, src_ptr: cute.Pointer):
    """Store to TMEM using a single 16x64b issue (low 16 rows only).

    N: x-count for the 16x64b instruction. Total input: N i32 regs.
    """
    _check_no_unpack16("tcgen05_st_16dp64bNx", unpack16)
    dst_addr = cutlass.Int32(tmem_start_col) + cutlass.Int32(tmem_col_offset)
    src_view = cute.make_tensor(cute.recast_ptr(src_ptr, dtype=cute.Int32), (N,))
    _emit_tmem_st(N, _TMEM_LD_MAX_LOG_N, "16x64b", 1, 2, dst_addr, src_view)
    _emit_tmem_st_fence()


@cute.jit
def tcgen05_st_16dp128bNx(N: Constexpr[int], unpack16: Constexpr[bool], tmem_start_col: int, tmem_col_offset: int, src_ptr: cute.Pointer):
    """Store to TMEM using a single 16x128b issue (low 16 rows only).

    N: x-count for the 16x128b instruction. Total input: 2*N i32 regs.
    """
    _check_no_unpack16("tcgen05_st_16dp128bNx", unpack16)
    dst_addr = cutlass.Int32(tmem_start_col) + cutlass.Int32(tmem_col_offset)
    src_view = cute.make_tensor(cute.recast_ptr(src_ptr, dtype=cute.Int32), (N * 2,))
    _emit_tmem_st(N, min(_TMEM_LD_MAX_LOG_N, 6), "16x128b", 2, 4, dst_addr, src_view)
    _emit_tmem_st_fence()


@cute.jit
def tcgen05_st_16dp256bNx(N: Constexpr[int], unpack16: Constexpr[bool], tmem_start_col: int, tmem_col_offset: int, src_ptr: cute.Pointer):
    """Store to TMEM using a single 16x256b issue (low 16 rows only).

    N: x-count for the 16x256b instruction. Total input: 4*N i32 regs.
    """
    _check_no_unpack16("tcgen05_st_16dp256bNx", unpack16)
    dst_addr = cutlass.Int32(tmem_start_col) + cutlass.Int32(tmem_col_offset)
    src_view = cute.make_tensor(cute.recast_ptr(src_ptr, dtype=cute.Int32), (N * 4,))
    _emit_tmem_st(N, min(_TMEM_LD_MAX_LOG_N, 6), "16x256b", 4, 8, dst_addr, src_view)
    _emit_tmem_st_fence()
