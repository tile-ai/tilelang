from __future__ import annotations
import tilelang.language as T
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable
from .mma_macro_generator import TensorCoreIntrinEmitter as MMAIntrinEmitter
from tvm import DataType
from tvm.tir import PrimExpr, Buffer, Var, IndexMap, BufferRegion
from tilelang.utils import is_fragment, retrive_ptr_from_buffer_region, is_full_region
from math import gcd
from tilelang.layout import (
    Layout,
    make_full_bank_swizzled_layout,
    make_half_bank_swizzled_layout,
    make_quarter_bank_swizzled_layout,
    make_linear_layout,
)
from tvm.runtime import convert
from tilelang.cuda.intrinsics.layout.mma_layout import (
    shared_16x8_to_mma_32x4_layout_sr_a,
    shared_16x16_to_mma_32x8_layout_sr_a,
    shared_16x32_to_mma_32x16_layout_sr_a,
)

lift = convert


@dataclass(frozen=True)
class WGMMADescriptorParams:
    """Pre-computed parameters for WGMMA descriptor initialization and atom offset computation.

    Returned by ``compute_wgmma_*_desc_params()`` and consumed by
    ``init_wgmma_*_desc()`` and ``wgmma_*_atom()`` methods.
    """

    swizzle_mode: int
    """SwizzleMode enum value (passed directly to ``T.initialize_wgmma_descriptor``)."""
    leading_byte_offset: int
    """LBO >> 4, ready to pass to ``T.initialize_wgmma_descriptor``."""
    stride_byte_offset: int
    """SBO >> 4, ready to pass to ``T.initialize_wgmma_descriptor``."""
    swizzle_atom_elems: int
    """Number of elements per swizzle atom along the non-K dimension."""
    k_atom_size: int
    """``max(swizzle_atom_elems // micro_size_k, 1)``."""
    elems_in_bytes: int
    """Byte width of a single element: ``DataType(dtype).bits // 8``."""
    is_k_major: bool
    """Whether the matrix is stored in K-major order (affects offset formula branching)."""


class SwizzleMode(IntEnum):
    # SWIZZLE_NONE = 0, SWIZZLE_32B = 3, SWIZZLE_64B = 2, SWIZZLE_128B = 1
    NONE = 0
    SWIZZLE_128B = 1
    SWIZZLE_64B = 2
    SWIZZLE_32B = 3

    def is_none(self) -> bool:
        return self == SwizzleMode.NONE

    def is_swizzle_32b(self) -> bool:
        return self == SwizzleMode.SWIZZLE_32B

    def is_swizzle_64b(self) -> bool:
        return self == SwizzleMode.SWIZZLE_64B

    def is_swizzle_128b(self) -> bool:
        return self == SwizzleMode.SWIZZLE_128B

    def swizzle_byte_size(self) -> int:
        if self.is_swizzle_32b():
            return 32
        elif self.is_swizzle_64b():
            return 64
        elif self.is_swizzle_128b():
            return 128
        else:
            return 1

    def swizzle_atom_size(self) -> int:
        if self.is_swizzle_32b():
            return 32 // 16
        elif self.is_swizzle_64b():
            return 64 // 16
        elif self.is_swizzle_128b():
            return 128 // 16
        else:
            return 1


# derive from MMAIntrinEmitter as some layouts are the same
class TensorCoreIntrinEmitter(MMAIntrinEmitter):
    """
    To eliminate Python syntax within TIR Macro.
    """

    # should be rewritten to support dynamic k_dim
    wgmma_prefix: str

    # wgmma instruction M dimension
    wgmma_inst_m: int
    # wgmma instruction N dimension
    wgmma_inst_n: int

    a_shared_layout: Layout = None
    b_shared_layout: Layout = None

    def __init__(
        self,
        a_dtype: str = T.float16,
        b_dtype: str = T.float16,
        accum_dtype: str = T.float16,
        a_transposed: bool = False,
        b_transposed: bool = False,
        block_row_warps: int = 2,
        block_col_warps: int = 2,
        warp_row_tiles: int = 8,
        warp_col_tiles: int = 8,
        chunk: int = 16,
        reduce_k: int = 1,
        num_elems_per_byte: int = 1,
        is_m_first: bool | None = False,
        thread_var: Var | None = None,
    ):
        super().__init__(
            a_dtype,
            b_dtype,
            accum_dtype,
            a_transposed,
            b_transposed,
            block_row_warps,
            block_col_warps,
            warp_row_tiles,
            warp_col_tiles,
            chunk,
            reduce_k,
            num_elems_per_byte,
            is_m_first,
            thread_var,
        )
        self._initialize_wgmma_prefix(self.n_dim)

    def _assign_a_shared_layout(self, layout: Layout):
        self.a_shared_layout = layout
        return self

    def _assign_b_shared_layout(self, layout: Layout):
        self.b_shared_layout = layout
        return self

    def _initialize_wgmma_prefix(self, n_dim: int = 16):
        inst_m, inst_n = 64, gcd(self.warp_col_tiles, 256)
        assert inst_n % 8 == 0, (
            f"inst_n must be a multiple of 8, got {inst_n} (block_col_warps={self.block_col_warps}, warp_col_tiles={self.warp_col_tiles})"
        )
        # Validate inst_n: Hopper WGMMA supports n in [8, 256] and multiple of 8
        assert 8 <= inst_n <= 256, (
            f"inst_n must be within [8, 256], got {inst_n} (block_col_warps={self.block_col_warps}, warp_col_tiles={self.warp_col_tiles})"
        )
        # 256 bits per instruction
        inst_k = 256 // DataType(self.a_dtype).bits
        self.wgmma_inst_m = inst_m
        self.wgmma_inst_n = inst_n
        self.wgmma_prefix = f"m{inst_m}n{inst_n}k{inst_k}"

    def _initialize_micro_size(self, m_dim: int = 16, k_dim: int = 16):
        warp_row_tiles = self.warp_row_tiles
        warp_col_tiles = self.warp_col_tiles
        assert warp_row_tiles >= 16, f"warp_row_tiles must be greater than 16, got {warp_row_tiles}"
        assert warp_row_tiles % 16 == 0, f"warp_row_tiles must be divisible by 16, got {warp_row_tiles}"
        assert warp_col_tiles >= 8, f"warp_col_tiles must be greater than 8, got {warp_col_tiles}"
        assert warp_col_tiles % 8 == 0, f"warp_col_tiles must be divisible by 8, got {warp_col_tiles}"

        # four warps per block
        self.warp_rows = warp_row_tiles // m_dim
        if warp_col_tiles % 16 == 0:
            self.n_dim = 16
            self.micro_size_y = 16
            self.warp_cols = warp_col_tiles // 16
        else:
            # must be divisible by 8
            self.n_dim = 8
            self.micro_size_y = 8
            self.warp_cols = warp_col_tiles // 8

        self.micro_size_x = m_dim
        self.micro_size_k = k_dim

    def _determinate_swizzle_mode(self, buffer: Buffer, layout: Layout) -> SwizzleMode:
        # same behavior to src/layout/gemm_layouts.cc::makeGemmABLayoutHopper
        if layout is None or layout.is_equal(make_linear_layout(buffer)):
            return SwizzleMode.NONE
        elif layout.is_equal(make_quarter_bank_swizzled_layout(buffer)):
            return SwizzleMode.SWIZZLE_32B
        elif layout.is_equal(make_half_bank_swizzled_layout(buffer)):
            return SwizzleMode.SWIZZLE_64B
        elif layout.is_equal(make_full_bank_swizzled_layout(buffer)):
            return SwizzleMode.SWIZZLE_128B
        else:
            raise ValueError(f"Unsupported swizzle mode: {layout}")

    def wgmma(
        self, A_region: BufferRegion, B_region: BufferRegion, C_region: BufferRegion, clear_accum: PrimExpr = False, wg_wait: int = 0
    ):
        if is_fragment(A_region):
            return self.wgmma_rs(A_region, B_region, C_region, clear_accum, wg_wait)

        k_dim = self.chunk
        micro_size_k = self.micro_size_k
        assert k_dim >= micro_size_k, f"k_dim must be greater than or equal to {micro_size_k}, got k_dim: {k_dim}"

        assert is_full_region(C_region), "Fragment output C must be a full region"
        C_buf = C_region.buffer

        num_inst_m = self.wgmma_num_inst_m
        num_inst_n = self.wgmma_num_inst_n
        num_k_atoms = self.wgmma_num_k_atoms
        a_params = self.compute_wgmma_a_desc_params(A_region)
        b_params = self.compute_wgmma_b_desc_params(B_region)

        @T.macro
        def _warp_mma(C_buf):
            desc_a = T.alloc_wgmma_desc()
            desc_b = T.alloc_wgmma_desc()
            self.init_wgmma_a_desc(desc_a, A_region, a_params)
            self.init_wgmma_b_desc(desc_b, B_region, b_params)
            self.wgmma_fence_c(C_buf)
            self.wgmma_arrive()

            for j in T.unroll(num_inst_n):
                for i in T.unroll(num_inst_m):
                    for ki in T.unroll(num_k_atoms):
                        self.wgmma_ss_atom(desc_a, desc_b, C_buf, i, j, ki, a_params, b_params, clear_accum)

            self.wgmma_commit()
            if wg_wait >= 0:
                self.wgmma_wait(wg_wait)
            self.wgmma_fence_c(C_buf)

        return _warp_mma(C_buf)

    def wgmma_rs(
        self, A_region: BufferRegion, B_region: BufferRegion, C_region: BufferRegion, clear_accum: PrimExpr = False, wg_wait: int = 0
    ):
        k_dim = self.chunk
        micro_size_k = self.micro_size_k
        assert k_dim >= micro_size_k, f"k_dim must be greater than or equal to {micro_size_k}, got k_dim: {k_dim}"

        assert is_full_region(A_region), "Fragment input A must be a full region"
        assert is_full_region(C_region), "Fragment output C must be a full region"
        A_buf = A_region.buffer
        C_buf = C_region.buffer

        num_inst_m = self.wgmma_num_inst_m
        num_inst_n = self.wgmma_num_inst_n
        num_k_atoms = self.wgmma_num_k_atoms
        b_params = self.compute_wgmma_b_desc_params(B_region)

        @T.macro
        def _warp_mma(A_buf, C_buf):
            desc_b = T.alloc_wgmma_desc()
            self.init_wgmma_b_desc(desc_b, B_region, b_params)
            self.wgmma_fence_a(A_buf)
            self.wgmma_fence_c(C_buf)
            self.wgmma_arrive()

            for j in T.unroll(0, num_inst_n):
                for i in T.unroll(num_inst_m):
                    for ki in T.unroll(0, num_k_atoms):
                        self.wgmma_rs_atom(A_buf, desc_b, C_buf, i, j, ki, b_params, clear_accum)

            self.wgmma_commit()
            if wg_wait >= 0:
                self.wgmma_wait(wg_wait)
            self.wgmma_fence_c(C_buf)
            self.wgmma_fence_a(A_buf)

        return _warp_mma(A_buf, C_buf)

    # ---- Atom-level interface ----

    @property
    def wgmma_num_inst_m(self) -> int:
        """Number of WGMMA instruction atoms along the M dimension."""
        return 4 * self.warp_row_tiles // self.wgmma_inst_m

    @property
    def wgmma_num_inst_n(self) -> int:
        """Number of WGMMA instruction atoms along the N dimension."""
        return self.warp_col_tiles // self.wgmma_inst_n

    @property
    def wgmma_num_k_atoms(self) -> int:
        """Number of K-dimension micro-steps (``chunk // micro_size_k``)."""
        return self.chunk // self.micro_size_k

    @property
    def wgmma_a_regs(self) -> int:
        """Number of 32-bit registers occupied by the A fragment (RS variant)."""
        a_bits = DataType(self.a_dtype).bits
        k_dim = self.chunk
        micro_size_k = self.micro_size_k
        return ((self.warp_rows * self.local_size_a * (k_dim // micro_size_k)) * a_bits + 31) // 32

    @property
    def wgmma_accum_regs(self) -> int:
        """Number of 32-bit registers occupied by the accumulator fragment."""
        m_dim = self.block_row_warps * self.warp_row_tiles
        accum_bits = DataType(self.accum_dtype).bits
        return ((m_dim // 64) * self.warp_cols * self.local_size_out * accum_bits + 31) // 32

    # -- Descriptor parameter computation (pure Python, no TIR) --

    def compute_wgmma_b_desc_params(self, B_region: BufferRegion) -> WGMMADescriptorParams:
        """Compute B descriptor parameters from the B shared buffer region.

        This is a pure-Python helper -- no TIR code is emitted.
        The returned ``WGMMADescriptorParams`` is passed to
        ``init_wgmma_b_desc()`` and ``wgmma_*_atom()`` methods.
        """
        n_dim = self.block_col_warps * self.warp_col_tiles
        k_dim = self.chunk
        micro_size_k = self.micro_size_k
        elems_in_bytes = DataType(self.a_dtype).bits // 8
        b_is_k_major = self.b_transposed

        b_swizzle_mode = self._determinate_swizzle_mode(B_region, self.b_shared_layout)
        b_swizzle_atom_elems = n_dim if b_swizzle_mode.is_none() else b_swizzle_mode.swizzle_byte_size() // elems_in_bytes

        b_leading_byte_offset = (8 * 8 * elems_in_bytes) if b_is_k_major else (8 * n_dim * elems_in_bytes)
        b_stride_byte_offset = (8 * k_dim * elems_in_bytes) if b_is_k_major else (0 if n_dim == 8 else (8 * 8 * elems_in_bytes))
        if not b_swizzle_mode.is_none():
            if b_is_k_major:
                b_leading_byte_offset = 16
                b_stride_byte_offset = 8 * b_swizzle_mode.swizzle_byte_size()
            else:
                b_n_axis_atoms = n_dim // b_swizzle_atom_elems
                if b_n_axis_atoms <= 1:
                    b_leading_byte_offset = 0
                else:
                    b_leading_byte_offset = 8 * 8 * elems_in_bytes * k_dim
                if b_n_axis_atoms <= 1:
                    b_stride_byte_offset = 8 * elems_in_bytes * n_dim
                else:
                    b_stride_byte_offset = 8 * elems_in_bytes * b_swizzle_atom_elems

        return WGMMADescriptorParams(
            swizzle_mode=int(b_swizzle_mode),
            leading_byte_offset=int(b_leading_byte_offset >> 4),
            stride_byte_offset=int(b_stride_byte_offset >> 4),
            swizzle_atom_elems=b_swizzle_atom_elems,
            k_atom_size=max(b_swizzle_atom_elems // micro_size_k, 1),
            elems_in_bytes=elems_in_bytes,
            is_k_major=b_is_k_major,
        )

    def compute_wgmma_a_desc_params(self, A_region: BufferRegion) -> WGMMADescriptorParams:
        """Compute A descriptor parameters from the A shared buffer region (SS variant).

        This is a pure-Python helper -- no TIR code is emitted.
        The returned ``WGMMADescriptorParams`` is passed to
        ``init_wgmma_a_desc()`` and ``wgmma_ss_atom()`` methods.
        """
        m_dim = self.block_row_warps * self.warp_row_tiles
        k_dim = self.chunk
        micro_size_k = self.micro_size_k
        elems_in_bytes = DataType(self.a_dtype).bits // 8
        a_is_k_major = not self.a_transposed

        a_swizzle_mode = self._determinate_swizzle_mode(A_region, self.a_shared_layout)
        a_swizzle_atom_elems = a_swizzle_mode.swizzle_byte_size() // elems_in_bytes

        a_leading_byte_offset = (8 * 8 * elems_in_bytes) if a_is_k_major else (8 * m_dim * elems_in_bytes)
        a_stride_byte_offset = (8 * k_dim * elems_in_bytes) if a_is_k_major else (8 * 8 * elems_in_bytes)
        if not a_swizzle_mode.is_none():
            if a_is_k_major:
                a_leading_byte_offset = 16
                a_stride_byte_offset = 8 * a_swizzle_mode.swizzle_byte_size()
            else:
                a_m_axis_atoms = m_dim // a_swizzle_atom_elems
                if a_m_axis_atoms <= 1:
                    a_leading_byte_offset = 0
                else:
                    a_leading_byte_offset = 8 * a_swizzle_mode.swizzle_atom_size() * (a_swizzle_mode.swizzle_byte_size() // elems_in_bytes)
                if a_m_axis_atoms <= 1:
                    a_stride_byte_offset = 8 * elems_in_bytes * m_dim
                else:
                    a_stride_byte_offset = 8 * elems_in_bytes * a_swizzle_atom_elems

        return WGMMADescriptorParams(
            swizzle_mode=int(a_swizzle_mode),
            leading_byte_offset=int(a_leading_byte_offset >> 4),
            stride_byte_offset=int(a_stride_byte_offset >> 4),
            swizzle_atom_elems=a_swizzle_atom_elems,
            k_atom_size=max(a_swizzle_atom_elems // micro_size_k, 1),
            elems_in_bytes=elems_in_bytes,
            is_k_major=a_is_k_major,
        )

    # -- Descriptor initialization (emit TIR) --

    def init_wgmma_b_desc(
        self,
        desc_b: Buffer,
        B_region: BufferRegion,
        b_params: WGMMADescriptorParams,
    ):
        """Emit TIR to initialize a pre-allocated WGMMA B descriptor.

        Parameters
        ----------
        desc_b : Buffer
            A descriptor buffer allocated via ``T.alloc_wgmma_desc()``.
        B_region : BufferRegion
            The B operand shared memory region.
        b_params : WGMMADescriptorParams
            Pre-computed parameters from ``compute_wgmma_b_desc_params()``.
        """
        B_ptr = retrive_ptr_from_buffer_region(B_region)
        swizzle_mode = b_params.swizzle_mode
        lbo = b_params.leading_byte_offset
        sbo = b_params.stride_byte_offset

        @T.macro
        def _init_b_desc(desc_b, B_ptr):
            T.initialize_wgmma_descriptor(desc_b, B_ptr, swizzle_mode, lbo, sbo)

        return _init_b_desc(desc_b, B_ptr)

    def init_wgmma_a_desc(
        self,
        desc_a: Buffer,
        A_region: BufferRegion,
        a_params: WGMMADescriptorParams,
    ):
        """Emit TIR to initialize a pre-allocated WGMMA A descriptor (SS variant).

        Parameters
        ----------
        desc_a : Buffer
            A descriptor buffer allocated via ``T.alloc_wgmma_desc()``.
        A_region : BufferRegion
            The A operand shared memory region.
        a_params : WGMMADescriptorParams
            Pre-computed parameters from ``compute_wgmma_a_desc_params()``.
        """
        A_ptr = retrive_ptr_from_buffer_region(A_region)
        swizzle_mode = a_params.swizzle_mode
        lbo = a_params.leading_byte_offset
        sbo = a_params.stride_byte_offset

        @T.macro
        def _init_a_desc(desc_a, A_ptr):
            T.initialize_wgmma_descriptor(desc_a, A_ptr, swizzle_mode, lbo, sbo)

        return _init_a_desc(desc_a, A_ptr)

    # -- Fence / Arrive / Commit / Wait primitives --

    def wgmma_fence_a(self, A_buf: Buffer):
        """Emit ``warpgroup_fence_operand`` for the A fragment buffer."""
        a_regs = self.wgmma_a_regs

        @T.macro
        def _fence_a(A_buf):
            T.warpgroup_fence_operand(A_buf, num_regs=a_regs)

        return _fence_a(A_buf)

    def wgmma_fence_c(self, C_buf: Buffer):
        """Emit ``warpgroup_fence_operand`` for the accumulator buffer."""
        accum_regs = self.wgmma_accum_regs

        @T.macro
        def _fence_c(C_buf):
            T.warpgroup_fence_operand(C_buf, num_regs=accum_regs)

        return _fence_c(C_buf)

    def wgmma_arrive(self):
        """Emit ``warpgroup_arrive()``."""

        @T.macro
        def _arrive():
            T.warpgroup_arrive()

        return _arrive()

    def wgmma_commit(self):
        """Emit ``warpgroup_commit_batch()``."""

        @T.macro
        def _commit():
            T.warpgroup_commit_batch()

        return _commit()

    def wgmma_wait(self, n: int = 0):
        """Emit ``warpgroup_wait(n)``."""

        @T.macro
        def _wait():
            T.warpgroup_wait(n)

        return _wait()

    # -- Atom emission --

    def wgmma_rs_atom(
        self,
        A_buf: Buffer,
        desc_b: Buffer,
        C_buf: Buffer,
        inst_m_idx: int,
        inst_n_idx: int,
        ki: int,
        b_params: WGMMADescriptorParams,
        clear_accum: PrimExpr = False,
    ):
        """Emit a single WGMMA RS instruction for atom ``(inst_m_idx, inst_n_idx, ki)``.

        Must be called between a ``wgmma_fence_a``/``wgmma_fence_c``/``wgmma_arrive``
        sequence and a ``wgmma_commit``/``wgmma_wait`` sequence.

        Calling this for every ``(j, i, ki)`` in
        ``T.grid(wgmma_num_inst_n, wgmma_num_inst_m, wgmma_num_k_atoms)``
        produces identical TIR to ``wgmma_rs()``.

        Parameters
        ----------
        A_buf : Buffer
            Fragment buffer for operand A (in registers).
        desc_b : Buffer
            Initialized B descriptor (from ``init_wgmma_b_desc``).
        C_buf : Buffer
            Accumulator fragment buffer.
        inst_m_idx : int
            M-dimension atom index (0 .. wgmma_num_inst_m - 1).
        inst_n_idx : int
            N-dimension atom index (0 .. wgmma_num_inst_n - 1).
        ki : int
            K-dimension atom index (0 .. wgmma_num_k_atoms - 1).
        b_params : WGMMADescriptorParams
            Pre-computed B descriptor parameters.
        clear_accum : PrimExpr
            Whether to zero the accumulator on the first K atom.
        """
        local_size_a = self.local_size_a
        local_size_out = self.local_size_out
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        micro_size_k = self.micro_size_k
        n_dim = self.block_col_warps * self.warp_col_tiles
        k_dim = self.chunk
        wgmma_inst_n = self.wgmma_inst_n
        num_inst_n = self.wgmma_num_inst_n
        a_dtype_abbrv = self.a_dtype_abbrv
        b_dtype_abbrv = self.b_dtype_abbrv
        accum_dtype = self.accum_dtype
        accum_dtype_abbrv = self.accum_dtype_abbrv
        wgmma_prefix = self.wgmma_prefix
        b_transposed = self.b_transposed
        elems_in_bytes = b_params.elems_in_bytes
        bk_atom_size = b_params.k_atom_size
        b_swizzle_atom_elems = b_params.swizzle_atom_elems

        thread_binding = self.get_thread_binding()

        A_offset = ki * warp_rows * local_size_a + inst_m_idx * local_size_a
        C_offset = inst_m_idx * warp_cols * local_size_out + inst_n_idx * warp_cols * local_size_out // num_inst_n

        @T.macro
        def _rs_atom(A_buf, desc_b, C_buf):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            warp_j = warp_n * num_inst_n + inst_n_idx
            scale_out = T.Select(ki != 0, 1, T.Select(clear_accum, 0, 1))

            B_offset = (
                (ki // bk_atom_size) * n_dim * b_swizzle_atom_elems
                + warp_j * wgmma_inst_n * b_swizzle_atom_elems
                + (ki % bk_atom_size) * micro_size_k
                if b_params.is_k_major
                else (
                    ki * b_swizzle_atom_elems * micro_size_k + warp_j * wgmma_inst_n * (k_dim if n_dim // b_swizzle_atom_elems > 1 else 1)
                )
            )

            T.ptx_wgmma_rs(
                accum_dtype,
                wgmma_prefix,
                b_transposed,
                a_dtype_abbrv,
                b_dtype_abbrv,
                accum_dtype_abbrv,
                A_buf.data,
                A_offset,
                desc_b.data,
                (B_offset * elems_in_bytes) >> 4,
                C_buf.data,
                C_offset,
                scale_out,
                1,
                1,
            )

        return _rs_atom(A_buf, desc_b, C_buf)

    def wgmma_ss_atom(
        self,
        desc_a: Buffer,
        desc_b: Buffer,
        C_buf: Buffer,
        inst_m_idx: int,
        inst_n_idx: int,
        ki: int,
        a_params: WGMMADescriptorParams,
        b_params: WGMMADescriptorParams,
        clear_accum: PrimExpr = False,
    ):
        """Emit a single WGMMA SS instruction for atom ``(inst_m_idx, inst_n_idx, ki)``.

        Must be called between fence/arrive and commit/wait sequences.

        Parameters
        ----------
        desc_a : Buffer
            Initialized A descriptor (from ``init_wgmma_a_desc``).
        desc_b : Buffer
            Initialized B descriptor (from ``init_wgmma_b_desc``).
        C_buf : Buffer
            Accumulator fragment buffer.
        inst_m_idx : int
            M-dimension atom index (0 .. wgmma_num_inst_m - 1).
        inst_n_idx : int
            N-dimension atom index (0 .. wgmma_num_inst_n - 1).
        ki : int
            K-dimension atom index (0 .. wgmma_num_k_atoms - 1).
        a_params : WGMMADescriptorParams
            Pre-computed A descriptor parameters.
        b_params : WGMMADescriptorParams
            Pre-computed B descriptor parameters.
        clear_accum : PrimExpr
            Whether to zero the accumulator on the first K atom.
        """
        local_size_out = self.local_size_out
        warp_cols = self.warp_cols
        micro_size_k = self.micro_size_k
        m_dim = self.block_row_warps * self.warp_row_tiles
        n_dim = self.block_col_warps * self.warp_col_tiles
        k_dim = self.chunk
        wgmma_inst_n = self.wgmma_inst_n
        num_inst_m = self.wgmma_num_inst_m
        num_inst_n = self.wgmma_num_inst_n
        a_dtype_abbrv = self.a_dtype_abbrv
        b_dtype_abbrv = self.b_dtype_abbrv
        accum_dtype = self.accum_dtype
        accum_dtype_abbrv = self.accum_dtype_abbrv
        wgmma_prefix = self.wgmma_prefix
        a_is_k_major = not self.a_transposed
        b_is_k_major = self.b_transposed
        a_elems_in_bytes = a_params.elems_in_bytes
        b_elems_in_bytes = b_params.elems_in_bytes
        ak_atom_size = a_params.k_atom_size
        bk_atom_size = b_params.k_atom_size
        a_swizzle_atom_elems = a_params.swizzle_atom_elems
        b_swizzle_atom_elems = b_params.swizzle_atom_elems

        thread_binding = self.get_thread_binding()

        C_offset = inst_m_idx * warp_cols * local_size_out + inst_n_idx * warp_cols * local_size_out // num_inst_n

        @T.macro
        def _ss_atom(desc_a, desc_b, C_buf):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            scale_out = T.Select(ki != 0, 1, T.Select(clear_accum, 0, 1))
            warp_i = (warp_m // 4) * num_inst_m + inst_m_idx
            warp_j = warp_n * num_inst_n + inst_n_idx

            A_offset = (
                (ki % ak_atom_size) * micro_size_k
                + warp_i * 64 * a_swizzle_atom_elems
                + (ki // ak_atom_size) * m_dim * a_swizzle_atom_elems
                if a_is_k_major
                else warp_i * 64 * k_dim + ki * a_swizzle_atom_elems * micro_size_k
            )
            B_offset = (
                (ki // bk_atom_size) * n_dim * b_swizzle_atom_elems
                + (ki % bk_atom_size) * micro_size_k
                + warp_j * wgmma_inst_n * b_swizzle_atom_elems
                if b_is_k_major
                else (
                    ki * b_swizzle_atom_elems * micro_size_k + warp_j * wgmma_inst_n * (k_dim if n_dim // b_swizzle_atom_elems > 1 else 1)
                )
            )

            T.ptx_wgmma_ss(
                accum_dtype,
                wgmma_prefix,
                a_is_k_major,
                b_is_k_major,
                a_dtype_abbrv,
                b_dtype_abbrv,
                accum_dtype_abbrv,
                desc_a.data,
                (A_offset * a_elems_in_bytes) >> 4,
                desc_b.data,
                (B_offset * b_elems_in_bytes) >> 4,
                C_buf.data,
                C_offset,
                scale_out,
                1,
                1,
            )

        return _ss_atom(desc_a, desc_b, C_buf)

    def make_mma_load_layout(self, local_buf: Buffer, matrix: str = "A") -> T.Fragment:
        """
        Create a layout function for storing MMA results into a fragment buffer.
        This layout is used in conjunction with `inverse_mma_store_layout` to
        map fragment indices to threads and local indices.

        Parameters
        ----------
        local_buf : tir.Buffer
            The local buffer representing a fragment of a matrix.

        Returns
        -------
        T.Fragment
            A fragment object that describes how threads and indices
            in `local_buf` are laid out.

        Raises
        ------
        AssertionError
            If `local_buf` is not detected to be a fragment buffer.
        """
        from tilelang.utils import is_fragment

        assert matrix in ["A"], "matrix should be A for WGMMA"
        dtype = self.a_dtype
        dtype_bits = DataType(dtype).bits
        transposed = self.a_transposed

        # s represents spatial axis
        # r represents reduction axis
        # sr represents the two dims are spatial + reduction
        # rs represents the two dims are reduction + spatial
        # sr also can represent a non-transposed basic layout
        # then rs also can represent a transposed basic layout
        transform_func_sr_a: Callable = None
        if dtype_bits == 32:
            transform_func_sr_a = shared_16x8_to_mma_32x4_layout_sr_a
        elif dtype_bits == 16:
            transform_func_sr_a = shared_16x16_to_mma_32x8_layout_sr_a
        elif dtype_bits == 8:
            transform_func_sr_a = shared_16x32_to_mma_32x16_layout_sr_a
        else:
            raise ValueError(f"Unsupported dtype {dtype}")

        is_sr_conditions = [False]
        is_sr_conditions.append(not transposed)
        is_sr_axis_order = any(is_sr_conditions)

        # the layout of mma.sync is row.col.
        # so the b matrix expected a transposed basic layout
        transform_func: Callable = None
        transform_func = transform_func_sr_a if is_sr_axis_order else lambda i, j: transform_func_sr_a(j, i)

        assert is_fragment(local_buf), f"local_buf must be a fragment, but got {local_buf.scope()}"

        micro_size_s, micro_size_r = self.micro_size_x, self.micro_size_k

        block_row_warps, block_col_warps = (
            self.block_row_warps,
            self.block_col_warps,
        )

        inverse_mma_load_layout = IndexMap.from_func(transform_func, index_dtype=T.int32)

        def forward_thread(i: int, j: int) -> int:
            """
            Given the row index `i` and column index `j` in the fragment,
            """
            lane_id, _ = inverse_mma_load_layout.map_indices([i, j])
            return lane_id

        def forward_index(i: int, j: int) -> int:
            """
            Given the row index `i` and column index `j` in the fragment,
            """
            _, local_id = inverse_mma_load_layout.map_indices([i, j])
            return local_id

        base_fragment = T.Fragment(
            [micro_size_s, micro_size_r] if is_sr_axis_order else [micro_size_r, micro_size_s],
            forward_thread_fn=forward_thread,
            forward_index_fn=forward_index,
        )

        warp_rows = self.warp_rows
        chunk = self.chunk

        warp_s = warp_rows
        warp_r = chunk // micro_size_r
        block_s = block_row_warps
        replicate = block_col_warps

        if is_sr_axis_order:
            warp_fragment = base_fragment.repeat([block_s, 1], repeat_on_thread=True, lower_dim_first=False).replicate(replicate)
            block_fragment = warp_fragment.repeat([warp_s, warp_r], repeat_on_thread=False, lower_dim_first=False)
        else:
            # rs condition, transposed_a matrix
            warp_fragment = base_fragment.repeat([1, block_s], repeat_on_thread=True, lower_dim_first=False).replicate(replicate)
            block_fragment = warp_fragment.repeat([warp_r, warp_s], repeat_on_thread=False, lower_dim_first=True)

        return block_fragment

    def make_mma_store_layout(self, local_buf: Buffer) -> T.Fragment:
        """
        Create a layout function for storing MMA results into a fragment buffer.
        This layout is used in conjunction with `inverse_mma_store_layout` to
        map fragment indices to threads and local indices.

        Parameters
        ----------
        local_buf : tir.Buffer
            The local buffer representing a fragment of a matrix.

        Returns
        -------
        T.Fragment
            A fragment object that describes how threads and indices
            in `local_buf` are laid out.

        Raises
        ------
        AssertionError
            If `local_buf` is not detected to be a fragment buffer.
        """
        inverse_mma_store_layout = self.get_store_index_map(inverse=True)
        assert is_fragment(local_buf), "local_buf must be a fragment"
        micro_size_x, micro_size_y = self.micro_size_x, self.micro_size_y
        block_row_warps, block_col_warps = self.block_row_warps, self.block_col_warps
        warp_rows, warp_cols = self.warp_rows, self.warp_cols

        def forward_thread(i: int, j: int) -> int:
            """
            Given the row index `i` and column index `j` in the fragment,
            map them to a thread index according to `inverse_mma_store_layout`.
            """
            lane_id, _ = inverse_mma_store_layout.map_indices([i, j])
            return lane_id

        def forward_index(i: int, j: int) -> int:
            """
            Given the row index `i` and column index `j` in the fragment,
            map them to a local index in a single thread according
            to `inverse_mma_store_layout`.
            """
            _, local_id = inverse_mma_store_layout.map_indices([i, j])
            return local_id

        # reproduce src/layout/gemm_layouts.cc::makeGemmFragmentCHopper
        base_fragment = T.Fragment(
            [micro_size_x, micro_size_y],
            forward_thread_fn=forward_thread,
            forward_index_fn=forward_index,
        )
        warp_n_layout = base_fragment.repeat([1, warp_cols], False, False)
        block_layout = warp_n_layout.repeat([block_row_warps, block_col_warps], True, False)
        warp_m_layout = block_layout.repeat([warp_rows, 1], False, False)
        return warp_m_layout
