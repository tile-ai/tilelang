from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
import tilelang.language as T
from .mma_macro_generator import TensorCoreIntrinEmitter as MMAIntrinEmitter
from tvm import DataType
from tvm.tir import PrimExpr, Buffer, Var, BufferLoad, BufferRegion
from tilelang import tvm as tvm
from tilelang import _ffi_api
from tilelang.utils import is_tensor_memory
from tilelang.layout import (
    Layout,
    make_full_bank_swizzled_layout,
    make_half_bank_swizzled_layout,
    make_quarter_bank_swizzled_layout,
    make_linear_layout,
)
from tvm.runtime import convert

lift = convert


@dataclass(frozen=True)
class TCGEN05DescriptorParams:
    """Pre-computed parameters for TCGEN05 descriptor initialization and atom offset computation.

    Returned by ``compute_tcgen05_*_desc_params()`` and consumed by
    ``init_tcgen05_*_desc()`` and ``tcgen05_*_atom()`` methods.
    """

    swizzle_mode: int
    """SwizzleMode enum value (passed directly to ``T.initialize_tcgen05_descriptor``)."""
    leading_byte_offset: int
    """LBO >> 4, ready to pass to ``T.initialize_tcgen05_descriptor``."""
    stride_byte_offset: int
    """SBO >> 4, ready to pass to ``T.initialize_tcgen05_descriptor``."""
    swizzle_atom_elems: int
    """Number of elements per swizzle atom along the non-K dimension."""
    k_atom_size: int
    """``max(swizzle_atom_elems // micro_size_k, 1)``."""
    elems_in_bytes: int
    """Byte width of a single element: ``(DataType(dtype).bits + 7) // 8``."""
    is_k_major: bool
    """Whether the matrix is stored in K-major order (affects offset formula branching)."""


class SwizzleMode(IntEnum):
    # SWIZZLE_NONE = 0, SWIZZLE_32B = 3, SWIZZLE_64B = 2, SWIZZLE_128B = 1
    NONE = 0
    SWIZZLE_128B = 2
    SWIZZLE_64B = 4
    SWIZZLE_32B = 6

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
    """Intrinsic emitter for Blackwell (SM100) TCGEN5MMA instructions.

    Generates TIR macros that lower to ``tcgen05.mma`` PTX instructions for
    both the SS (Shared-Shared) and TS (TensorMemory-Shared) GEMM variants.
    Also provides layout helpers for tensor-memory (TMEM) buffers.
    """

    # should be rewritten to support dynamic k_dim
    tcgen05_prefix: str

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
        is_m_first: bool = False,
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

    def _assign_a_shared_layout(self, layout: Layout):
        self.a_shared_layout = layout
        return self

    def _assign_b_shared_layout(self, layout: Layout):
        self.b_shared_layout = layout
        return self

    def _initialize_micro_size(self, m_dim: int = 16, k_dim: int = 16):
        # tcgen05 doesn't care about warp partitioning
        self.micro_size_x = m_dim
        self.micro_size_k = k_dim

    def _initialize_k_dim(self, a_dtype=T.float16):
        if isinstance(a_dtype, str):
            a_dtype = DataType(a_dtype)
        if a_dtype.bits == 6 or str(a_dtype) == "float4_e2m1fn":
            if self.chunk % 32 != 0:
                raise ValueError(f"TCGEN5MMA FP{a_dtype.bits} requires chunk to be a multiple of 32, got {self.chunk}")
            self.k_dim = 32
            return
        super()._initialize_k_dim(a_dtype)

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

    def tcgen05mma(self, A_buf: Buffer, B_buf: Buffer, C_local_buf: Buffer, mbar, clear_accum: PrimExpr = False):
        """Emit a TCGEN5MMA operation, dispatching to SS or TS variant based on A's memory scope.

        If *A_buf* resides in tensor memory (``shared.tmem``), the TS variant is
        emitted; otherwise the SS variant is used (both A and B from shared memory).

        Parameters
        ----------
        A_buf : Buffer
            Operand A — either in shared memory (SS) or tensor memory (TS).
        B_buf : Buffer
            Operand B in shared memory.
        C_local_buf : Buffer
            Accumulator buffer in tensor memory.
        mbar : PrimExpr
            Memory barrier used for MMA completion signalling.
        clear_accum : PrimExpr
            Whether to zero the accumulator before the first MMA.
        """
        if is_tensor_memory(A_buf):
            return self.tcgen05mma_ts(A_buf, B_buf, C_local_buf, mbar, clear_accum)
        return self.tcgen05mma_ss(A_buf, B_buf, C_local_buf, mbar, clear_accum)

    def tcgen05mma_ss(self, A_buf: Buffer, B_buf: Buffer, C_local_buf: Buffer, mbar, clear_accum: PrimExpr = False):
        """Emit the SS (Shared-Shared) variant of TCGEN5MMA.

        Reads operand A and B from shared memory via a descriptor.

        Parameters
        ----------
        A_buf : Buffer
            Operand A in shared memory.
        B_buf : Buffer
            Operand B in shared memory.
        C_local_buf : Buffer
            Accumulator buffer in tensor memory.
        mbar : PrimExpr
            Memory barrier for MMA completion signalling.
        clear_accum : PrimExpr
            Whether to zero the accumulator before the first MMA.
        """
        micro_size_k = self.micro_size_k
        k_dim = self.chunk
        assert k_dim >= micro_size_k, f"k_dim must be greater than or equal to {micro_size_k}, got k_dim: {k_dim}"

        num_inst_m = self.tcgen05_num_inst_m
        num_inst_n = self.tcgen05_num_inst_n
        num_k_atoms = self.tcgen05_num_k_atoms
        a_params = self.compute_tcgen05_a_desc_params(A_buf)
        b_params = self.compute_tcgen05_b_desc_params(B_buf)
        instr_desc = self.compute_tcgen05_instr_desc()

        @T.macro
        def _warp_mma_ss(A_buf, B_buf, C_local_buf, mbar):
            desc_a = T.alloc_tcgen05_smem_desc()
            desc_b = T.alloc_tcgen05_smem_desc()
            self.init_tcgen05_a_desc(desc_a, A_buf, a_params)
            self.init_tcgen05_b_desc(desc_b, B_buf, b_params)

            for j in T.unroll(num_inst_n):
                for i in T.unroll(num_inst_m):
                    for ki in T.unroll(0, num_k_atoms):
                        self.tcgen05_ss_atom(desc_a, desc_b, C_local_buf, i, j, ki, a_params, b_params, instr_desc, clear_accum)
            self.tcgen05_atom_arrive(mbar)

        return _warp_mma_ss(A_buf, B_buf, C_local_buf, mbar)

    def tcgen05mma_ts(self, A_buf, B_buf, C_local_buf, mbar, clear_accum: PrimExpr = False):
        """Emit the TS (TensorMemory-Shared) variant of TCGEN5MMA.

        Reads operand A directly from tensor memory (TMEM) and operand B from
        shared memory via a descriptor.  The TMEM column offset for A is
        computed assuming packed storage (e.g. two ``bfloat16`` values per
        ``uint32`` column) to match the output of ``tcgen05.st``.

        Parameters
        ----------
        A_buf : Buffer
            Operand A residing in tensor memory (``shared.tmem``).
        B_buf : Buffer
            Operand B in shared memory.
        C_local_buf : Buffer
            Accumulator buffer in tensor memory.
        mbar : PrimExpr
            Memory barrier for MMA completion signalling.
        clear_accum : PrimExpr
            Whether to zero the accumulator before the first MMA.
        """
        micro_size_k = self.micro_size_k
        k_dim = self.chunk
        assert k_dim >= micro_size_k, f"k_dim must be >= {micro_size_k}, got {k_dim}"

        num_inst_m = self.tcgen05_num_inst_m
        num_inst_n = self.tcgen05_num_inst_n
        num_k_atoms = self.tcgen05_num_k_atoms
        b_params = self.compute_tcgen05_b_desc_params(B_buf)
        instr_desc = self.compute_tcgen05_instr_desc()

        # Resolve the TMEM data pointer for A
        if isinstance(A_buf, BufferRegion):
            a_tmem_data = A_buf.buffer.data
        elif isinstance(A_buf, Buffer):
            a_tmem_data = A_buf.data
        else:
            raise ValueError(f"Unsupported A_buf type for TS variant: {type(A_buf)}")

        @T.macro
        def _warp_mma_ts(a_data, B_buf, C_local_buf, mbar):
            desc_b = T.alloc_tcgen05_smem_desc()
            self.init_tcgen05_b_desc(desc_b, B_buf, b_params)

            for j in T.unroll(num_inst_n):
                for i in T.unroll(num_inst_m):
                    for ki in T.unroll(0, num_k_atoms):
                        self.tcgen05_ts_atom(a_data, desc_b, C_local_buf, i, j, ki, b_params, instr_desc, clear_accum)
            self.tcgen05_atom_arrive(mbar)

        return _warp_mma_ts(a_tmem_data, B_buf, C_local_buf, mbar)

    def tcgen05mma_blockscaled(
        self,
        A_buf: Buffer,
        B_buf: Buffer,
        C_local_buf: Buffer,
        SFA_tmem,
        SFB_tmem,
        mbar,
        clear_accum: PrimExpr = False,
        sf_a_id=0,
        sf_b_id=0,
    ):
        """Emit a block-scaled TCGEN5MMA (SS variant with TMEM scale factors).

        Uses ``tcgen05.mma.cta_group::1|2.kind::mxf8f6f4.block_scale`` PTX instruction.
        Scale factors must already reside in tensor memory.
        """
        m_dim = self.block_row_warps * self.warp_row_tiles
        micro_size_k = self.micro_size_k
        k_dim, n_dim = self.chunk, self.block_col_warps * self.warp_col_tiles

        assert k_dim >= micro_size_k

        a_is_k_major = not self.a_transposed
        b_is_k_major = self.b_transposed
        a_swizzle_mode = self._determinate_swizzle_mode(A_buf, self.a_shared_layout)

        elems_in_bytes = (DataType(self.a_dtype).bits + 7) // 8

        if len(self.meta) != 5:
            self.get_tcgen5_mma_meta(m_dim, n_dim, k_dim, disable_2cta=False)
        if len(self.meta) != 5:
            raise ValueError(
                f"Unsupported TCGEN5MMA configuration for block-scaled: M={m_dim}, N={n_dim}, "
                f"K={k_dim}, A dtype={self.a_dtype}, accum dtype={self.accum_dtype}"
            )
        atom_m, atom_n, _, _, enable_2cta = self.tcgen05_meta_unpacked
        atom_m_per_cta = atom_m // 2 if enable_2cta else atom_m

        a_swizzle_atom_elems = a_swizzle_mode.swizzle_byte_size() // elems_in_bytes

        # Block-scaled A LBO/SBO differ from regular SS (uses atom_m_per_cta instead of m_dim)
        a_leading_byte_offset = (8 * 8 * elems_in_bytes) if a_is_k_major else (8 * atom_m_per_cta * elems_in_bytes)
        a_stride_byte_offset = (8 * k_dim * elems_in_bytes) if a_is_k_major else (8 * 8 * elems_in_bytes)
        if not a_swizzle_mode.is_none():
            if a_is_k_major:
                a_leading_byte_offset = 16
                a_stride_byte_offset = 8 * a_swizzle_mode.swizzle_byte_size()
            else:
                a_m_axis_atoms = atom_m_per_cta // a_swizzle_atom_elems
                a_leading_byte_offset = k_dim * a_swizzle_mode.swizzle_byte_size() if a_m_axis_atoms > 1 else 0
                a_stride_byte_offset = (
                    8 * elems_in_bytes * a_swizzle_atom_elems if a_m_axis_atoms > 1 else 8 * elems_in_bytes * atom_m_per_cta
                )

        a_params = TCGEN05DescriptorParams(
            swizzle_mode=int(a_swizzle_mode),
            leading_byte_offset=int(a_leading_byte_offset >> 4),
            stride_byte_offset=int(a_stride_byte_offset >> 4),
            swizzle_atom_elems=a_swizzle_atom_elems,
            k_atom_size=max(a_swizzle_atom_elems // micro_size_k, 1),
            elems_in_bytes=elems_in_bytes,
            is_k_major=a_is_k_major,
        )
        b_params = self.compute_tcgen05_b_desc_params(B_buf)

        base_instr_desc = self.get_tcgen5_blockscaled_instr_desc(
            atom_m,
            atom_n,
            a_is_k_major,
            b_is_k_major,
            1,
            1,
            0,
            0,
        )

        num_inst_m = m_dim // atom_m_per_cta
        num_inst_n = n_dim // atom_n
        num_k_atoms = self.tcgen05_num_k_atoms

        if isinstance(SFA_tmem, BufferRegion):
            sfa_data = SFA_tmem.buffer.data
        elif isinstance(SFA_tmem, Buffer):
            sfa_data = SFA_tmem.data
        else:
            raise ValueError(f"Unsupported SFA_tmem type: {type(SFA_tmem)}")

        if isinstance(SFB_tmem, BufferRegion):
            sfb_data = SFB_tmem.buffer.data
        elif isinstance(SFB_tmem, Buffer):
            sfb_data = SFB_tmem.data
        else:
            raise ValueError(f"Unsupported SFB_tmem type: {type(SFB_tmem)}")

        @T.macro
        def _warp_mma_blockscaled(A_buf, B_buf, C_local_buf, sfa_data, sfb_data, mbar):
            desc_a = T.alloc_tcgen05_smem_desc()
            desc_b = T.alloc_tcgen05_smem_desc()
            self.init_tcgen05_a_desc(desc_a, A_buf, a_params)
            self.init_tcgen05_b_desc(desc_b, B_buf, b_params)

            _sf_a = tvm.tir.const(sf_a_id, "int32") if isinstance(sf_a_id, int) else sf_a_id
            _sf_b = tvm.tir.const(sf_b_id, "int32") if isinstance(sf_b_id, int) else sf_b_id
            runtime_instr_desc = base_instr_desc | (_sf_a << 29) | (_sf_b << 4)
            for j in T.unroll(num_inst_n):
                for i in T.unroll(num_inst_m):
                    for ki in T.unroll(0, num_k_atoms):
                        self.tcgen05_blockscaled_atom(
                            desc_a,
                            desc_b,
                            C_local_buf,
                            sfa_data,
                            sfb_data,
                            i,
                            j,
                            ki,
                            a_params,
                            b_params,
                            runtime_instr_desc,
                            clear_accum,
                        )
            self.tcgen05_atom_arrive(mbar)

        return _warp_mma_blockscaled(A_buf, B_buf, C_local_buf, sfa_data, sfb_data, mbar)

    def get_tcgen5_blockscaled_instr_desc(
        self,
        atom_m: int,
        atom_n: int,
        a_is_k_major: bool,
        b_is_k_major: bool,
        scale_in_a: int,
        scale_in_b: int,
        a_sf_id: int,
        b_sf_id: int,
    ) -> PrimExpr:
        """Build the block-scaled instruction descriptor via FFI."""
        desc = _ffi_api.get_tcgen5_blockscaled_instr_desc(
            atom_m,
            atom_n,
            DataType(self.a_dtype),
            a_is_k_major,
            b_is_k_major,
            scale_in_a,
            scale_in_b,
            a_sf_id,
            b_sf_id,
        )
        return lift(desc)

    def make_mma_load_layout(self, local_buf: Buffer, matrix: str = "A") -> T.Fragment:
        raise NotImplementedError

    def make_mma_store_layout(self, tmem_buf: Buffer) -> Layout:
        """
        Create the TCGEN5 tensor-memory layout used to store MMA accumulators.

        Parameters
        ----------
        tmem_buf : tir.Buffer
            The local buffer representing tensormemory of a mma's output

        Returns
        -------
        Layout
            Layout object describing how logical (i, j) coordinates map to the
            swizzled tensor-memory offsets required by TCGEN5MMA.

        Raises
        ------
        AssertionError
            If `tmem_buf` is not detected to be a tensor-memory buffer.
        """
        assert is_tensor_memory(tmem_buf), "tmem_buf must reside in tensor memory (shared.tmem)"
        if len(tmem_buf.shape) != 2:
            raise ValueError(f"TCGEN5MMA expects a 2-D tensor-memory buffer, got shape {tmem_buf.shape}")

        m = int(tmem_buf.shape[0])
        n = int(tmem_buf.shape[1])
        k = int(self.chunk)

        meta = self.meta
        if len(meta) != 5:
            raise ValueError(
                f"Unsupported TCGEN5MMA configuration: M={m}, N={n}, K={k}, A dtype={self.a_dtype}, accum dtype={self.accum_dtype}"
            )
        atom_m, atom_n, _, _, enable_2cta = (int(x) for x in meta)
        atom_m_per_cta = atom_m // 2 if enable_2cta else atom_m

        if m % atom_m_per_cta != 0 or n % atom_n != 0:
            raise ValueError(f"Invalid TCGEN5MMA store layout for shape ({m}, {n}) with atoms ({atom_m}, {atom_n})")

        def forward(i: PrimExpr, j: PrimExpr):
            atom_idx = (i // atom_m_per_cta) + (j // atom_n) * (m // atom_m_per_cta)
            ai = i % atom_m_per_cta
            aj = j % atom_n

            # NOTE: Currently not all 7 layout are supported
            if atom_m == 256:
                # Layout A (2 cta)
                assert enable_2cta, "atom_m=256 for TCGEN5MMA must use 2cta"
                return [
                    ai % 128,
                    aj + atom_idx * atom_n,
                ]
            if atom_m == 128:
                if enable_2cta:
                    # Layout B
                    half_atom_n = atom_n // 2
                    return [
                        ai + (aj // half_atom_n) * 64,
                        (aj % half_atom_n) + atom_idx * half_atom_n,
                    ]
                else:
                    # Layout D
                    return [
                        ai,
                        aj + atom_idx * atom_n,
                    ]
            if atom_m == 64:
                # Layout E (.ws variant)
                half_atom_n = atom_n // 2
                return [
                    (ai // 32) * 32 + ai % 32 + (aj // half_atom_n) * 64,
                    (aj % half_atom_n) + atom_idx * half_atom_n,
                ]
            if atom_m == 32:
                # Layout G
                quarter_atom_n = atom_n // 4
                return [
                    ai % 32 + (aj // quarter_atom_n) * 32,
                    (aj % quarter_atom_n) + atom_idx * quarter_atom_n,
                ]

            raise ValueError(f"Unsupported TCGEN5 atom_m={atom_m}")

        return Layout([m, n], forward)

    def get_tcgen5_mma_meta(self, m: int, n: int, k: int, disable_2cta: bool):
        """Query the FFI for TCGEN5MMA atom metadata (atom_m, atom_n, atom_k, enable_ws, enable_2cta), and record them in `self.meta`."""
        self.meta = _ffi_api.get_tcgen5_mma_meta(
            int(m), int(n), int(k), DataType(self.a_dtype), DataType(self.accum_dtype), bool(disable_2cta)
        )

    def get_tcgen5_instr_desc(
        self, atom_m: int, atom_n: int, atom_k: int, a_is_k_major: bool, b_is_k_major: bool, scale_in_a: int, scale_in_b: int
    ) -> PrimExpr:
        """Build the 64-bit instruction descriptor for a ``tcgen05.mma`` PTX call."""
        desc = _ffi_api.get_tcgen5_instr_desc(
            atom_m,
            atom_n,
            atom_k,
            DataType(self.a_dtype),
            DataType(self.accum_dtype),
            a_is_k_major,
            b_is_k_major,
            scale_in_a,
            scale_in_b,
        )
        return lift(desc)

    # ---- Atom-level interface ----

    @property
    def tcgen05_meta_unpacked(self) -> tuple:
        """Return ``(atom_m, atom_n, atom_k, enable_ws, enable_2cta)`` as ints.

        Requires ``self.meta`` to have been set via ``get_tcgen5_mma_meta()``.
        """
        assert len(self.meta) == 5, "TCGEN05 meta not initialized; call get_tcgen5_mma_meta() first"
        return tuple(int(x) for x in self.meta)

    @property
    def tcgen05_num_inst_m(self) -> int:
        """Number of TCGEN05MMA instruction atoms along M (SS variant)."""
        atom_m, _, _, _, enable_2cta = self.tcgen05_meta_unpacked
        atom_m_per_cta = atom_m // 2 if enable_2cta else atom_m
        return self.block_row_warps * self.warp_row_tiles // atom_m_per_cta

    @property
    def tcgen05_num_inst_n(self) -> int:
        """Number of TCGEN05MMA instruction atoms along N."""
        _, atom_n, _, _, _ = self.tcgen05_meta_unpacked
        return self.block_col_warps * self.warp_col_tiles // atom_n

    @property
    def tcgen05_num_k_atoms(self) -> int:
        """Number of K-dimension micro-steps (``chunk // micro_size_k``)."""
        return self.chunk // self.micro_size_k

    @staticmethod
    def _access_ptr_from(buffer_or_load_or_region, access_type: str = "r"):
        """Resolve an access pointer from a Buffer, BufferLoad, or BufferRegion."""
        if isinstance(buffer_or_load_or_region, Buffer):
            return buffer_or_load_or_region.access_ptr(access_type)
        elif isinstance(buffer_or_load_or_region, BufferLoad):
            buffer_load = buffer_or_load_or_region
            offset, stride = 0, 1
            buffer = buffer_load.buffer
            for i, shape in enumerate(reversed(buffer.shape)):
                indice = buffer_load.indices[len(buffer_load.indices) - i - 1]
                if isinstance(indice, tvm.tir.Ramp):
                    offset += indice.base * stride
                elif isinstance(indice, (tvm.tir.IntImm, tvm.tir.PrimExpr)):
                    offset += indice * stride
                else:
                    raise ValueError(f"Unsupported index type: {type(indice)}")
                stride *= shape
            return buffer.access_ptr(access_type, offset=offset)
        elif isinstance(buffer_or_load_or_region, BufferRegion):
            buffer_region = buffer_or_load_or_region
            buffer = buffer_region.buffer
            offset, stride = 0, 1
            for i, shape in enumerate(reversed(buffer.shape)):
                offset += buffer_region.region[len(buffer_region.region) - i - 1].min * stride
                stride *= shape
            return buffer.access_ptr(access_type, offset=offset)
        else:
            raise ValueError(f"Unsupported buffer type: {type(buffer_or_load_or_region)}")

    # -- Descriptor parameter computation (pure Python, no TIR) --

    def compute_tcgen05_b_desc_params(self, B_buf) -> TCGEN05DescriptorParams:
        """Compute B descriptor parameters from the B shared buffer.

        This is a pure-Python helper -- no TIR code is emitted.
        The returned ``TCGEN05DescriptorParams`` is passed to
        ``init_tcgen05_b_desc()`` and ``tcgen05_*_atom()`` methods.

        Parameters
        ----------
        B_buf : Buffer or BufferRegion
            The B operand in shared memory.
        """
        atom_m, atom_n, _, _, enable_2cta = self.tcgen05_meta_unpacked
        n_dim = self.block_col_warps * self.warp_col_tiles
        n_dim_per_cta = n_dim // 2 if enable_2cta else n_dim
        k_dim = self.chunk
        micro_size_k = self.micro_size_k
        elems_in_bytes = (DataType(self.a_dtype).bits + 7) // 8
        b_is_k_major = self.b_transposed

        b_swizzle_mode = self._determinate_swizzle_mode(B_buf, self.b_shared_layout)
        b_swizzle_atom_elems = n_dim_per_cta if b_swizzle_mode.is_none() else b_swizzle_mode.swizzle_byte_size() // elems_in_bytes

        b_leading_byte_offset = (8 * 8 * elems_in_bytes) if b_is_k_major else (8 * n_dim_per_cta * elems_in_bytes)
        b_stride_byte_offset = (8 * k_dim * elems_in_bytes) if b_is_k_major else (0 if n_dim_per_cta == 8 else (8 * 8 * elems_in_bytes))
        if not b_swizzle_mode.is_none():
            if b_is_k_major:
                b_leading_byte_offset = 16
                b_stride_byte_offset = 8 * b_swizzle_mode.swizzle_byte_size()
            else:
                b_n_axis_atoms = n_dim_per_cta // b_swizzle_atom_elems
                if b_n_axis_atoms <= 1:
                    b_leading_byte_offset = 0
                else:
                    b_leading_byte_offset = 8 * 8 * elems_in_bytes * k_dim
                if b_n_axis_atoms <= 1:
                    b_stride_byte_offset = 8 * elems_in_bytes * n_dim_per_cta
                else:
                    b_stride_byte_offset = 8 * elems_in_bytes * b_swizzle_atom_elems

        return TCGEN05DescriptorParams(
            swizzle_mode=int(b_swizzle_mode),
            leading_byte_offset=int(b_leading_byte_offset >> 4),
            stride_byte_offset=int(b_stride_byte_offset >> 4),
            swizzle_atom_elems=b_swizzle_atom_elems,
            k_atom_size=max(b_swizzle_atom_elems // micro_size_k, 1),
            elems_in_bytes=elems_in_bytes,
            is_k_major=b_is_k_major,
        )

    def compute_tcgen05_a_desc_params(self, A_buf) -> TCGEN05DescriptorParams:
        """Compute A descriptor parameters from the A shared buffer (SS variant).

        This is a pure-Python helper -- no TIR code is emitted.

        Parameters
        ----------
        A_buf : Buffer or BufferRegion
            The A operand in shared memory.
        """
        m_dim = self.block_row_warps * self.warp_row_tiles
        k_dim = self.chunk
        micro_size_k = self.micro_size_k
        elems_in_bytes = (DataType(self.a_dtype).bits + 7) // 8
        a_is_k_major = not self.a_transposed

        a_swizzle_mode = self._determinate_swizzle_mode(A_buf, self.a_shared_layout)
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
                    a_leading_byte_offset = k_dim * a_swizzle_mode.swizzle_byte_size()
                if a_m_axis_atoms <= 1:
                    a_stride_byte_offset = 8 * elems_in_bytes * m_dim
                else:
                    a_stride_byte_offset = 8 * elems_in_bytes * a_swizzle_atom_elems

        return TCGEN05DescriptorParams(
            swizzle_mode=int(a_swizzle_mode),
            leading_byte_offset=int(a_leading_byte_offset >> 4),
            stride_byte_offset=int(a_stride_byte_offset >> 4),
            swizzle_atom_elems=a_swizzle_atom_elems,
            k_atom_size=max(a_swizzle_atom_elems // micro_size_k, 1),
            elems_in_bytes=elems_in_bytes,
            is_k_major=a_is_k_major,
        )

    # -- Descriptor initialization (emit TIR) --

    def init_tcgen05_b_desc(self, desc_b, B_buf, b_params: TCGEN05DescriptorParams):
        """Emit TIR to initialize a pre-allocated TCGEN05 B descriptor.

        Parameters
        ----------
        desc_b : Buffer
            A descriptor buffer allocated via ``T.alloc_tcgen05_smem_desc()``.
        B_buf : Buffer or BufferRegion
            The B operand in shared memory.
        b_params : TCGEN05DescriptorParams
            Pre-computed parameters from ``compute_tcgen05_b_desc_params()``.
        """
        access_ptr_from = self._access_ptr_from
        lbo = b_params.leading_byte_offset
        sbo = b_params.stride_byte_offset
        swizzle_mode = b_params.swizzle_mode
        B_ptr = access_ptr_from(B_buf, "r")

        @T.macro
        def _init_b(desc_b, B_ptr):
            T.initialize_tcgen05_descriptor(desc_b, B_ptr, lbo, sbo, 0, False, swizzle_mode)

        return _init_b(desc_b, B_ptr)

    def init_tcgen05_a_desc(self, desc_a, A_buf, a_params: TCGEN05DescriptorParams):
        """Emit TIR to initialize a pre-allocated TCGEN05 A descriptor (SS variant).

        Parameters
        ----------
        desc_a : Buffer
            A descriptor buffer allocated via ``T.alloc_tcgen05_smem_desc()``.
        A_buf : Buffer or BufferRegion
            The A operand in shared memory.
        a_params : TCGEN05DescriptorParams
            Pre-computed parameters from ``compute_tcgen05_a_desc_params()``.
        """
        access_ptr_from = self._access_ptr_from
        lbo = a_params.leading_byte_offset
        sbo = a_params.stride_byte_offset
        swizzle_mode = a_params.swizzle_mode
        A_ptr = access_ptr_from(A_buf, "r")

        @T.macro
        def _init_a(desc_a, A_ptr):
            T.initialize_tcgen05_descriptor(desc_a, A_ptr, lbo, sbo, 0, False, swizzle_mode)

        return _init_a(desc_a, A_ptr)

    # -- Instruction descriptor computation --

    def compute_tcgen05_instr_desc(self) -> PrimExpr:
        """Compute the 64-bit instruction descriptor using current meta.

        Requires ``self.meta`` to have been set via ``get_tcgen5_mma_meta()``.
        """
        atom_m, atom_n, atom_k, _, _ = self.tcgen05_meta_unpacked
        a_is_k_major = not self.a_transposed
        b_is_k_major = self.b_transposed
        return self.get_tcgen5_instr_desc(atom_m, atom_n, atom_k, a_is_k_major, b_is_k_major, 1, 1)

    # -- Arrive --

    def tcgen05_atom_arrive(self, mbar):
        """Emit ``tcgen05_mma_arrive(mbar)``."""
        _, _, _, _, enable_2cta = self.tcgen05_meta_unpacked

        @T.macro
        def _arrive(mbar):
            T.tcgen05_mma_arrive(mbar, arrive_2cta=bool(enable_2cta))

        return _arrive(mbar)

    # -- Atom emission --

    def tcgen05_ss_atom(
        self,
        desc_a,
        desc_b,
        C_local_buf: Buffer,
        inst_m_idx: int,
        inst_n_idx: int,
        ki: int,
        a_params: TCGEN05DescriptorParams,
        b_params: TCGEN05DescriptorParams,
        instr_desc: PrimExpr,
        clear_accum: PrimExpr = False,
    ):
        """Emit a single TCGEN05MMA SS instruction for atom ``(inst_m_idx, inst_n_idx, ki)``.

        Must be called after descriptor initialization and before ``tcgen05_atom_arrive()``.

        Parameters
        ----------
        desc_a, desc_b : Buffer
            Initialized A and B descriptors.
        C_local_buf : Buffer
            Accumulator buffer in tensor memory.
        inst_m_idx : int
            M-dimension atom index (0 .. tcgen05_num_inst_m - 1).
        inst_n_idx : int
            N-dimension atom index (0 .. tcgen05_num_inst_n - 1).
        ki : int
            K-dimension atom index (0 .. tcgen05_num_k_atoms - 1).
        a_params : TCGEN05DescriptorParams
            Pre-computed A descriptor parameters.
        b_params : TCGEN05DescriptorParams
            Pre-computed B descriptor parameters.
        instr_desc : PrimExpr
            Instruction descriptor from ``compute_tcgen05_instr_desc()``.
        clear_accum : PrimExpr
            Whether to zero the accumulator on the first K atom.
        """
        atom_m, atom_n, _, enable_ws, enable_2cta = self.tcgen05_meta_unpacked
        atom_m_per_cta = atom_m // 2 if enable_2cta else atom_m
        n_dim = self.block_col_warps * self.warp_col_tiles
        n_dim_per_cta = n_dim // 2 if enable_2cta else n_dim
        m_dim = self.block_row_warps * self.warp_row_tiles
        micro_size_k = self.micro_size_k
        k_dim = self.chunk
        accum_dtype_in_bits = DataType(self.accum_dtype).bits
        a_dtype_abbrv = self.a_dtype_abbrv
        a_elems_in_bytes = a_params.elems_in_bytes
        b_elems_in_bytes = b_params.elems_in_bytes
        ak_atom_size = a_params.k_atom_size
        bk_atom_size = b_params.k_atom_size
        a_swizzle_atom_elems = a_params.swizzle_atom_elems
        b_swizzle_atom_elems = b_params.swizzle_atom_elems
        mask_zero = T.cast(0, T.int32)

        # Pre-compute offsets
        if a_params.is_k_major:
            A_elem_offset = (
                (ki % ak_atom_size) * micro_size_k
                + inst_m_idx * atom_m_per_cta * a_swizzle_atom_elems
                + (ki // ak_atom_size) * m_dim * a_swizzle_atom_elems
            )
        else:
            A_elem_offset = inst_m_idx * atom_m_per_cta * k_dim + ki * a_swizzle_atom_elems * micro_size_k

        if b_params.is_k_major:
            B_elem_offset = (
                (ki // bk_atom_size) * n_dim_per_cta * b_swizzle_atom_elems
                + (ki % bk_atom_size) * micro_size_k
                + inst_n_idx * atom_n * b_swizzle_atom_elems
            )
        else:
            B_elem_offset = ki * b_swizzle_atom_elems * micro_size_k + inst_n_idx * atom_n * (
                k_dim if n_dim_per_cta // b_swizzle_atom_elems > 1 else 1
            )

        A_byte_offset = A_elem_offset * a_elems_in_bytes
        B_byte_offset = B_elem_offset * b_elems_in_bytes
        tmem_col_step = atom_n // (128 // atom_m_per_cta)
        C_offset = (inst_m_idx * n_dim + inst_n_idx * tmem_col_step) * accum_dtype_in_bits // 32

        @T.macro
        def _ss_atom(desc_a, desc_b, C_local_buf):
            scale_out = T.Select(ki != 0, 1, T.Select(clear_accum, 0, 1))
            T.ptx_tcgen05_mma_ss(
                a_dtype_abbrv,
                desc_a.data,
                A_byte_offset,
                desc_b.data,
                B_byte_offset,
                C_local_buf.data,
                C_offset,
                instr_desc,
                scale_out,
                mask_zero,
                mask_zero,
                mask_zero,
                mask_zero,
                enable_ws,
                enable_2cta,
            )

        return _ss_atom(desc_a, desc_b, C_local_buf)

    def tcgen05_ts_atom(
        self,
        a_tmem_data,
        desc_b,
        C_local_buf: Buffer,
        inst_m_idx: int,
        inst_n_idx: int,
        ki: int,
        b_params: TCGEN05DescriptorParams,
        instr_desc: PrimExpr,
        clear_accum: PrimExpr = False,
    ):
        """Emit a single TCGEN05MMA TS instruction for atom ``(inst_m_idx, inst_n_idx, ki)``.

        A resides in tensor memory; B in shared memory.

        Parameters
        ----------
        a_tmem_data : Var
            Data pointer for the A operand in tensor memory (e.g., ``A_buf.data``).
        desc_b : Buffer
            Initialized B descriptor.
        C_local_buf : Buffer
            Accumulator buffer in tensor memory.
        inst_m_idx : int
            M-dimension atom index.
        inst_n_idx : int
            N-dimension atom index.
        ki : int
            K-dimension atom index.
        b_params : TCGEN05DescriptorParams
            Pre-computed B descriptor parameters.
        instr_desc : PrimExpr
            Instruction descriptor from ``compute_tcgen05_instr_desc()``.
        clear_accum : PrimExpr
            Whether to zero the accumulator on the first K atom.
        """
        atom_m, atom_n, atom_k, _, enable_2cta = self.tcgen05_meta_unpacked
        atom_m_per_cta = atom_m // 2 if enable_2cta else atom_m
        n_dim = self.block_col_warps * self.warp_col_tiles
        n_dim_per_cta = n_dim // 2 if enable_2cta else n_dim
        micro_size_k = self.micro_size_k
        k_dim = self.chunk
        a_dtype_in_bits = DataType(self.a_dtype).bits
        accum_dtype_in_bits = DataType(self.accum_dtype).bits
        a_dtype_abbrv = self.a_dtype_abbrv
        b_elems_in_bytes = b_params.elems_in_bytes
        bk_atom_size = b_params.k_atom_size
        b_swizzle_atom_elems = b_params.swizzle_atom_elems
        mask_zero = T.cast(0, T.int32)

        # TMEM column geometry for A
        interleave = max(128 // atom_m, 1)
        a_tmem_cols_per_k_atom = atom_k * a_dtype_in_bits // 32 // interleave
        a_tmem_k_stride = k_dim * a_dtype_in_bits // 32 // interleave

        A_tmem_offset = inst_m_idx * a_tmem_k_stride + ki * a_tmem_cols_per_k_atom

        if b_params.is_k_major:
            B_elem_offset = (
                (ki // bk_atom_size) * n_dim_per_cta * b_swizzle_atom_elems
                + (ki % bk_atom_size) * micro_size_k
                + inst_n_idx * atom_n * b_swizzle_atom_elems
            )
        else:
            B_elem_offset = ki * b_swizzle_atom_elems * micro_size_k + inst_n_idx * atom_n * (
                k_dim if n_dim_per_cta // b_swizzle_atom_elems > 1 else 1
            )
        B_byte_offset = B_elem_offset * b_elems_in_bytes

        tmem_col_step = atom_n // (128 // atom_m_per_cta)
        C_offset = (inst_m_idx * n_dim + inst_n_idx * tmem_col_step) * accum_dtype_in_bits // 32

        @T.macro
        def _ts_atom(a_data, desc_b, C_local_buf):
            scale_out = T.Select(ki != 0, 1, T.Select(clear_accum, 0, 1))
            T.ptx_tcgen05_mma_ts(
                a_dtype_abbrv,
                a_data,
                A_tmem_offset,
                desc_b.data,
                B_byte_offset,
                C_local_buf.data,
                C_offset,
                instr_desc,
                scale_out,
                mask_zero,
                mask_zero,
                mask_zero,
                mask_zero,
            )

        return _ts_atom(a_tmem_data, desc_b, C_local_buf)

    def tcgen05_blockscaled_atom(
        self,
        desc_a,
        desc_b,
        C_local_buf: Buffer,
        sfa_data,
        sfb_data,
        inst_m_idx: int,
        inst_n_idx: int,
        ki: int,
        a_params: TCGEN05DescriptorParams,
        b_params: TCGEN05DescriptorParams,
        instr_desc: PrimExpr,
        clear_accum: PrimExpr = False,
    ):
        """Emit a single TCGEN05MMA block-scaled SS instruction.

        Parameters
        ----------
        desc_a, desc_b : Buffer
            Initialized A and B descriptors.
        C_local_buf : Buffer
            Accumulator buffer in tensor memory.
        sfa_data, sfb_data : Var
            Scale factor data pointers in tensor memory.
        inst_m_idx, inst_n_idx, ki : int
            Atom indices.
        a_params, b_params : TCGEN05DescriptorParams
            Pre-computed descriptor parameters.
        instr_desc : PrimExpr
            Block-scaled instruction descriptor (with SF IDs already encoded).
        clear_accum : PrimExpr
            Whether to zero the accumulator on the first K atom.
        """
        atom_m, atom_n, _, enable_ws, enable_2cta = self.tcgen05_meta_unpacked
        atom_m_per_cta = atom_m // 2 if enable_2cta else atom_m
        n_dim = self.block_col_warps * self.warp_col_tiles
        n_dim_per_cta = n_dim // 2 if enable_2cta else n_dim
        m_dim = self.block_row_warps * self.warp_row_tiles
        micro_size_k = self.micro_size_k
        k_dim = self.chunk
        accum_dtype_in_bits = DataType(self.accum_dtype).bits
        a_dtype_abbrv = self.a_dtype_abbrv
        a_elems_in_bytes = a_params.elems_in_bytes
        b_elems_in_bytes = b_params.elems_in_bytes
        ak_atom_size = a_params.k_atom_size
        bk_atom_size = b_params.k_atom_size
        a_swizzle_atom_elems = a_params.swizzle_atom_elems
        b_swizzle_atom_elems = b_params.swizzle_atom_elems

        if a_params.is_k_major:
            A_elem_offset = (
                (ki % ak_atom_size) * micro_size_k
                + inst_m_idx * atom_m_per_cta * a_swizzle_atom_elems
                + (ki // ak_atom_size) * m_dim * a_swizzle_atom_elems
            )
        else:
            A_elem_offset = inst_m_idx * atom_m_per_cta * k_dim + ki * a_swizzle_atom_elems * micro_size_k

        if b_params.is_k_major:
            B_elem_offset = (
                (ki // bk_atom_size) * n_dim_per_cta * b_swizzle_atom_elems
                + (ki % bk_atom_size) * micro_size_k
                + inst_n_idx * atom_n * b_swizzle_atom_elems
            )
        else:
            B_elem_offset = ki * b_swizzle_atom_elems * micro_size_k + inst_n_idx * atom_n * (
                k_dim if n_dim_per_cta // b_swizzle_atom_elems > 1 else 1
            )

        A_byte_offset = A_elem_offset * a_elems_in_bytes
        B_byte_offset = B_elem_offset * b_elems_in_bytes
        tmem_col_step = atom_n // (128 // atom_m_per_cta)
        C_offset = (inst_m_idx * n_dim + inst_n_idx * tmem_col_step) * accum_dtype_in_bits // 32

        @T.macro
        def _bs_atom(desc_a, desc_b, C_local_buf, sfa_data, sfb_data):
            scale_out = T.Select(ki != 0, 1, T.Select(clear_accum, 0, 1))
            T.ptx_tcgen05_mma_blockscaled_ss(
                a_dtype_abbrv,
                desc_a.data,
                A_byte_offset,
                desc_b.data,
                B_byte_offset,
                C_local_buf.data,
                C_offset,
                instr_desc,
                scale_out,
                sfa_data,
                0,
                sfb_data,
                0,
                0,
                0,
                enable_ws,
                enable_2cta,
            )

        return _bs_atom(desc_a, desc_b, C_local_buf, sfa_data, sfb_data)
