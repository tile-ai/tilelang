from __future__ import annotations
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
    """
    To eliminate Python syntax within TIR Macro.
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
        warp_row_tiles = self.warp_row_tiles
        warp_col_tiles = self.warp_col_tiles
        # For tcgen05, warp_row_tiles is 8 as we can use .ws to support m32
        assert warp_row_tiles >= 8, f"warp_row_tiles must be greater than 8, got {warp_row_tiles}"
        assert warp_row_tiles % 8 == 0, f"warp_row_tiles must be divisible by 8, got {warp_row_tiles}"
        assert warp_col_tiles >= 8, f"warp_col_tiles must be greater than 8, got {warp_col_tiles}"
        assert warp_col_tiles % 8 == 0, f"warp_col_tiles must be divisible by 8, got {warp_col_tiles}"

        # four warps per block
        self.warp_rows = warp_row_tiles // 8
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

    def tcgen05mma(self, A_buf: Buffer, B_buf: Buffer, C_local_buf: Buffer, mbar, clear_accum: PrimExpr = False):
        if is_tensor_memory(A_buf):
            return self.tcgen05mma_rs(A_buf, B_buf, C_local_buf, clear_accum)

        accum_dtype = self.accum_dtype
        m_dim = self.block_row_warps * self.warp_row_tiles
        micro_size_k = self.micro_size_k
        k_dim, n_dim = self.chunk, self.block_col_warps * self.warp_col_tiles
        scale_in_a = 1
        scale_in_b = 1

        assert k_dim >= micro_size_k, f"k_dim must be greater than or equal to {micro_size_k}, got k_dim: {k_dim}"

        a_is_k_major = not self.a_transposed
        b_is_k_major = self.b_transposed
        a_swizzle_mode = self._determinate_swizzle_mode(A_buf, self.a_shared_layout)
        b_swizzle_mode = self._determinate_swizzle_mode(B_buf, self.b_shared_layout)

        elems_in_bits = DataType(self.a_dtype).bits
        elems_in_bytes = elems_in_bits // 8
        a_swizzle_atom_elems = a_swizzle_mode.swizzle_byte_size() // elems_in_bytes
        b_swizzle_atom_elems = n_dim if b_swizzle_mode.is_none() else b_swizzle_mode.swizzle_byte_size() // elems_in_bytes
        accum_dtype_in_bits = DataType(accum_dtype).bits

        meta = self.get_tcgen5_mma_meta(m_dim, n_dim, k_dim)
        if len(meta) != 5:
            raise ValueError(
                f"Unsupported TCGEN5MMA configuration for desc generation: M={m_dim}, N={n_dim}, "
                f"K={k_dim}, A dtype={self.a_dtype}, accum dtype={self.accum_dtype}"
            )
        atom_m, atom_n, atom_k, enable_ws, enable_2cta = (int(x) for x in meta)

        # by default, we utilize non-swizzle layout offset
        a_leading_byte_offset = (8 * 8 * elems_in_bytes) if a_is_k_major else (8 * m_dim * elems_in_bytes)
        a_stride_byte_offset = (8 * k_dim * elems_in_bytes) if a_is_k_major else (8 * 8 * elems_in_bytes)

        if not a_swizzle_mode.is_none():
            # swizzle mode doesn't require LBO/SBO to be 1
            # https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-leading-dimension-byte-offset
            if a_is_k_major:
                a_leading_byte_offset = 16
                a_stride_byte_offset = 8 * a_swizzle_mode.swizzle_byte_size()
            else:
                # MN Major
                # LBO represents the distance between two atoms along the M dimension
                # SBO represents the distance between two atoms along the K dimension
                a_m_axis_atoms = m_dim // a_swizzle_atom_elems
                if a_m_axis_atoms <= 1:
                    a_leading_byte_offset = 0
                else:
                    a_leading_byte_offset = k_dim * a_swizzle_mode.swizzle_byte_size()

                if a_m_axis_atoms <= 1:
                    a_stride_byte_offset = 8 * elems_in_bytes * m_dim
                else:
                    a_stride_byte_offset = 8 * elems_in_bytes * a_swizzle_atom_elems

        b_leading_byte_offset = (8 * 8 * elems_in_bytes) if b_is_k_major else (8 * n_dim * elems_in_bytes)
        b_stride_byte_offset = (8 * k_dim * elems_in_bytes) if b_is_k_major else (0 if n_dim == 8 else (8 * 8 * elems_in_bytes))
        if not b_swizzle_mode.is_none():
            # swizzle mode doesn't require LBO/SBO to be 1
            # https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-leading-dimension-byte-offset
            if b_is_k_major:
                b_leading_byte_offset = 16
                b_stride_byte_offset = 8 * b_swizzle_mode.swizzle_byte_size()
            else:
                # MN Major, K * N
                # LBO represents the distance between two atoms along the N dimension
                # SBO represents the distance between two atoms along the K dimension
                b_n_axis_atoms = n_dim // b_swizzle_atom_elems
                if b_n_axis_atoms <= 1:
                    b_leading_byte_offset = 0
                else:
                    b_leading_byte_offset = 8 * 8 * elems_in_bytes * k_dim
                if b_n_axis_atoms <= 1:
                    b_stride_byte_offset = 8 * elems_in_bytes * n_dim
                else:
                    b_stride_byte_offset = 8 * elems_in_bytes * b_swizzle_atom_elems

        # for example, if [n, k] where k is 128, we should split it into 2 atoms
        # where max specially handles the case when n_dim is 8.
        ak_atom_size = max(a_swizzle_atom_elems // micro_size_k, 1)
        bk_atom_size = max(b_swizzle_atom_elems // micro_size_k, 1)

        instr_desc = self.get_tcgen5_instr_desc(
            atom_m,
            atom_n,
            atom_k,
            a_is_k_major,
            b_is_k_major,
            scale_in_a,
            scale_in_b,
        )
        # Allocate an instruction descriptor wrapper and initialize it
        a_dtype_abbrv = self.a_dtype_abbrv
        mask_zero = T.Cast(T.int32, 0)
        mask0 = mask1 = mask2 = mask3 = mask_zero

        # TCGEN05 only has one warp group
        num_inst_m = self.block_row_warps * self.warp_row_tiles // atom_m
        num_inst_n = self.block_col_warps * self.warp_col_tiles // atom_n

        # Helper to allow BufferRegion/BufferLoad as inputs
        def access_ptr_from(buffer_or_load_or_region, access_type: str = "r"):
            if isinstance(buffer_or_load_or_region, Buffer):
                return buffer_or_load_or_region.access_ptr(access_type)
            elif isinstance(buffer_or_load_or_region, BufferLoad):
                buffer_load = buffer_or_load_or_region
                offset, stride = 0, 1
                buffer = buffer_load.buffer
                for i, shape in enumerate(reversed(buffer.shape)):
                    indice = buffer_load.indices[len(buffer_load.indices) - i - 1]
                    if isinstance(indice, (tvm.tir.IntImm, tvm.tir.PrimExpr)):
                        offset += indice * stride
                    elif isinstance(indice, tvm.tir.Ramp):
                        offset += indice.base * stride
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

        @T.macro
        def _warp_mma(A_buf, B_buf, C_local_buf, mbar):
            # Allocate SMEM descriptors for A and B
            desc_a = T.alloc_tcgen05_smem_desc()
            desc_b = T.alloc_tcgen05_smem_desc()
            A_ptr = access_ptr_from(A_buf, "r")
            B_ptr = access_ptr_from(B_buf, "r")

            T.initialize_tcgen05_descriptor(
                desc_a,
                A_ptr,
                int(a_leading_byte_offset >> 4),
                int(a_stride_byte_offset >> 4),
                0,
                False,
                int(a_swizzle_mode),
            )
            T.initialize_tcgen05_descriptor(
                desc_b,
                B_ptr,
                int(b_leading_byte_offset >> 4),
                int(b_stride_byte_offset >> 4),
                0,
                False,
                int(b_swizzle_mode),
            )

            tmem_col_step = atom_n // (128 // atom_m)
            for j in T.unroll(num_inst_n):
                for i in T.unroll(num_inst_m):
                    for ki in T.unroll(0, (k_dim // micro_size_k)):
                        scale_out = T.Select(ki != 0, 1, T.Select(clear_accum, 0, 1))
                        A_elem_offset = (
                            (ki % ak_atom_size) * micro_size_k
                            + i * atom_m * a_swizzle_atom_elems
                            + (ki // ak_atom_size) * m_dim * a_swizzle_atom_elems
                            if a_is_k_major
                            else i * atom_m * k_dim + ki * a_swizzle_atom_elems * micro_size_k
                        )

                        B_elem_offset = (
                            (ki // bk_atom_size) * n_dim * b_swizzle_atom_elems
                            + (ki % bk_atom_size) * micro_size_k
                            + j * atom_n * b_swizzle_atom_elems
                            if b_is_k_major
                            else (
                                ki * b_swizzle_atom_elems * micro_size_k + j * atom_n * (k_dim if n_dim // b_swizzle_atom_elems > 1 else 1)
                            )
                        )

                        A_byte_offset = A_elem_offset * elems_in_bytes
                        B_byte_offset = B_elem_offset * elems_in_bytes
                        C_offset = (i * n_dim + j * tmem_col_step) * accum_dtype_in_bits // 32  # 32 bits per tmem bank

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
                            mask0,
                            mask1,
                            mask2,
                            mask3,
                            enable_ws,
                        )
            T.tcgen05_mma_arrive(mbar)

        return _warp_mma(A_buf, B_buf, C_local_buf, mbar)

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

        meta = self.get_tcgen5_mma_meta(m, n, k)
        if len(meta) != 5:
            raise ValueError(
                f"Unsupported TCGEN5MMA configuration: M={m}, N={n}, K={k}, A dtype={self.a_dtype}, accum dtype={self.accum_dtype}"
            )
        atom_m, atom_n, _, _, _ = (int(x) for x in meta)

        if m % atom_m != 0 or n % atom_n != 0:
            raise ValueError(f"Invalid TCGEN5MMA store layout for shape ({m}, {n}) with atoms ({atom_m}, {atom_n})")

        def forward(i: PrimExpr, j: PrimExpr):
            atom_idx = (i // atom_m) + (j // atom_n) * (m // atom_m)
            ai = i % atom_m
            aj = j % atom_n

            if atom_m == 128:
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

    def get_tcgen5_mma_meta(self, m: int, n: int, k: int):
        return _ffi_api.get_tcgen5_mma_meta(int(m), int(n), int(k), DataType(self.a_dtype), DataType(self.accum_dtype))

    def get_tcgen5_instr_desc(
        self, atom_m: int, atom_n: int, atom_k: int, a_is_k_major: bool, b_is_k_major: bool, scale_in_a: int, scale_in_b: int
    ) -> PrimExpr:
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
