import tilelang.language as T
from enum import IntEnum
from typing import Optional
from .mma_macro_generator import TensorCoreIntrinEmitter as MMAIntrinEmitter
from tvm import DataType
from tvm.tir import PrimExpr, Buffer, Var
from tilelang.utils import is_fragment
from tilelang.layout import (
    Layout,
    make_full_bank_swizzled_layout,
    make_half_bank_swizzled_layout,
    make_quarter_bank_swizzled_layout,
)
from tvm.runtime import convert

lift = convert


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
            return 32 // 8
        elif self.is_swizzle_64b():
            return 64 // 8
        elif self.is_swizzle_128b():
            return 128 // 8
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

    a_shared_layout: Layout = None
    b_shared_layout: Layout = None

    def __init__(
        self,
        a_dtype: str = "float16",
        b_dtype: str = "float16",
        accum_dtype: str = "float16",
        a_transposed: bool = False,
        b_transposed: bool = False,
        block_row_warps: int = 2,
        block_col_warps: int = 2,
        warp_row_tiles: int = 8,
        warp_col_tiles: int = 8,
        chunk: int = 16,
        reduce_k: int = 1,
        num_elems_per_byte: int = 1,
        is_m_first: Optional[bool] = False,
        thread_var: Optional[Var] = None,
    ):
        super().__init__(a_dtype, b_dtype, accum_dtype, a_transposed, b_transposed, block_row_warps,
                         block_col_warps, warp_row_tiles, warp_col_tiles, chunk, reduce_k,
                         num_elems_per_byte, is_m_first, thread_var)
        self._initialize_wgmma_prefix(self.n_dim)

    def _assign_a_shared_layout(self, layout: Layout):
        self.a_shared_layout = layout
        return self

    def _assign_b_shared_layout(self, layout: Layout):
        self.b_shared_layout = layout
        return self

    def _initialize_wgmma_prefix(self, n_dim: int = 16):
        inst_m, inst_n = 64, self.block_col_warps * self.warp_col_tiles
        # 256 bits per instruction
        inst_k = 256 // DataType(self.a_dtype).bits
        assert inst_k * n_dim == 256, f"Failed to initialize wgmma prefix, dtype: {self.a_dtype}, n_dim: {n_dim}, inst_k: {inst_k}"
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
        if layout is None:
            return SwizzleMode.NONE
        elif layout.is_equal(make_quarter_bank_swizzled_layout(buffer)):
            return SwizzleMode.SWIZZLE_32B
        elif layout.is_equal(make_half_bank_swizzled_layout(buffer)):
            return SwizzleMode.SWIZZLE_64B
        elif layout.is_equal(make_full_bank_swizzled_layout(buffer)):
            return SwizzleMode.SWIZZLE_128B
        else:
            raise ValueError(f"Unsupported swizzle mode: {layout}")

    def wgmma(self,
              A_buf: Buffer,
              B_buf: Buffer,
              C_local_buf: Buffer,
              clear_accum: PrimExpr = False):
        local_size_out = self.local_size_out
        a_dtype_abbrv = self.a_dtype_abbrv
        b_dtype_abbrv = self.b_dtype_abbrv
        accum_dtype = self.accum_dtype
        accum_dtype_abbrv = self.accum_dtype_abbrv
        m_dim = self.block_row_warps * self.warp_row_tiles
        warp_cols = self.warp_cols
        k_dim, n_dim = self.chunk, self.block_col_warps * self.warp_col_tiles
        wgmma_prefix = self.wgmma_prefix
        scale_out = not clear_accum
        scale_in_a = 1
        scale_in_b = 1

        a_is_k_major = not self.a_transposed
        b_is_k_major = self.b_transposed

        a_swizzle_mode = self._determinate_swizzle_mode(A_buf, self.a_shared_layout)
        b_swizzle_mode = self._determinate_swizzle_mode(B_buf, self.b_shared_layout)

        elems_in_bytes = DataType(self.a_dtype).bits // 8

        # by default, we utilize non-swizzle layout offset
        a_leading_byte_offset = (8 * 8 * elems_in_bytes) if a_is_k_major else (8 * m_dim *
                                                                               elems_in_bytes)
        a_stride_byte_offset = (8 * k_dim * elems_in_bytes) if a_is_k_major else (8 * 8 *
                                                                                  elems_in_bytes)

        if not a_swizzle_mode.is_none():
            # swizzle mode doesn't require LBO/SBO to be 1
            # https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-leading-dimension-byte-offset
            if a_is_k_major:
                a_leading_byte_offset = 16
            else:
                # MN Major
                # LBO represents the distance between two atoms along the M dimension
                # SBO represents the distance between two atoms along the K dimension
                a_leading_byte_offset = a_swizzle_mode.swizzle_atom_size()
                a_stride_byte_offset = 8 * 64 * elems_in_bytes

        b_leading_byte_offset = (8 * 8 * elems_in_bytes) if b_is_k_major else (8 * n_dim *
                                                                               elems_in_bytes)
        b_stride_byte_offset = (8 * k_dim * elems_in_bytes) if b_is_k_major else (8 * 8 *
                                                                                  elems_in_bytes)
        if not b_swizzle_mode.is_none():
            # swizzle mode doesn't require LBO/SBO to be 1
            # https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-leading-dimension-byte-offset
            if b_is_k_major:
                b_leading_byte_offset = 16
            else:
                # MN Major
                # LBO represents the distance between two atoms along the N dimension
                # SBO represents the distance between two atoms along the K dimension
                b_leading_byte_offset = b_swizzle_mode.swizzle_atom_size()
                b_stride_byte_offset = 8 * n_dim * elems_in_bytes

        @T.macro
        def _warp_mma(A_buf, B_buf, C_local_buf):
            desc_a = T.alloc_descriptor()
            desc_b = T.alloc_descriptor()
            T.initialize_descriptor(desc_a, A_buf.access_ptr("w"), a_swizzle_mode, int(a_leading_byte_offset >> 4), int(a_stride_byte_offset >> 4))
            T.initialize_descriptor(desc_b, B_buf.access_ptr("w"), b_swizzle_mode, int(b_leading_byte_offset >> 4), int(b_stride_byte_offset >> 4))
            for ki in T.serial(0, (k_dim // self.micro_size_k)):
                for i in T.serial(m_dim // 64):
                    k_dim_offset = ki * self.micro_size_k
                    A_offset = i * 64 * A_buf.shape[
                        -1] + k_dim_offset if a_is_k_major else ki * self.micro_size_k * 64 + i * 64 * k_dim
                    B_offset = k_dim_offset if b_is_k_major else k_dim_offset * B_buf.shape[-1]
                    C_offset = i * warp_cols * local_size_out  # 4 warps as an unit
                    T.ptx_wgmma(
                        accum_dtype,
                        wgmma_prefix,
                        self.a_transposed,
                        not self.b_transposed,
                        a_dtype_abbrv,
                        b_dtype_abbrv,
                        accum_dtype_abbrv,
                        desc_a.data,
                        (A_offset * elems_in_bytes) >> 4,
                        desc_b.data,
                        (B_offset * elems_in_bytes) >> 4,
                        C_local_buf.data,
                        C_offset,
                        scale_out,
                        scale_in_a,
                        scale_in_b,
                        a_swizzle_mode,
                        int(a_leading_byte_offset >> 4),
                        int(a_stride_byte_offset >> 4),
                        b_swizzle_mode,
                        int(b_leading_byte_offset >> 4),
                        int(b_stride_byte_offset >> 4),
                    )

                        

        return _warp_mma(A_buf, B_buf, C_local_buf)

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
