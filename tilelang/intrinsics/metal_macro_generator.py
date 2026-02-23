from __future__ import annotations

import tilelang.language as T
from tvm import tir
from tvm.tir import Buffer, BufferRegion


class MPSIntrinEmitter:
    WARP_SIZE = 32

    def __init__(
        self,
        a_dtype: str = "float16",
        b_dtype: str = "float16",
        accum_dtype: str = "float32",
        a_transposed: bool = False,
        b_transposed: bool = False,
        block_row_warps: int = 1,
        block_col_warps: int = 1,
        warp_row_tiles: int = 8,
        warp_col_tiles: int = 8,
        chunk: int = 32,
        thread_var: tir.Var | None = None,
    ):
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.accum_dtype = accum_dtype
        self.a_transposed = a_transposed
        self.b_transposed = b_transposed
        self.block_row_warps = block_row_warps
        self.block_col_warps = block_col_warps
        self.warp_row_tiles = warp_row_tiles
        self.warp_col_tiles = warp_col_tiles
        self.chunk = chunk
        self.thread_var = thread_var

        # Metal simdgroup matrix size (always 8x8)
        self.micro_size_x = 8
        self.micro_size_y = 8
        self.micro_size_k = 8

        # Number of 8x8 tiles per warp
        self.warp_rows = warp_row_tiles // self.micro_size_x
        self.warp_cols = warp_col_tiles // self.micro_size_y

    def get_thread_binding(self):
        if self.thread_var is None:
            current_frame = T.KernelLaunchFrame.Current()
            assert current_frame is not None, "Must be called in a T.Kernel Frame"
            return current_frame.get_thread_binding()
        else:
            return self.thread_var

    def _get_warp_indices(self):
        thread_binding = self.get_thread_binding()
        WARP_SIZE = self.WARP_SIZE
        block_row_warps = self.block_row_warps
        block_col_warps = self.block_col_warps

        warp_m = (thread_binding // WARP_SIZE) % block_row_warps
        warp_n = (thread_binding // (WARP_SIZE * block_row_warps)) % block_col_warps
        return warp_m, warp_n

    @staticmethod
    def _parse_buffer_2d(buf):
        """Extract (buffer, row_offset, col_offset, stride) from Buffer or BufferRegion."""
        if isinstance(buf, BufferRegion):
            buffer = buf.buffer
            off_row = buf.region[-2].min
            off_col = buf.region[-1].min
        else:
            buffer = buf
            off_row = 0
            off_col = 0
        stride = buffer.shape[-1]
        return buffer, off_row, off_col, stride

    def ldmatrix_a(self, A_local_buf, A_shared_buf: Buffer | BufferRegion, ki):
        warp_rows = self.warp_rows
        micro_size_x = self.micro_size_x
        micro_size_k = self.micro_size_k
        a_transposed = self.a_transposed

        warp_m, _ = self._get_warp_indices()

        buffer, offset_m, offset_k, stride = self._parse_buffer_2d(A_shared_buf)

        @T.macro
        def _warp_ldmatrix_a(A_local_buf, buffer, offset_m, offset_k, stride, warp_m, ki):
            for i in T.serial(warp_rows):
                if a_transposed:
                    row_idx = offset_k + ki * micro_size_k
                    col_idx = offset_m + warp_m * (self.warp_row_tiles) + i * micro_size_x
                else:
                    row_idx = offset_m + warp_m * (self.warp_row_tiles) + i * micro_size_x
                    col_idx = offset_k + ki * micro_size_k

                ptr = T.access_ptr(buffer[row_idx, col_idx], "r")

                T.simdgroup_load(
                    A_local_buf.data,
                    i,
                    ptr,
                    stride,
                    micro_size_x,
                    micro_size_k,
                    T.bool(a_transposed),
                )

        return _warp_ldmatrix_a(A_local_buf, buffer, offset_m, offset_k, stride, warp_m, ki)

    def ldmatrix_b(self, B_local_buf, B_shared_buf: Buffer | BufferRegion, ki):
        warp_cols = self.warp_cols
        micro_size_y = self.micro_size_y
        micro_size_k = self.micro_size_k
        b_transposed = self.b_transposed

        _, warp_n = self._get_warp_indices()

        buffer, offset_k, offset_n, stride = self._parse_buffer_2d(B_shared_buf)

        @T.macro
        def _warp_ldmatrix_b(B_local_buf, buffer, offset_k, offset_n, stride, warp_n, ki):
            for j in T.serial(warp_cols):
                if b_transposed:
                    row_idx = offset_n + warp_n * (self.warp_col_tiles) + j * micro_size_y
                    col_idx = offset_k + ki * micro_size_k
                else:
                    row_idx = offset_k + ki * micro_size_k
                    col_idx = offset_n + warp_n * (self.warp_col_tiles) + j * micro_size_y

                ptr = T.access_ptr(buffer[row_idx, col_idx], "r")

                T.simdgroup_load(
                    B_local_buf.data,
                    j,
                    ptr,
                    stride,
                    micro_size_k,
                    micro_size_y,
                    T.bool(b_transposed),
                )

        return _warp_ldmatrix_b(B_local_buf, buffer, offset_k, offset_n, stride, warp_n, ki)

    def mma(self, A_local_buf, B_local_buf, C_local_buf):
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols

        @T.macro
        def _warp_mma(A_local_buf, B_local_buf, C_local_buf):
            for i, j in T.grid(warp_rows, warp_cols):
                T.simdgroup_multiply_accumulate(
                    C_local_buf.data,
                    i * warp_cols + j,
                    A_local_buf.data,
                    i,
                    B_local_buf.data,
                    j,
                    C_local_buf.data,
                    i * warp_cols + j,
                )

        return _warp_mma(A_local_buf, B_local_buf, C_local_buf)

    def simdgroup_copy(self, C_simd_buf, C_dst, is_store=True):
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        micro_size_x = self.micro_size_x
        micro_size_y = self.micro_size_y

        warp_m, warp_n = self._get_warp_indices()

        buffer, offset_m, offset_n, stride = self._parse_buffer_2d(C_dst)

        simd_op = T.simdgroup_store if is_store else T.simdgroup_load
        access_mode = "w" if is_store else "r"

        @T.macro
        def _simdgroup_copy(C_simd_buf, buffer, offset_m, offset_n, stride, warp_m, warp_n):
            for i, j in T.grid(warp_rows, warp_cols):
                row = offset_m + warp_m * self.warp_row_tiles + i * micro_size_x
                col = offset_n + warp_n * self.warp_col_tiles + j * micro_size_y

                index_c = i * warp_cols + j

                simd_op(
                    C_simd_buf.data,
                    index_c,
                    T.access_ptr(buffer[row, col], access_mode),
                    stride,
                    micro_size_x,
                    micro_size_y,
                    T.bool(False),
                )

        return _simdgroup_copy(C_simd_buf, buffer, offset_m, offset_n, stride, warp_m, warp_n)

    def simd_store(self, C_simd_buf, C_dst):
        return self.simdgroup_copy(C_simd_buf, C_dst, is_store=True)

    def simd_load(self, C_simd_buf, C_src):
        return self.simdgroup_copy(C_simd_buf, C_src, is_store=False)
