from __future__ import annotations

import tilelang.language as T
from tvm import tirx as tir
from tvm.tirx import Buffer, BufferRegion

OPERAND_LEFT = 0
OPERAND_RIGHT = 1
OPERAND_DEST = 2


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
        a_stride_override: int | None = None,
        b_stride_override: int | None = None,
        inner_k_steps: int = 1,
        use_cooperative_tensor: bool = True,
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
        self.a_stride_override = a_stride_override
        self.b_stride_override = b_stride_override
        self.inner_k_steps = inner_k_steps
        self.use_cooperative_tensor = use_cooperative_tensor

        if use_cooperative_tensor:
            self.micro_size_x = 16
            self.micro_size_y = 32
            self.micro_size_k = 16
        else:
            self.micro_size_x = 8
            self.micro_size_y = 8
            self.micro_size_k = 8

        self.warp_rows = warp_row_tiles // self.micro_size_x
        self.warp_cols = warp_col_tiles // self.micro_size_y

    def get_thread_binding(self):
        if self.thread_var is None:
            current_frame = T.KernelLaunchFrame.Current()
            assert current_frame is not None, "Must be called in a T.Kernel Frame"
            return current_frame.get_thread_binding()
        return self.thread_var

    def _get_warp_indices(self):
        thread_binding = self.get_thread_binding()
        warp_m = (thread_binding // self.WARP_SIZE) % self.block_row_warps
        warp_n = (thread_binding // (self.WARP_SIZE * self.block_row_warps)) % self.block_col_warps
        return warp_m, warp_n

    @staticmethod
    def _parse_buffer_2d(buf):
        if isinstance(buf, BufferRegion):
            buffer = buf.buffer
            leading = tuple(r.min for r in buf.region[:-2])
            off_row = buf.region[-2].min
            off_col = buf.region[-1].min
        else:
            buffer = buf
            leading = ()
            off_row = 0
            off_col = 0
        stride = buffer.strides[-2] if buffer.strides and len(buffer.strides) >= 2 else buffer.shape[-1]
        return buffer, off_row, off_col, stride, leading

    @staticmethod
    def _buf_idx(leading, row, col):
        return (*leading, row, col)

    def ldmatrix_a(self, A_local_buf, A_shared_buf: Buffer | BufferRegion, ki, k_inner: int = 0):
        warp_m, _ = self._get_warp_indices()
        buffer, offset_m, offset_k, stride, leading = self._parse_buffer_2d(A_shared_buf)
        if self.a_stride_override is not None:
            stride = self.a_stride_override
        buf_idx = self._buf_idx

        @T.macro
        def _warp_ldmatrix_a(A_local_buf, buffer, offset_m, offset_k, stride, warp_m, ki):
            for i in T.serial(self.warp_rows):
                if self.a_transposed:
                    row_idx = offset_k + ki * self.micro_size_k
                    col_idx = offset_m + warp_m * self.warp_row_tiles + i * self.micro_size_x
                else:
                    row_idx = offset_m + warp_m * self.warp_row_tiles + i * self.micro_size_x
                    col_idx = offset_k + ki * self.micro_size_k
                ptr = T.access_ptr(buffer[buf_idx(leading, row_idx, col_idx)], "r")
                if self.use_cooperative_tensor:
                    T.cooperative_tensor_load(
                        A_local_buf.data,
                        k_inner * self.warp_rows + i,
                        ptr,
                        stride,
                        self.micro_size_x,
                        self.micro_size_k,
                        T.bool(self.a_transposed),
                        self.micro_size_x,
                        self.micro_size_y,
                        self.micro_size_k,
                        OPERAND_LEFT,
                    )
                else:
                    T.simdgroup_load(
                        A_local_buf.data,
                        i,
                        ptr,
                        stride,
                        self.micro_size_x,
                        self.micro_size_k,
                        T.bool(self.a_transposed),
                    )

        return _warp_ldmatrix_a(A_local_buf, buffer, offset_m, offset_k, stride, warp_m, ki)

    def ldmatrix_b(self, B_local_buf, B_shared_buf: Buffer | BufferRegion, ki, k_inner: int = 0):
        _, warp_n = self._get_warp_indices()
        warp_n_offset = warp_n * self.warp_col_tiles
        buffer, offset_k, offset_n, stride, leading = self._parse_buffer_2d(B_shared_buf)
        if self.b_stride_override is not None:
            stride = self.b_stride_override
        buf_idx = self._buf_idx

        @T.macro
        def _warp_ldmatrix_b(B_local_buf, buffer, offset_k, offset_n, stride, warp_n_offset, ki):
            for j in T.serial(self.warp_cols):
                if self.b_transposed:
                    row_idx = offset_n + warp_n_offset + j * self.micro_size_y
                    col_idx = offset_k + ki * self.micro_size_k
                else:
                    row_idx = offset_k + ki * self.micro_size_k
                    col_idx = offset_n + warp_n_offset + j * self.micro_size_y
                ptr = T.access_ptr(buffer[buf_idx(leading, row_idx, col_idx)], "r")
                if self.use_cooperative_tensor:
                    T.cooperative_tensor_load(
                        B_local_buf.data,
                        k_inner * self.warp_cols + j,
                        ptr,
                        stride,
                        self.micro_size_k,
                        self.micro_size_y,
                        T.bool(self.b_transposed),
                        self.micro_size_x,
                        self.micro_size_y,
                        self.micro_size_k,
                        OPERAND_RIGHT,
                    )
                else:
                    T.simdgroup_load(
                        B_local_buf.data,
                        j,
                        ptr,
                        stride,
                        self.micro_size_k,
                        self.micro_size_y,
                        T.bool(self.b_transposed),
                    )

        return _warp_ldmatrix_b(B_local_buf, buffer, offset_k, offset_n, stride, warp_n_offset, ki)

    def mma(self, A_local_buf, B_local_buf, C_local_buf, k_inner: int = 0):
        @T.macro
        def _warp_mma(A_local_buf, B_local_buf, C_local_buf):
            for i, j in T.grid(self.warp_rows, self.warp_cols):
                if self.use_cooperative_tensor:
                    T.cooperative_tensor_multiply_accumulate(
                        C_local_buf.data,
                        i * self.warp_cols + j,
                        A_local_buf.data,
                        k_inner * self.warp_rows + i,
                        B_local_buf.data,
                        k_inner * self.warp_cols + j,
                        C_local_buf.data,
                        i * self.warp_cols + j,
                        self.micro_size_x,
                        self.micro_size_y,
                        self.micro_size_k,
                        T.bool(self.a_transposed),
                        T.bool(self.b_transposed),
                    )
                else:
                    T.simdgroup_multiply_accumulate(
                        C_local_buf.data,
                        i * self.warp_cols + j,
                        A_local_buf.data,
                        i,
                        B_local_buf.data,
                        j,
                        C_local_buf.data,
                        i * self.warp_cols + j,
                    )

        return _warp_mma(A_local_buf, B_local_buf, C_local_buf)

    def simdgroup_copy(self, C_simd_buf, C_dst, is_store=True):
        warp_m, warp_n = self._get_warp_indices()
        warp_m_offset = warp_m * self.warp_row_tiles
        warp_n_offset = warp_n * self.warp_col_tiles
        buffer, offset_m, offset_n, stride, leading = self._parse_buffer_2d(C_dst)
        ct_op = T.cooperative_tensor_store if is_store else T.cooperative_tensor_load
        simd_op = T.simdgroup_store if is_store else T.simdgroup_load
        access_mode = "w" if is_store else "r"
        buf_idx = self._buf_idx

        @T.macro
        def _simdgroup_copy(C_simd_buf, buffer, offset_m, offset_n, stride, warp_m_offset, warp_n_offset):
            for i, j in T.grid(self.warp_rows, self.warp_cols):
                row = offset_m + warp_m_offset + i * self.micro_size_x
                col = offset_n + warp_n_offset + j * self.micro_size_y
                index_c = i * self.warp_cols + j
                if self.use_cooperative_tensor:
                    ct_op(
                        C_simd_buf.data,
                        index_c,
                        T.access_ptr(buffer[buf_idx(leading, row, col)], access_mode),
                        stride,
                        self.micro_size_x,
                        self.micro_size_y,
                        T.bool(False),
                        self.micro_size_x,
                        self.micro_size_y,
                        self.micro_size_k,
                        OPERAND_DEST,
                    )
                else:
                    simd_op(
                        C_simd_buf.data,
                        index_c,
                        T.access_ptr(buffer[buf_idx(leading, row, col)], access_mode),
                        stride,
                        self.micro_size_x,
                        self.micro_size_y,
                        T.bool(False),
                    )

        return _simdgroup_copy(C_simd_buf, buffer, offset_m, offset_n, stride, warp_m_offset, warp_n_offset)

    def make_cooperative_tensor_store_layout(self, local_buf):
        from tilelang.utils.language import is_fragment
        from tilelang.cuda.intrinsics.layout.mma_layout import metal_ct_store_index_map

        assert is_fragment(local_buf), f"{local_buf} must be a fragment"
        shape = local_buf.shape
        inverse_index_map = metal_ct_store_index_map().inverse([self.WARP_SIZE, 16])

        def forward_thread(i: int, j: int) -> int:
            warp_m = (i // self.micro_size_x) // self.warp_rows
            warp_n = (j // self.micro_size_y) // self.warp_cols
            mma_i = i % self.micro_size_x
            mma_j = j % self.micro_size_y
            lane_id, _ = inverse_index_map.map_indices([mma_i, mma_j])
            return warp_m * (self.block_col_warps * self.WARP_SIZE) + warp_n * self.WARP_SIZE + lane_id

        def forward_index(i: int, j: int) -> int:
            warp_i = (i // self.micro_size_x) % self.warp_rows
            warp_j = (j // self.micro_size_y) % self.warp_cols
            mma_i = i % self.micro_size_x
            mma_j = j % self.micro_size_y
            _, local_id = inverse_index_map.map_indices([mma_i, mma_j])
            return warp_i * (self.warp_cols * 16) + warp_j * 16 + local_id

        return T.Fragment(shape, forward_thread_fn=forward_thread, forward_index_fn=forward_index)

    def simd_store(self, C_simd_buf, C_dst):
        return self.simdgroup_copy(C_simd_buf, C_dst, is_store=True)

    def simd_load(self, C_simd_buf, C_src):
        return self.simdgroup_copy(C_simd_buf, C_src, is_store=False)
