from __future__ import annotations

import tilelang.language as T
from tvm import tirx as tir
from tvm.tirx import Buffer, BufferRegion

OPERAND_LEFT = 0
OPERAND_RIGHT = 1
OPERAND_DEST = 2


class MPSIntrinEmitter:
    """Metal simdgroup/cooperative tensor intrinsic emitter for GEMM operations."""

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
        """Initialize the Metal GEMM intrinsic emitter."""
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
        """Return the thread index expression for the current kernel."""
        if self.thread_var is None:
            current_frame = T.KernelLaunchFrame.Current()
            assert current_frame is not None, "Must be called in a T.Kernel Frame"
            return current_frame.get_thread_binding()
        return self.thread_var

    def _get_warp_indices(self):
        """Compute (warp_m, warp_n) from the current thread binding."""
        thread_binding = self.get_thread_binding()
        warp_m = (thread_binding // self.WARP_SIZE) % self.block_row_warps
        warp_n = (thread_binding // (self.WARP_SIZE * self.block_row_warps)) % self.block_col_warps
        return warp_m, warp_n

    @staticmethod
    def _parse_buffer_nd(buf):
        """Extract (buffer, extra_indices, row_offset, col_offset, stride)."""
        if isinstance(buf, BufferRegion):
            buffer = buf.buffer
            extra = tuple(r.min for r in buf.region[:-2])
            off_row = buf.region[-2].min
            off_col = buf.region[-1].min
        else:
            buffer = buf
            extra = ()
            off_row = 0
            off_col = 0
        stride = buffer.strides[-2] if buffer.strides and len(buffer.strides) >= 2 else buffer.shape[-1]
        return buffer, extra, off_row, off_col, stride

    def ldmatrix_a(self, A_local_buf, A_shared_buf: Buffer | BufferRegion, ki, k_inner: int = 0):
        """Load matrix A tiles from memory into simdgroup/cooperative tensor buffers."""
        warp_rows = self.warp_rows
        warp_row_tiles = self.warp_row_tiles
        micro_size_x = self.micro_size_x
        micro_size_y = self.micro_size_y
        micro_size_k = self.micro_size_k
        a_transposed = self.a_transposed
        use_cooperative_tensor = self.use_cooperative_tensor

        warp_m, _ = self._get_warp_indices()
        buffer, extra, offset_m, offset_k, stride = self._parse_buffer_nd(A_shared_buf)
        if self.a_stride_override is not None:
            stride = self.a_stride_override

        @T.macro
        def _warp_ldmatrix_a(A_local_buf, buffer, offset_m, offset_k, stride, warp_m, ki):
            for i in T.serial(warp_rows):
                if a_transposed:
                    row_idx = offset_k + ki * micro_size_k
                    col_idx = offset_m + warp_m * warp_row_tiles + i * micro_size_x
                else:
                    row_idx = offset_m + warp_m * warp_row_tiles + i * micro_size_x
                    col_idx = offset_k + ki * micro_size_k
                ptr = T.access_ptr(buffer[extra + (row_idx, col_idx)], "r")
                if use_cooperative_tensor:
                    T.cooperative_tensor_load(
                        A_local_buf.data,
                        k_inner * warp_rows + i,
                        ptr,
                        stride,
                        micro_size_x,
                        micro_size_k,
                        T.bool(a_transposed),
                        micro_size_x,
                        micro_size_y,
                        micro_size_k,
                        OPERAND_LEFT,
                    )
                else:
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

    def ldmatrix_b(self, B_local_buf, B_shared_buf: Buffer | BufferRegion, ki, k_inner: int = 0):
        """Load matrix B tiles from memory into simdgroup/cooperative tensor buffers."""
        warp_cols = self.warp_cols
        warp_col_tiles = self.warp_col_tiles
        micro_size_x = self.micro_size_x
        micro_size_y = self.micro_size_y
        micro_size_k = self.micro_size_k
        b_transposed = self.b_transposed
        use_cooperative_tensor = self.use_cooperative_tensor

        _, warp_n = self._get_warp_indices()
        buffer, extra, offset_k, offset_n, stride = self._parse_buffer_nd(B_shared_buf)
        if self.b_stride_override is not None:
            stride = self.b_stride_override

        @T.macro
        def _warp_ldmatrix_b(B_local_buf, buffer, offset_k, offset_n, stride, warp_n, ki):
            for j in T.serial(warp_cols):
                if b_transposed:
                    row_idx = offset_n + warp_n * warp_col_tiles + j * micro_size_y
                    col_idx = offset_k + ki * micro_size_k
                else:
                    row_idx = offset_k + ki * micro_size_k
                    col_idx = offset_n + warp_n * warp_col_tiles + j * micro_size_y
                ptr = T.access_ptr(buffer[extra + (row_idx, col_idx)], "r")
                if use_cooperative_tensor:
                    T.cooperative_tensor_load(
                        B_local_buf.data,
                        k_inner * warp_cols + j,
                        ptr,
                        stride,
                        micro_size_k,
                        micro_size_y,
                        T.bool(b_transposed),
                        micro_size_x,
                        micro_size_y,
                        micro_size_k,
                        OPERAND_RIGHT,
                    )
                else:
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

    def mma(self, A_local_buf, B_local_buf, C_local_buf, k_inner: int = 0):
        """Perform matrix multiply-accumulate: C += A * B."""
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        micro_size_x = self.micro_size_x
        micro_size_y = self.micro_size_y
        micro_size_k = self.micro_size_k
        a_transposed = self.a_transposed
        b_transposed = self.b_transposed
        use_cooperative_tensor = self.use_cooperative_tensor

        @T.macro
        def _warp_mma(A_local_buf, B_local_buf, C_local_buf):
            for i, j in T.grid(warp_rows, warp_cols):
                index_c = i * warp_cols + j
                if use_cooperative_tensor:
                    T.cooperative_tensor_multiply_accumulate(
                        C_local_buf.data,
                        index_c,
                        A_local_buf.data,
                        k_inner * warp_rows + i,
                        B_local_buf.data,
                        k_inner * warp_cols + j,
                        C_local_buf.data,
                        index_c,
                        micro_size_x,
                        micro_size_y,
                        micro_size_k,
                        T.bool(a_transposed),
                        T.bool(b_transposed),
                    )
                else:
                    T.simdgroup_multiply_accumulate(
                        C_local_buf.data,
                        index_c,
                        A_local_buf.data,
                        i,
                        B_local_buf.data,
                        j,
                        C_local_buf.data,
                        index_c,
                    )

        return _warp_mma(A_local_buf, B_local_buf, C_local_buf)

    def simdgroup_copy(self, C_simd_buf, C_dst, is_store=True):
        """Copy between register-backed Metal matrix buffers and memory."""
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        warp_row_tiles = self.warp_row_tiles
        warp_col_tiles = self.warp_col_tiles
        micro_size_x = self.micro_size_x
        micro_size_y = self.micro_size_y
        micro_size_k = self.micro_size_k
        use_cooperative_tensor = self.use_cooperative_tensor

        warp_m, warp_n = self._get_warp_indices()
        buffer, extra, offset_m, offset_n, stride = self._parse_buffer_nd(C_dst)

        ct_op = T.cooperative_tensor_store if is_store else T.cooperative_tensor_load
        simd_op = T.simdgroup_store if is_store else T.simdgroup_load
        access_mode = "w" if is_store else "r"

        @T.macro
        def _simdgroup_copy(C_simd_buf, buffer, offset_m, offset_n, stride, warp_m, warp_n):
            for i, j in T.grid(warp_rows, warp_cols):
                row = offset_m + warp_m * warp_row_tiles + i * micro_size_x
                col = offset_n + warp_n * warp_col_tiles + j * micro_size_y
                index_c = i * warp_cols + j
                ptr = T.access_ptr(buffer[extra + (row, col)], access_mode)
                if use_cooperative_tensor:
                    ct_op(
                        C_simd_buf.data,
                        index_c,
                        ptr,
                        stride,
                        micro_size_x,
                        micro_size_y,
                        T.bool(False),
                        micro_size_x,
                        micro_size_y,
                        micro_size_k,
                        OPERAND_DEST,
                    )
                else:
                    simd_op(
                        C_simd_buf.data,
                        index_c,
                        ptr,
                        stride,
                        micro_size_x,
                        micro_size_y,
                        T.bool(False),
                    )

        return _simdgroup_copy(C_simd_buf, buffer, offset_m, offset_n, stride, warp_m, warp_n)

    def make_cooperative_tensor_store_layout(self, local_buf):
        from tilelang.utils.language import is_fragment

        assert is_fragment(local_buf), f"{local_buf} must be a fragment"
        shape = local_buf.shape

        def inverse_metal_ct_index(i, j):
            row = i % self.micro_size_x
            col = j % self.micro_size_y
            row_low = row % 8
            col_group = (col % 16) // 4
            lane_id = (col_group % 2) + (row_low % 4) * 2 + (col_group // 2) * 8 + (row_low // 4) * 16
            local_id = (col // 16) * 8 + (row // 8) * 4 + (col % 4)
            return lane_id, local_id

        def forward_thread(i: int, j: int) -> int:
            warp_m = (i // self.micro_size_x) // self.warp_rows
            warp_n = (j // self.micro_size_y) // self.warp_cols
            mma_i = i % self.micro_size_x
            mma_j = j % self.micro_size_y
            lane_id, _ = inverse_metal_ct_index(mma_i, mma_j)
            return warp_m * (self.block_col_warps * self.WARP_SIZE) + warp_n * self.WARP_SIZE + lane_id

        def forward_index(i: int, j: int) -> int:
            warp_i = (i // self.micro_size_x) % self.warp_rows
            warp_j = (j // self.micro_size_y) % self.warp_cols
            mma_i = i % self.micro_size_x
            mma_j = j % self.micro_size_y
            _, local_id = inverse_metal_ct_index(mma_i, mma_j)
            return warp_i * (self.warp_cols * 16) + warp_j * 16 + local_id

        return T.Fragment(shape, forward_thread_fn=forward_thread, forward_index_fn=forward_index)

    def simd_store(self, C_simd_buf, C_dst):
        """Store simdgroup/cooperative tensor local buffer to memory."""
        return self.simdgroup_copy(C_simd_buf, C_dst, is_store=True)

    def simd_load(self, C_simd_buf, C_src):
        """Load memory into simdgroup/cooperative tensor local buffer."""
        return self.simdgroup_copy(C_simd_buf, C_src, is_store=False)
