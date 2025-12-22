from __future__ import annotations
import tilelang.language as T
from typing import Literal, Callable
from tvm import DataType
from tvm.tir import PrimExpr, IndexMap, Buffer, Var, BufferRegion
from tilelang import tvm as tvm
from tvm.runtime import convert
from tilelang.utils import is_fragment
from tilelang.intrinsics.mma_sm70_layout import (
    shared_16x4_to_mma_a_32x4_layout,
    shared_4x16_to_mma_b_32x4_layout,
    shared_16x4_to_mma_b_32x4_layout_trans,
    mma_32x8_to_shared_16x16_layout_fp32,
    mma_32x8_to_shared_16x16_layout_fp16,
    mma_load_a_32x4_to_shared_16x4_layout,
    mma_load_b_32x4_to_shared_16x4_layout_trans,
    mma_load_b_32x4_to_shared_4x16_layout,
)

lift = convert


class TensorCoreIntrinEmitter:
    """
    To eliminate Python syntax within TIR Macro.
    """

    M_DIM = 16
    # use lowercase as n_dim can be dynamic
    # the smallest instructions can be m16n8k16, so the n_dim can also be 8
    n_dim = 16
    WARP_SIZE = 32
    HALF_WARP_SIZE = WARP_SIZE // 2
    dtype_abbrv = {
        "float16": "fp16",
        "bfloat16": "bf16",
        "float32": "fp32",
        "int8": "int8",
        "int32": "int32",
        "float8_e4m3": "e4m3",
        "float8_e5m2": "e5m2",
    }

    # Represent the thread binding in the form of (tx, warp_n, warp_m)
    is_m_first = False

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
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.accum_dtype = accum_dtype
        self.a_transposed = a_transposed
        self.b_transposed = b_transposed
        # Hint Information
        self.block_row_warps = block_row_warps
        self.block_col_warps = block_col_warps
        self.warp_row_tiles = warp_row_tiles
        self.warp_col_tiles = warp_col_tiles
        self.chunk = chunk
        self._initialize_k_dim(a_dtype)
        self._initialize_abbrev(a_dtype, b_dtype, accum_dtype)
        self._initialize_micro_size(self.M_DIM, self.k_dim)
        self._initialize_local_size(self.M_DIM, self.n_dim, self.k_dim)
        self._initialize_mma_prefix(self.k_dim)
        self._initialize_is_m_first(is_m_first)

        self.reduce_k = reduce_k
        self.threads = self.WARP_SIZE * (block_row_warps * block_col_warps) * reduce_k
        self.num_elems_per_byte = num_elems_per_byte
        self.thread_var = thread_var

        if self.warp_rows == 0 or self.warp_cols == 0:
            raise ValueError(
                f"Invalid threads configuration for this tile shape, {self.warp_rows} x {self.warp_cols} with threads {self.threads}"
            )

    def _initialize_k_dim(self, a_dtype=T.float16):
        self.k_dim = 4

    def _initialize_local_size(self, m_dim=16, n_dim=16, k_dim=16):
        self.local_size_a = (m_dim * k_dim) // self.HALF_WARP_SIZE
        self.local_size_b = (n_dim * k_dim) // self.HALF_WARP_SIZE
        self.local_size_out = (m_dim * n_dim) // self.WARP_SIZE

    def _initialize_abbrev(self, a_dtype, b_dtype, accum_dtype):
        self.a_dtype_abbrv = self._get_dtype_abbrv(a_dtype)
        self.b_dtype_abbrv = self._get_dtype_abbrv(b_dtype)
        self.accum_dtype_abbrv = self._get_dtype_abbrv(accum_dtype)

    def _get_dtype_abbrv(self, dtype: str) -> str:
        try:
            return self.dtype_abbrv[dtype]
        except KeyError as err:
            raise ValueError(f"Unsupported dtype: {dtype}") from err

    def _initialize_mma_prefix(self, k_dim: int = 16):
        if k_dim == 4:
            # typically used for float16
            self.mma_prefix = "m16n16k4"
        else:
            raise ValueError(f"Unsupported k_dim: {k_dim}")

    def _initialize_micro_size(self, m_dim: int = 16, k_dim: int = 16):
        warp_row_tiles = self.warp_row_tiles
        warp_col_tiles = self.warp_col_tiles
        assert warp_row_tiles >= 16, f"warp_row_tiles must be greater than 16, got {warp_row_tiles}"
        assert warp_row_tiles % 16 == 0, f"warp_row_tiles must be divisible by 16, got {warp_row_tiles}"
        assert warp_col_tiles >= 16, f"warp_col_tiles must be greater than 16, got {warp_col_tiles}"
        assert warp_col_tiles % 16 == 0, f"warp_col_tiles must be divisible by 16, got {warp_col_tiles}"

        self.warp_rows = warp_row_tiles // m_dim

        self.n_dim = 16
        self.micro_size_y = 16
        self.warp_cols = warp_col_tiles // 16

        self.micro_size_x = m_dim
        self.micro_size_k = k_dim

    def _initialize_is_m_first(self, is_m_first: bool | None = False):
        if is_m_first is not None:
            self.is_m_first = is_m_first

    def get_thread_binding(self):
        if self.thread_var is None:
            current_frame = T.KernelLaunchFrame.Current()
            assert current_frame is not None, "Must be called in a T.Kernel Frame"
            return current_frame.get_thread_binding()
        else:
            return self.thread_var

    def get_store_index_map(self, inverse: bool = False) -> IndexMap:
        warp_size, local_size_c = self.WARP_SIZE, self.local_size_out
        index_map = IndexMap.from_func(
            mma_32x8_to_shared_16x16_layout_fp32 if self.accum_dtype == T.float32 else mma_32x8_to_shared_16x16_layout_fp16,
            index_dtype=T.int32,
        )
        if not inverse:
            return index_map
        inverse_index_map = index_map.inverse([warp_size, local_size_c])
        return inverse_index_map

    def extract_thread_binding(self, thread_id: PrimExpr, is_m_first: bool | None = None) -> tuple[PrimExpr, PrimExpr, PrimExpr]:
        """
        is_m_first: True if the thread binding is in the form of (tx, warp_n, warp_m)
        which represents [warp_size, block_row_warps (split n), block_col_warps (split m)]
        Otherwise, it is in the form of [warp_size, block_col_warps (split m), block_row_warps (split n)]
        """
        WARP_SIZE = self.WARP_SIZE
        block_row_warps = self.block_row_warps
        block_col_warps = self.block_col_warps

        # if is_m_first is None, then use the default value
        if is_m_first is None:
            is_m_first = self.is_m_first

        if is_m_first:
            lane_id, warp_n, warp_m = (
                thread_id % WARP_SIZE,
                (thread_id // WARP_SIZE) % block_col_warps,
                (thread_id // (WARP_SIZE * block_col_warps)) % block_row_warps,
            )
            return lane_id, warp_n, warp_m
        else:
            lane_id, warp_m, warp_n = (
                thread_id % WARP_SIZE,
                (thread_id // WARP_SIZE) % block_row_warps,
                (thread_id // (WARP_SIZE * block_row_warps)) % block_col_warps,
            )
            return lane_id, warp_n, warp_m

    def ldmatrix_a(self, A_local_buf: Buffer, A_shared_buf: Buffer | BufferRegion, ki: PrimExpr, rk: PrimExpr | None = 0):
        warp_row_tiles = self.warp_row_tiles
        warp_rows = self.warp_rows
        chunk = self.chunk
        micro_size_x = self.micro_size_x
        micro_size_k = self.micro_size_k
        local_size_a = self.local_size_a
        a_transposed = self.a_transposed

        thread_binding = self.get_thread_binding()

        assert not a_transposed, "A must be not transposed"

        mma_load_layout = mma_load_a_32x4_to_shared_16x4_layout

        # legalize shared buffer to region
        A_region = self._legalize_to_buffer_region(A_shared_buf)
        A_buf = A_region.buffer
        A_base0 = A_region.region[-2].min
        A_base1 = A_region.region[-1].min

        @T.macro
        def _warp_ldmatrix_a(
            A_local_buf,
            A_shared_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            tx, _, warp_m = self.extract_thread_binding(thread_binding)

            for i in T.serial(warp_rows):
                # Assign A_shared_buf_elem
                wi, wk = warp_m * warp_row_tiles + i * micro_size_x, rk * chunk + ki * micro_size_k
                for j in T.vectorized(local_size_a):
                    mi, mk = mma_load_layout(tx, j)
                    A_local_buf[i * local_size_a + j] = A_buf[A_base0 + wi + mi, A_base1 + wk + mk]

        return _warp_ldmatrix_a(A_local_buf, A_region, ki, thread_binding, rk)

    def ldmatrix_b(self, B_local_buf: Buffer, B_shared_buf: Buffer | BufferRegion, ki: PrimExpr, rk: PrimExpr | None = 0):
        warp_col_tiles = self.warp_col_tiles
        warp_cols = self.warp_cols
        chunk = self.chunk
        micro_size_y = self.micro_size_y
        micro_size_k = self.micro_size_k
        local_size_b = self.local_size_b
        b_transposed = self.b_transposed
        thread_binding = self.get_thread_binding()

        mma_load_layout = mma_load_b_32x4_to_shared_16x4_layout_trans if b_transposed else mma_load_b_32x4_to_shared_4x16_layout

        # legalize shared buffer to region
        B_region = self._legalize_to_buffer_region(B_shared_buf)
        B_buf = B_region.buffer
        B_base0 = B_region.region[-2].min
        B_base1 = B_region.region[-1].min

        @T.macro
        def _warp_ldmatrix_b(
            B_local_buf,
            B_shared_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            tx, warp_n, _ = self.extract_thread_binding(thread_binding)

            for i in T.serial(warp_cols):
                # Assign B_shared_elem
                wi, wk = (
                    warp_n * warp_col_tiles + i * micro_size_y,
                    rk * chunk + ki * micro_size_k,
                )
                # load 16x32 data from shared buffer to local buffer
                # must be transposed.
                for j in T.vectorized(local_size_b):
                    if b_transposed:
                        mi, mk = mma_load_layout(tx, j)
                        B_local_buf[i * local_size_b + j] = B_buf[B_base0 + wi + mi, B_base1 + wk + mk]
                    else:
                        mk, mi = mma_load_layout(tx, j)
                        B_local_buf[i * local_size_b + j] = B_buf[B_base0 + wk + mk, B_base1 + wi + mi]

        return _warp_ldmatrix_b(B_local_buf, B_region, ki, thread_binding, rk)

    def mma(self, A_local_buf: Buffer, B_local_buf: Buffer, C_local_buf: Buffer, k_inner: PrimExpr | None = 0):
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_a = self.local_size_a
        local_size_b = self.local_size_b
        local_size_out = self.local_size_out
        a_dtype_abbrv = self.a_dtype_abbrv
        b_dtype_abbrv = self.b_dtype_abbrv
        accum_dtype_abbrv = self.accum_dtype_abbrv
        mma_prefix = self.mma_prefix

        a_is_fragment = is_fragment(A_local_buf)
        b_is_fragment = is_fragment(B_local_buf)
        a_local_stride: PrimExpr = k_inner * warp_rows * local_size_a if a_is_fragment else 0
        b_local_stride: PrimExpr = k_inner * warp_cols * local_size_b if b_is_fragment else 0

        a_major = "col" if self.a_transposed else "row"
        b_major = "col" if self.b_transposed else "row"

        @T.macro
        def _warp_mma(A_local_buf, B_local_buf, C_local_buf):
            for i, j in T.grid(warp_rows, warp_cols):
                T.ptx_mma_sm70(
                    mma_prefix,
                    a_major,
                    b_major,
                    a_dtype_abbrv,
                    b_dtype_abbrv,
                    accum_dtype_abbrv,
                    A_local_buf.data,
                    a_local_stride + i * local_size_a,
                    B_local_buf.data,
                    b_local_stride + j * local_size_b,
                    C_local_buf.data,
                    i * warp_cols * local_size_out + j * local_size_out,
                )

        return _warp_mma(A_local_buf, B_local_buf, C_local_buf)

    def make_mma_load_layout(self, local_buf: Buffer, matrix: Literal["A", "B"] = "A") -> T.Fragment:
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

        assert matrix in ["A", "B"], "matrix should be either A or B"
        matrix_is_a: bool = matrix == "A"
        matrix_is_b: bool = matrix == "B"
        dtype = self.a_dtype if matrix_is_a else self.b_dtype
        dtype_bits = DataType(dtype).bits
        transposed = self.a_transposed if matrix_is_a else self.b_transposed

        # s represents spatial axis
        # r represents reduction axis
        # sr represents the two dims are spatial + reduction
        # rs represents the two dims are reduction + spatial
        # sr also can represent a non-transposed basic layout
        # then rs also can represent a transposed basic layout
        transform_func_sr_a: Callable = None
        transform_func_sr_b: Callable = None
        transform_func_rs_b: Callable = None
        if dtype_bits == 16:
            transform_func_sr_a = shared_16x4_to_mma_a_32x4_layout
            transform_func_sr_b = shared_16x4_to_mma_b_32x4_layout_trans
            transform_func_rs_b = shared_4x16_to_mma_b_32x4_layout
        else:
            raise ValueError(f"Unsupported dtype {dtype}")

        is_sr_conditions = [False]
        is_sr_conditions.append(matrix_is_a and not transposed)
        is_sr_conditions.append(matrix_is_b and transposed)
        is_sr_axis_order = any(is_sr_conditions)

        # the layout of mma.sync is row.col.
        # so the b matrix expected a transposed basic layout
        transform_func: Callable = None
        if matrix_is_a:
            transform_func = transform_func_sr_a if is_sr_axis_order else lambda i, j: transform_func_sr_a(j, i)
        elif matrix_is_b:
            transform_func = transform_func_sr_b if is_sr_axis_order else lambda i, j: transform_func_rs_b(i, j)
        else:
            raise ValueError(f"Unsupported matrix {matrix}")

        assert is_fragment(local_buf), f"local_buf must be a fragment, but got {local_buf.scope()}"

        if matrix_is_a:
            micro_size_s, micro_size_r = self.micro_size_x, self.micro_size_k
        else:
            micro_size_r, micro_size_s = self.micro_size_k, self.micro_size_y

        block_row_warps, block_col_warps = (
            self.block_row_warps,
            self.block_col_warps,
        )

        inverse_mma_load_layout = IndexMap.from_func(transform_func, index_dtype=T.int32)

        def forward(i: int, j: int, rep: int) -> int:
            """
            Given the row index `i` and column index `j` in the fragment,
            """
            lane_id, local_id = inverse_mma_load_layout.map_indices([i, j, rep])
            return lane_id, local_id

        base_fragment = T.Fragment(
            [micro_size_s, micro_size_r] if is_sr_axis_order else [micro_size_r, micro_size_s], forward_fn=forward, replicate=2
        )

        warp_rows, warp_cols = self.warp_rows, self.warp_cols
        chunk = self.chunk

        warp_s = warp_rows if matrix_is_a else warp_cols
        warp_r = chunk // micro_size_r
        block_s = block_row_warps if matrix_is_a else block_col_warps
        replicate = block_col_warps if matrix_is_a else block_row_warps

        if is_sr_axis_order:
            warp_fragment = base_fragment.repeat([warp_s, warp_r], repeat_on_thread=False, lower_dim_first=False)
            if matrix_is_a:
                block_fragment = warp_fragment.repeat([block_s, 1], repeat_on_thread=True, lower_dim_first=True).replicate(replicate)
            elif matrix_is_b:
                block_fragment = warp_fragment.replicate(replicate).repeat([block_s, 1], repeat_on_thread=True, lower_dim_first=True)
            else:
                raise ValueError(f"Unsupported matrix type {matrix}")
        else:
            warp_fragment = base_fragment.repeat([warp_r, warp_s], repeat_on_thread=False, lower_dim_first=True)
            if matrix_is_a:
                block_fragment = warp_fragment.repeat([1, block_s], repeat_on_thread=True, lower_dim_first=True).replicate(replicate)
            elif matrix_is_b:
                block_fragment = warp_fragment.replicate(replicate).repeat([1, block_s], repeat_on_thread=True, lower_dim_first=True)
            else:
                raise ValueError(f"Unsupported matrix type {matrix}")

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
        from tilelang.utils import is_fragment

        shape = local_buf.shape
        inverse_mma_store_layout = self.get_store_index_map(inverse=True)
        assert is_fragment(local_buf), "local_buf must be a fragment"
        micro_size_x, micro_size_y = self.micro_size_x, self.micro_size_y
        local_size_out = self.local_size_out
        block_row_warps, block_col_warps = self.block_row_warps, self.block_col_warps
        warp_rows, warp_cols = self.warp_rows, self.warp_cols
        warp_size = self.WARP_SIZE
        is_m_first = self.is_m_first

        def forward_thread(i: int, j: int) -> int:
            """
            Given the row index `i` and column index `j` in the fragment,
            map them to a thread index according to `inverse_mma_store_layout`.
            """
            # the upper bounds of i and j are block_row_warps * warp_rows * micro_size_x and block_col_warps * warp_cols * micro_size_y
            # the upper bounds of block_row_warps and block_col_warps are warp_rows and warp_cols
            block_i, block_j = (i // micro_size_x) // warp_rows, (j // micro_size_y) // warp_cols
            # upper bounds of mma_i and mma_j are micro_size_x and micro_size_y
            mma_i, mma_j = i % micro_size_x, j % micro_size_y
            lane_id, _ = inverse_mma_store_layout.map_indices([mma_i, mma_j])
            if is_m_first:
                thread_id = block_i * (block_col_warps * warp_cols) + block_j * warp_size + lane_id
            else:
                thread_id = block_j * (block_row_warps * warp_size) + block_i * warp_size + lane_id
            return thread_id

        def forward_index(i: int, j: int) -> int:
            """
            Given the row index `i` and column index `j` in the fragment,
            map them to a local index in a single thread according
            to `inverse_mma_store_layout`.
            """
            # the upper bounds of i and j are block_row_warps * warp_rows * micro_size_x and block_col_warps * warp_cols * micro_size_y
            # the upper bounds of warp_i and warp_j are warp_rows and warp_cols
            warp_i, warp_j = (i // micro_size_x) % warp_rows, (j // micro_size_y) % warp_cols
            # upper bounds of mma_i and mma_j are micro_size_x and micro_size_y
            mma_i, mma_j = i % micro_size_x, j % micro_size_y
            _, local_id = inverse_mma_store_layout.map_indices([mma_i, mma_j])
            return warp_i * (warp_cols * local_size_out) + warp_j * local_size_out + local_id

        return T.Fragment(
            shape,
            forward_thread_fn=forward_thread,
            forward_index_fn=forward_index,
        )
