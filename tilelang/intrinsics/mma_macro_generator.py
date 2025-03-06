# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tilelang.language as T
from typing import Union, Tuple, Optional, Literal, Callable
from tilelang.common import TransformKind
from tvm import DataType
from tvm.tir import PrimExpr, IndexMap, Buffer
from tvm.runtime import convert
from .utils import (
    mma_store_index_map,
    get_ldmatrix_offset,
)
from tilelang.utils import is_fragment

lift = convert


# TODO(lei): Add Typing for this file
class TensorCoreIntrinEmitter(object):
    """
    To eliminate Python syntax within TIR Macro.
    """

    M_DIM = 16
    N_DIM = 16
    WARP_SIZE = 32
    dtype_abbrv = {
        "float16": "fp16",
        "bfloat16": "bf16",
        "float32": "fp32",
        "int8": "int8",
        "int32": "int32",
        "e4m3_float8": "e4m3",
        "e5m2_float8": "e5m2",
    }

    # Represent the thread binding in the form of (tx, warp_n, warp_m)
    is_m_first = False

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
        self._initialize_local_size(self.M_DIM, self.N_DIM, self.k_dim, self.WARP_SIZE)
        self._initialize_mma_prefix(self.k_dim)
        self._initialize_micro_size(self.M_DIM, self.N_DIM, self.k_dim)
        self._initialize_is_m_first(is_m_first)

        self.warp_rows = warp_row_tiles // self.micro_size_x
        self.warp_cols = warp_col_tiles // self.micro_size_y
        self.reduce_k = reduce_k
        self.threads = self.WARP_SIZE * (block_row_warps * block_col_warps) * reduce_k
        self.num_elems_per_byte = num_elems_per_byte

        if self.warp_rows == 0 or self.warp_cols == 0:
            raise ValueError(
                f"Invalid threads configuration for this tile shape, {self.warp_rows} x {self.warp_cols} with threads {self.threads}"
            )

    def _initialize_k_dim(self, a_dtype="float16"):
        if isinstance(a_dtype, str):
            a_dtype = DataType(a_dtype)
        self.k_dim = 256 // a_dtype.bits

    def _initialize_local_size(self, m_dim=16, n_dim=16, k_dim=16, warp_size=32):
        self.local_size_a = (m_dim * k_dim) // warp_size
        self.local_size_b = (n_dim * k_dim) // warp_size
        self.local_size_out = (m_dim * n_dim) // warp_size

    def _initialize_abbrev(self, a_dtype, b_dtype, accum_dtype):
        self.a_dtype_abbrv = self.dtype_abbrv[a_dtype]
        self.b_dtype_abbrv = self.dtype_abbrv[b_dtype]
        self.accum_dtype_abbrv = self.dtype_abbrv[accum_dtype]

    def _initialize_mma_prefix(self, k_dim: int = 16):
        if k_dim == 16:
            self.mma_prefix = "m16n8k16"
        elif k_dim == 32:
            self.mma_prefix = "m16n8k32"
        else:
            raise ValueError("Unsupported k_dim")

    def _initialize_micro_size(self, m_dim: int = 16, n_dim: int = 16, k_dim: int = 16):
        self.micro_size_x = m_dim
        self.micro_size_y = n_dim
        self.micro_size_k = k_dim

    def _initialize_is_m_first(self, is_m_first: Optional[bool] = False):
        if is_m_first is not None:
            self.is_m_first = is_m_first

    def get_store_index_map(self, inverse: bool = False) -> IndexMap:
        warp_size, local_size_c = self.WARP_SIZE, self.local_size_out
        index_map = IndexMap.from_func(mma_store_index_map, index_dtype="int32")
        if not inverse:
            return index_map
        inverse_index_map = index_map.inverse([warp_size, local_size_c])
        return inverse_index_map

    def extract_thread_binding(
            self,
            thread_id: PrimExpr,
            is_m_first: Optional[bool] = None) -> Tuple[PrimExpr, PrimExpr, PrimExpr]:
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

    def ldmatrix_a(self,
                   A_local_buf: Buffer,
                   A_shared_buf: Buffer,
                   ki: PrimExpr,
                   rk: Optional[PrimExpr] = 0):
        warp_row_tiles = self.warp_row_tiles
        warp_rows = self.warp_rows
        chunk = self.chunk
        micro_size_x = self.micro_size_x
        micro_size_k = self.micro_size_k
        local_size_a = self.local_size_a
        a_dtype = self.a_dtype
        a_transposed = self.a_transposed

        current_frame = T.KernelLaunchFrame.Current()
        thread_binding = current_frame.get_thread_binding()

        @T.macro
        def _warp_ldmatrix_a(
            A_local_buf,
            A_shared_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            stride = A_shared_buf.shape[-1]
            tx, _, warp_m = self.extract_thread_binding(thread_binding)
            for i in T.serial(warp_rows):
                T.ptx_ldmatrix(
                    a_dtype,
                    T.bool(False),
                    4,
                    ".b16",
                    A_local_buf.data,
                    i * local_size_a,
                    T.address_of(A_shared_buf[
                        warp_m * warp_row_tiles + i * micro_size_x,
                        rk * chunk + ki * micro_size_k,
                    ]),
                    get_ldmatrix_offset("A", tx, 0, stride, a_dtype, a_transposed),
                )

        return _warp_ldmatrix_a(A_local_buf, A_shared_buf, ki, thread_binding, rk)

    def ldmatrix_b(self,
                   B_local_buf: Buffer,
                   B_shared_buf: Buffer,
                   ki: PrimExpr,
                   rk: Optional[PrimExpr] = 0):
        warp_col_tiles = self.warp_col_tiles
        warp_cols = self.warp_cols
        chunk = self.chunk
        micro_size_y = self.micro_size_y
        micro_size_k = self.micro_size_k
        local_size_b = self.local_size_b
        b_dtype = self.b_dtype
        b_transposed = self.b_transposed
        current_frame = T.KernelLaunchFrame.Current()
        thread_binding = current_frame.get_thread_binding()

        @T.macro
        def _warp_ldmatrix_b(
            B_local_buf,
            B_shared_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            stride = B_shared_buf.shape[-1]
            tx, warp_n, _ = self.extract_thread_binding(thread_binding)

            for j in T.serial(warp_cols):
                # Assign B_shared_elem
                ri, rj = (
                    warp_n * warp_col_tiles + j * micro_size_y,
                    rk * chunk + ki * micro_size_k,
                )
                B_shared_elem = B_shared_buf[ri, rj]

                T.ptx_ldmatrix(
                    b_dtype,
                    T.bool(False),  # TODO(lei): should be optimized
                    4,
                    ".b16",
                    B_local_buf.data,
                    j * local_size_b,
                    T.address_of(B_shared_elem),
                    get_ldmatrix_offset("B", tx, 0, stride, b_dtype, b_transposed),
                )

        return _warp_ldmatrix_b(B_local_buf, B_shared_buf, ki, thread_binding, rk)

    def mma(self,
            A_local_buf: Buffer,
            B_local_buf: Buffer,
            C_local_buf: Buffer,
            k_inner: Optional[PrimExpr] = 0):
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_a = self.local_size_a
        local_size_b = self.local_size_b
        local_size_out = self.local_size_out
        a_dtype_abbrv = self.a_dtype_abbrv
        b_dtype_abbrv = self.b_dtype_abbrv
        accum_dtype = self.accum_dtype
        accum_dtype_abbrv = self.accum_dtype_abbrv
        mma_prefix = self.mma_prefix

        a_is_fragment = is_fragment(A_local_buf)
        b_is_fragment = is_fragment(B_local_buf)
        a_local_stride: PrimExpr = k_inner * warp_rows * local_size_a if a_is_fragment else 0
        b_local_stride: PrimExpr = k_inner * warp_cols * local_size_b if b_is_fragment else 0

        @T.macro
        def _warp_mma(A_local_buf, B_local_buf, C_local_buf):
            for i, j in T.grid(warp_rows, warp_cols):
                T.ptx_mma(
                    accum_dtype,
                    mma_prefix,
                    "row",
                    "col",
                    a_dtype_abbrv,
                    b_dtype_abbrv,
                    accum_dtype_abbrv,
                    A_local_buf.data,
                    a_local_stride + i * local_size_a,
                    B_local_buf.data,
                    b_local_stride + j * local_size_b,
                    C_local_buf.data,
                    i * warp_cols * local_size_out + j * local_size_out,
                    T.bool(False),
                )

                T.ptx_mma(
                    accum_dtype,
                    mma_prefix,
                    "row",
                    "col",
                    a_dtype_abbrv,
                    b_dtype_abbrv,
                    accum_dtype_abbrv,
                    A_local_buf.data,
                    a_local_stride + i * local_size_a,
                    B_local_buf.data,
                    b_local_stride + j * local_size_b + lift(local_size_b) // 2,
                    C_local_buf.data,
                    i * warp_cols * local_size_out + j * local_size_out + lift(local_size_out) // 2,
                    T.bool(False),
                )

        return _warp_mma(A_local_buf, B_local_buf, C_local_buf)

    def stmatrix(self, C_local_buf, C_buf, pid_m=None, pid_n=None):
        block_row_warps = self.block_row_warps
        block_col_warps = self.block_col_warps
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_out = self.local_size_out

        is_global = pid_m is not None and pid_n is not None
        BLOCK_M = block_row_warps * warp_rows
        BLOCK_N = block_col_warps * warp_cols
        M_DIM, N_DIM = self.M_DIM, self.N_DIM
        C_buf_dims = len(C_buf.shape)
        assert C_buf_dims in {2, 4}, "C_buf should be 2D or 4D"

        current_frame = T.KernelLaunchFrame.Current()
        thread_binding = current_frame.get_thread_binding()

        # STS
        # MMA Store must be in simulated instead of TVM Intrins
        # As TVM Intrins is like a hack that the threadIdx.x should be always
        # equal to the warp_size
        @T.macro
        def _warp_stmatrix_shared(C_local_buf, C_buf, thread_binding):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            for i, j in T.grid(warp_rows, warp_cols):
                for local_id_o in T.serial(local_size_out // 2):
                    for local_id_i in T.vectorized(2):
                        local_id = local_id_o * 2 + local_id_i
                        row, col = T.meta_var(mma_store_index_map(tx, local_id))
                        if C_buf_dims == 2:
                            C_buf[(warp_m * warp_rows + i) * M_DIM + row,
                                  (warp_n * warp_cols + j) * N_DIM +
                                  col] = C_local_buf[i * (warp_cols * local_size_out) +
                                                     j * local_size_out + local_id]
                        else:
                            C_buf[warp_m * warp_rows + i, warp_n * warp_cols + j, row,
                                  col] = C_local_buf[i * (warp_cols * local_size_out) +
                                                     j * local_size_out + local_id]

        @T.macro
        def _warp_stmatrix_global(C_local_buf, C_buf, thread_binding):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            for i, j in T.grid(warp_rows, warp_cols):
                for local_id_o in T.serial(local_size_out // 2):
                    for local_id_i in T.vectorized(2):
                        local_id = local_id_o * 2 + local_id_i
                        row, col = T.meta_var(mma_store_index_map(tx, local_id))
                        C_buf[
                            (pid_m * BLOCK_M + warp_m * warp_rows + i) * M_DIM + row,
                            (pid_n * BLOCK_N + warp_n * warp_cols + j) * N_DIM + col,
                        ] = C_local_buf[i * warp_cols * local_size_out + j * local_size_out +
                                        local_id]

        return (_warp_stmatrix_global(C_local_buf, C_buf, thread_binding)
                if is_global else _warp_stmatrix_shared(C_local_buf, C_buf, thread_binding))

    def make_mma_load_layout(self,
                             local_buf: Buffer,
                             matrix: Literal["A", "B"] = "A") -> T.Fragment:
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
        from tilelang.intrinsics.mma_layout import (
            shared_16x16_to_mma_32x8_layout_sr,
            shared_16x16_to_mma_32x8_layout_rs,
            shared_16x32_to_mma_32x16_layout,
            shared_32x16_to_mma_32x16_layout,
        )
        assert matrix in ["A", "B"], "matrix should be either A or B"
        dtype = self.a_dtype if matrix == "A" else self.b_dtype
        dtype_bits = DataType(dtype).bits
        transposed = self.a_transposed
        assert transposed is False, "transposed is not supported yet"
        # s represents spatial axis
        # r represents reduction axis
        # sr represents the two dims are spatial + reduction
        # rs represents the two dims are reduction + spatial
        transform_func_sr: Callable = None
        transform_func_rs: Callable = None
        if dtype_bits == 16:
            transform_func_sr = shared_16x16_to_mma_32x8_layout_sr
            transform_func_rs = shared_16x16_to_mma_32x8_layout_rs
        elif dtype_bits == 8:
            transform_func_sr = shared_16x32_to_mma_32x16_layout
            transform_func_rs = shared_32x16_to_mma_32x16_layout
        else:
            raise ValueError(f"Unsupported dtype {dtype}")
        is_sr_conditions = [False]
        is_sr_conditions.append(matrix == "A" and not transposed)
        is_sr_conditions.append(matrix == "B" and transposed)
        is_sr_axis_order = any(is_sr_conditions)

        transform_func: Callable = transform_func_sr if is_sr_axis_order else transform_func_rs

        assert is_fragment(local_buf), "local_buf must be a fragment, but got {}".format(
            local_buf.scope())

        if matrix == "A":
            micro_size_s, micro_size_r = self.micro_size_x, self.micro_size_k
        else:
            micro_size_r, micro_size_s = self.micro_size_k, self.micro_size_y

        block_row_warps, block_col_warps = (
            self.block_row_warps,
            self.block_col_warps,
        )
        warp_rows, warp_cols = self.warp_rows, self.warp_cols
        warp_s = warp_rows if matrix == "A" else warp_cols
        chunk = self.chunk
        transform_func = transform_func
        inverse_mma_load_layout = IndexMap.from_func(transform_func, index_dtype="int32")

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
            [micro_size_r, micro_size_s],
            forward_thread_fn=forward_thread,
            forward_index_fn=forward_index,
        )
        warp_fragment = base_fragment.repeat([block_row_warps, 1],
                                             repeat_on_thread=True).replicate(block_col_warps)
        block_fragment = warp_fragment.repeat([warp_s, chunk // micro_size_r],
                                              repeat_on_thread=False,
                                              lower_dim_first=False)
        print(f"base_fragment: {base_fragment}")
        print(f"warp_fragment: {warp_fragment}")
        print(f"block_fragment: {block_fragment}")
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


class TensorCoreIntrinEmitterWithLadderTransform(TensorCoreIntrinEmitter):
    """
    To eliminate Python syntax within TIR Macro.
    With Ladder Transform Plugin.
    """

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
        transform_kind_a: Union[int, TransformKind] = 0,
        transform_kind_b: Union[int, TransformKind] = 0,
    ):
        super().__init__(
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            accum_dtype=accum_dtype,
            a_transposed=a_transposed,
            b_transposed=b_transposed,
            block_row_warps=block_row_warps,
            block_col_warps=block_col_warps,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=chunk,
            reduce_k=reduce_k,
            num_elems_per_byte=num_elems_per_byte,
            is_m_first=is_m_first,
        )
        self._initialize_transform_kind(transform_kind_a, transform_kind_b)

    def _initialize_k_dim(self, a_dtype="float16"):
        self.k_dim = 256 // DataType(a_dtype).bits

    def _initialize_local_size(self, m_dim=16, n_dim=16, k_dim=16, warp_size=32):
        self.local_size_a = (m_dim * k_dim) // warp_size
        self.local_size_b = (n_dim * k_dim) // warp_size
        self.local_size_out = (m_dim * n_dim) // warp_size

    def _initialize_abbrev(self, a_dtype, b_dtype, accum_dtype):
        self.a_dtype_abbrv = self.dtype_abbrv[a_dtype]
        self.b_dtype_abbrv = self.dtype_abbrv[b_dtype]
        self.accum_dtype_abbrv = self.dtype_abbrv[accum_dtype]

    def _initialize_mma_prefix(self, k_dim=16):
        if k_dim == 16:
            self.mma_prefix = "m16n8k16"
        elif k_dim == 32:
            self.mma_prefix = "m16n8k32"
        else:
            raise ValueError("Unsupported k_dim")

    def _initialize_micro_size(self, m_dim=16, n_dim=16, k_dim=16):
        self.micro_size_x = m_dim
        self.micro_size_y = n_dim
        self.micro_size_k = k_dim

    def _initialize_transform_kind(self, transform_kind_a, transform_kind_b):
        if isinstance(transform_kind_a, int):
            self.transform_kind_a = TransformKind(transform_kind_a)
        elif isinstance(transform_kind_a, TransformKind):
            self.transform_kind_a = transform_kind_a
        else:
            raise ValueError("Unsupported transform_kind_a")

        if isinstance(transform_kind_b, int):
            self.transform_kind_b = TransformKind(transform_kind_b)
        elif isinstance(transform_kind_b, TransformKind):
            self.transform_kind_b = transform_kind_b
        else:
            raise ValueError("Unsupported transform_kind_b")

        assert transform_kind_a in [0, 1, 2, 3], "Input transform stage should be 0, 1, 2, or 3"
        assert transform_kind_b in [0, 1, 2, 3], "Weight transform stage should be 0, 1, 2, or 3"

    def ldmatrix_a(self, A_local_buf, A_shared_buf, ki, rk=0):
        warp_row_tiles = self.warp_row_tiles
        warp_rows = self.warp_rows
        chunk = self.chunk
        micro_size_x = self.micro_size_x
        micro_size_k = self.micro_size_k
        local_size_a = self.local_size_a
        a_dtype = self.a_dtype
        a_transposed = self.a_transposed
        transform_kind_a = self.transform_kind_a

        current_frame = T.KernelLaunchFrame.Current()
        thread_binding = current_frame.get_thread_binding()

        @T.macro
        def _warp_ldmatrix_a(
            A_local_buf,
            A_shared_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            stride = A_shared_buf.shape[-1]
            tx, _, warp_m = self.extract_thread_binding(thread_binding)
            if transform_kind_a == TransformKind.NonTransform:
                for i in T.serial(warp_rows):
                    T.ptx_ldmatrix(
                        a_dtype,
                        T.bool(False),
                        4,
                        ".b16",
                        A_local_buf.data,
                        i * local_size_a,
                        T.address_of(A_shared_buf[
                            warp_m * warp_row_tiles + i * micro_size_x,
                            rk * chunk + ki * micro_size_k,
                        ]),
                        get_ldmatrix_offset("A", tx, 0, stride, a_dtype, a_transposed),
                    )
            elif transform_kind_a == TransformKind.InterWarpTransform:
                for i in T.serial(warp_rows):
                    # Assign B_shared_elem
                    ri, rj = (
                        warp_m * warp_row_tiles + i * micro_size_x,
                        rk * chunk + ki * micro_size_k,
                    )
                    ni, nj, nii, njj = (
                        (ri) // micro_size_x,
                        (rj) // micro_size_k,
                        (ri) % micro_size_x,
                        (rj) % micro_size_k,
                    )
                    args = (ni, nj, nii, njj) if transform_kind_a > 0 else (ri, rj)
                    A_shared_elem = A_shared_buf[args]

                    T.ptx_ldmatrix(
                        a_dtype,
                        T.bool(False),
                        4,
                        ".b16",
                        A_local_buf.data,
                        i * local_size_a,
                        T.address_of(A_shared_elem),
                        get_ldmatrix_offset("A", tx, 0, stride, a_dtype, a_transposed),
                    )
            elif transform_kind_a == TransformKind.IntraWarpTransform:
                for i in T.serial(warp_rows):
                    # Assign B_shared_elem
                    ri, rj = (
                        warp_m * warp_row_tiles + i * micro_size_x,
                        rk * chunk + ki * micro_size_k,
                    )
                    ni, nj, nii, njj = (
                        (ri) // micro_size_x,
                        (rj) // micro_size_k,
                        (ri) % micro_size_x,
                        (rj) % micro_size_k,
                    )
                    A_shared_elem = A_shared_buf[ni, nj, nii, njj]

                    T.ptx_ldmatrix(
                        a_dtype,
                        T.bool(False),
                        4,
                        ".b16",
                        A_local_buf.data,
                        i * local_size_a,
                        T.address_of(A_shared_elem),
                        tx * local_size_a,
                    )
            elif transform_kind_a == TransformKind.LDMatrixTransform:
                for j in T.serial(warp_rows):
                    for local_id in T.vectorized(local_size_a):
                        # Assign A_shared_elem
                        ri, rj = (
                            warp_m * warp_rows + j,
                            rk * (chunk // micro_size_k) + ki,
                        )
                        rii, rjj = (tx * local_size_a +
                                    local_id) // micro_size_k, (tx * local_size_a + local_id) % (
                                        micro_size_k)
                        A_local_buf[j * local_size_a + local_id] = (A_shared_buf[ri, rj, rii, rjj])
            else:
                raise ValueError("Unsupported TransformKind for Input A")

        return _warp_ldmatrix_a(A_local_buf, A_shared_buf, ki, thread_binding, rk)

    def ldmatrix_b(self, B_local_buf, B_shared_buf, ki, rk=0):
        warp_col_tiles = self.warp_col_tiles
        warp_cols = self.warp_cols
        chunk = self.chunk
        micro_size_y = self.micro_size_y
        micro_size_k = self.micro_size_k
        local_size_b = self.local_size_b
        b_dtype = self.b_dtype
        transform_kind_b = self.transform_kind_b
        b_transposed = self.b_transposed
        num_elems_per_byte = self.num_elems_per_byte

        current_frame = T.KernelLaunchFrame.Current()
        thread_binding = current_frame.get_thread_binding()

        @T.macro
        def _warp_ldmatrix_b(
            B_local_buf,
            B_shared_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            stride = B_shared_buf.shape[-1]
            tx, warp_n, _ = self.extract_thread_binding(thread_binding)

            if transform_kind_b == TransformKind.NonTransform:
                for j in T.serial(warp_cols):
                    # Assign B_shared_elem
                    ri, rj = (
                        warp_n * warp_col_tiles + j * micro_size_y,
                        rk * chunk + ki * micro_size_k,
                    )
                    B_shared_elem = B_shared_buf[ri, rj]

                    T.ptx_ldmatrix(
                        b_dtype,
                        T.bool(False),
                        4,
                        ".b16",
                        B_local_buf.data,
                        j * local_size_b,
                        T.address_of(B_shared_elem),
                        get_ldmatrix_offset("B", tx, 0, stride, b_dtype, b_transposed),
                    )
            elif transform_kind_b == TransformKind.InterWarpTransform:
                for j in T.serial(warp_cols):
                    # Assign B_shared_elem
                    ri, rj = (
                        warp_n * warp_col_tiles + j * micro_size_y,
                        rk * chunk + ki * micro_size_k,
                    )
                    ni, nj, nii, njj = (
                        (ri) // micro_size_y,
                        (rj) // micro_size_k,
                        (ri) % micro_size_y,
                        (rj) % micro_size_k,
                    )
                    B_shared_elem = B_shared_buf[ni, nj, nii, njj]

                    T.ptx_ldmatrix(
                        b_dtype,
                        T.bool(False),  # TODO(lei): should be optimized
                        4,
                        ".b16",
                        B_local_buf.data,
                        j * local_size_b,
                        T.address_of(B_shared_elem),
                        get_ldmatrix_offset("B", tx, 0, stride, b_dtype, b_transposed),
                    )
            elif transform_kind_b == TransformKind.IntraWarpTransform:
                for j in T.serial(warp_cols):
                    # Assign B_shared_elem
                    ri, rj = (
                        warp_n * warp_col_tiles + j * micro_size_y,
                        rk * chunk + ki * micro_size_k,
                    )
                    ni, nj, nii, njj = (
                        (ri) // micro_size_y,
                        (rj) // micro_size_k,
                        (ri) % micro_size_y,
                        (rj) % micro_size_k,
                    )
                    B_shared_elem = B_shared_buf[ni, nj, nii, njj]

                    T.ptx_ldmatrix(
                        b_dtype,
                        T.bool(False),  # TODO(lei): should be optimized
                        4,
                        ".b16",
                        B_local_buf.data,
                        j * local_size_b,
                        T.address_of(B_shared_elem),
                        tx * local_size_b,
                    )
            elif transform_kind_b == TransformKind.LDMatrixTransform:
                local_size_dequantize = local_size_b // num_elems_per_byte
                for j in T.serial(warp_cols):
                    for local_id in T.vectorized(local_size_dequantize):
                        # Assign B_shared_elem
                        ri, rj = (
                            warp_n * warp_cols + j,
                            rk * (chunk // micro_size_k) + ki,
                        )
                        rii, rjj = (tx * local_size_dequantize +
                                    local_id) // (micro_size_k // num_elems_per_byte), (
                                        tx * local_size_dequantize + local_id) % (
                                            micro_size_k // num_elems_per_byte)
                        B_local_buf[j * local_size_dequantize + local_id] = (
                            B_shared_buf[ri, rj, rii, rjj])
            else:
                raise ValueError("Unsupported TransformKind for Input B")

        return _warp_ldmatrix_b(B_local_buf, B_shared_buf, ki, thread_binding, rk)

    def mma(self, A_local_buf, B_local_buf, C_local_buf):
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_a = self.local_size_a
        local_size_b = self.local_size_b
        local_size_out = self.local_size_out
        a_dtype_abbrv = self.a_dtype_abbrv
        b_dtype_abbrv = self.b_dtype_abbrv
        accum_dtype = self.accum_dtype
        accum_dtype_abbrv = self.accum_dtype_abbrv
        mma_prefix = self.mma_prefix

        @T.macro
        def _warp_mma(A_local_buf, B_local_buf, C_local_buf):
            for i, j in T.grid(warp_rows, warp_cols):
                T.ptx_mma(
                    accum_dtype,
                    mma_prefix,
                    "row",
                    "col",
                    a_dtype_abbrv,
                    b_dtype_abbrv,
                    accum_dtype_abbrv,
                    A_local_buf.data,
                    i * local_size_a,
                    B_local_buf.data,
                    j * local_size_b,
                    C_local_buf.data,
                    i * warp_cols * local_size_out + j * local_size_out,
                    T.bool(False),
                )

                T.ptx_mma(
                    accum_dtype,
                    mma_prefix,
                    "row",
                    "col",
                    a_dtype_abbrv,
                    b_dtype_abbrv,
                    accum_dtype_abbrv,
                    A_local_buf.data,
                    i * local_size_a,
                    B_local_buf.data,
                    j * local_size_b + lift(local_size_b) // 2,
                    C_local_buf.data,
                    i * warp_cols * local_size_out + j * local_size_out + lift(local_size_out) // 2,
                    T.bool(False),
                )

        return _warp_mma(A_local_buf, B_local_buf, C_local_buf)


class INT4TensorCoreIntrinEmitter(TensorCoreIntrinEmitter):

    def mma(self, A_local_buf, B_local_buf, C_local_buf):
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_a = self.local_size_a
        local_size_b = self.local_size_b
        local_size_out = self.local_size_out
        a_dtype_abbrv = "int4"
        b_dtype_abbrv = "int4"
        accum_dtype = self.accum_dtype
        accum_dtype_abbrv = accum_dtype
        mma_prefix = "m16n8k32"

        @T.macro
        def _warp_mma(A_local_buf, B_local_buf, C_local_buf):
            for i, j in T.grid(warp_rows, warp_cols):
                """
                A[16, 32], B[16, 32], C[16, 16]
                A_local_size -> 16
                B_local_size -> 16
                C_local_size -> 8
                For each m16n8k32 inst
                For A: m16k32 consume 16 int4 elements -> 8 A_local_size
                For A: n8k32 consume 8 int4 elements -> 4 B_local_size
                For C: m16n8 consume 4 int32 elements -> 4 C_local_size
                """

                # A[0:16, 0:16] * B[0:8, 0:16] -> C[0:16, 0:8]
                T.ptx_mma(
                    accum_dtype,
                    mma_prefix,
                    "row",
                    "col",
                    a_dtype_abbrv,
                    b_dtype_abbrv,
                    accum_dtype_abbrv,
                    A_local_buf.data,
                    i * local_size_a,
                    B_local_buf.data,
                    j * local_size_b,
                    C_local_buf.data,
                    i * warp_cols * local_size_out + j * local_size_out,
                    T.bool(False),
                )

                # A[0:16, 0:16] * B[8:16, 0:16] -> C[0:16, 8:16]
                T.ptx_mma(
                    accum_dtype,
                    mma_prefix,
                    "row",
                    "col",
                    a_dtype_abbrv,
                    b_dtype_abbrv,
                    accum_dtype_abbrv,
                    A_local_buf.data,
                    i * local_size_a,
                    B_local_buf.data,
                    j * local_size_b + lift(local_size_b) // 2,
                    C_local_buf.data,
                    i * warp_cols * local_size_out + j * local_size_out + lift(local_size_out) // 2,
                    T.bool(False),
                )

                # A[0:16, 16:32] * B[0:8, 16:32] -> C[0:16, 0:8]
                T.ptx_mma(
                    accum_dtype,
                    mma_prefix,
                    "row",
                    "col",
                    a_dtype_abbrv,
                    b_dtype_abbrv,
                    accum_dtype_abbrv,
                    A_local_buf.data,
                    i * local_size_a + lift(local_size_a) // 2,
                    B_local_buf.data,
                    j * local_size_b + lift(local_size_b) // 4,
                    C_local_buf.data,
                    i * warp_cols * local_size_out + j * local_size_out,
                    T.bool(False),
                )

                # A[0:16, 16:32] * B[8:16, 16:32] -> C[0:16, 8:16]
                T.ptx_mma(
                    accum_dtype,
                    mma_prefix,
                    "row",
                    "col",
                    a_dtype_abbrv,
                    b_dtype_abbrv,
                    accum_dtype_abbrv,
                    A_local_buf.data,
                    i * local_size_a + lift(local_size_b) // 2,
                    B_local_buf.data,
                    j * local_size_b + lift(local_size_b) // 2 + lift(local_size_b) // 4,
                    C_local_buf.data,
                    i * warp_cols * local_size_out + j * local_size_out + lift(local_size_out) // 2,
                    T.bool(False),
                )

        return _warp_mma(A_local_buf, B_local_buf, C_local_buf)


class INT4TensorCoreIntrinEmitterWithLadderTransform(TensorCoreIntrinEmitterWithLadderTransform):

    def mma(self, A_local_buf, B_local_buf, C_local_buf):

        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_a = self.local_size_a
        local_size_b = self.local_size_b
        local_size_out = self.local_size_out
        a_dtype_abbrv = "int4"
        b_dtype_abbrv = "int4"
        accum_dtype = self.accum_dtype
        accum_dtype_abbrv = "int32"
        mma_prefix = "m16n8k32"

        @T.macro
        def _warp_mma(A_local_buf, B_local_buf, C_local_buf):
            for i, j in T.grid(warp_rows, warp_cols):
                """
                A[16, 32], B[16, 32], C[16, 16]
                A_local_size -> 16
                B_local_size -> 16
                C_local_size -> 8
                For each m16n8k32 inst
                For A: m16k32 consume 16 int4 elements -> 8 A_local_size
                For A: n8k32 consume 8 int4 elements -> 4 B_local_size
                For C: m16n8 consume 4 int32 elements -> 4 C_local_size
                """

                # A[0:16, 0:16] * B[0:8, 0:16] -> C[0:16, 0:8]
                T.ptx_mma(
                    accum_dtype,
                    mma_prefix,
                    "row",
                    "col",
                    a_dtype_abbrv,
                    b_dtype_abbrv,
                    accum_dtype_abbrv,
                    A_local_buf.data,
                    i * local_size_a,
                    B_local_buf.data,
                    j * local_size_b,
                    C_local_buf.data,
                    i * warp_cols * local_size_out + j * local_size_out,
                    T.bool(False),
                )

                # A[0:16, 0:16] * B[8:16, 0:16] -> C[0:16, 8:16]
                T.ptx_mma(
                    accum_dtype,
                    mma_prefix,
                    "row",
                    "col",
                    a_dtype_abbrv,
                    b_dtype_abbrv,
                    accum_dtype_abbrv,
                    A_local_buf.data,
                    i * local_size_a,
                    B_local_buf.data,
                    j * local_size_b + lift(local_size_b) // 2,
                    C_local_buf.data,
                    i * warp_cols * local_size_out + j * local_size_out + lift(local_size_out) // 2,
                    T.bool(False),
                )

                # A[0:16, 16:32] * B[0:8, 16:32] -> C[0:16, 0:8]
                T.ptx_mma(
                    accum_dtype,
                    mma_prefix,
                    "row",
                    "col",
                    a_dtype_abbrv,
                    b_dtype_abbrv,
                    accum_dtype_abbrv,
                    A_local_buf.data,
                    i * local_size_a + lift(local_size_a) // 2,
                    B_local_buf.data,
                    j * local_size_b + lift(local_size_b) // 4,
                    C_local_buf.data,
                    i * warp_cols * local_size_out + j * local_size_out,
                    T.bool(False),
                )

                # A[0:16, 16:32] * B[8:16, 16:32] -> C[0:16, 8:16]
                T.ptx_mma(
                    accum_dtype,
                    mma_prefix,
                    "row",
                    "col",
                    a_dtype_abbrv,
                    b_dtype_abbrv,
                    accum_dtype_abbrv,
                    A_local_buf.data,
                    i * local_size_a + lift(local_size_b) // 2,
                    B_local_buf.data,
                    j * local_size_b + lift(local_size_b) // 2 + lift(local_size_b) // 4,
                    C_local_buf.data,
                    i * warp_cols * local_size_out + j * local_size_out + lift(local_size_out) // 2,
                    T.bool(False),
                )

        return _warp_mma(A_local_buf, B_local_buf, C_local_buf)
