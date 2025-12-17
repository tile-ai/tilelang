from __future__ import annotations
import tilelang.language as T
from typing import Literal, Callable
from tilelang.common import TransformKind
from tvm import DataType
from tvm import tir
from tvm.ir import Range
from tvm.tir import PrimExpr, IndexMap, Buffer, Var, BufferRegion, BufferLoad
from tilelang import tvm as tvm
from tvm.runtime import convert
from .utils import (
    mma_store_index_map,
    get_ldmatrix_offset,
)
from tilelang.utils import is_fragment, get_buffer_region_from_load
from tilelang.intrinsics.mma_layout import (
    shared_16x8_to_mma_32x4_layout_sr_a,
    shared_16x8_to_mma_32x4_layout_sr_b,
    shared_16x16_to_mma_32x8_layout_sr_a,
    shared_16x16_to_mma_32x8_layout_sr_b,
    shared_16x32_to_mma_32x16_layout_sr_a,
    shared_16x32_to_mma_32x16_layout_sr_b,
    mma_load_a_32x4_to_shared_16x8_layout,
    mma_load_b_32x4_to_shared_16x8_layout,
    mma_load_b_32x8_to_shared_16x16_layout,
    mma_load_a_32x16_to_shared_16x32_layout,
    mma_load_b_32x16_to_shared_16x32_layout,
    mma_load_a_32x8_to_shared_16x16_layout,
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
    dtype_abbrv = {
        "float16": "fp16",
        "bfloat16": "bf16",
        "float32": "fp32",
        "float64": "fp64",
        "int8": "int8",
        "int32": "int32",
        "float8_e4m3": "e4m3",
        "float8_e4m3fn": "e4m3",
        "float8_e4m3fnuz": "e4m3",
        "float8_e5m2": "e5m2",
        "float8_e5m2fnuz": "e5m2",
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
        # For FP64, MMA shape is m8n8k4; adjust instance dims early
        if DataType(a_dtype).bits == 64:
            # Override default M/N dims for fp64 MMA
            self.M_DIM = 8
            # n_dim will be set to 8 in _initialize_micro_size via k_dim==4
        self._initialize_abbrev(a_dtype, b_dtype, accum_dtype)
        self._initialize_micro_size(self.M_DIM, self.k_dim)
        self._initialize_local_size(self.M_DIM, self.n_dim, self.k_dim, self.WARP_SIZE)
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
        if isinstance(a_dtype, str):
            a_dtype = DataType(a_dtype)
        self.k_dim = 256 // a_dtype.bits

    def _initialize_local_size(self, m_dim=16, n_dim=16, k_dim=16, warp_size=32):
        self.local_size_a = (m_dim * k_dim) // warp_size
        self.local_size_b = (n_dim * k_dim) // warp_size
        self.local_size_out = (m_dim * n_dim) // warp_size

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
            # fp64
            self.mma_prefix = "m8n8k4"
        elif k_dim == 8:
            # typically used for tfloat32
            self.mma_prefix = "m16n8k8"
        elif k_dim == 16:
            # typically used for float16/bfloat16
            self.mma_prefix = "m16n8k16"
        elif k_dim == 32:
            # typically used for int8/fp8
            self.mma_prefix = "m16n8k32"
        else:
            raise ValueError("Unsupported k_dim")

    def _initialize_micro_size(self, m_dim: int = 16, k_dim: int = 16):
        warp_row_tiles = self.warp_row_tiles
        warp_col_tiles = self.warp_col_tiles
        # For fp64 (k_dim==4), micro tile is 8x8, otherwise keep 16x{8|16}
        if k_dim == 4:
            # fp64 path: m_dim must be 8, n_dim 8
            assert m_dim == 8, f"For fp64 MMA, m_dim must be 8, got {m_dim}"
            self.n_dim = 8
            self.micro_size_y = 8
            self.warp_rows = warp_row_tiles // m_dim
            self.warp_cols = warp_col_tiles // 8
        else:
            assert warp_row_tiles >= 16, f"warp_row_tiles must be greater than 16, got {warp_row_tiles}"
            assert warp_row_tiles % 16 == 0, f"warp_row_tiles must be divisible by 16, got {warp_row_tiles}"
            assert warp_col_tiles >= 8, f"warp_col_tiles must be greater than 8, got {warp_col_tiles}"
            assert warp_col_tiles % 8 == 0, f"warp_col_tiles must be divisible by 8, got {warp_col_tiles}"

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
        from .utils import mma_store_index_map, mma_store_index_map_fp64

        warp_size, local_size_c = self.WARP_SIZE, self.local_size_out
        if DataType(self.accum_dtype).bits == 64:
            index_map = IndexMap.from_func(mma_store_index_map_fp64, index_dtype=T.int32)
        else:
            index_map = IndexMap.from_func(mma_store_index_map, index_dtype=T.int32)
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
        # Fast path for fp64: no ldmatrix support, do direct per-lane loads
        if DataType(self.a_dtype).bits == 64:
            warp_row_tiles = self.warp_row_tiles
            warp_rows = self.warp_rows
            chunk = self.chunk
            micro_size_x = self.micro_size_x  # 8
            micro_size_k = self.micro_size_k  # 4
            local_size_a = self.local_size_a  # 1
            a_transposed = self.a_transposed

            thread_binding = self.get_thread_binding()
            # legalize shared buffer to region
            A_region = self._legalize_to_buffer_region(A_shared_buf)
            A_buf = A_region.buffer
            A_base0 = A_region.region[-2].min
            A_base1 = A_region.region[-1].min

            @T.macro
            def _warp_ld_a_fp64(
                A_local_buf,
                A_shared_buf,
                ki,
                thread_binding,
                rk=0,
            ):
                tx, _, warp_m = self.extract_thread_binding(thread_binding)
                for i in T.serial(warp_rows):
                    wi = warp_m * warp_row_tiles + i * micro_size_x
                    wk = rk * chunk + ki * micro_size_k
                    mi = tx // micro_size_k
                    mk = tx % micro_size_k
                    if a_transposed:
                        A_local_buf[i * local_size_a] = A_buf[A_base0 + wk + mk, A_base1 + wi + mi]
                    else:
                        A_local_buf[i * local_size_a] = A_buf[A_base0 + wi + mi, A_base1 + wk + mk]

            return _warp_ld_a_fp64(A_local_buf, A_region, ki, thread_binding, rk)

        warp_row_tiles = self.warp_row_tiles
        warp_rows = self.warp_rows
        chunk = self.chunk
        micro_size_x = self.micro_size_x
        micro_size_k = self.micro_size_k
        local_size_a = self.local_size_a
        a_dtype = self.a_dtype
        a_transposed = self.a_transposed
        # ldmatrix cannot be used for int8 + trans case.
        ldmatrix_available = not (DataType(a_dtype).bits != 16 and a_transposed)

        def mma_load_layout(i, j):
            return i, j

        if not ldmatrix_available:
            if DataType(a_dtype).bits == 8:
                mma_load_layout = mma_load_a_32x16_to_shared_16x32_layout
            elif DataType(a_dtype).bits == 16:
                mma_load_layout = mma_load_a_32x8_to_shared_16x16_layout
            elif DataType(a_dtype).bits == 32:
                mma_load_layout = mma_load_a_32x4_to_shared_16x8_layout
            else:
                raise ValueError(f"Unsupported dtype: {a_dtype}")

        thread_binding = self.get_thread_binding()

        # legalize shared buffer to region
        A_region = self._legalize_to_buffer_region(A_shared_buf)
        A_buf = A_region.buffer
        A_base0 = A_region.region[-2].min
        A_base1 = A_region.region[-1].min
        A_stride_last = A_buf.shape[-1]

        @T.macro
        def _warp_ldmatrix_a(
            A_local_buf,
            A_shared_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            stride = A_stride_last
            tx, _, warp_m = self.extract_thread_binding(thread_binding)
            trans = self.a_transposed

            for i in T.serial(warp_rows):
                # Assign A_shared_buf_elem
                wi, wk = warp_m * warp_row_tiles + i * micro_size_x, rk * chunk + ki * micro_size_k
                A_shared_buf_elem = A_buf[A_base0 + wk, A_base1 + wi] if a_transposed else A_buf[A_base0 + wi, A_base1 + wk]

                if ldmatrix_available:
                    T.ptx_ldmatrix(
                        a_dtype,
                        T.bool(trans),
                        4,
                        ".b16",
                        A_local_buf.data,
                        i * local_size_a,
                        T.address_of(A_shared_buf_elem),
                        get_ldmatrix_offset("A", tx, 0, stride, a_dtype, a_transposed),
                    )
                else:
                    for j in T.serial(local_size_a):
                        mi, mk = mma_load_layout(tx, j)
                        if a_transposed:
                            A_local_buf[i * local_size_a + j] = A_buf[A_base0 + wk + mk, A_base1 + wi + mi]
                        else:
                            A_local_buf[i * local_size_a + j] = A_buf[A_base0 + wi + mi, A_base1 + wk + mk]

        return _warp_ldmatrix_a(A_local_buf, A_region, ki, thread_binding, rk)

    def ldmatrix_b(self, B_local_buf: Buffer, B_shared_buf: Buffer | BufferRegion, ki: PrimExpr, rk: PrimExpr | None = 0):
        # Fast path for fp64: no ldmatrix support, do direct per-lane loads
        if DataType(self.b_dtype).bits == 64:
            warp_col_tiles = self.warp_col_tiles
            warp_cols = self.warp_cols
            chunk = self.chunk
            micro_size_y = self.micro_size_y  # 8
            micro_size_k = self.micro_size_k  # 4
            local_size_b = self.local_size_b  # 1
            b_transposed = self.b_transposed
            thread_binding = self.get_thread_binding()

            # legalize shared buffer to region
            B_region = self._legalize_to_buffer_region(B_shared_buf)
            B_buf = B_region.buffer
            B_base0 = B_region.region[-2].min
            B_base1 = B_region.region[-1].min

            @T.macro
            def _warp_ld_b_fp64(
                B_local_buf,
                B_shared_buf,
                ki,
                thread_binding,
                rk=0,
            ):
                tx, warp_n, _ = self.extract_thread_binding(thread_binding)
                for j in T.serial(warp_cols):
                    wi = warp_n * warp_col_tiles + j * micro_size_y
                    wk = rk * chunk + ki * micro_size_k
                    mi = tx // micro_size_k
                    mk = tx % micro_size_k
                    if b_transposed:
                        B_local_buf[j * local_size_b] = B_buf[B_base0 + wi + mi, B_base1 + wk + mk]
                    else:
                        B_local_buf[j * local_size_b] = B_buf[B_base0 + wk + mk, B_base1 + wi + mi]

            return _warp_ld_b_fp64(B_local_buf, B_region, ki, thread_binding, rk)

        warp_col_tiles = self.warp_col_tiles
        warp_cols = self.warp_cols
        chunk = self.chunk
        micro_size_y = self.micro_size_y
        micro_size_k = self.micro_size_k
        local_size_b = self.local_size_b
        b_dtype = self.b_dtype
        b_transposed = self.b_transposed
        thread_binding = self.get_thread_binding()

        # legalize shared buffer to region
        B_region = self._legalize_to_buffer_region(B_shared_buf)
        B_buf = B_region.buffer
        B_base0 = B_region.region[-2].min
        B_base1 = B_region.region[-1].min
        B_stride_last = B_buf.shape[-1]
        replicate_b = self.n_dim == 16
        # ldmatrix cannot be used for int8 + trans case.
        ldmatrix_available = not (DataType(b_dtype).bits != 16 and not b_transposed)

        def mma_load_layout(i, j):
            return i, j

        if not ldmatrix_available:
            if DataType(b_dtype).bits == 8:
                mma_load_layout = mma_load_b_32x16_to_shared_16x32_layout
            elif DataType(b_dtype).bits == 16:
                mma_load_layout = mma_load_b_32x8_to_shared_16x16_layout
            elif DataType(b_dtype).bits == 32:
                mma_load_layout = mma_load_b_32x4_to_shared_16x8_layout
            else:
                raise ValueError(f"Unsupported dtype: {b_dtype}")

        @T.macro
        def _warp_ldmatrix_b(
            B_local_buf,
            B_shared_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            stride = B_stride_last
            tx, warp_n, _ = self.extract_thread_binding(thread_binding)
            trans = not b_transposed

            for i in T.serial(warp_cols):
                # Assign B_shared_elem
                wi, wk = (
                    warp_n * warp_col_tiles + i * micro_size_y,
                    rk * chunk + ki * micro_size_k,
                )

                if ldmatrix_available:
                    B_shared_buf_elem = B_buf[B_base0 + wi, B_base1 + wk] if b_transposed else B_buf[B_base0 + wk, B_base1 + wi]

                    T.ptx_ldmatrix(
                        b_dtype,
                        T.bool(trans),
                        4 if replicate_b else 2,
                        ".b16",
                        B_local_buf.data,
                        i * local_size_b,
                        T.address_of(B_shared_buf_elem),
                        get_ldmatrix_offset("B", tx, 0, stride, b_dtype, b_transposed),
                    )

                else:
                    # load 16x32 data from shared buffer to local buffer
                    # must be transposed.
                    for j in T.serial(local_size_b):
                        mi, mk = mma_load_layout(tx, j)
                        if b_transposed:
                            B_local_buf[i * local_size_b + j] = B_buf[B_base0 + wi + mi, B_base1 + wk + mk]
                        else:
                            B_local_buf[i * local_size_b + j] = B_buf[B_base0 + wk + mk, B_base1 + wi + mi]

        return _warp_ldmatrix_b(B_local_buf, B_shared_buf, ki, thread_binding, rk)

    def mma(self, A_local_buf: Buffer, B_local_buf: Buffer, C_local_buf: Buffer, k_inner: PrimExpr | None = 0):
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
        replicate_b = self.n_dim == 16

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
                    T.bool(False),  # saturate
                )
                if replicate_b:
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
                        T.bool(False),  # saturate
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
        M_DIM, n_dim = self.M_DIM, self.n_dim
        C_buf_dims = len(C_buf.shape)
        assert C_buf_dims in {2, 4}, "C_buf should be 2D or 4D"

        thread_binding = self.get_thread_binding()

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
                            C_buf[(warp_m * warp_rows + i) * M_DIM + row, (warp_n * warp_cols + j) * n_dim + col] = C_local_buf[
                                i * (warp_cols * local_size_out) + j * local_size_out + local_id
                            ]
                        else:
                            C_buf[warp_m * warp_rows + i, warp_n * warp_cols + j, row, col] = C_local_buf[
                                i * (warp_cols * local_size_out) + j * local_size_out + local_id
                            ]

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
                            (pid_n * BLOCK_N + warp_n * warp_cols + j) * n_dim + col,
                        ] = C_local_buf[i * warp_cols * local_size_out + j * local_size_out + local_id]

        return (
            _warp_stmatrix_global(C_local_buf, C_buf, thread_binding)
            if is_global
            else _warp_stmatrix_shared(C_local_buf, C_buf, thread_binding)
        )

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
        if dtype_bits == 32:
            transform_func_sr_a = shared_16x8_to_mma_32x4_layout_sr_a
            transform_func_sr_b = shared_16x8_to_mma_32x4_layout_sr_b
        elif dtype_bits == 16:
            transform_func_sr_a = shared_16x16_to_mma_32x8_layout_sr_a
            transform_func_sr_b = shared_16x16_to_mma_32x8_layout_sr_b
        elif dtype_bits == 8:
            transform_func_sr_a = shared_16x32_to_mma_32x16_layout_sr_a
            transform_func_sr_b = shared_16x32_to_mma_32x16_layout_sr_b
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
            transform_func = transform_func_sr_b if is_sr_axis_order else lambda i, j: transform_func_sr_b(j, i)
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
        assert is_fragment(local_buf), f"local_buf {local_buf} must be a fragment, but got {local_buf.scope()}"
        inverse_mma_store_layout = self.get_store_index_map(inverse=True)

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

    @staticmethod
    def _legalize_to_buffer_region(obj: Buffer | BufferLoad | BufferRegion) -> BufferRegion:
        """
        Convert Buffer/BufferRegion/BufferLoad to a BufferRegion.

        - Buffer -> full-region BufferRegion covering entire shape
        - BufferRegion -> returned as-is
        - BufferLoad -> best-effort convert via get_buffer_region_from_load;
        if scalar, fall back to 1-sized ranges at given indices
        """
        if isinstance(obj, BufferRegion):
            return obj
        if isinstance(obj, Buffer):
            mins = [tir.IntImm("int32", 0) for _ in obj.shape]
            ranges = [Range.from_min_extent(m, e) for m, e in zip(mins, obj.shape)]
            return BufferRegion(obj, ranges)
        if isinstance(obj, BufferLoad):
            region = get_buffer_region_from_load(obj)
            if region is not None:
                return region
            # Fallback: scalar load -> 1-sized ranges at indices
            mins = [idx for idx in obj.indices]
            ones = [tir.IntImm("int32", 1) for _ in obj.indices]
            ranges = [Range.from_min_extent(m, e) for m, e in zip(mins, ones)]
            return BufferRegion(obj.buffer, ranges)
        raise ValueError(f"Unsupported argument type for BufferRegion: {type(obj)}")


class TensorCoreIntrinEmitterWithLadderTransform(TensorCoreIntrinEmitter):
    """
    To eliminate Python syntax within TIR Macro.
    With Ladder Transform Plugin.
    """

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
        transform_kind_a: int | TransformKind = 0,
        transform_kind_b: int | TransformKind = 0,
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

    def _initialize_k_dim(self, a_dtype=T.float16):
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

        thread_binding = self.get_thread_binding()

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
                        T.address_of(
                            A_shared_buf[
                                warp_m * warp_row_tiles + i * micro_size_x,
                                rk * chunk + ki * micro_size_k,
                            ]
                        ),
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
                        rii, rjj = (tx * local_size_a + local_id) // micro_size_k, (tx * local_size_a + local_id) % (micro_size_k)
                        A_local_buf[j * local_size_a + local_id] = A_shared_buf[ri, rj, rii, rjj]
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

        thread_binding = self.get_thread_binding()

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
                        rii, rjj = (
                            (tx * local_size_dequantize + local_id) // (micro_size_k // num_elems_per_byte),
                            (tx * local_size_dequantize + local_id) % (micro_size_k // num_elems_per_byte),
                        )
                        B_local_buf[j * local_size_dequantize + local_id] = B_shared_buf[ri, rj, rii, rjj]
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
        accum_dtype_abbrv = T.int32
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
