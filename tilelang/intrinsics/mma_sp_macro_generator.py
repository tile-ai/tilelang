from __future__ import annotations

import tilelang.language as T
from typing import Literal, Callable
from tvm import DataType
from tvm.tir import PrimExpr, IndexMap, Buffer, Var
from tvm.runtime import convert
from .utils import (
    mma_store_index_map,
    get_ldmatrix_offset,
)
from tilelang.utils import is_fragment

from tilelang.intrinsics.mma_sp_layout import (
    shared_16x16_to_mma_sp_layout_sr_a,
    shared_16x16_to_mma_sp_layout_sr_b,
    shared_16x32_to_mma_sp_layout_sr_a,
    shared_16x32_to_mma_sp_layout_sr_b,
    shared_16x64_to_mma_sp_layout_sr_a,
    shared_16x64_to_mma_sp_layout_sr_b,
    mma_sp_load_a_32x4_to_shared_16x16_layout,
    mma_sp_load_a_32x8_to_shared_16x32_layout,
    mma_sp_load_a_32x16_to_shared_16x64_layout,
    mma_sp_load_b_32x8_to_shared_16x16_layout,
    mma_sp_load_b_32x16_to_shared_16x32_layout,
    mma_sp_load_b_32x32_to_shared_16x64_layout,
    metadata_8bit_load_32x4_to_shared_16x4_layout_32bit,
    metadata_16bit_load_32x2_to_shared_16x2_layout_32bit,
    metadata_8bit_load_32x4_to_shared_16x4_layout_16bit,
    metadata_16bit_load_32x2_to_shared_16x2_layout_16bit,
    metadata_8bit_load_32x4_to_shared_16x4_layout_8bit,
    metadata_16bit_load_32x2_to_shared_16x4_layout_8bit,
    metadata_32bit_load_32x1_to_shared_16x2_layout_8bit,
    get_ldmatrix_offset_b,
)

lift = convert


class SparseTensorCoreIntrinEmitter:
    """
    To eliminate Python syntax within TIR Macro.
    """

    M_DIM = 16
    SPARSE_FACTOR = 2  # 1:2 for tfloat12, 2:4 for 16-bit and 8-bit datatypes
    SPARSE_SELECTOR = 0  # always use lower threads to provide metadata
    # use lowercase as n_dim can be dynamic
    # the smallest instructions can be m16n8k16, so the n_dim can also be 8
    n_dim = 16
    WARP_SIZE = 32
    dtype_abbrv = {
        "float16": "fp16",
        "bfloat16": "bf16",
        "float32": "fp32",
        "int8": "int8",
        "int32": "int32",
        "float8_e4m3": "e4m3",
        "float8_e5m2": "e5m2",
    }

    E_FACTOR_MAP = {  # e_kdim = mma_kdim // e_factor
        "float": {
            "int16": 8,
            "uint16": 8,
        },
        "float32": {
            "int16": 8,
            "uint16": 8,
        },
        "float16": {
            "int8": 8,
            "uint8": 8,
            "int16": 16,
            "uint16": 16,
            "int32": 32,
            "uint32": 32,
        },
        "bfloat16": {
            "int8": 8,
            "uint8": 8,
            "int16": 16,
            "uint16": 16,
            "int32": 32,
            "uint32": 32,
        },
        "int8": {
            "int8": 8,
            "uint8": 8,
            "int16": 16,
            "uint16": 16,
            "int32": 32,
            "uint32": 32,
        },
        "uint8": {
            "int8": 8,
            "uint8": 8,
            "int16": 16,
            "uint16": 16,
            "int32": 32,
            "uint32": 32,
        },
        "float8_e4m3": {
            "int8": 8,
            "uint8": 8,
            "int16": 16,
            "uint16": 16,
            "int32": 32,
            "uint32": 32,
        },
        "float8_e5m2": {
            "int8": 8,
            "uint8": 8,
            "int16": 16,
            "uint16": 16,
            "int32": 32,
            "uint32": 32,
        },
    }

    E_REPLICATE_FACTOR = {  # metadata replicate every 4 consecutive threads
        "float32": 2,
        "float16": 2,  # 2 of 4 consecutive threads provides
        "bfloat16": 2,
        "int8": 1,  # 4 of 4 consecutive threads provides
        "uint8": 1,
        "float8_e4m3": 1,
        "float8_e5m2": 1,
    }

    # Represent the thread binding in the form of (tx, warp_n, warp_m)
    is_m_first = False

    def __init__(
        self,
        a_dtype: str = T.float16,
        e_dtype: str = T.uint8,
        b_dtype: str = T.float16,
        accum_dtype: str = T.float16,
        a_transposed: bool = False,
        b_transposed: bool = False,
        e_transposed: bool = False,
        block_row_warps: int = 2,
        block_col_warps: int = 2,
        warp_row_tiles: int = 8,
        warp_col_tiles: int = 8,
        warp_k: int = 16,
        reduce_k: int = 1,
        num_elems_per_byte: int = 1,
        is_m_first: bool = False,
        thread_var: Var | None = None,
    ):
        self.a_dtype = a_dtype
        self.e_dtype = e_dtype
        self.b_dtype = b_dtype
        self.accum_dtype = accum_dtype
        self.a_transposed = a_transposed
        self.b_transposed = b_transposed
        self.e_transposed = e_transposed
        # Hint Information
        self.block_row_warps = block_row_warps
        self.block_col_warps = block_col_warps
        self.warp_row_tiles = warp_row_tiles
        self.warp_col_tiles = warp_col_tiles
        self.warp_k = warp_k
        self.e_factor = self.E_FACTOR_MAP[self.a_dtype][self.e_dtype]
        self._initialize_k_dim(a_dtype)
        self._initialize_abbrev(a_dtype, b_dtype, accum_dtype)
        self._initialize_micro_size(self.M_DIM, self.k_dim)
        self._initialize_local_size(self.M_DIM, self.n_dim, self.k_dim, self.WARP_SIZE)
        self._initialize_mma_sp_prefix(self.k_dim)
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
        # NOTE: k_dim here represents the logical shape of the MMA operation.
        # When referring to the physical data movement, it should be divided by sparse_factor.
        self.k_dim = 256 // a_dtype.bits * self.SPARSE_FACTOR

    def _initialize_local_size(self, m_dim=16, n_dim=16, k_dim=16, warp_size=32):
        self.local_size_a = (m_dim * k_dim) // warp_size // self.SPARSE_FACTOR
        self.local_size_e = (m_dim * k_dim) // self.e_factor // warp_size * self.E_REPLICATE_FACTOR[self.a_dtype]
        self.local_size_b = (n_dim * k_dim) // warp_size
        self.local_size_out = (m_dim * n_dim) // warp_size

    def _initialize_abbrev(self, a_dtype, b_dtype, accum_dtype):
        self.a_dtype_abbrv = self.dtype_abbrv[a_dtype]
        self.b_dtype_abbrv = self.dtype_abbrv[b_dtype]
        self.accum_dtype_abbrv = self.dtype_abbrv[accum_dtype]

    def _initialize_mma_sp_prefix(self, k_dim: int = 16):
        if k_dim == 16:
            # typically used for tfloat32
            self.mma_prefix = "m16n8k16"
        elif k_dim == 32:
            # typically used for float16/bfloat16
            self.mma_prefix = "m16n8k32"
        elif k_dim == 64:
            # typically used for int8/fp8
            self.mma_prefix = "m16n8k64"
        else:
            raise ValueError("Unsupported k_dim")

    def _initialize_micro_size(self, m_dim: int = 16, k_dim: int = 16):
        warp_row_tiles = self.warp_row_tiles
        warp_col_tiles = self.warp_col_tiles
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
        # NOTE: k_dim here represents the logical shape of the MMA operation.
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

    def ldmatrix_a(self, A_local_buf: Buffer, A_shared_buf: Buffer, ki: PrimExpr, rk: PrimExpr = 0):
        warp_row_tiles = self.warp_row_tiles
        warp_rows = self.warp_rows
        warp_k = self.warp_k
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
                mma_load_layout = mma_sp_load_a_32x16_to_shared_16x64_layout
            elif DataType(a_dtype).bits == 16:
                mma_load_layout = mma_sp_load_a_32x8_to_shared_16x32_layout
            elif DataType(a_dtype).bits == 32:
                mma_load_layout = mma_sp_load_a_32x4_to_shared_16x16_layout
            else:
                raise ValueError(f"Unsupported dtype: {a_dtype}")

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
            trans = self.a_transposed

            for i in T.serial(warp_rows):
                # Assign A_shared_buf_elem
                wi, wk = warp_m * warp_row_tiles + i * micro_size_x, (rk * warp_k + ki * micro_size_k) // self.SPARSE_FACTOR
                A_shared_buf_elem = A_shared_buf[wk, wi] if a_transposed else A_shared_buf[wi, wk]

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
                        A_local_buf[i * local_size_a + j] = (
                            A_shared_buf[wk + mk, wi + mi] if a_transposed else A_shared_buf[wi + mi, wk + mk]
                        )

        return _warp_ldmatrix_a(A_local_buf, A_shared_buf, ki, thread_binding, rk)

    def ldmatrix_e(self, E_local_buf: Buffer, E_shared_buf: Buffer, ki: PrimExpr, rk: PrimExpr = 0):
        warp_row_tiles = self.warp_row_tiles
        warp_rows = self.warp_rows
        warp_k = self.warp_k
        micro_size_x = self.micro_size_x
        micro_size_k = self.micro_size_k
        local_size_e = self.local_size_e
        a_dtype = self.a_dtype
        e_dtype = self.e_dtype
        trans = self.e_transposed
        # ldmatrix cannot be used for int8 + trans case.
        # include/cutlass/gemm/warp/mma_tensor_op_tile_iterator_sparse.h
        ldmatrix_available = False  # TODO: use ldmatrix when possible

        def mma_load_layout(i, j):
            return i, j

        if not ldmatrix_available:
            if DataType(e_dtype).bits == 8:
                if DataType(a_dtype).bits == 8:
                    mma_load_layout = metadata_8bit_load_32x4_to_shared_16x4_layout_8bit
                elif DataType(a_dtype).bits == 16:
                    mma_load_layout = metadata_8bit_load_32x4_to_shared_16x4_layout_16bit
                elif DataType(a_dtype).bits == 32:
                    mma_load_layout = metadata_8bit_load_32x4_to_shared_16x4_layout_32bit
                else:
                    raise ValueError(f"Unsupported a_dtype for e_dtype 8bit: {a_dtype}")
            elif DataType(e_dtype).bits == 16:
                if DataType(a_dtype).bits == 8:
                    mma_load_layout = metadata_16bit_load_32x2_to_shared_16x4_layout_8bit
                elif DataType(a_dtype).bits == 16:
                    mma_load_layout = metadata_16bit_load_32x2_to_shared_16x2_layout_16bit
                elif DataType(a_dtype).bits == 32:
                    mma_load_layout = metadata_16bit_load_32x2_to_shared_16x2_layout_32bit
                else:
                    raise ValueError(f"Unsupported a_dtype for e_dtype 16bit: {a_dtype}")
            elif DataType(e_dtype).bits == 32:
                if DataType(a_dtype).bits == 8:
                    mma_load_layout = metadata_32bit_load_32x1_to_shared_16x2_layout_8bit
                else:
                    raise ValueError(f"Unsupported a_dtype for e_dtype 32bit: {a_dtype}")
            else:
                raise ValueError(f"Unsupported dtype: {e_dtype}")

        thread_binding = self.get_thread_binding()

        @T.macro
        def _warp_ldmatrix_e(
            E_local_buf,
            E_shared_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            tx, _, warp_m = self.extract_thread_binding(thread_binding)
            for i in T.serial(warp_rows):
                # Assign E_shared_buf_elem
                wi, wk = warp_m * warp_row_tiles + i * micro_size_x, (rk * warp_k + ki * micro_size_k) // self.e_factor
                for j in T.serial(local_size_e):
                    mi, mk = mma_load_layout(tx, j)
                    E_local_buf[i * local_size_e + j] = E_shared_buf[wk + mk, wi + mi] if trans else E_shared_buf[wi + mi, wk + mk]

        return _warp_ldmatrix_e(E_local_buf, E_shared_buf, ki, thread_binding, rk)

    def ldmatrix_b(self, B_local_buf: Buffer, B_shared_buf: Buffer, ki: PrimExpr, rk: PrimExpr = 0):
        warp_col_tiles = self.warp_col_tiles
        warp_cols = self.warp_cols
        warp_k = self.warp_k
        micro_size_y = self.micro_size_y
        micro_size_k = self.micro_size_k
        local_size_b = self.local_size_b
        b_dtype = self.b_dtype
        b_transposed = self.b_transposed
        thread_binding = self.get_thread_binding()
        replicate_b = self.n_dim == 16
        # ldmatrix cannot be used for int8 + trans case.
        ldmatrix_available = not (DataType(b_dtype).bits != 16 and not b_transposed)

        def mma_load_layout(i, j):
            return i, j

        if not ldmatrix_available:
            if DataType(b_dtype).bits == 8:
                mma_load_layout = mma_sp_load_b_32x32_to_shared_16x64_layout
            elif DataType(b_dtype).bits == 16:
                mma_load_layout = mma_sp_load_b_32x16_to_shared_16x32_layout
            elif DataType(b_dtype).bits == 32:
                mma_load_layout = mma_sp_load_b_32x8_to_shared_16x16_layout
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
            stride = B_shared_buf.shape[-1]
            tx, warp_n, _ = self.extract_thread_binding(thread_binding)
            trans = not b_transposed

            for i in T.serial(warp_cols):
                # Assign B_shared_elem
                wi, wk = (
                    warp_n * warp_col_tiles + i * micro_size_y,
                    rk * warp_k + ki * micro_size_k,
                )

                if ldmatrix_available:
                    B_shared_buf_elem = B_shared_buf[wi, wk] if b_transposed else B_shared_buf[wk, wi]

                    if replicate_b:
                        T.ptx_ldmatrix(
                            b_dtype,
                            T.bool(trans),
                            4,
                            ".b16",
                            B_local_buf.data,
                            i * local_size_b,
                            T.address_of(B_shared_buf_elem),
                            get_ldmatrix_offset_b("B", tx, 0, stride, b_dtype, b_transposed),
                        )

                        T.ptx_ldmatrix(
                            b_dtype,
                            T.bool(trans),
                            4,
                            ".b16",
                            B_local_buf.data,
                            i * local_size_b + lift(local_size_b) // 2,
                            T.address_of(B_shared_buf_elem),
                            get_ldmatrix_offset_b("B", tx, lift(local_size_b) // 2, stride, b_dtype, b_transposed),
                        )
                    else:
                        T.ptx_ldmatrix(
                            b_dtype,
                            T.bool(trans),
                            4,
                            ".b16",
                            B_local_buf.data,
                            i * local_size_b,
                            T.address_of(B_shared_buf_elem),
                            get_ldmatrix_offset_b("B", tx, 0, stride, b_dtype, b_transposed),
                        )

                else:
                    # load 16x32 data from shared buffer to local buffer
                    # must be transposed.
                    for j in T.serial(local_size_b):
                        mi, mk = mma_load_layout(tx, j)
                        B_local_buf[i * local_size_b + j] = (
                            B_shared_buf[wi + mi, wk + mk] if b_transposed else B_shared_buf[wk + mk, wi + mi]
                        )

        return _warp_ldmatrix_b(B_local_buf, B_shared_buf, ki, thread_binding, rk)

    def mma_sp(self, A_local_buf: Buffer, E_local_buf: Buffer, B_local_buf: Buffer, C_local_buf: Buffer, k_inner: PrimExpr = 0):
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_a = self.local_size_a
        local_size_e = self.local_size_e
        local_size_b = self.local_size_b
        local_size_out = self.local_size_out
        a_dtype_abbrv = self.a_dtype_abbrv
        b_dtype_abbrv = self.b_dtype_abbrv
        accum_dtype = self.accum_dtype
        accum_dtype_abbrv = self.accum_dtype_abbrv
        mma_prefix = self.mma_prefix
        replicate_b = self.n_dim == 16

        a_is_fragment = is_fragment(A_local_buf)
        e_is_fragment = is_fragment(E_local_buf)
        b_is_fragment = is_fragment(B_local_buf)
        assert not e_is_fragment, f"currently E_local_buf must be a local allocation, found {E_local_buf.scope()}"
        a_local_stride: PrimExpr = k_inner * warp_rows * local_size_a if a_is_fragment else 0
        e_local_stride: PrimExpr = k_inner * warp_rows * local_size_e if e_is_fragment else 0
        b_local_stride: PrimExpr = k_inner * warp_cols * local_size_b if b_is_fragment else 0

        @T.macro
        def _warp_mma_sp(A_local_buf, E_local_buf, B_local_buf, C_local_buf):
            for i, j in T.grid(warp_rows, warp_cols):
                T.ptx_mma_sp(
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
                    E_local_buf.data,  # metadata
                    e_local_stride + i * local_size_e,  # metadata offset
                    self.SPARSE_SELECTOR,  # sparse_selector
                    T.bool(False),  # saturate
                )
                if replicate_b:
                    T.ptx_mma_sp(
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
                        E_local_buf.data,  # metadata
                        e_local_stride + i * local_size_e,  # metadata offset
                        self.SPARSE_SELECTOR,  # sparse_selector
                        T.bool(False),  # saturate
                    )

        return _warp_mma_sp(A_local_buf, E_local_buf, B_local_buf, C_local_buf)

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
            transform_func_sr_a = shared_16x16_to_mma_sp_layout_sr_a
            transform_func_sr_b = shared_16x16_to_mma_sp_layout_sr_b
        elif dtype_bits == 16:
            transform_func_sr_a = shared_16x32_to_mma_sp_layout_sr_a
            transform_func_sr_b = shared_16x32_to_mma_sp_layout_sr_b
        elif dtype_bits == 8:
            transform_func_sr_a = shared_16x64_to_mma_sp_layout_sr_a
            transform_func_sr_b = shared_16x64_to_mma_sp_layout_sr_b
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
            [micro_size_s, micro_size_r // 2 if matrix_is_a else micro_size_r]
            if is_sr_axis_order
            else [micro_size_r // 2 if matrix_is_a else micro_size_r, micro_size_s],
            forward_thread_fn=forward_thread,
            forward_index_fn=forward_index,
        )

        warp_rows, warp_cols = self.warp_rows, self.warp_cols
        chunk = self.warp_k

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
