from __future__ import annotations
from tilelang import tvm as tvm
import tilelang.language as T
from tvm import DataType
from tvm import tir
from tvm.ir import Range
from tvm.tir import PrimExpr, IndexMap, Buffer, Var, BufferRegion, BufferLoad
from tvm.runtime import convert
from .utils import mfma_store_index_map
from typing import Literal, Callable

from tilelang.utils import is_fragment
from tilelang.utils.language import get_buffer_region_from_load
from .mfma_layout import (
    shared_16x4_to_local_64x1_layout_A,
    shared_4x16_to_local_64x1_layout_B,
    shared_16x16_to_local_64x4_layout_A,
    shared_16x16_to_local_64x4_layout_B,
    shared_16x32_to_local_64x8_layout_A,
    shared_16x32_to_local_64x8_layout_B,
    shared_16x64_to_local_64x16_layout_A,
    shared_16x64_to_local_64x16_layout_B,
    thread_id_shared_access_64x1_to_16x4_layout_A,
    thread_id_shared_access_64x1_to_4x16_layout_B,
    thread_id_shared_access_64x4_to_16x16_layout_A,
    thread_id_shared_access_64x4_to_16x16_layout_B,
    thread_id_shared_access_64x8_to_16x32_layout_A,
    thread_id_shared_access_64x8_to_16x32_layout_B,
    thread_id_shared_access_64x16_to_16x64_layout_A,
    thread_id_shared_access_64x16_to_16x64_layout_B,
)

lift = convert


class MatrixCoreIntrinEmitter:
    """
    To eliminate Python syntax within TIR Macro.
    """

    M_DIM = 16
    N_DIM = 16
    WARP_SIZE = 64
    dtype_abbrv = {
        "float16": "fp16",
        "bfloat16": "bf16",
        "float32": "fp32",
        "int8": "int8",
        "int32": "int32",
        "float8_e4m3": "e4m3",
        "float8_e5m2": "e5m2",
        "float8_e4m3fnuz": "e4m3fnuz",
    }

    # k_pack represents the number of elements in a vectorized instruction
    # Detail information can be found in the triton documentation
    # https://github.com/triton-lang/triton/blob/433037206d8870f0b82a3cd669097001084a29ed/third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp#L419
    k_pack = 1
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
        k_pack: int | None = None,
        is_m_first: bool | None = False,
        b_preshuffle: bool | None = False,
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
        self._initialize_local_size(self.M_DIM, self.N_DIM, self.k_dim, self.WARP_SIZE)
        self._initialize_mfma_prefix(self.k_dim)
        self._initialize_micro_size(self.M_DIM, self.N_DIM, self.k_dim)
        self._initialize_k_pack(k_pack)
        self._initialize_is_m_first(is_m_first)
        self._initialize_b_preshuffle(b_preshuffle)

        self.warp_rows = warp_row_tiles // self.micro_size_x
        self.warp_cols = warp_col_tiles // self.micro_size_y
        self.reduce_k = reduce_k
        self.threads = self.WARP_SIZE * (block_row_warps * block_col_warps) * reduce_k
        self.num_elems_per_byte = num_elems_per_byte
        self.thread_var = thread_var

    def _initialize_k_dim(self, a_dtype=T.float16):
        if isinstance(a_dtype, str):
            if a_dtype in ["float8_e4m3fnuz", T.int8]:
                self.k_dim = 32
                return
            a_dtype = DataType(a_dtype)

        if a_dtype.bits == 32:
            self.k_dim = 4
        elif a_dtype.bits in {16, 8}:
            self.k_dim = 16
        else:
            raise ValueError(f"Unsupported a_dtype = {a_dtype}")

    def _initialize_local_size(self, m_dim=16, n_dim=16, k_dim=16, warp_size=32):
        self.local_size_a = (m_dim * k_dim) // warp_size
        self.local_size_b = (n_dim * k_dim) // warp_size
        self.local_size_out = (m_dim * n_dim) // warp_size

    def _initialize_abbrev(self, a_dtype, b_dtype, accum_dtype):
        self.a_dtype_abbrv = self.dtype_abbrv[a_dtype]
        self.b_dtype_abbrv = self.dtype_abbrv[b_dtype]
        self.accum_dtype_abbrv = self.dtype_abbrv[accum_dtype]

    def _initialize_mfma_prefix(self, k_dim=16):
        in_dtype, out_dtype = self.a_dtype, self.accum_dtype
        M_DIM, N_DIM = self.M_DIM, self.N_DIM
        out_dtype_abbrv = {T.float16: "f16", T.float32: "f32", T.int8: "i8", T.int32: "i32"}[out_dtype]

        in_dtype_abbrv = {
            "bfloat16": "bf16",
            "float16": "f16",
            "float32": "f32",
            "int8": "i8",
            "int32": "i32",
            "float8_e4m3fnuz": "fp8",
        }[in_dtype]

        if in_dtype_abbrv == "fp8":
            self.mfma_suffix = f"{out_dtype_abbrv}_{M_DIM}x{N_DIM}x{k_dim}_fp8_fp8"
        elif in_dtype_abbrv == "i8":
            self.mfma_suffix = f"{out_dtype_abbrv}_{M_DIM}x{N_DIM}x{k_dim}_i8"
        elif in_dtype_abbrv == "bf16":
            # HIP intrinsic uses ...x{K}bf16_1k without an underscore before bf16
            self.mfma_suffix = f"{out_dtype_abbrv}_{M_DIM}x{N_DIM}x{k_dim}bf16_1k"
        else:
            self.mfma_suffix = f"{out_dtype_abbrv}_{M_DIM}x{N_DIM}x{k_dim}{in_dtype_abbrv}"

    def _initialize_micro_size(self, m_dim=16, n_dim=16, k_dim=16):
        self.micro_size_x = m_dim
        self.micro_size_y = n_dim
        self.micro_size_k = k_dim

    def _initialize_k_pack(self, k_pack: int | None = None):
        if k_pack is not None:
            self.k_pack = k_pack

    def _initialize_is_m_first(self, is_m_first: bool | None = False):
        if is_m_first is not None:
            self.is_m_first = is_m_first

    def _initialize_b_preshuffle(self, b_preshuffle: bool | None = False):
        if b_preshuffle is not None:
            self.b_preshuffle = b_preshuffle

    def get_ldmatrix_index_map(self, is_b=False):
        k_dim = self.k_dim * self.k_pack
        transposed = self.a_transposed if not is_b else self.b_transposed
        if k_dim == 4:
            index_map = shared_16x4_to_local_64x1_layout_A
            reverse_index_map = thread_id_shared_access_64x1_to_16x4_layout_A
            if is_b:
                index_map = shared_16x4_to_local_64x1_layout_A if transposed else shared_4x16_to_local_64x1_layout_B
                reverse_index_map = (
                    thread_id_shared_access_64x1_to_16x4_layout_A if transposed else thread_id_shared_access_64x1_to_4x16_layout_B
                )
        elif k_dim == 16:
            index_map = shared_16x16_to_local_64x4_layout_B if transposed else shared_16x16_to_local_64x4_layout_A
            reverse_index_map = (
                thread_id_shared_access_64x4_to_16x16_layout_B if transposed else thread_id_shared_access_64x4_to_16x16_layout_A
            )

            if is_b:
                index_map = shared_16x16_to_local_64x4_layout_A if transposed else shared_16x16_to_local_64x4_layout_B
                reverse_index_map = (
                    thread_id_shared_access_64x4_to_16x16_layout_A if transposed else thread_id_shared_access_64x4_to_16x16_layout_B
                )
        elif k_dim == 32:
            index_map = shared_16x32_to_local_64x8_layout_B if transposed else shared_16x32_to_local_64x8_layout_A
            reverse_index_map = (
                thread_id_shared_access_64x8_to_16x32_layout_B if transposed else thread_id_shared_access_64x8_to_16x32_layout_A
            )

            if is_b:
                index_map = shared_16x32_to_local_64x8_layout_A if transposed else shared_16x32_to_local_64x8_layout_B
                reverse_index_map = (
                    thread_id_shared_access_64x8_to_16x32_layout_A if transposed else thread_id_shared_access_64x8_to_16x32_layout_B
                )
        elif k_dim == 64:
            index_map = shared_16x64_to_local_64x16_layout_B if transposed else shared_16x64_to_local_64x16_layout_A
            reverse_index_map = (
                thread_id_shared_access_64x16_to_16x64_layout_B if transposed else thread_id_shared_access_64x16_to_16x64_layout_A
            )

            if is_b:
                index_map = shared_16x64_to_local_64x16_layout_A if transposed else shared_16x64_to_local_64x16_layout_B
                reverse_index_map = (
                    thread_id_shared_access_64x16_to_16x64_layout_A if transposed else thread_id_shared_access_64x16_to_16x64_layout_B
                )
        else:
            raise ValueError("k_dim must be 4 or 16 or 32 or 64 currently")

        return index_map, reverse_index_map

    def get_store_index_map(self, inverse: bool = False) -> IndexMap:
        warp_size, local_size_c = self.WARP_SIZE, self.local_size_out
        index_map = IndexMap.from_func(mfma_store_index_map, index_dtype=T.int32)
        if not inverse:
            return index_map
        inverse_index_map = index_map.inverse([warp_size, local_size_c])
        return inverse_index_map

    def get_thread_binding(self):
        if self.thread_var is None:
            current_frame = T.KernelLaunchFrame.Current()
            assert current_frame is not None, "Must be called in a T.Kernel Frame"
            return current_frame.get_thread_binding()
        else:
            return self.thread_var

    def extract_thread_binding(self, thread_id, is_m_first=None) -> tuple[PrimExpr, PrimExpr, PrimExpr]:
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

    def ldmatrix_a(self, A_local_buf, A_shared_buf: Buffer | BufferRegion, ki, rk=0):
        warp_row_tiles = self.warp_row_tiles
        warp_rows = self.warp_rows
        chunk = self.chunk
        micro_size_x = self.micro_size_x
        micro_size_k = self.micro_size_k
        local_size_a = self.local_size_a
        k_pack = self.k_pack
        is_transposed = self.a_transposed
        thread_binding = self.get_thread_binding()
        _, reverse_index_map = self.get_ldmatrix_index_map(is_b=False)

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
            if is_transposed:
                for i in T.serial(warp_rows):
                    for local_id in T.vectorized(k_pack * local_size_a):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (rk * chunk + ki * (k_pack * micro_size_k), warp_m * warp_row_tiles + i * micro_size_x)
                        A_local_buf[i * k_pack * local_size_a + local_id] = A_buf[A_base0 + l + row, A_base1 + r + col]
            else:
                for i in T.serial(warp_rows):
                    for local_id in T.vectorized(k_pack * local_size_a):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (warp_m * warp_row_tiles + i * micro_size_x, rk * chunk + ki * (k_pack * micro_size_k))
                        A_local_buf[i * k_pack * local_size_a + local_id] = A_buf[A_base0 + l + row, A_base1 + r + col]

        return _warp_ldmatrix_a(A_local_buf, A_shared_buf, ki, thread_binding, rk)

    def ldmatrix_b(self, B_local_buf, B_shared_buf: Buffer | BufferRegion, ki, rk=0):
        warp_col_tiles = self.warp_col_tiles
        warp_cols = self.warp_cols
        chunk = self.chunk
        micro_size_y = self.micro_size_y
        micro_size_k = self.micro_size_k
        local_size_b = self.local_size_b
        k_pack = self.k_pack
        is_transposed = self.b_transposed
        thread_binding = self.get_thread_binding()
        _, reverse_index_map = self.get_ldmatrix_index_map(is_b=True)

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
            if is_transposed:
                for j in T.serial(warp_cols):
                    for local_id in T.vectorized(k_pack * local_size_b):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (
                            warp_n * warp_col_tiles + j * micro_size_y,
                            rk * chunk + ki * (k_pack * micro_size_k),
                        )
                        B_local_buf[j * k_pack * local_size_b + local_id] = B_buf[B_base0 + l + row, B_base1 + r + col]

            else:
                for j in T.serial(warp_cols):
                    for local_id in T.vectorized(k_pack * local_size_b):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (
                            rk * chunk + ki * (k_pack * micro_size_k),
                            warp_n * warp_col_tiles + j * micro_size_y,
                        )
                        B_local_buf[j * k_pack * local_size_b + local_id] = B_buf[B_base0 + l + row, B_base1 + r + col]

        return _warp_ldmatrix_b(B_local_buf, B_shared_buf, ki, thread_binding, rk)

    def mfma(self, A_local_buf: Buffer, B_local_buf: Buffer, C_local_buf: Buffer, k_inner: PrimExpr | None = 0):
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_a = self.local_size_a
        local_size_b = self.local_size_b
        local_size_out = self.local_size_out
        k_pack = self.k_pack
        mfma_suffix = self.mfma_suffix
        a_dtype, b_dtype, out_dtype = self.a_dtype, self.b_dtype, self.accum_dtype
        compute_a_dtype = a_dtype if local_size_a == 1 else f"{a_dtype}x{local_size_a}"
        compute_b_dtype = b_dtype if local_size_b == 1 else f"{b_dtype}x{local_size_b}"
        compute_out_dtype = out_dtype if local_size_out == 1 else f"{out_dtype}x{local_size_out}"

        a_is_fragment = is_fragment(A_local_buf)
        b_is_fragment = is_fragment(B_local_buf)
        a_local_stride: PrimExpr = k_inner * warp_rows * k_pack * local_size_a if a_is_fragment else 0
        b_local_stride: PrimExpr = k_inner * warp_cols * k_pack * local_size_b if b_is_fragment else 0

        @T.macro
        def _warp_mfma(A_local_buf, B_local_buf, C_local_buf):
            for kp, i, j in T.grid(k_pack, warp_rows, warp_cols):
                T.tvm_mfma(
                    mfma_suffix,
                    "row",
                    "row",
                    compute_a_dtype,
                    compute_b_dtype,
                    compute_out_dtype,
                    B_local_buf.data,
                    (b_local_stride + (j * k_pack + kp) * local_size_b) // local_size_b,
                    A_local_buf.data,
                    (a_local_stride + (i * k_pack + kp) * local_size_a) // local_size_a,
                    C_local_buf.data,
                    (i * warp_cols * local_size_out + j * local_size_out) // local_size_out,
                    dtype=compute_out_dtype,
                )

        return _warp_mfma(A_local_buf, B_local_buf, C_local_buf)

    def stmatrix(self, C_local_buf, C_buf, pid_m=None, pid_n=None):
        block_row_warps = self.block_row_warps
        block_col_warps = self.block_col_warps
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_out = self.local_size_out
        thread_binding = self.get_thread_binding()
        is_global = pid_m is not None and pid_n is not None
        BLOCK_M = block_row_warps * warp_rows
        BLOCK_N = block_col_warps * warp_cols
        M_DIM, N_DIM = self.M_DIM, self.N_DIM
        C_buf_dims = len(C_buf.shape)
        assert C_buf_dims in {2, 4}, "C_buf should be 2D or 4D"

        # STS
        # MFMA Store must be in simulated instead of TVM Intrins
        # As TVM Intrins is like a hack that the threadIdx.x should be always
        # equal to the warp_size
        @T.macro
        def _warp_stmatrix_shared(C_local_buf, C_buf, thread_binding):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            for i, j in T.grid(warp_rows, warp_cols):
                for local_id in T.vectorized(local_size_out):
                    row, col = T.meta_var(mfma_store_index_map(tx, local_id))
                    if C_buf_dims == 2:
                        C_buf[(warp_m * warp_rows + i) * M_DIM + row, (warp_n * warp_cols + j) * N_DIM + col] = C_local_buf[
                            i * (warp_cols * local_size_out) + j * local_size_out + local_id
                        ]
                    else:
                        C_buf[warp_m * warp_rows + i, warp_n * warp_cols + j, row, col] = C_local_buf[
                            i * warp_cols * local_size_out + j * local_size_out + local_id
                        ]

        @T.macro
        def _warp_stmatrix_global(C_local_buf, C_buf, thread_binding):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            for i, j in T.grid(warp_rows, warp_cols):
                for local_id in T.vectorized(local_size_out):
                    row, col = T.meta_var(mfma_store_index_map(tx, local_id))
                    C_buf[
                        (pid_m * BLOCK_M + warp_m * warp_rows + i) * M_DIM + row, (pid_n * BLOCK_N + warp_n * warp_cols + j) * N_DIM + col
                    ] = C_local_buf[i * warp_cols * local_size_out + j * local_size_out + local_id]

        return (
            _warp_stmatrix_global(C_local_buf, C_buf, thread_binding)
            if is_global
            else _warp_stmatrix_shared(C_local_buf, C_buf, thread_binding)
        )

    def make_mfma_load_layout(self, local_buf: Buffer, matrix: Literal["A", "B"] = "A") -> T.Fragment:
        """
        Create a layout function for storing MFMA results into a fragment buffer.

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
        transposed = self.a_transposed if matrix_is_a else self.b_transposed

        # s represents spatial axis
        # r represents reduction axis
        # sr represents the two dims are spatial + reduction
        # rs represents the two dims are reduction + spatial
        # sr also can represent a non-transposed basic layout
        # then rs also can represent a transposed basic layout
        transform_func_sr_a: Callable = None
        transform_func_sr_b: Callable = None

        k_dim = self.k_dim * self.k_pack

        if k_dim == 4:
            transform_func_sr_a = shared_16x4_to_local_64x1_layout_A
            transform_func_sr_b = shared_16x4_to_local_64x1_layout_A
        elif k_dim == 16:
            transform_func_sr_a = shared_16x16_to_local_64x4_layout_A
            transform_func_sr_b = shared_16x16_to_local_64x4_layout_A
        elif k_dim == 32:
            transform_func_sr_a = shared_16x32_to_local_64x8_layout_A
            transform_func_sr_b = shared_16x32_to_local_64x8_layout_A
        elif k_dim == 64:
            transform_func_sr_a = shared_16x64_to_local_64x16_layout_A
            transform_func_sr_b = shared_16x64_to_local_64x16_layout_A
        else:
            raise ValueError("k_dim must be 4 or 16 or 32 or 64 currently")

        is_sr_conditions = [False]
        is_sr_conditions.append(matrix_is_a and not transposed)
        is_sr_conditions.append(matrix_is_b and transposed)
        is_sr_axis_order = any(is_sr_conditions)

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

        inverse_mfma_load_layout = IndexMap.from_func(transform_func, index_dtype=T.int32)

        def forward_thread(i: int, j: int) -> int:
            """
            Given the row index `i` and column index `j` in the fragment,
            """
            lane_id, _ = inverse_mfma_load_layout.map_indices([i, j])
            return lane_id

        def forward_index(i: int, j: int) -> int:
            """
            Given the row index `i` and column index `j` in the fragment,
            """
            _, local_id = inverse_mfma_load_layout.map_indices([i, j])
            return local_id

        base_fragment = T.Fragment(
            [micro_size_s, micro_size_r * self.k_pack] if is_sr_axis_order else [micro_size_r * self.k_pack, micro_size_s],
            forward_thread_fn=forward_thread,
            forward_index_fn=forward_index,
        )

        warp_rows, warp_cols = self.warp_rows, self.warp_cols
        chunk = self.chunk

        warp_s = warp_rows if matrix_is_a else warp_cols
        warp_r = chunk // (micro_size_r * self.k_pack)
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

    def make_mfma_store_layout(self, local_buf: Buffer) -> T.Fragment:
        """
        Create a layout function for storing MFMA results into a fragment buffer.

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
        inverse_mfma_store_layout = self.get_store_index_map(inverse=True)
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
            map them to a thread index according to `inverse_mfma_store_layout`.
            """
            # the upper bounds of i and j are block_row_warps * warp_rows * micro_size_x and block_col_warps * warp_cols * micro_size_y
            # the upper bounds of block_row_warps and block_col_warps are warp_rows and warp_cols
            block_i, block_j = (i // micro_size_x) // warp_rows, (j // micro_size_y) // warp_cols
            # upper bounds of mfma_i and mfma_j are micro_size_x and micro_size_y
            mfma_i, mfma_j = i % micro_size_x, j % micro_size_y
            lane_id, _ = inverse_mfma_store_layout.map_indices([mfma_i, mfma_j])
            if is_m_first:
                thread_id = block_i * (block_col_warps * warp_cols) + block_j * warp_size + lane_id
            else:
                thread_id = block_j * (block_row_warps * warp_size) + block_i * warp_size + lane_id
            return thread_id

        def forward_index(i: int, j: int) -> int:
            """
            Given the row index `i` and column index `j` in the fragment,
            map them to a local index in a single thread according
            to `inverse_mfma_store_layout`.
            """
            # the upper bounds of i and j are block_row_warps * warp_rows * micro_size_x and block_col_warps * warp_cols * micro_size_y
            # the upper bounds of warp_i and warp_j are warp_rows and warp_cols
            warp_i, warp_j = (i // micro_size_x) % warp_rows, (j // micro_size_y) % warp_cols
            # upper bounds of mfma_i and mfma_j are micro_size_x and micro_size_y
            mfma_i, mfma_j = i % micro_size_x, j % micro_size_y
            _, local_id = inverse_mfma_store_layout.map_indices([mfma_i, mfma_j])
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


class MatrixCorePreshuffleIntrinEmitter(MatrixCoreIntrinEmitter):
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
        k_pack: int | None = None,
        is_m_first: bool | None = False,
        a_preshuffle: bool | None = False,
        b_preshuffle: bool | None = False,
        thread_var: Var | None = None,
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
            k_pack=k_pack,
            is_m_first=is_m_first,
            thread_var=thread_var,
        )
        self._initialize_preshuffle(a_preshuffle, b_preshuffle)

    def _initialize_preshuffle(self, a_preshuffle: bool, b_preshuffle: bool):
        if a_preshuffle is not None:
            self.a_preshuffle = a_preshuffle
        if b_preshuffle is not None:
            self.b_preshuffle = b_preshuffle

    def ldmatrix_a(self, A_local_buf, A_buf, ki, rk=0, pid_m=None, pid_n=None):
        warp_rows = self.warp_rows
        chunk = self.chunk
        micro_size_k = self.micro_size_k
        local_size_a = self.local_size_a
        k_pack = self.k_pack
        is_transposed = self.a_transposed
        current_frame = T.KernelLaunchFrame.Current()
        thread_binding = current_frame.get_thread_binding()
        _, reverse_index_map = self.get_ldmatrix_index_map(is_b=False)
        is_global = pid_m is not None and pid_n is not None

        # no preshuffle, use the default implementation
        if self.a_preshuffle is False:
            return super().ldmatrix_a(A_local_buf, A_buf, ki, rk)

        def _warp_ldmatrix_a_global(
            A_local_buf,
            A_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            tx, _, warp_m = self.extract_thread_binding(thread_binding)
            if is_transposed:
                for i in T.serial(warp_rows):
                    for local_id in T.vectorized(k_pack * local_size_a):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (
                            rk * (chunk // micro_size_k) + ki,
                            (pid_m * self.block_row_warps + warp_m) * warp_rows + i,
                        )
                        A_local_buf[i * k_pack * local_size_a + local_id] = A_buf[l, r, row, col]
            else:
                for i in T.serial(warp_rows):
                    for local_id in T.vectorized(k_pack * local_size_a):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (
                            (pid_m * self.block_row_warps + warp_m) * warp_rows + i,
                            rk * (chunk // micro_size_k) + ki,
                        )
                        A_local_buf[i * k_pack * local_size_a + local_id] = A_buf[l, r, row, col]

        @T.macro
        def _warp_ldmatrix_a_shared(
            A_local_buf,
            A_shared_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            tx, _, warp_m = self.extract_thread_binding(thread_binding)
            if is_transposed:
                for i in T.serial(warp_rows):
                    for local_id in T.vectorized(k_pack * local_size_a):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (
                            rk * (chunk // micro_size_k) + ki,
                            warp_m * warp_rows + i,
                        )
                        A_local_buf[i * k_pack * local_size_a + local_id] = A_shared_buf[l, r, row, col]
            else:
                print(self.a_preshuffle)
                for i in T.serial(warp_rows):
                    for local_id in T.vectorized(k_pack * local_size_a):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (warp_m * warp_rows + i, rk * (chunk // micro_size_k) + ki)
                        A_local_buf[i * k_pack * local_size_a + local_id] = A_shared_buf[l, r, row, col]

        return (
            _warp_ldmatrix_a_global(A_local_buf, A_buf, ki, thread_binding, rk)
            if is_global
            else _warp_ldmatrix_a_shared(A_local_buf, A_buf, ki, thread_binding, rk)
        )

    def ldmatrix_b(self, B_local_buf, B_buf, ki, rk=0, pid_m=None, pid_n=None):
        warp_cols = self.warp_cols
        chunk = self.chunk
        micro_size_k = self.micro_size_k
        local_size_b = self.local_size_b
        k_pack = self.k_pack
        is_transposed = self.b_transposed
        current_frame = T.KernelLaunchFrame.Current()
        thread_binding = current_frame.get_thread_binding()
        _, reverse_index_map = self.get_ldmatrix_index_map(is_b=True)
        is_global = pid_m is not None and pid_n is not None

        if self.b_preshuffle is False:
            return super().ldmatrix_b(B_local_buf, B_buf, ki, rk, pid_m, pid_n)

        @T.macro
        def _warp_ldmatrix_b_global(
            B_local_buf,
            B_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            tx, warp_n, _ = self.extract_thread_binding(thread_binding)
            if is_transposed:
                for j in T.serial(warp_cols):
                    for local_id in T.vectorized(k_pack * local_size_b):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (
                            (pid_n * self.block_col_warps + warp_n) * warp_cols + j,
                            rk * (chunk // micro_size_k) + ki,
                        )
                        B_local_buf[j * k_pack * local_size_b + local_id] = B_buf[l, r, row, col]
            else:
                for j in T.serial(warp_cols):
                    for local_id in T.vectorized(k_pack * local_size_b):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (
                            rk * (chunk // micro_size_k) + ki,
                            (pid_n * self.block_col_warps + warp_n) * warp_cols + j,
                        )
                        B_local_buf[j * k_pack * local_size_b + local_id] = B_buf[l, r, row, col]

        @T.macro
        def _warp_ldmatrix_b_shared(
            B_local_buf,
            B_shared_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            tx, warp_n, _ = self.extract_thread_binding(thread_binding)
            if is_transposed:
                for j in T.serial(warp_cols):
                    for local_id in T.vectorized(k_pack * local_size_b):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (
                            warp_n * warp_cols + j,
                            rk * (chunk // micro_size_k) + ki,
                        )
                        B_local_buf[j * k_pack * local_size_b + local_id] = B_shared_buf[l, r, row, col]
            else:
                for j in T.serial(warp_cols):
                    for local_id in T.vectorized(k_pack * local_size_b):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (
                            rk * (chunk // micro_size_k) + ki,
                            warp_n * warp_cols + j,
                        )
                        B_local_buf[j * k_pack * local_size_b + local_id] = B_shared_buf[l, r, row, col]

        return (
            _warp_ldmatrix_b_global(B_local_buf, B_buf, ki, thread_binding, rk)
            if is_global
            else _warp_ldmatrix_b_shared(B_local_buf, B_buf, ki, thread_binding, rk)
        )
