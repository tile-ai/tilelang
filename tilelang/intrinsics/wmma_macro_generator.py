"""WMMA intrinsic emitter for AMD RDNA architectures (gfx11 / gfx12).

Only supports the f16->f32, 16x16x16 variant with warp-size=32.

Thread-data mapping (per AMDGPU ISA):
  A[16][K=16]: thread t holds A[t//2][(t%2)*8 : (t%2)*8+8]  (8 fp16 = 4 f32 per thread)
  B[K=16][16]: same mapping as A for the transposed dimension
  C/D[16][16]: thread t holds D[t//2][(t%2)*8 : (t%2)*8+8]  (8 f32 per thread)
"""

from __future__ import annotations

from typing import Literal

import tilelang.language as T
from tilelang import tvm as tvm
from tvm import tir
from tvm.ir import Range
from tvm.target import Target
from tvm.tir import Buffer, BufferLoad, BufferRegion, IndexMap, PrimExpr, Var
from tvm.runtime import convert

from tilelang.language.utils import get_buffer_region_from_load
from tilelang.utils import is_fragment
from .wmma_layout import (
    shared_16x16_to_local_32x8_layout_A,
    shared_16x16_to_local_32x8_layout_B,
    shared_16x16_to_local_32x8_layout_B_colmajor,
    thread_id_shared_access_32x8_to_16x16_layout_A,
    thread_id_shared_access_32x8_to_16x16_layout_B,
    thread_id_shared_access_32x8_to_16x16_layout_B_colmajor,
    wmma_store_index_map,
)

lift = convert


class WMMAIntrinEmitter:
    """Intrinsic emitter for AMD RDNA WMMA (16×16×16, warp-size=32).

    Supports:
      - fp16 -> fp32  (f32_16x16x16_f16_w32 / _gfx12)
    """

    M_DIM = 16
    N_DIM = 16
    K_DIM = 16
    WARP_SIZE = 32

    def __init__(
        self,
        a_dtype: str = "float16",
        b_dtype: str = "float16",
        accum_dtype: str = "float32",
        a_transposed: bool = False,
        b_transposed: bool = False,
        block_row_warps: int = 2,
        block_col_warps: int = 2,
        warp_row_tiles: int = 16,
        warp_col_tiles: int = 16,
        chunk: int = 16,
        k_pack: int = 1,
        thread_var: Var | None = None,
        target: Target | None = None,
    ):
        assert a_dtype in ("float16", "bfloat16"), f"Unsupported a_dtype: {a_dtype}"
        assert accum_dtype == "float32", f"Unsupported accum_dtype: {accum_dtype}"

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
        self.k_pack = k_pack
        self.thread_var = thread_var
        self.target = target

        self.micro_size_x = self.M_DIM
        self.micro_size_y = self.N_DIM
        self.micro_size_k = self.K_DIM

        # Each thread holds 8 fp16 (A/B) or 8 fp32 (C/D)
        self.local_size_a = (self.M_DIM * self.K_DIM) // self.WARP_SIZE  # = 8
        self.local_size_b = (self.N_DIM * self.K_DIM) // self.WARP_SIZE  # = 8
        self.local_size_out = (self.M_DIM * self.N_DIM) // self.WARP_SIZE  # = 8

        self.warp_rows = warp_row_tiles // self.M_DIM
        self.warp_cols = warp_col_tiles // self.N_DIM
        self.threads = self.WARP_SIZE * block_row_warps * block_col_warps

        # Build the wmma shape string used by T.tvm_rdna_wmma
        # shape = "f32_16x16x16_f16_w32" (or _gfx12 suffix is handled in codegen)
        dtype_in_abbrv = {"float16": "f16", "bfloat16": "bf16"}[a_dtype]
        dtype_out_abbrv = "f32"
        self.wmma_shape = f"{dtype_out_abbrv}_{self.M_DIM}x{self.N_DIM}x{self.K_DIM}_{dtype_in_abbrv}_w{self.WARP_SIZE}"

    # ─────────────────────────────────────────────────────────────────────────
    # Thread binding helpers
    # ─────────────────────────────────────────────────────────────────────────

    def get_thread_binding(self) -> PrimExpr:
        if self.thread_var is not None:
            return self.thread_var
        current_frame = T.KernelLaunchFrame.Current()
        assert current_frame is not None, "Must be called inside T.Kernel"
        return current_frame.get_thread_binding()

    def extract_thread_binding(self, thread_id):
        """Return (lane_id, warp_n, warp_m)."""
        WARP_SIZE = self.WARP_SIZE
        block_row_warps = self.block_row_warps
        lane_id = thread_id % WARP_SIZE
        warp_m = (thread_id // WARP_SIZE) % block_row_warps
        warp_n = (thread_id // (WARP_SIZE * block_row_warps)) % self.block_col_warps
        return lane_id, warp_n, warp_m

    # ─────────────────────────────────────────────────────────────────────────
    # Layout queries
    # ─────────────────────────────────────────────────────────────────────────

    def get_ldmatrix_index_map(self, is_b: bool = False):
        """Return (forward, reverse) index maps for shared→local loading.

        For WMMA gfx12:
          - A is stored row-major [M, K]. Thread t loads A[t%16][(t//16)*8+local].
          - B (non-transposed) is stored row-major [K, N].
            Thread t loads B[t%16][(t//16)*8+local] (same shape/pattern as A).
          - B (transposed) is stored [N, K].
            Thread t loads B_T[t%16][(t//16)*8+local] (N-row, K-col).
        """
        transposed = self.b_transposed if is_b else self.a_transposed
        if not is_b:
            # A matrix [M, K]
            if transposed:
                # A stored as [K, M]: row=K, col=M → same mapping but rotated
                # In this case row index runs over K, col over M
                # thread t: K-row = t%16, M-col = (t//16)*8+local
                index_map = shared_16x16_to_local_32x8_layout_A
                reverse_index_map = thread_id_shared_access_32x8_to_16x16_layout_A
            else:
                # A stored as [M, K]: row=M, col=K
                index_map = shared_16x16_to_local_32x8_layout_A
                reverse_index_map = thread_id_shared_access_32x8_to_16x16_layout_A
        else:
            # B matrix
            if transposed:
                # B stored as [N, K]: thread t: N-row = t%16, K-col = (t//16)*8+local
                index_map = shared_16x16_to_local_32x8_layout_B_colmajor
                reverse_index_map = thread_id_shared_access_32x8_to_16x16_layout_B_colmajor
            else:
                # B stored as [K, N]: thread t: K-row = t%16, N-col = (t//16)*8+local
                index_map = shared_16x16_to_local_32x8_layout_B
                reverse_index_map = thread_id_shared_access_32x8_to_16x16_layout_B
        return index_map, reverse_index_map

    def get_store_index_map(self, inverse: bool = False) -> IndexMap:
        """Return the store index map.

        The forward map is (thread_id, local_id) -> (i, j), which is affine.
        The inverse map is (i, j) -> (thread_id, local_id).
        """
        warp_size, local_size_c = self.WARP_SIZE, self.local_size_out
        # forward: (thread_id, local_id) -> (row, col)
        index_map = IndexMap.from_func(wmma_store_index_map, index_dtype=T.int32)
        if not inverse:
            return index_map
        # inverse: (row, col) -> (thread_id, local_id)
        return index_map.inverse([warp_size, local_size_c])

    # ─────────────────────────────────────────────────────────────────────────
    # Load A from shared memory
    # ─────────────────────────────────────────────────────────────────────────

    def ldmatrix_a(self, A_local_buf, A_shared_buf, ki, rk=0):
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

        A_region = self._legalize_to_buffer_region(A_shared_buf)
        A_buf = A_region.buffer
        A_base0 = A_region.region[-2].min
        A_base1 = A_region.region[-1].min

        @T.macro
        def _warp_ldmatrix_a(A_local_buf, A_shared_buf, ki, thread_binding, rk=0):
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

    # ─────────────────────────────────────────────────────────────────────────
    # Load B from shared memory
    # ─────────────────────────────────────────────────────────────────────────

    def ldmatrix_b(self, B_local_buf, B_shared_buf, ki, rk=0):
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

        B_region = self._legalize_to_buffer_region(B_shared_buf)
        B_buf = B_region.buffer
        B_base0 = B_region.region[-2].min
        B_base1 = B_region.region[-1].min

        @T.macro
        def _warp_ldmatrix_b(B_local_buf, B_shared_buf, ki, thread_binding, rk=0):
            tx, warp_n, _ = self.extract_thread_binding(thread_binding)
            if is_transposed:
                for j in T.serial(warp_cols):
                    for local_id in T.vectorized(k_pack * local_size_b):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (warp_n * warp_col_tiles + j * micro_size_y, rk * chunk + ki * (k_pack * micro_size_k))
                        B_local_buf[j * k_pack * local_size_b + local_id] = B_buf[B_base0 + l + row, B_base1 + r + col]
            else:
                for j in T.serial(warp_cols):
                    for local_id in T.vectorized(k_pack * local_size_b):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (rk * chunk + ki * (k_pack * micro_size_k), warp_n * warp_col_tiles + j * micro_size_y)
                        B_local_buf[j * k_pack * local_size_b + local_id] = B_buf[B_base0 + l + row, B_base1 + r + col]

        return _warp_ldmatrix_b(B_local_buf, B_shared_buf, ki, thread_binding, rk)

    # ─────────────────────────────────────────────────────────────────────────
    # Issue WMMA
    # ─────────────────────────────────────────────────────────────────────────

    def wmma(self, A_local_buf: Buffer, B_local_buf: Buffer, C_local_buf: Buffer, k_inner: PrimExpr | None = 0):
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_a = self.local_size_a
        local_size_b = self.local_size_b
        local_size_out = self.local_size_out
        k_pack = self.k_pack
        wmma_shape = self.wmma_shape
        a_dtype = self.a_dtype
        b_dtype = self.b_dtype
        out_dtype = self.accum_dtype

        # vectorized dtype strings for the intrinsic
        compute_a_dtype = f"{a_dtype}x{local_size_a}"
        compute_b_dtype = f"{b_dtype}x{local_size_b}"
        compute_out_dtype = f"{out_dtype}x{local_size_out}"

        a_is_fragment = is_fragment(A_local_buf)
        b_is_fragment = is_fragment(B_local_buf)
        a_local_stride: PrimExpr = k_inner * warp_rows * k_pack * local_size_a if a_is_fragment else 0
        b_local_stride: PrimExpr = k_inner * warp_cols * k_pack * local_size_b if b_is_fragment else 0

        @T.macro
        def _warp_wmma(A_local_buf, B_local_buf, C_local_buf):
            for kp, i, j in T.grid(k_pack, warp_rows, warp_cols):
                # With hardware D layout: no A/B swap needed for either case.
                # Both transposed and non-transposed B give correct results
                # with A first, B second.
                T.tvm_rdna_wmma(
                    compute_out_dtype,
                    wmma_shape,
                    "row",
                    "row",
                    compute_a_dtype,
                    compute_b_dtype,
                    compute_out_dtype,
                    A_local_buf.data,
                    (a_local_stride + (i * k_pack + kp) * local_size_a) // local_size_a,
                    B_local_buf.data,
                    (b_local_stride + (j * k_pack + kp) * local_size_b) // local_size_b,
                    C_local_buf.data,
                    (i * warp_cols * local_size_out + j * local_size_out) // local_size_out,
                )

        return _warp_wmma(A_local_buf, B_local_buf, C_local_buf)

    # ─────────────────────────────────────────────────────────────────────────
    # Store C to shared/global memory
    # ─────────────────────────────────────────────────────────────────────────

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

        @T.macro
        def _warp_stmatrix_shared(C_local_buf, C_buf, thread_binding):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            for i, j in T.grid(warp_rows, warp_cols):
                for local_id in T.vectorized(local_size_out):
                    row, col = T.meta_var(wmma_store_index_map(tx, local_id))
                    if C_buf_dims == 2:
                        C_buf[
                            (warp_m * warp_rows + i) * M_DIM + row,
                            (warp_n * warp_cols + j) * N_DIM + col,
                        ] = C_local_buf[i * (warp_cols * local_size_out) + j * local_size_out + local_id]
                    else:
                        C_buf[warp_m * warp_rows + i, warp_n * warp_cols + j, row, col] = C_local_buf[
                            i * warp_cols * local_size_out + j * local_size_out + local_id
                        ]

        @T.macro
        def _warp_stmatrix_global(C_local_buf, C_buf, thread_binding):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            for i, j in T.grid(warp_rows, warp_cols):
                for local_id in T.vectorized(local_size_out):
                    row, col = T.meta_var(wmma_store_index_map(tx, local_id))
                    C_buf[
                        (pid_m * BLOCK_M + warp_m * warp_rows + i) * M_DIM + row,
                        (pid_n * BLOCK_N + warp_n * warp_cols + j) * N_DIM + col,
                    ] = C_local_buf[i * warp_cols * local_size_out + j * local_size_out + local_id]

        return (
            _warp_stmatrix_global(C_local_buf, C_buf, thread_binding)
            if is_global
            else _warp_stmatrix_shared(C_local_buf, C_buf, thread_binding)
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Layout inference helpers (used by GemmWMMA.infer_layout)
    # ─────────────────────────────────────────────────────────────────────────

    def make_wmma_load_layout(self, local_buf: Buffer, matrix: Literal["A", "B"] = "A") -> T.Fragment:
        assert is_fragment(local_buf), "local_buf must be a fragment"
        assert matrix in ("A", "B")

        matrix_is_a = matrix == "A"
        transposed = self.a_transposed if matrix_is_a else self.b_transposed
        index_map, _ = self.get_ldmatrix_index_map(is_b=not matrix_is_a)

        inverse_load_layout = IndexMap.from_func(index_map, index_dtype=T.int32)

        def forward_thread(i, j):
            lane_id, _ = inverse_load_layout.map_indices([i, j])
            return lane_id

        def forward_index(i, j):
            _, local_id = inverse_load_layout.map_indices([i, j])
            return local_id

        micro_size_k = self.micro_size_k * self.k_pack
        if matrix_is_a:
            if transposed:
                shape_atom = [micro_size_k, self.micro_size_x]
            else:
                shape_atom = [self.micro_size_x, micro_size_k]
        else:
            if transposed:
                shape_atom = [self.micro_size_y, micro_size_k]
            else:
                shape_atom = [micro_size_k, self.micro_size_y]

        base_fragment = T.Fragment(
            shape_atom,
            forward_thread_fn=forward_thread,
            forward_index_fn=forward_index,
        )

        warp_s = self.warp_rows if matrix_is_a else self.warp_cols
        warp_r = self.chunk // micro_size_k
        block_s = self.block_row_warps if matrix_is_a else self.block_col_warps
        replicate = self.block_col_warps if matrix_is_a else self.block_row_warps

        if (matrix_is_a and not transposed) or (not matrix_is_a and transposed):
            warp_fragment = base_fragment.repeat([warp_s, warp_r], repeat_on_thread=False, lower_dim_first=False)
            if matrix_is_a:
                block_fragment = warp_fragment.repeat([block_s, 1], repeat_on_thread=True, lower_dim_first=True).replicate(replicate)
            else:
                block_fragment = warp_fragment.replicate(replicate).repeat([block_s, 1], repeat_on_thread=True, lower_dim_first=True)
        else:
            warp_fragment = base_fragment.repeat([warp_r, warp_s], repeat_on_thread=False, lower_dim_first=True)
            if matrix_is_a:
                block_fragment = warp_fragment.repeat([1, block_s], repeat_on_thread=True, lower_dim_first=True).replicate(replicate)
            else:
                block_fragment = warp_fragment.replicate(replicate).repeat([1, block_s], repeat_on_thread=True, lower_dim_first=True)

        return block_fragment

    def make_wmma_store_layout(self, local_buf: Buffer) -> T.Fragment:
        assert is_fragment(local_buf), "local_buf must be a fragment"
        shape = local_buf.shape
        # inverse_store_layout: (row, col) -> (thread_id, local_id) within 16x16 atom
        inverse_store_layout = self.get_store_index_map(inverse=True)
        micro_size_x, micro_size_y = self.micro_size_x, self.micro_size_y
        local_size_out = self.local_size_out
        block_row_warps, _ = self.block_row_warps, self.block_col_warps
        warp_rows, warp_cols = self.warp_rows, self.warp_cols
        warp_size = self.WARP_SIZE

        def forward_thread(i, j):
            block_i = (i // micro_size_x) // warp_rows
            block_j = (j // micro_size_y) // warp_cols
            atom_i, atom_j = i % micro_size_x, j % micro_size_y
            lane_id, _ = inverse_store_layout.map_indices([atom_i, atom_j])
            # block layout: [block_row_warps, block_col_warps, warp_size]
            return block_j * (block_row_warps * warp_size) + block_i * warp_size + lane_id

        def forward_index(i, j):
            warp_i = (i // micro_size_x) % warp_rows
            warp_j = (j // micro_size_y) % warp_cols
            atom_i, atom_j = i % micro_size_x, j % micro_size_y
            _, local_id = inverse_store_layout.map_indices([atom_i, atom_j])
            return warp_i * (warp_cols * local_size_out) + warp_j * local_size_out + local_id

        return T.Fragment(shape, forward_thread_fn=forward_thread, forward_index_fn=forward_index)

    # ─────────────────────────────────────────────────────────────────────────
    # Static helper
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _legalize_to_buffer_region(obj) -> BufferRegion:
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
            mins = list(obj.indices)
            ones = [tir.IntImm("int32", 1) for _ in obj.indices]
            ranges = [Range.from_min_extent(m, e) for m, e in zip(mins, ones)]
            return BufferRegion(obj.buffer, ranges)
        raise ValueError(f"Unsupported argument type: {type(obj)}")
