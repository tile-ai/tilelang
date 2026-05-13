from __future__ import annotations
from tilelang.cuda.intrinsics.macro.wgmma_macro_generator import SwizzleMode, gcd
from tilelang.cuda.intrinsics.macro.mma_sp_macro_generator import SparseTensorCoreIntrinEmitter
import tilelang.language as T
from tvm import DataType
from tvm.tir import PrimExpr, Buffer, Var, BufferRegion, IndexMap
from tilelang.utils import is_fragment, is_shared, retrive_ptr_from_buffer_region, is_full_region
from tilelang.cuda.intrinsics.layout.mma_layout import (
    shared_16x8_to_mma_32x4_layout_sr_a,
    shared_16x16_to_mma_32x8_layout_sr_a,
    shared_16x32_to_mma_32x16_layout_sr_a,
)
from tilelang.layout import (
    Layout,
    make_full_bank_swizzled_layout,
    make_half_bank_swizzled_layout,
    make_quarter_bank_swizzled_layout,
    make_linear_layout,
)
from tilelang.cuda.intrinsics.layout.mma_sp_layout import (
    metadata_8bit_load_32x4_to_shared_16x4_layout_32bit,
    metadata_16bit_load_32x2_to_shared_16x2_layout_32bit,
    metadata_8bit_load_32x4_to_shared_16x4_layout_16bit,
    metadata_16bit_load_32x2_to_shared_16x2_layout_16bit,
    metadata_8bit_load_32x4_to_shared_16x8_layout_8bit,
    metadata_16bit_load_32x2_to_shared_16x4_layout_8bit,
    metadata_32bit_load_32x1_to_shared_16x2_layout_8bit,
)


class WGSparseTensorCoreIntrinEmitter(SparseTensorCoreIntrinEmitter):
    wgmma_prefix: str

    wgmma_inst_m: int

    wgmma_inst_n: int

    a_shared_layout: Layout = None
    b_shared_layout: Layout = None

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
        is_m_first: bool | None = False,
        thread_var: Var | None = None,
    ):
        assert reduce_k == 1, f"{reduce_k=} is not supported"
        super().__init__(
            a_dtype=a_dtype,
            e_dtype=e_dtype,
            b_dtype=b_dtype,
            accum_dtype=accum_dtype,
            a_transposed=a_transposed,
            b_transposed=b_transposed,
            e_transposed=e_transposed,
            block_row_warps=block_row_warps,
            block_col_warps=block_col_warps,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            warp_k=warp_k,
            reduce_k=reduce_k,
            num_elems_per_byte=num_elems_per_byte,
            is_m_first=is_m_first,
            thread_var=thread_var,
        )
        self._initialize_wgmma_prefix(self.n_dim)

    def _assign_a_shared_layout(self, layout: Layout):
        self.a_shared_layout = layout
        return self

    def _assign_b_shared_layout(self, layout: Layout):
        self.b_shared_layout = layout
        return self

    def _initialize_wgmma_prefix(self, n_dim: int = 16):
        inst_m, inst_n = 64, gcd(self.warp_col_tiles, 256)
        assert inst_n % 8 == 0, (
            f"inst_n must be a multiple of 8, got {inst_n} (block_col_warps={self.block_col_warps}, warp_col_tiles={self.warp_col_tiles})"
        )
        # Validate inst_n: Hopper WGMMA supports n in [8, 256] and multiple of 8
        assert 8 <= inst_n <= 256, (
            f"inst_n must be within [8, 256], got {inst_n} (block_col_warps={self.block_col_warps}, warp_col_tiles={self.warp_col_tiles})"
        )
        # 512 bits per instruction for sparse wgmma
        inst_k = 512 // DataType(self.a_dtype).bits
        self.wgmma_inst_m = inst_m
        self.wgmma_inst_n = inst_n
        self.wgmma_prefix = f"m{inst_m}n{inst_n}k{inst_k}"

    def _determinate_swizzle_mode(self, buffer: Buffer, layout: Layout) -> SwizzleMode:
        # same behavior to src/layout/gemm_layouts.cc::makeGemmABLayoutHopper
        if layout is None or layout.is_equal(make_linear_layout(buffer)):
            return SwizzleMode.NONE
        elif layout.is_equal(make_quarter_bank_swizzled_layout(buffer)):
            return SwizzleMode.SWIZZLE_32B
        elif layout.is_equal(make_half_bank_swizzled_layout(buffer)):
            return SwizzleMode.SWIZZLE_64B
        elif layout.is_equal(make_full_bank_swizzled_layout(buffer)):
            return SwizzleMode.SWIZZLE_128B
        else:
            raise ValueError(f"Unsupported swizzle mode: {layout}")

    def wgmma_ss(
        self,
        A_region: BufferRegion,
        E_region: BufferRegion,
        B_region: BufferRegion,
        C_region: BufferRegion,
        clear_accum: PrimExpr = False,
        wg_wait: int = 0,
    ):
        assert is_shared(A_region), "A operand must be a shared buffer for wgmma_ss"
        assert is_shared(E_region), "E operand must be a shared buffer for wgmma_ss"

        local_size_out = self.local_size_out
        local_size_e = self.local_size_e
        a_dtype_abbrv = self.a_dtype_abbrv
        b_dtype_abbrv = self.b_dtype_abbrv
        accum_dtype = self.accum_dtype
        accum_dtype_abbrv = self.accum_dtype_abbrv
        m_dim = self.block_row_warps * self.warp_row_tiles
        warp_cols = self.warp_cols
        micro_size_k = self.micro_size_k
        k_dim, n_dim = self.warp_k, self.block_col_warps * self.warp_col_tiles
        wgmma_prefix = self.wgmma_prefix
        scale_in_a = 1
        scale_in_b = 1

        assert k_dim >= micro_size_k, f"k_dim must be greater than or equal to {micro_size_k}, got k_dim: {k_dim}"

        a_is_k_major = not self.a_transposed
        b_is_k_major = self.b_transposed

        a_swizzle_mode = self._determinate_swizzle_mode(A_region, self.a_shared_layout)
        b_swizzle_mode = self._determinate_swizzle_mode(B_region, self.b_shared_layout)

        elems_in_bits = DataType(self.a_dtype).bits
        elems_in_bytes = elems_in_bits // 8

        a_swizzle_atom_elems = a_swizzle_mode.swizzle_byte_size() // elems_in_bytes
        b_swizzle_atom_elems = n_dim if b_swizzle_mode.is_none() else b_swizzle_mode.swizzle_byte_size() // elems_in_bytes
        accum_bits = DataType(accum_dtype).bits
        accum_regs = ((m_dim // 64) * warp_cols * local_size_out * accum_bits + 31) // 32

        a_leading_byte_offset = (8 * 8 * elems_in_bytes) if a_is_k_major else (8 * m_dim * elems_in_bytes)
        a_stride_byte_offset = (8 * k_dim * elems_in_bytes) if a_is_k_major else (8 * 8 * elems_in_bytes)

        if not a_swizzle_mode.is_none():
            # swizzle mode doesn't require LBO/SBO to be 1
            # https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-leading-dimension-byte-offset
            if a_is_k_major:
                a_leading_byte_offset = 16
                a_stride_byte_offset = 8 * a_swizzle_mode.swizzle_byte_size()
            else:
                # MN Major
                # LBO represents the distance between two atoms along the M dimension
                # SBO represents the distance between two atoms along the K dimension
                a_m_axis_atoms = m_dim // a_swizzle_atom_elems
                if a_m_axis_atoms <= 1:
                    a_leading_byte_offset = 0
                else:
                    a_leading_byte_offset = 8 * a_swizzle_mode.swizzle_atom_size() * (a_swizzle_mode.swizzle_byte_size() // elems_in_bytes)

                if a_m_axis_atoms <= 1:
                    a_stride_byte_offset = 8 * elems_in_bytes * m_dim
                else:
                    a_stride_byte_offset = 8 * elems_in_bytes * a_swizzle_atom_elems

        b_leading_byte_offset = (8 * 8 * elems_in_bytes) if b_is_k_major else (8 * n_dim * elems_in_bytes)
        b_stride_byte_offset = (8 * k_dim * elems_in_bytes) if b_is_k_major else (0 if n_dim == 8 else (8 * 8 * elems_in_bytes))
        if not b_swizzle_mode.is_none():
            # swizzle mode doesn't require LBO/SBO to be 1
            # https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-leading-dimension-byte-offset
            if b_is_k_major:
                b_leading_byte_offset = 16
                b_stride_byte_offset = 8 * b_swizzle_mode.swizzle_byte_size()
            else:
                # MN Major, K * N
                # LBO represents the distance between two atoms along the N dimension
                # SBO represents the distance between two atoms along the K dimension
                b_n_axis_atoms = n_dim // b_swizzle_atom_elems
                if b_n_axis_atoms <= 1:
                    b_leading_byte_offset = 0
                else:
                    b_leading_byte_offset = 8 * 8 * elems_in_bytes * k_dim
                if b_n_axis_atoms <= 1:
                    b_stride_byte_offset = 8 * elems_in_bytes * n_dim
                else:
                    b_stride_byte_offset = 8 * elems_in_bytes * b_swizzle_atom_elems

        # for example, if [n, k] where k is 128, we should split it into 2 atoms
        # where max specially handles the case when n_dim is 8.
        ak_atom_size = max(a_swizzle_atom_elems // (micro_size_k // self.SPARSE_FACTOR), 1)
        bk_atom_size = max(b_swizzle_atom_elems // micro_size_k, 1)
        wgmma_inst_m, wgmma_inst_n = self.wgmma_inst_m, self.wgmma_inst_n
        num_inst_m = 4 * self.warp_row_tiles // wgmma_inst_m
        num_inst_n = self.warp_col_tiles // wgmma_inst_n

        thread_binding = self.get_thread_binding()

        A_ptr = retrive_ptr_from_buffer_region(A_region)
        B_ptr = retrive_ptr_from_buffer_region(B_region)
        assert is_full_region(C_region), "Fragment output C must be a full region"
        C_buf = C_region.buffer

        @T.macro
        def _warp_mma(A_ptr, B_ptr, C_buf):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            k_blocks = k_dim // micro_size_k
            e_stage_elems = self.warp_rows * self.local_size_e
            E_local = T.alloc_local((k_blocks * e_stage_elems), self.e_dtype)

            desc_a = T.alloc_wgmma_desc()
            desc_b = T.alloc_wgmma_desc()
            T.initialize_wgmma_descriptor(desc_a, A_ptr, a_swizzle_mode, int(a_leading_byte_offset >> 4), int(a_stride_byte_offset >> 4))
            T.initialize_wgmma_descriptor(desc_b, B_ptr, b_swizzle_mode, int(b_leading_byte_offset >> 4), int(b_stride_byte_offset >> 4))

            for ki in T.unroll(k_blocks):
                for i in T.unroll(num_inst_m):
                    self.ldmatrix_e(E_local, E_region, i, warp_m, ki, ki)

            # NOTE: cutlass doesn't fence metadata, we follow the same here
            T.warpgroup_fence_operand(C_buf, num_regs=accum_regs)
            T.warpgroup_arrive()

            for ki in T.unroll(k_blocks):
                for j in T.unroll(num_inst_n):
                    for i in T.unroll(num_inst_m):
                        e_local_offset = ki * e_stage_elems + i * local_size_e
                        scale_out = T.Select(ki != 0, 1, T.Select(clear_accum, 0, 1))
                        warp_i = (warp_m // 4) * num_inst_m + i
                        warp_j = warp_n * num_inst_n + j
                        A_offset = (
                            (ki % ak_atom_size) * (micro_size_k // self.SPARSE_FACTOR)
                            + warp_i * 64 * a_swizzle_atom_elems
                            + (ki // ak_atom_size) * m_dim * a_swizzle_atom_elems
                            if a_is_k_major
                            else warp_i * 64 * (k_dim // self.SPARSE_FACTOR)
                            + ki * a_swizzle_atom_elems * (micro_size_k // self.SPARSE_FACTOR)
                        )
                        B_offset = (
                            (ki // bk_atom_size) * n_dim * b_swizzle_atom_elems
                            + (ki % bk_atom_size) * micro_size_k
                            + warp_j * wgmma_inst_n * b_swizzle_atom_elems
                            if b_is_k_major
                            else (
                                ki * b_swizzle_atom_elems * micro_size_k
                                + warp_j * wgmma_inst_n * (k_dim if n_dim // b_swizzle_atom_elems > 1 else 1)
                            )
                        )
                        C_offset = i * warp_cols * local_size_out + j * warp_cols * local_size_out // num_inst_n  # 4 warps as an unit
                        T.ptx_wgmma_sp_ss(
                            accum_dtype,
                            wgmma_prefix,
                            a_is_k_major,
                            b_is_k_major,
                            a_dtype_abbrv,
                            b_dtype_abbrv,
                            accum_dtype_abbrv,
                            desc_a.data,
                            (A_offset * elems_in_bytes) >> 4,
                            E_local.data,
                            e_local_offset,
                            self.SPARSE_SELECTOR,
                            desc_b.data,
                            (B_offset * elems_in_bytes) >> 4,
                            C_buf.data,
                            C_offset,
                            scale_out,
                            scale_in_a,
                            scale_in_b,
                        )

            T.warpgroup_commit_batch()
            if wg_wait >= 0:
                T.warpgroup_wait(wg_wait)
            T.warpgroup_fence_operand(C_buf, num_regs=accum_regs)

        return _warp_mma(A_ptr, B_ptr, C_buf)

    def wgmma_rs(
        self,
        A_region: BufferRegion,
        E_region: BufferRegion,
        B_region: BufferRegion,
        C_region: BufferRegion,
        clear_accum: PrimExpr = False,
        wg_wait: int = 0,
    ):
        assert is_fragment(A_region), "A operand must be a fragment buffer for wgmma_rs"
        assert is_shared(E_region), "E operand must be a shared buffer for wgmma_rs"

        local_size_a = self.local_size_a
        local_size_out = self.local_size_out
        local_size_e = self.local_size_e
        a_dtype_abbrv = self.a_dtype_abbrv
        b_dtype_abbrv = self.b_dtype_abbrv
        accum_dtype = self.accum_dtype
        accum_dtype_abbrv = self.accum_dtype_abbrv
        m_dim = self.block_row_warps * self.warp_row_tiles
        warp_rows, warp_cols = self.warp_rows, self.warp_cols
        micro_size_k = self.micro_size_k
        k_dim, n_dim = self.warp_k, self.block_col_warps * self.warp_col_tiles
        wgmma_prefix = self.wgmma_prefix
        scale_in_a = 1
        scale_in_b = 1

        assert k_dim >= micro_size_k, f"k_dim must be greater than or equal to {micro_size_k}, got k_dim: {k_dim}"

        elems_in_bytes = DataType(self.a_dtype).bits // 8
        a_bits = DataType(self.a_dtype).bits
        accum_bits = DataType(accum_dtype).bits
        a_regs = ((warp_rows * local_size_a * (k_dim // micro_size_k)) * a_bits + 31) // 32
        accum_regs = ((m_dim // 64) * warp_cols * local_size_out * accum_bits + 31) // 32
        b_is_k_major = self.b_transposed

        b_swizzle_mode = self._determinate_swizzle_mode(B_region, self.b_shared_layout)
        b_swizzle_atom_elems = n_dim if b_swizzle_mode.is_none() else b_swizzle_mode.swizzle_byte_size() // elems_in_bytes

        b_leading_byte_offset = (8 * 8 * elems_in_bytes) if b_is_k_major else (8 * n_dim * elems_in_bytes)
        b_stride_byte_offset = (8 * k_dim * elems_in_bytes) if b_is_k_major else (0 if n_dim == 8 else (8 * 8 * elems_in_bytes))
        if not b_swizzle_mode.is_none():
            if b_is_k_major:
                b_leading_byte_offset = 16
                b_stride_byte_offset = 8 * b_swizzle_mode.swizzle_byte_size()
            else:
                b_n_axis_atoms = n_dim // b_swizzle_atom_elems
                if b_n_axis_atoms <= 1:
                    b_leading_byte_offset = 0
                else:
                    b_leading_byte_offset = 8 * 8 * elems_in_bytes * k_dim
                if b_n_axis_atoms <= 1:
                    b_stride_byte_offset = 8 * elems_in_bytes * n_dim
                else:
                    b_stride_byte_offset = 8 * elems_in_bytes * b_swizzle_atom_elems

        bk_atom_size = max(b_swizzle_atom_elems // micro_size_k, 1)
        wgmma_inst_m, wgmma_inst_n = self.wgmma_inst_m, self.wgmma_inst_n
        num_inst_m = 4 * self.warp_row_tiles // wgmma_inst_m
        num_inst_n = self.warp_col_tiles // wgmma_inst_n

        thread_binding = self.get_thread_binding()

        assert is_full_region(A_region), "Fragment input A must be a full region"
        assert is_full_region(C_region), "Fragment output C must be a full region"
        A_buf = A_region.buffer
        B_ptr = retrive_ptr_from_buffer_region(B_region)
        C_buf = C_region.buffer

        k_blocks = k_dim // micro_size_k
        e_stage_elems = self.warp_rows * self.local_size_e

        @T.macro
        def _warp_mma(A_buf, B_ptr, C_buf):
            _, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            E_local = T.alloc_local((k_blocks * e_stage_elems), self.e_dtype)

            desc_b = T.alloc_wgmma_desc()
            T.initialize_wgmma_descriptor(desc_b, B_ptr, b_swizzle_mode, int(b_leading_byte_offset >> 4), int(b_stride_byte_offset >> 4))

            for ki in T.unroll(k_blocks):
                for i in T.unroll(num_inst_m):
                    self.ldmatrix_e(E_local, E_region, i, warp_m, ki, ki)

            # NOTE: cutlass doesn't fence metadata, we follow the same here
            T.warpgroup_fence_operand(A_buf, num_regs=a_regs)
            T.warpgroup_fence_operand(C_buf, num_regs=accum_regs)
            T.warpgroup_arrive()

            for ki in T.unroll(k_blocks):
                for j in T.unroll(num_inst_n):
                    for i in T.unroll(num_inst_m):
                        e_local_offset = ki * e_stage_elems + i * local_size_e
                        scale_out = T.Select(ki != 0, 1, T.Select(clear_accum, 0, 1))
                        warp_j = warp_n * num_inst_n + j
                        A_offset = ki * warp_rows * local_size_a + i * local_size_a
                        B_offset = (
                            (ki // bk_atom_size) * n_dim * b_swizzle_atom_elems
                            + warp_j * wgmma_inst_n * b_swizzle_atom_elems
                            + (ki % bk_atom_size) * micro_size_k
                            if b_is_k_major
                            else (
                                ki * b_swizzle_atom_elems * micro_size_k
                                + warp_j * wgmma_inst_n * (k_dim if n_dim // b_swizzle_atom_elems > 1 else 1)
                            )
                        )
                        C_offset = i * warp_cols * local_size_out + j * warp_cols * local_size_out // num_inst_n
                        T.ptx_wgmma_sp_rs(
                            accum_dtype,
                            wgmma_prefix,
                            b_is_k_major,
                            a_dtype_abbrv,
                            b_dtype_abbrv,
                            accum_dtype_abbrv,
                            A_buf.data,
                            A_offset,
                            E_local.data,
                            e_local_offset,
                            self.SPARSE_SELECTOR,
                            desc_b.data,
                            (B_offset * elems_in_bytes) >> 4,
                            C_buf.data,
                            C_offset,
                            scale_out,
                            scale_in_a,
                            scale_in_b,
                        )

            T.warpgroup_commit_batch()
            if wg_wait >= 0:
                T.warpgroup_wait(wg_wait)
            T.warpgroup_fence_operand(C_buf, num_regs=accum_regs)
            T.warpgroup_fence_operand(A_buf, num_regs=a_regs)

        return _warp_mma(A_buf, B_ptr, C_buf)

    def ldmatrix_e(self, E_local_buf: Buffer, E_shared_buf: Buffer, inst_i: PrimExpr, warp_m: PrimExpr, ki: PrimExpr, ki_slot: PrimExpr):
        num_inst_m = 4 * self.warp_row_tiles // self.wgmma_inst_m
        micro_size_k = self.micro_size_k
        local_size_e = self.local_size_e
        e_stage_elems = self.warp_rows * local_size_e
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
                    mma_load_layout = metadata_8bit_load_32x4_to_shared_16x8_layout_8bit
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

        E_region = self._legalize_to_buffer_region(E_shared_buf)
        E_buf = E_region.buffer
        E_base0 = E_region.region[-2].min
        E_base1 = E_region.region[-1].min
        E_other = [r.min for r in E_region.region[:-2]]

        @T.macro
        def _warp_ldmatrix_e(
            E_local_buf,
            E_shared_buf,
            inst_i,
            ki,
            thread_binding,
        ):
            wi = ((warp_m // 4) * num_inst_m + inst_i) * 64 + (warp_m % 4) * 16
            wk = (ki * micro_size_k) // self.e_factor
            e_local_base = ki_slot * e_stage_elems
            tx, _, _ = self.extract_thread_binding(thread_binding)
            for j in T.serial(local_size_e):
                mi, mk = mma_load_layout(tx, j)
                E_local_buf[e_local_base + inst_i * local_size_e + j] = (
                    E_shared_buf[tuple(E_other) + (E_base0 + wk + mk, E_base1 + wi + mi)]
                    if trans
                    else E_shared_buf[tuple(E_other) + (E_base0 + wi + mi, E_base1 + wk + mk)]
                )

        return _warp_ldmatrix_e(E_local_buf, E_buf, inst_i, ki, thread_binding)

    def make_mma_load_layout(self, local_buf: Buffer, matrix: str = "A") -> T.Fragment:
        assert matrix == "A", "WGMMA sparse only supports A matrix load layout"
        assert is_fragment(local_buf), f"local_buf must be a fragment, but got {local_buf.scope()}"

        dtype = self.a_dtype
        dtype_bits = DataType(dtype).bits
        transposed = self.a_transposed

        if dtype_bits == 32:
            transform_func_sr_a = shared_16x8_to_mma_32x4_layout_sr_a
        elif dtype_bits == 16:
            transform_func_sr_a = shared_16x16_to_mma_32x8_layout_sr_a
        elif dtype_bits == 8:
            transform_func_sr_a = shared_16x32_to_mma_32x16_layout_sr_a
        else:
            raise ValueError(f"Unsupported dtype {dtype}")

        is_sr_axis_order = not transposed

        transform_func = transform_func_sr_a if is_sr_axis_order else lambda i, j: transform_func_sr_a(j, i)

        inverse_mma_load_layout = IndexMap.from_func(transform_func, index_dtype=T.int32)

        def forward_thread(i: int, j: int) -> int:
            lane_id, _ = inverse_mma_load_layout.map_indices([i, j])
            return lane_id

        def forward_index(i: int, j: int) -> int:
            _, local_id = inverse_mma_load_layout.map_indices([i, j])
            return local_id

        micro_size_s = self.micro_size_x
        # sparse: each instruction holds micro_size_k/SPARSE_FACTOR actual K elements
        micro_size_r = self.micro_size_k // self.SPARSE_FACTOR

        base_fragment = T.Fragment(
            [micro_size_s, micro_size_r] if is_sr_axis_order else [micro_size_r, micro_size_s],
            forward_thread_fn=forward_thread,
            forward_index_fn=forward_index,
        )

        warp_rows = self.warp_rows
        # number of instructions in K direction
        warp_r = self.warp_k // self.micro_size_k
        block_s = self.block_row_warps
        replicate = self.block_col_warps

        if is_sr_axis_order:
            warp_fragment = base_fragment.repeat([block_s, 1], repeat_on_thread=True, lower_dim_first=False).replicate(replicate)
            block_fragment = warp_fragment.repeat([warp_rows, warp_r], repeat_on_thread=False, lower_dim_first=False)
        else:
            warp_fragment = base_fragment.repeat([1, block_s], repeat_on_thread=True, lower_dim_first=False).replicate(replicate)
            block_fragment = warp_fragment.repeat([warp_r, warp_rows], repeat_on_thread=False, lower_dim_first=True)

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
