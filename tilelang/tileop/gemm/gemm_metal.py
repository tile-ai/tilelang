from __future__ import annotations

from .gemm_base import GemmBase
from tilelang.layout import Layout
from tilelang.utils.language import is_shared, is_global, is_full_region, is_metal_cooperative_tensor, is_fragment, is_metal_simdgroup
from tilelang import tvm as tvm
from tvm.target import Target
from tvm.ir import Range
from tvm import tirx as tir
from tilelang import language as T
from tilelang.transform.simplify import _Simplify


GEMM_INST_METAL = "metal.simdgroup"
GEMM_INST_METAL_COOPERATIVE_TENSOR = "metal.cooperative_tensor"


def _make_padded_layout(buffer):
    shape = buffer.shape
    stride = int(shape[-2])
    continuous = int(shape[-1])
    element_bits = int(tvm.DataType(buffer.dtype).bits)
    padded = continuous
    if (element_bits * continuous) % 256 == 0:
        padded += 128 // element_bits
    return Layout([stride, continuous], lambda i, j: i * padded + j)


class GemmMetalSimdGroup(GemmBase):
    def is_gemm_ss(self) -> bool:
        return is_shared(self.A) and is_shared(self.B)

    def infer_layout(self, target: Target, thread_nums: int):
        return {}

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_var: tir.Var,
        mbar_phase_expr: tir.PrimExpr | None = None,
    ):
        thread_nums = thread_bounds.extent
        m_warp, n_warp = self.policy.compute_warp_partition(self.M, self.N, thread_nums, target, GEMM_INST_METAL)
        warp_row_tiles = int(self.M // m_warp)
        warp_col_tiles = int(self.N // n_warp)

        from tilelang.intrinsics.metal_macro_generator import MPSIntrinEmitter

        mps_emitter = MPSIntrinEmitter(
            a_dtype=self.in_dtype,
            b_dtype=self.in_dtype,
            accum_dtype=self.accum_dtype,
            a_transposed=self.trans_A,
            b_transposed=self.trans_B,
            block_row_warps=m_warp,
            block_col_warps=n_warp,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=self.chunk,
            thread_var=thread_var,
            use_cooperative_tensor=False,
        )

        in_dtype = self.in_dtype
        accum_dtype = self.accum_dtype
        warp_rows = mps_emitter.warp_rows
        warp_cols = mps_emitter.warp_cols
        num_simd_c = warp_rows * warp_cols
        block_K = mps_emitter.chunk
        micro_size_k = mps_emitter.micro_size_k

        A_region = self.ARegion
        B_region = self.BRegion
        C_region = self.CRegion
        C_buf = C_region.buffer
        clear_accum = self.clear_accum
        c_in_register = is_fragment(C_buf) or is_metal_simdgroup(C_buf)

        assert block_K >= micro_size_k, f"block_K ({block_K}) must be >= micro_size_k ({micro_size_k})"
        assert is_full_region(C_region), "Fragment output C must be a full region"
        assert c_in_register or is_shared(C_buf), (
            f"Metal GEMM requires C in local.fragment, metal.simdgroup or shared scope, got {C_buf.scope()}"
        )

        if not self.is_gemm_ss():
            raise ValueError(f"Unsupported gemm combination, A: {self.A.scope()}, B: {self.B.scope()}")

        if c_in_register:

            @T.prim_func
            def _gemm_ss_simdgroup() -> None:
                A_local = T.alloc_local((warp_rows * 64), in_dtype, scope="metal.simdgroup")
                B_local = T.alloc_local((warp_cols * 64), in_dtype, scope="metal.simdgroup")
                if clear_accum:
                    for _i in T.serial(num_simd_c):
                        T.make_filled_simdgroup_matrix(C_buf.data, _i, T.cast(0, accum_dtype))
                for ki in T.serial(0, (block_K // micro_size_k)):
                    mps_emitter.ldmatrix_a(A_local, A_region, ki)
                    mps_emitter.ldmatrix_b(B_local, B_region, ki)
                    mps_emitter.mma(A_local, B_local, C_buf)

            return _Simplify(_gemm_ss_simdgroup, inline_let=True)

        @T.prim_func
        def _gemm_ss_shared() -> None:
            A_local = T.alloc_local((warp_rows * 64), in_dtype, scope="metal.simdgroup")
            B_local = T.alloc_local((warp_cols * 64), in_dtype, scope="metal.simdgroup")
            C_simd = T.alloc_local((num_simd_c * 64), accum_dtype, scope="metal.simdgroup")
            if clear_accum:
                for _i in T.serial(num_simd_c):
                    T.make_filled_simdgroup_matrix(C_simd.data, _i, T.cast(0, accum_dtype))
            else:
                mps_emitter.simd_load(C_simd, C_buf)
            for ki in T.serial(0, (block_K // micro_size_k)):
                mps_emitter.ldmatrix_a(A_local, A_region, ki)
                mps_emitter.ldmatrix_b(B_local, B_region, ki)
                mps_emitter.mma(A_local, B_local, C_simd)
            mps_emitter.simd_store(C_simd, C_buf)

        return _Simplify(_gemm_ss_shared, inline_let=True)


class GemmMetal(GemmBase):
    def is_gemm_ss(self) -> bool:
        return is_shared(self.A) and is_shared(self.B)

    def is_gemm_gg(self) -> bool:
        return is_global(self.A) and is_global(self.B)

    def _make_mps_emitter(self, target: Target, thread_nums: int):
        from tilelang.intrinsics.metal_macro_generator import MPSIntrinEmitter

        m_warp, n_warp = self.policy.compute_warp_partition(self.M, self.N, thread_nums, target, GEMM_INST_METAL)
        if self.is_gemm_gg():
            num_warps = int(thread_nums) // 32
            k_n_per_warp = 32
            m_warp, n_warp = 1, num_warps
            if int(self.N) % (n_warp * k_n_per_warp) != 0:
                n_warp = int(self.N) // k_n_per_warp
                m_warp = num_warps // n_warp
                if m_warp == 0:
                    m_warp = 1
        warp_row_tiles = int(self.M // m_warp)
        warp_col_tiles = int(self.N // n_warp)
        return MPSIntrinEmitter(
            a_dtype=self.in_dtype,
            b_dtype=self.in_dtype,
            accum_dtype=self.accum_dtype,
            a_transposed=self.trans_A,
            b_transposed=self.trans_B,
            block_row_warps=m_warp,
            block_col_warps=n_warp,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=self.chunk,
        ), m_warp, n_warp

    def infer_layout(self, target: Target, thread_nums: int):
        result = {}
        if self.is_gemm_ss():
            result[self.A] = _make_padded_layout(self.A)
            result[self.B] = _make_padded_layout(self.B)
        if is_fragment(self.C):
            emitter, _, _ = self._make_mps_emitter(target, thread_nums)
            result[self.C] = emitter.make_cooperative_tensor_store_layout(self.C)
        return result

    @staticmethod
    def _get_padded_stride(buffer):
        continuous = int(buffer.shape[-1])
        element_bits = int(tvm.DataType(buffer.dtype).bits)
        padded = continuous
        if (element_bits * continuous) % 256 == 0:
            padded += 128 // element_bits
        return padded

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_var: tir.Var,
        mbar_phase_expr: tir.PrimExpr | None = None,
    ):
        thread_nums = thread_bounds.extent
        _, m_warp, n_warp = self._make_mps_emitter(target, int(thread_nums))
        warp_row_tiles = int(self.M // m_warp)
        warp_col_tiles = int(self.N // n_warp)

        from tilelang.intrinsics.metal_macro_generator import MPSIntrinEmitter

        a_stride = self._get_padded_stride(self.A) if self.is_gemm_ss() else None
        b_stride = self._get_padded_stride(self.B) if self.is_gemm_ss() else None

        c_bytes_per_thread = warp_row_tiles * warp_col_tiles * 64
        inner_k_steps = 2 if c_bytes_per_thread <= 128 else 1
        mps_emitter = MPSIntrinEmitter(
            a_dtype=self.in_dtype,
            b_dtype=self.in_dtype,
            accum_dtype=self.accum_dtype,
            a_transposed=self.trans_A,
            b_transposed=self.trans_B,
            block_row_warps=m_warp,
            block_col_warps=n_warp,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=self.chunk,
            thread_var=thread_var,
            a_stride_override=a_stride,
            b_stride_override=b_stride,
            inner_k_steps=inner_k_steps,
        )

        in_dtype = self.in_dtype
        accum_dtype = self.accum_dtype
        warp_rows = mps_emitter.warp_rows
        warp_cols = mps_emitter.warp_cols
        num_simd_c = warp_rows * warp_cols
        block_K = mps_emitter.chunk
        micro_size_x = mps_emitter.micro_size_x
        micro_size_y = mps_emitter.micro_size_y
        micro_size_k = mps_emitter.micro_size_k
        inner_k_steps = mps_emitter.inner_k_steps
        a_tile_elems = micro_size_x * micro_size_k
        b_tile_elems = micro_size_k * micro_size_y
        c_tile_elems = micro_size_x * micro_size_y

        A_region = self.ARegion
        B_region = self.BRegion
        C_region = self.CRegion
        C_buf = C_region.buffer
        clear_accum = self.clear_accum
        assert block_K >= micro_size_k, f"block_K ({block_K}) must be >= micro_size_k ({micro_size_k})"
        assert is_full_region(C_region), "Fragment output C must be a full region"

        if not (self.is_gemm_ss() or self.is_gemm_gg()):
            raise ValueError(f"Unsupported gemm combination, A: {self.A.scope()}, B: {self.B.scope()}")

        @T.prim_func
        def _gemm_with_c_writeback() -> None:
            A_local = T.alloc_local((warp_rows * a_tile_elems * inner_k_steps), in_dtype)
            B_local = T.alloc_local((warp_cols * b_tile_elems * inner_k_steps), in_dtype)
            C_ct = T.alloc_local((num_simd_c * c_tile_elems), accum_dtype)
            if clear_accum:
                for _i in T.serial(num_simd_c):
                    T.cooperative_tensor_fill(C_ct.data, _i, T.cast(0, accum_dtype), micro_size_x, micro_size_y)
            else:
                mps_emitter.simd_load(C_ct, C_buf)
            for k_outer in T.serial(0, (block_K // (micro_size_k * inner_k_steps))):
                for k_inner in T.serial(0, inner_k_steps):
                    ki = k_outer * inner_k_steps + k_inner
                    mps_emitter.ldmatrix_a(A_local, A_region, ki, k_inner)
                    mps_emitter.ldmatrix_b(B_local, B_region, ki, k_inner)
                for k_inner in T.serial(0, inner_k_steps):
                    mps_emitter.mma(A_local, B_local, C_ct, k_inner)
            mps_emitter.simd_store(C_ct, C_buf)

        return _Simplify(_gemm_with_c_writeback, inline_let=True)
