"""GEMM implementation using AMD RDNA WMMA instructions (gfx11/gfx12)."""

from .gemm_base import GemmBase
from .inst import GemmInst
from tilelang.layout import make_swizzled_layout
from tilelang.intrinsics.wmma_macro_generator import WMMAIntrinEmitter
from tilelang.utils.language import is_shared, is_fragment, is_full_region
from tilelang import tvm as tvm
from tvm.target import Target
from tvm.ir import Range
from tvm import tir
from tilelang import language as T
from tilelang.transform.simplify import _Simplify


class GemmWMMA(GemmBase):
    """GEMM using AMD RDNA WMMA instructions (16×16×16, warp-size=32)."""

    def _make_emitter(self, target: Target, thread_nums: int, thread_var=None) -> WMMAIntrinEmitter:
        m_warp, n_warp = self.policy.compute_warp_partition(self.M, self.N, thread_nums, target, GemmInst.WMMA)
        warp_row_tiles = int(self.M // m_warp)
        warp_col_tiles = int(self.N // n_warp)
        return WMMAIntrinEmitter(
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
            k_pack=self.k_pack,
            thread_var=thread_var,
            target=target,
        )

    def infer_layout(self, target: Target, thread_nums: int):
        wmma_emitter = self._make_emitter(target, thread_nums)

        if self.is_gemm_ss():
            return {
                self.A: make_swizzled_layout(self.A),
                self.B: make_swizzled_layout(self.B),
                self.C: wmma_emitter.make_wmma_store_layout(self.C),
            }
        elif self.is_gemm_sr():
            return {
                self.A: make_swizzled_layout(self.A),
                self.B: wmma_emitter.make_wmma_load_layout(self.B, matrix="B"),
                self.C: wmma_emitter.make_wmma_store_layout(self.C),
            }
        elif self.is_gemm_rs():
            return {
                self.A: wmma_emitter.make_wmma_load_layout(self.A, matrix="A"),
                self.B: make_swizzled_layout(self.B),
                self.C: wmma_emitter.make_wmma_store_layout(self.C),
            }
        elif self.is_gemm_rr():
            return {
                self.A: wmma_emitter.make_wmma_load_layout(self.A, matrix="A"),
                self.B: wmma_emitter.make_wmma_load_layout(self.B, matrix="B"),
                self.C: wmma_emitter.make_wmma_store_layout(self.C),
            }
        else:
            raise ValueError(f"Unsupported gemm combination, A: {self.A.scope()}, B: {self.B.scope()}")

    def lower(self, layout_map: dict, target: Target, thread_bounds: Range, thread_var: tir.Var):
        thread_nums = thread_bounds.extent
        wmma_emitter = self._make_emitter(target, thread_nums, thread_var=thread_var)

        block_K = wmma_emitter.chunk
        micro_size_k = wmma_emitter.micro_size_k
        in_dtype = self.in_dtype
        warp_rows = wmma_emitter.warp_rows
        warp_cols = wmma_emitter.warp_cols
        local_size_a = wmma_emitter.local_size_a
        local_size_b = wmma_emitter.local_size_b
        k_pack = wmma_emitter.k_pack

        A_region = self.ARegion
        B_region = self.BRegion
        C_region = self.CRegion
        A_buf = A_region.buffer
        B_buf = B_region.buffer
        C_buf = C_region.buffer
        clear_accum = self.clear_accum

        assert block_K >= micro_size_k * k_pack
        assert block_K % (micro_size_k * k_pack) == 0
        assert is_full_region(C_region), "Fragment output C must be a full region"

        if self.is_gemm_ss():

            @T.prim_func
            def _gemm_ssr() -> None:
                A_local = T.alloc_local((warp_rows * local_size_a * k_pack), in_dtype)
                B_local = T.alloc_local((warp_cols * local_size_b * k_pack), in_dtype)
                if clear_accum:
                    T.clear(C_buf)
                for ki in T.serial(0, (block_K // (micro_size_k * k_pack))):
                    wmma_emitter.ldmatrix_a(A_local, A_region, ki)
                    wmma_emitter.ldmatrix_b(B_local, B_region, ki)
                    wmma_emitter.wmma(A_local, B_local, C_buf, ki)

            return _Simplify(_gemm_ssr, inline_let=True)

        elif self.is_gemm_sr():
            assert is_full_region(B_region), "Fragment input B must be a full region"

            @T.prim_func
            def _gemm_srr() -> None:
                A_local = T.alloc_local((warp_rows * local_size_a * k_pack), in_dtype)
                if clear_accum:
                    T.clear(C_buf)
                for ki in T.serial(0, (block_K // (micro_size_k * k_pack))):
                    wmma_emitter.ldmatrix_a(A_local, A_region, ki)
                    wmma_emitter.wmma(A_local, B_buf, C_buf, ki)

            return _Simplify(_gemm_srr, inline_let=True)

        elif self.is_gemm_rs():
            assert is_full_region(A_region), "Fragment input A must be a full region"

            @T.prim_func
            def _gemm_rsr() -> None:
                B_local = T.alloc_local((warp_cols * local_size_b * k_pack), in_dtype)
                if clear_accum:
                    T.clear(C_buf)
                for ki in T.serial(0, (block_K // (micro_size_k * k_pack))):
                    wmma_emitter.ldmatrix_b(B_local, B_region, ki)
                    wmma_emitter.wmma(A_buf, B_local, C_buf, ki)

            return _Simplify(_gemm_rsr, inline_let=True)

        elif self.is_gemm_rr():
            assert is_full_region(A_region), "Fragment input A must be a full region"
            assert is_full_region(B_region), "Fragment input B must be a full region"

            @T.prim_func
            def _gemm_rrr() -> None:
                for ki in T.serial(0, (block_K // (micro_size_k * self.k_pack))):
                    wmma_emitter.wmma(A_buf, B_buf, C_buf, ki)

            return _Simplify(_gemm_rrr, inline_let=True)

        else:
            raise ValueError(f"Unsupported gemm combination, A: {self.A.scope()}, B: {self.B.scope()}")

    def is_gemm_ss(self) -> bool:
        return is_shared(self.A) and is_shared(self.B)

    def is_gemm_sr(self) -> bool:
        return is_shared(self.A) and is_fragment(self.B)

    def is_gemm_rs(self) -> bool:
        return is_fragment(self.A) and is_shared(self.B)

    def is_gemm_rr(self) -> bool:
        return is_fragment(self.A) and is_fragment(self.B)
