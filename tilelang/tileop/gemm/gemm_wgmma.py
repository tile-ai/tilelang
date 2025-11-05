from .gemm_base import GemmBase
from tilelang.layout import make_wgmma_swizzled_layout
from tilelang.intrinsics.wgmma_macro_generator import (
    TensorCoreIntrinEmitter,)
from tilelang.utils.language import is_shared, is_fragment
from tilelang import tvm as tvm
from tvm.target import Target
from tvm import tir
from tilelang import language as T
from tilelang.transform.simplify import _Simplify


class GemmWGMMA(GemmBase):

    def infer_layout(self, target: Target, thread_nums: int):
        m_warp, n_warp = self.policy.compute_warp_partition(self.M, self.N, thread_nums, target,
                                                            True)
        warp_row_tiles = int(self.M // m_warp)
        warp_col_tiles = int(self.N // n_warp)
        mma_emitter = TensorCoreIntrinEmitter(
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
        )
        a_is_k_major = not self.trans_A
        b_is_k_major = self.trans_B

        if self.is_gemm_ss():
            a_continuity = self.M if a_is_k_major else 4 * self.K // m_warp
            b_continuity = self.K if b_is_k_major else self.N // n_warp

            return {
                # WGMMA does not support padding
                self.A:
                    make_wgmma_swizzled_layout(
                        self.A, continuity=a_continuity, k_major=a_is_k_major),
                self.B:
                    make_wgmma_swizzled_layout(
                        self.B, continuity=b_continuity, k_major=b_is_k_major),
                self.C:
                    mma_emitter.make_mma_store_layout(self.C),
            }
        elif self.is_gemm_rs():
            b_continuity = self.N if b_is_k_major else 4 * self.K // n_warp
            return {
                self.A:
                    mma_emitter.make_mma_load_layout(self.A, matrix="A"),
                self.B:
                    make_wgmma_swizzled_layout(
                        self.B, continuity=b_continuity, k_major=b_is_k_major),
                self.C:
                    mma_emitter.make_mma_store_layout(self.C),
            }
        else:
            raise ValueError(
                f"Unsupported gemm combination for wgmma, A: {self.A.scope()}, B: {self.B.scope()}")

    def lower(self, layout_map: dict, target: Target, thread_nums: int, thread_var: tir.Var):
        m_warp, n_warp = self.policy.compute_warp_partition(self.M, self.N, thread_nums, target,
                                                            True)

        warp_row_tiles = int(self.M // m_warp)
        warp_col_tiles = int(self.N // n_warp)
        mma_emitter = TensorCoreIntrinEmitter(
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
        )

        if self.A in layout_map:
            mma_emitter._assign_a_shared_layout(layout_map[self.A])
        if self.B in layout_map:
            mma_emitter._assign_b_shared_layout(layout_map[self.B])

        A_shared = self.A
        B_shared = self.B
        C_local = self.C
        clear_accum = self.clear_accum
        wg_wait = self.wg_wait

        if self.is_gemm_ss():

            @T.prim_func
            def _gemm_ssr() -> None:
                """
                The inner macro that loads data from shared buffers A_shared and
                B_shared into local fragments, then issues Tensor Core mma ops,
                accumulating into C_local.
                """
                # Perform Matrix Multiplication
                mma_emitter.wgmma(A_shared, B_shared, C_local, clear_accum, wg_wait)

            # Simplify to optimize the index computing
            # Must inline let statements to simplify the analysis
            return _Simplify(_gemm_ssr, inline_let=True)
        elif self.is_gemm_rs():
            A_local = self.A

            @T.prim_func
            def _gemm_rsr() -> None:
                """
                The inner macro that loads data from shared buffers A_shared and
                B_shared into local fragments, then issues Tensor Core mma ops,
                accumulating into C_local.
                """
                mma_emitter.wgmma(A_local, B_shared, C_local, clear_accum, wg_wait)

            # Simplify to optimize the index computing
            # Must inline let statements to simplify the analysis
            return _Simplify(_gemm_rsr, inline_let=True)
        raise ValueError(
            f"Unsupported gemm combination for wgmma, A: {self.A.scope()}, B: {self.B.scope()}")

    def is_gemm_ss(self) -> bool:
        return is_shared(self.A) and is_shared(self.B)

    def is_gemm_sr(self) -> bool:
        return is_shared(self.A) and is_fragment(self.B)

    def is_gemm_rs(self) -> bool:
        return is_fragment(self.A) and is_shared(self.B)

    def is_gemm_rr(self) -> bool:
        return is_fragment(self.A) and is_fragment(self.B)
