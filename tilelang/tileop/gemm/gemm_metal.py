from .gemm_base import GemmBase
from .inst import GemmInst
from tilelang.utils.language import is_shared, is_full_region
from tilelang import tvm as tvm
from tvm.target import Target
from tvm.ir import Range
from tvm import tir
from tilelang import language as T
from tilelang.transform.simplify import _Simplify


class GemmMetal(GemmBase):
    def is_gemm_ss(self) -> bool:
        return is_shared(self.A) and is_shared(self.B)

    def infer_layout(self, target: Target, thread_nums: int):
        return {}

    def lower(self, layout_map: dict, target: Target, thread_bounds: Range, thread_var: tir.Var):
        thread_nums = thread_bounds.extent
        m_warp, n_warp = self.policy.compute_warp_partition(self.M, self.N, thread_nums, target, GemmInst.METAL)
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

        assert block_K >= micro_size_k, f"block_K ({block_K}) must be >= micro_size_k ({micro_size_k})"
        assert is_full_region(C_region), "Fragment output C must be a full region"

        if self.is_gemm_ss():

            @T.prim_func
            def _gemm_ssr() -> None:
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

            return _Simplify(_gemm_ssr, inline_let=True)
        else:
            raise ValueError(f"Unsupported gemm combination, A: {self.A.scope()}, B: {self.B.scope()}")
