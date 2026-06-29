from __future__ import annotations

from tilelang.tileop.gemm.gemm_base import GemmBase
from tilelang.layout import make_swizzled_layout
from tilelang.cuda.intrinsics.macro.mma_macro_generator import (
    SM120BlockScaledOperandPackage,
    TensorCoreIntrinEmitter,
    TensorCoreIntrinEmitterWithBlockScale,
)
from tilelang.utils.language import is_shared, is_fragment, is_full_region
from tilelang import tvm as tvm
from tvm.target import Target
from tvm.ir import Range
from tvm import tirx
from tilelang import language as T
from tilelang.transform.simplify import _Simplify


GEMM_INST_MMA = "cuda.mma"


def _is_explicit_non_sm120_cuda(target: Target) -> bool:
    if target.kind.name != "cuda":
        return False
    arch = target.attrs.get("arch", None)
    if arch is None:
        return False
    arch_str = str(arch)
    if not arch_str.startswith("sm_"):
        return False
    arch_int = int(arch_str[3:].rstrip("af"))
    return arch_int < 120 or arch_int >= 130


class GemmMMA(GemmBase):
    intrin_emitter_cls = TensorCoreIntrinEmitter

    def _make_mma_emitter(self, target: Target, thread_nums: int, thread_var: tirx.Var | None = None):
        if self.is_blockscaled and _is_explicit_non_sm120_cuda(target):
            raise ValueError("T.mma_gemm_blockscaled requires SM120 CUDA target")
        m_warp, n_warp = self.policy.compute_warp_partition(self.M, self.N, thread_nums, target, GEMM_INST_MMA)
        warp_row_tiles = int(self.M // m_warp)
        warp_col_tiles = int(self.N // n_warp)
        emitter_cls = TensorCoreIntrinEmitterWithBlockScale if self.is_blockscaled else self.intrin_emitter_cls
        emitter = emitter_cls(
            a_dtype=self.a_dtype,
            b_dtype=self.b_dtype,
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
        return emitter

    def infer_layout(self, target: Target, thread_nums: int):
        mma_emitter = self._make_mma_emitter(target, thread_nums)
        if self.is_blockscaled and not self.is_gemm_ss():
            raise ValueError("T.mma_gemm_blockscaled supports shared-memory A/B operands only")
        if self.is_gemm_ss():
            return {
                self.A: make_swizzled_layout(self.A),
                self.B: make_swizzled_layout(self.B),
                self.C: mma_emitter.make_mma_store_layout(self.C),
            }
        elif self.is_gemm_sr():
            return {
                self.A: make_swizzled_layout(self.A),
                self.B: mma_emitter.make_mma_load_layout(self.B, matrix="B"),
                self.C: mma_emitter.make_mma_store_layout(self.C),
            }
        elif self.is_gemm_rs():
            return {
                self.A: mma_emitter.make_mma_load_layout(self.A, matrix="A"),
                self.B: make_swizzled_layout(self.B),
                self.C: mma_emitter.make_mma_store_layout(self.C),
            }
        elif self.is_gemm_rr():
            return {
                self.A: mma_emitter.make_mma_load_layout(self.A, matrix="A"),
                self.B: mma_emitter.make_mma_load_layout(self.B, matrix="B"),
                self.C: mma_emitter.make_mma_store_layout(self.C),
            }
        else:
            raise ValueError(f"Unsupported gemm combination, A: {self.A.scope()}, B: {self.B.scope()}")

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_var: tirx.Var,
        mbar_phase_expr: tirx.PrimExpr | None = None,
    ):
        thread_nums = thread_bounds.extent
        # Emitter lane/warp math uses zero-based ids within the current thread bounds.
        local_thread_var = thread_var - thread_bounds.min
        mma_emitter = self._make_mma_emitter(target, thread_nums, thread_var=local_thread_var)

        a_dtype = self.a_dtype
        b_dtype = self.b_dtype
        warp_rows = mma_emitter.warp_rows
        warp_cols = mma_emitter.warp_cols
        local_size_a = mma_emitter.local_size_a
        local_size_b = mma_emitter.local_size_b
        block_K = mma_emitter.chunk
        micro_size_k = mma_emitter.micro_size_k
        # We use region for memory input to support strided gemm
        # T.gemm(A_shared[0:128, :], B_shared, C_local)
        A_region = self.ARegion
        B_region = self.BRegion
        C_region = self.CRegion

        A_buf = A_region.buffer
        B_buf = B_region.buffer
        C_buf = C_region.buffer

        clear_accum = self.clear_accum
        assert block_K >= micro_size_k, f"block_K ({block_K}) must be >= micro_size_k ({micro_size_k})"

        assert is_full_region(C_region), "Fragment output C must be a full region"

        if self.is_blockscaled:
            if not self.is_gemm_ss():
                raise ValueError("T.mma_gemm_blockscaled supports shared-memory A/B operands only")
            annotations = getattr(self.gemm_node, "annotations", {})
            sf_a_granularity_k = annotations.get("sf_a_granularity_k")
            sf_b_granularity_k = annotations.get("sf_b_granularity_k")
            micro_pipeline = annotations.get("micro_pipeline")
            sf_layout = annotations.get("sf_layout", "rowmajor")
            if sf_layout not in ("rowmajor", "cutlass_128x4"):
                raise ValueError(f"Unsupported SM120 scale layout: {sf_layout}")
            if sf_a_granularity_k is None or sf_b_granularity_k is None:
                raise ValueError("Block-scaled MMA GEMM requires sf_a_granularity_k and sf_b_granularity_k")

            if micro_pipeline == "b_pingpong_scale_prefetch":

                @T.prim_func
                def _gemm_ss_blockscaled_b_pingpong_scale_prefetch() -> None:
                    A_local = T.alloc_local((warp_rows * local_size_a), a_dtype)
                    B_local_0 = T.alloc_local((local_size_b), b_dtype)
                    B_local_1 = T.alloc_local((local_size_b), b_dtype)
                    SFA_local = T.alloc_local((warp_rows), "uint32")
                    SFB_local = T.alloc_local((warp_cols), "uint32")
                    SFB_rep_local = T.alloc_local((warp_cols), "uint32")
                    if clear_accum:
                        T.clear(C_buf)
                    for ki in T.serial(0, (block_K // micro_size_k)):
                        mma_emitter.ldmatrix_a(A_local, A_region, ki)
                        mma_emitter.ldscale_fragment(
                            SFA_local,
                            SFB_local,
                            SFB_rep_local,
                            self.SFARegion,
                            self.SFBRegion,
                            ki=ki,
                            k_start=self.sf_k_start,
                            sf_a_granularity_k=int(sf_a_granularity_k),
                            sf_b_granularity_k=int(sf_b_granularity_k),
                        )
                        mma_emitter.ldmatrix_b_atom(B_local_0, B_region, ki, 0)
                        if warp_cols > 1:
                            mma_emitter.ldmatrix_b_atom(B_local_1, B_region, ki, 1)
                        for j in T.unroll(warp_cols):
                            if j % 2 == 0:
                                for i in T.unroll(warp_rows):
                                    mma_emitter.mma_atom_with_scale_fragments(
                                        A_local,
                                        B_local_0,
                                        C_buf,
                                        SFA_local,
                                        SFB_local,
                                        SFB_rep_local,
                                        i,
                                        j,
                                    )
                            else:
                                for i in T.unroll(warp_rows):
                                    mma_emitter.mma_atom_with_scale_fragments(
                                        A_local,
                                        B_local_1,
                                        C_buf,
                                        SFA_local,
                                        SFB_local,
                                        SFB_rep_local,
                                        i,
                                        j,
                                    )
                            next_j = j + 2
                            if next_j < warp_cols:
                                if next_j % 2 == 0:
                                    mma_emitter.ldmatrix_b_atom(B_local_0, B_region, ki, next_j)
                                else:
                                    mma_emitter.ldmatrix_b_atom(B_local_1, B_region, ki, next_j)

                return _Simplify(_gemm_ss_blockscaled_b_pingpong_scale_prefetch, inline_let=True)

            if micro_pipeline == "b_pingpong":

                @T.prim_func
                def _gemm_ss_blockscaled_b_pingpong() -> None:
                    A_local = T.alloc_local((warp_rows * local_size_a), a_dtype)
                    B_local_0 = T.alloc_local((local_size_b), b_dtype)
                    B_local_1 = T.alloc_local((local_size_b), b_dtype)
                    if clear_accum:
                        T.clear(C_buf)
                    for ki in T.serial(0, (block_K // micro_size_k)):
                        mma_emitter.ldmatrix_a(A_local, A_region, ki)
                        mma_emitter.ldmatrix_b_atom(B_local_0, B_region, ki, 0)
                        if warp_cols > 1:
                            mma_emitter.ldmatrix_b_atom(B_local_1, B_region, ki, 1)
                        for j in T.unroll(warp_cols):
                            if j % 2 == 0:
                                for i in T.unroll(warp_rows):
                                    mma_emitter.mma_atom(
                                        A_local,
                                        B_local_0,
                                        C_buf,
                                        self.SFARegion,
                                        self.SFBRegion,
                                        i,
                                        j,
                                        ki=ki,
                                        k_start=self.sf_k_start,
                                        sf_a_granularity_k=int(sf_a_granularity_k),
                                        sf_b_granularity_k=int(sf_b_granularity_k),
                                    )
                            else:
                                for i in T.unroll(warp_rows):
                                    mma_emitter.mma_atom(
                                        A_local,
                                        B_local_1,
                                        C_buf,
                                        self.SFARegion,
                                        self.SFBRegion,
                                        i,
                                        j,
                                        ki=ki,
                                        k_start=self.sf_k_start,
                                        sf_a_granularity_k=int(sf_a_granularity_k),
                                        sf_b_granularity_k=int(sf_b_granularity_k),
                                    )
                            next_j = j + 2
                            if next_j < warp_cols:
                                if next_j % 2 == 0:
                                    mma_emitter.ldmatrix_b_atom(B_local_0, B_region, ki, next_j)
                                else:
                                    mma_emitter.ldmatrix_b_atom(B_local_1, B_region, ki, next_j)

                return _Simplify(_gemm_ss_blockscaled_b_pingpong, inline_let=True)

            if micro_pipeline == "k_static_b_pingpong_scale_stream":
                num_k_blocks = int(block_K // micro_size_k)
                if num_k_blocks != 4:
                    raise ValueError(
                        "micro_pipeline='k_static_b_pingpong_scale_stream' currently requires block_K / micro_size_k == 4"
                    )
                if int(warp_rows) != 1 or int(warp_cols) != 8:
                    raise ValueError(
                        "micro_pipeline='k_static_b_pingpong_scale_stream' currently targets warp_rows=1, warp_cols=8"
                    )

                @T.prim_func
                def _gemm_ss_blockscaled_k_static_b_pingpong_scale_stream() -> None:
                    A_local_0 = T.alloc_local((warp_rows * local_size_a), a_dtype)
                    A_local_1 = T.alloc_local((warp_rows * local_size_a), a_dtype)
                    B_local_0 = T.alloc_local((local_size_b), b_dtype)
                    B_local_1 = T.alloc_local((local_size_b), b_dtype)
                    SFA_local_0 = T.alloc_local((warp_rows), "uint32")
                    SFA_local_1 = T.alloc_local((warp_rows), "uint32")
                    SFB_local_0 = T.alloc_local((warp_cols), "uint32")
                    SFB_local_1 = T.alloc_local((warp_cols), "uint32")
                    SFB_rep_local_0 = T.alloc_local((warp_cols), "uint32")
                    SFB_rep_local_1 = T.alloc_local((warp_cols), "uint32")
                    if clear_accum:
                        T.clear(C_buf)

                    mma_emitter.ldmatrix_a(A_local_0, A_region, 0)
                    mma_emitter.ldscale_fragment(
                        SFA_local_0,
                        SFB_local_0,
                        SFB_rep_local_0,
                        self.SFARegion,
                        self.SFBRegion,
                        ki=0,
                        k_start=self.sf_k_start,
                        sf_a_granularity_k=int(sf_a_granularity_k),
                        sf_b_granularity_k=int(sf_b_granularity_k),
                        sf_layout=sf_layout,
                    )
                    mma_emitter.ldmatrix_a(A_local_1, A_region, 1)
                    mma_emitter.ldscale_fragment(
                        SFA_local_1,
                        SFB_local_1,
                        SFB_rep_local_1,
                        self.SFARegion,
                        self.SFBRegion,
                        ki=1,
                        k_start=self.sf_k_start,
                        sf_a_granularity_k=int(sf_a_granularity_k),
                        sf_b_granularity_k=int(sf_b_granularity_k),
                        sf_layout=sf_layout,
                    )

                    mma_emitter.ldmatrix_b_atom(B_local_0, B_region, 0, 0)
                    mma_emitter.ldmatrix_b_atom(B_local_1, B_region, 0, 1)
                    mma_emitter.mma_atom_with_scale_fragments(
                        A_local_0, B_local_0, C_buf, SFA_local_0, SFB_local_0, SFB_rep_local_0, 0, 0
                    )
                    mma_emitter.ldmatrix_b_atom(B_local_0, B_region, 0, 2)
                    mma_emitter.mma_atom_with_scale_fragments(
                        A_local_0, B_local_1, C_buf, SFA_local_0, SFB_local_0, SFB_rep_local_0, 0, 1
                    )
                    mma_emitter.ldmatrix_b_atom(B_local_1, B_region, 0, 3)
                    mma_emitter.mma_atom_with_scale_fragments(
                        A_local_0, B_local_0, C_buf, SFA_local_0, SFB_local_0, SFB_rep_local_0, 0, 2
                    )
                    mma_emitter.ldmatrix_b_atom(B_local_0, B_region, 0, 4)
                    mma_emitter.mma_atom_with_scale_fragments(
                        A_local_0, B_local_1, C_buf, SFA_local_0, SFB_local_0, SFB_rep_local_0, 0, 3
                    )
                    mma_emitter.ldmatrix_b_atom(B_local_1, B_region, 0, 5)
                    mma_emitter.mma_atom_with_scale_fragments(
                        A_local_0, B_local_0, C_buf, SFA_local_0, SFB_local_0, SFB_rep_local_0, 0, 4
                    )
                    mma_emitter.ldmatrix_b_atom(B_local_0, B_region, 0, 6)
                    mma_emitter.mma_atom_with_scale_fragments(
                        A_local_0, B_local_1, C_buf, SFA_local_0, SFB_local_0, SFB_rep_local_0, 0, 5
                    )
                    mma_emitter.ldmatrix_b_atom(B_local_1, B_region, 0, 7)
                    mma_emitter.mma_atom_with_scale_fragments(
                        A_local_0, B_local_0, C_buf, SFA_local_0, SFB_local_0, SFB_rep_local_0, 0, 6
                    )
                    mma_emitter.mma_atom_with_scale_fragments(
                        A_local_0, B_local_1, C_buf, SFA_local_0, SFB_local_0, SFB_rep_local_0, 0, 7
                    )

                    mma_emitter.ldmatrix_a(A_local_0, A_region, 2)
                    mma_emitter.ldscale_fragment(
                        SFA_local_0,
                        SFB_local_0,
                        SFB_rep_local_0,
                        self.SFARegion,
                        self.SFBRegion,
                        ki=2,
                        k_start=self.sf_k_start,
                        sf_a_granularity_k=int(sf_a_granularity_k),
                        sf_b_granularity_k=int(sf_b_granularity_k),
                        sf_layout=sf_layout,
                    )
                    mma_emitter.ldmatrix_b_atom(B_local_0, B_region, 1, 0)
                    mma_emitter.ldmatrix_b_atom(B_local_1, B_region, 1, 1)
                    mma_emitter.mma_atom_with_scale_fragments(
                        A_local_1, B_local_0, C_buf, SFA_local_1, SFB_local_1, SFB_rep_local_1, 0, 0
                    )
                    mma_emitter.ldmatrix_b_atom(B_local_0, B_region, 1, 2)
                    mma_emitter.mma_atom_with_scale_fragments(
                        A_local_1, B_local_1, C_buf, SFA_local_1, SFB_local_1, SFB_rep_local_1, 0, 1
                    )
                    mma_emitter.ldmatrix_b_atom(B_local_1, B_region, 1, 3)
                    mma_emitter.mma_atom_with_scale_fragments(
                        A_local_1, B_local_0, C_buf, SFA_local_1, SFB_local_1, SFB_rep_local_1, 0, 2
                    )
                    mma_emitter.ldmatrix_b_atom(B_local_0, B_region, 1, 4)
                    mma_emitter.mma_atom_with_scale_fragments(
                        A_local_1, B_local_1, C_buf, SFA_local_1, SFB_local_1, SFB_rep_local_1, 0, 3
                    )
                    mma_emitter.ldmatrix_b_atom(B_local_1, B_region, 1, 5)
                    mma_emitter.mma_atom_with_scale_fragments(
                        A_local_1, B_local_0, C_buf, SFA_local_1, SFB_local_1, SFB_rep_local_1, 0, 4
                    )
                    mma_emitter.ldmatrix_b_atom(B_local_0, B_region, 1, 6)
                    mma_emitter.mma_atom_with_scale_fragments(
                        A_local_1, B_local_1, C_buf, SFA_local_1, SFB_local_1, SFB_rep_local_1, 0, 5
                    )
                    mma_emitter.ldmatrix_b_atom(B_local_1, B_region, 1, 7)
                    mma_emitter.mma_atom_with_scale_fragments(
                        A_local_1, B_local_0, C_buf, SFA_local_1, SFB_local_1, SFB_rep_local_1, 0, 6
                    )
                    mma_emitter.mma_atom_with_scale_fragments(
                        A_local_1, B_local_1, C_buf, SFA_local_1, SFB_local_1, SFB_rep_local_1, 0, 7
                    )

                    mma_emitter.ldmatrix_a(A_local_1, A_region, 3)
                    mma_emitter.ldscale_fragment(
                        SFA_local_1,
                        SFB_local_1,
                        SFB_rep_local_1,
                        self.SFARegion,
                        self.SFBRegion,
                        ki=3,
                        k_start=self.sf_k_start,
                        sf_a_granularity_k=int(sf_a_granularity_k),
                        sf_b_granularity_k=int(sf_b_granularity_k),
                        sf_layout=sf_layout,
                    )
                    mma_emitter.ldmatrix_b_atom(B_local_0, B_region, 2, 0)
                    mma_emitter.ldmatrix_b_atom(B_local_1, B_region, 2, 1)
                    mma_emitter.mma_atom_with_scale_fragments(
                        A_local_0, B_local_0, C_buf, SFA_local_0, SFB_local_0, SFB_rep_local_0, 0, 0
                    )
                    mma_emitter.ldmatrix_b_atom(B_local_0, B_region, 2, 2)
                    mma_emitter.mma_atom_with_scale_fragments(
                        A_local_0, B_local_1, C_buf, SFA_local_0, SFB_local_0, SFB_rep_local_0, 0, 1
                    )
                    mma_emitter.ldmatrix_b_atom(B_local_1, B_region, 2, 3)
                    mma_emitter.mma_atom_with_scale_fragments(
                        A_local_0, B_local_0, C_buf, SFA_local_0, SFB_local_0, SFB_rep_local_0, 0, 2
                    )
                    mma_emitter.ldmatrix_b_atom(B_local_0, B_region, 2, 4)
                    mma_emitter.mma_atom_with_scale_fragments(
                        A_local_0, B_local_1, C_buf, SFA_local_0, SFB_local_0, SFB_rep_local_0, 0, 3
                    )
                    mma_emitter.ldmatrix_b_atom(B_local_1, B_region, 2, 5)
                    mma_emitter.mma_atom_with_scale_fragments(
                        A_local_0, B_local_0, C_buf, SFA_local_0, SFB_local_0, SFB_rep_local_0, 0, 4
                    )
                    mma_emitter.ldmatrix_b_atom(B_local_0, B_region, 2, 6)
                    mma_emitter.mma_atom_with_scale_fragments(
                        A_local_0, B_local_1, C_buf, SFA_local_0, SFB_local_0, SFB_rep_local_0, 0, 5
                    )
                    mma_emitter.ldmatrix_b_atom(B_local_1, B_region, 2, 7)
                    mma_emitter.mma_atom_with_scale_fragments(
                        A_local_0, B_local_0, C_buf, SFA_local_0, SFB_local_0, SFB_rep_local_0, 0, 6
                    )
                    mma_emitter.mma_atom_with_scale_fragments(
                        A_local_0, B_local_1, C_buf, SFA_local_0, SFB_local_0, SFB_rep_local_0, 0, 7
                    )

                    mma_emitter.ldmatrix_b_atom(B_local_0, B_region, 3, 0)
                    mma_emitter.ldmatrix_b_atom(B_local_1, B_region, 3, 1)
                    mma_emitter.mma_atom_with_scale_fragments(
                        A_local_1, B_local_0, C_buf, SFA_local_1, SFB_local_1, SFB_rep_local_1, 0, 0
                    )
                    mma_emitter.ldmatrix_b_atom(B_local_0, B_region, 3, 2)
                    mma_emitter.mma_atom_with_scale_fragments(
                        A_local_1, B_local_1, C_buf, SFA_local_1, SFB_local_1, SFB_rep_local_1, 0, 1
                    )
                    mma_emitter.ldmatrix_b_atom(B_local_1, B_region, 3, 3)
                    mma_emitter.mma_atom_with_scale_fragments(
                        A_local_1, B_local_0, C_buf, SFA_local_1, SFB_local_1, SFB_rep_local_1, 0, 2
                    )
                    mma_emitter.ldmatrix_b_atom(B_local_0, B_region, 3, 4)
                    mma_emitter.mma_atom_with_scale_fragments(
                        A_local_1, B_local_1, C_buf, SFA_local_1, SFB_local_1, SFB_rep_local_1, 0, 3
                    )
                    mma_emitter.ldmatrix_b_atom(B_local_1, B_region, 3, 5)
                    mma_emitter.mma_atom_with_scale_fragments(
                        A_local_1, B_local_0, C_buf, SFA_local_1, SFB_local_1, SFB_rep_local_1, 0, 4
                    )
                    mma_emitter.ldmatrix_b_atom(B_local_0, B_region, 3, 6)
                    mma_emitter.mma_atom_with_scale_fragments(
                        A_local_1, B_local_1, C_buf, SFA_local_1, SFB_local_1, SFB_rep_local_1, 0, 5
                    )
                    mma_emitter.ldmatrix_b_atom(B_local_1, B_region, 3, 7)
                    mma_emitter.mma_atom_with_scale_fragments(
                        A_local_1, B_local_0, C_buf, SFA_local_1, SFB_local_1, SFB_rep_local_1, 0, 6
                    )
                    mma_emitter.mma_atom_with_scale_fragments(
                        A_local_1, B_local_1, C_buf, SFA_local_1, SFB_local_1, SFB_rep_local_1, 0, 7
                    )

                return _Simplify(_gemm_ss_blockscaled_k_static_b_pingpong_scale_stream, inline_let=True)

            if micro_pipeline == "k_static_b_atom_n8_stream":
                num_k_blocks = int(block_K // micro_size_k)
                if num_k_blocks != 4:
                    raise ValueError(
                        "micro_pipeline='k_static_b_atom_n8_stream' currently requires block_K / micro_size_k == 4"
                    )
                if int(warp_cols) != 8:
                    raise ValueError("micro_pipeline='k_static_b_atom_n8_stream' currently targets warp_cols=8")

                def copy_a_scale_kblock(A_frag, SFA_frag, SFB_frag, SFB_rep_frag, k_block):
                    mma_emitter.ldmatrix_a(A_frag, A_region, k_block)
                    mma_emitter.ldscale_fragment(
                        SFA_frag,
                        SFB_frag,
                        SFB_rep_frag,
                        self.SFARegion,
                        self.SFBRegion,
                        ki=k_block,
                        k_start=self.sf_k_start,
                        sf_a_granularity_k=int(sf_a_granularity_k),
                        sf_b_granularity_k=int(sf_b_granularity_k),
                        sf_layout=sf_layout,
                    )

                @T.macro
                def gemm_b_atom_stream(A_frag, B_atom_0, B_atom_1, SFA_frag, SFB_frag, SFB_rep_frag, k_block):
                    mma_emitter.ldmatrix_b_atom(B_atom_0, B_region, k_block, 0)
                    mma_emitter.ldmatrix_b_atom(B_atom_1, B_region, k_block, 1)
                    for j in T.unroll(warp_cols):
                        if j % 2 == 0:
                            mma_emitter.mma_b_atom_n8_serpentine_with_prefetched_scales(
                                A_frag,
                                B_atom_0,
                                C_buf,
                                SFA_frag,
                                SFB_frag,
                                SFB_rep_frag,
                                j,
                            )
                        else:
                            mma_emitter.mma_b_atom_n8_serpentine_with_prefetched_scales(
                                A_frag,
                                B_atom_1,
                                C_buf,
                                SFA_frag,
                                SFB_frag,
                                SFB_rep_frag,
                                j,
                            )
                        next_j = j + 2
                        if next_j < warp_cols:
                            if next_j % 2 == 0:
                                mma_emitter.ldmatrix_b_atom(B_atom_0, B_region, k_block, next_j)
                            else:
                                mma_emitter.ldmatrix_b_atom(B_atom_1, B_region, k_block, next_j)

                @T.prim_func
                def _gemm_ss_blockscaled_k_static_b_atom_n8_stream() -> None:
                    A_local_0 = T.alloc_local((warp_rows * local_size_a), a_dtype)
                    A_local_1 = T.alloc_local((warp_rows * local_size_a), a_dtype)
                    B_atom_0 = T.alloc_local((local_size_b), b_dtype)
                    B_atom_1 = T.alloc_local((local_size_b), b_dtype)
                    SFA_local_0 = T.alloc_local((warp_rows), "uint32")
                    SFA_local_1 = T.alloc_local((warp_rows), "uint32")
                    SFB_local_0 = T.alloc_local((warp_cols), "uint32")
                    SFB_local_1 = T.alloc_local((warp_cols), "uint32")
                    SFB_rep_local_0 = T.alloc_local((warp_cols), "uint32")
                    SFB_rep_local_1 = T.alloc_local((warp_cols), "uint32")
                    if clear_accum:
                        T.clear(C_buf)

                    copy_a_scale_kblock(A_local_0, SFA_local_0, SFB_local_0, SFB_rep_local_0, 0)
                    copy_a_scale_kblock(A_local_1, SFA_local_1, SFB_local_1, SFB_rep_local_1, 1)
                    gemm_b_atom_stream(A_local_0, B_atom_0, B_atom_1, SFA_local_0, SFB_local_0, SFB_rep_local_0, 0)
                    copy_a_scale_kblock(A_local_0, SFA_local_0, SFB_local_0, SFB_rep_local_0, 2)
                    gemm_b_atom_stream(A_local_1, B_atom_0, B_atom_1, SFA_local_1, SFB_local_1, SFB_rep_local_1, 1)
                    copy_a_scale_kblock(A_local_1, SFA_local_1, SFB_local_1, SFB_rep_local_1, 3)
                    gemm_b_atom_stream(A_local_0, B_atom_0, B_atom_1, SFA_local_0, SFB_local_0, SFB_rep_local_0, 2)
                    gemm_b_atom_stream(A_local_1, B_atom_0, B_atom_1, SFA_local_1, SFB_local_1, SFB_rep_local_1, 3)

                return _Simplify(_gemm_ss_blockscaled_k_static_b_atom_n8_stream, inline_let=True)

            if micro_pipeline == "k_static_b_atom_n8_stream_vecsf":
                num_k_blocks = int(block_K // micro_size_k)
                if num_k_blocks != 4:
                    raise ValueError(
                        "micro_pipeline='k_static_b_atom_n8_stream_vecsf' currently requires block_K / micro_size_k == 4"
                    )
                if int(warp_cols) != 8:
                    raise ValueError("micro_pipeline='k_static_b_atom_n8_stream_vecsf' currently targets warp_cols=8")

                @T.macro
                def gemm_b_atom_stream_vecsf(A_frag, B_atom_0, B_atom_1, SFA_pack, SFB_pack, SFB_rep_pack, k_block):
                    mma_emitter.ldmatrix_b_atom(B_atom_0, B_region, k_block, 0)
                    mma_emitter.ldmatrix_b_atom(B_atom_1, B_region, k_block, 1)
                    for j in T.unroll(warp_cols):
                        if j % 2 == 0:
                            mma_emitter.mma_b_atom_n8_serpentine_with_scale_pack(
                                A_frag,
                                B_atom_0,
                                C_buf,
                                SFA_pack,
                                SFB_pack,
                                SFB_rep_pack,
                                k_block,
                                j,
                                num_k_blocks,
                            )
                        else:
                            mma_emitter.mma_b_atom_n8_serpentine_with_scale_pack(
                                A_frag,
                                B_atom_1,
                                C_buf,
                                SFA_pack,
                                SFB_pack,
                                SFB_rep_pack,
                                k_block,
                                j,
                                num_k_blocks,
                            )
                        next_j = j + 2
                        if next_j < warp_cols:
                            if next_j % 2 == 0:
                                mma_emitter.ldmatrix_b_atom(B_atom_0, B_region, k_block, next_j)
                            else:
                                mma_emitter.ldmatrix_b_atom(B_atom_1, B_region, k_block, next_j)

                @T.prim_func
                def _gemm_ss_blockscaled_k_static_b_atom_n8_stream_vecsf() -> None:
                    A_local_0 = T.alloc_local((warp_rows * local_size_a), a_dtype)
                    A_local_1 = T.alloc_local((warp_rows * local_size_a), a_dtype)
                    B_atom_0 = T.alloc_local((local_size_b), b_dtype)
                    B_atom_1 = T.alloc_local((local_size_b), b_dtype)
                    SFA_pack = T.alloc_local((warp_rows * num_k_blocks), "uint32")
                    SFB_pack = T.alloc_local((warp_cols * num_k_blocks), "uint32")
                    SFB_rep_pack = T.alloc_local((warp_cols * num_k_blocks), "uint32")
                    if clear_accum:
                        T.clear(C_buf)

                    mma_emitter.ldscale_fragment_kpack(
                        SFA_pack,
                        SFB_pack,
                        SFB_rep_pack,
                        self.SFARegion,
                        self.SFBRegion,
                        num_k_blocks=num_k_blocks,
                        k_start=self.sf_k_start,
                        sf_a_granularity_k=int(sf_a_granularity_k),
                        sf_b_granularity_k=int(sf_b_granularity_k),
                        sf_layout=sf_layout,
                    )
                    mma_emitter.ldmatrix_a(A_local_0, A_region, 0)
                    mma_emitter.ldmatrix_a(A_local_1, A_region, 1)
                    gemm_b_atom_stream_vecsf(A_local_0, B_atom_0, B_atom_1, SFA_pack, SFB_pack, SFB_rep_pack, 0)
                    mma_emitter.ldmatrix_a(A_local_0, A_region, 2)
                    gemm_b_atom_stream_vecsf(A_local_1, B_atom_0, B_atom_1, SFA_pack, SFB_pack, SFB_rep_pack, 1)
                    mma_emitter.ldmatrix_a(A_local_1, A_region, 3)
                    gemm_b_atom_stream_vecsf(A_local_0, B_atom_0, B_atom_1, SFA_pack, SFB_pack, SFB_rep_pack, 2)
                    gemm_b_atom_stream_vecsf(A_local_1, B_atom_0, B_atom_1, SFA_pack, SFB_pack, SFB_rep_pack, 3)

                return _Simplify(_gemm_ss_blockscaled_k_static_b_atom_n8_stream_vecsf, inline_let=True)

            if micro_pipeline == "k_static_full_b_atom_scale_stream":
                num_k_blocks = int(block_K // micro_size_k)
                if num_k_blocks != 4:
                    raise ValueError(
                        "micro_pipeline='k_static_full_b_atom_scale_stream' currently requires block_K / micro_size_k == 4"
                    )

                @T.prim_func
                def _gemm_ss_blockscaled_k_static_full_b_atom_scale_stream() -> None:
                    A_local_0 = T.alloc_local((warp_rows * local_size_a), a_dtype)
                    A_local_1 = T.alloc_local((warp_rows * local_size_a), a_dtype)
                    B_local_0 = T.alloc_local((warp_cols * local_size_b), b_dtype)
                    B_local_1 = T.alloc_local((warp_cols * local_size_b), b_dtype)
                    SFA_local_0 = T.alloc_local((warp_rows), "uint32")
                    SFA_local_1 = T.alloc_local((warp_rows), "uint32")
                    SFB_local_0 = T.alloc_local((warp_cols), "uint32")
                    SFB_local_1 = T.alloc_local((warp_cols), "uint32")
                    SFB_rep_local_0 = T.alloc_local((warp_cols), "uint32")
                    SFB_rep_local_1 = T.alloc_local((warp_cols), "uint32")
                    if clear_accum:
                        T.clear(C_buf)

                    mma_emitter.ldmatrix_a(A_local_0, A_region, 0)
                    mma_emitter.ldmatrix_b(B_local_0, B_region, 0)
                    mma_emitter.ldscale_fragment(
                        SFA_local_0,
                        SFB_local_0,
                        SFB_rep_local_0,
                        self.SFARegion,
                        self.SFBRegion,
                        ki=0,
                        k_start=self.sf_k_start,
                        sf_a_granularity_k=int(sf_a_granularity_k),
                        sf_b_granularity_k=int(sf_b_granularity_k),
                        sf_layout=sf_layout,
                    )
                    mma_emitter.ldmatrix_a(A_local_1, A_region, 1)
                    mma_emitter.ldmatrix_b(B_local_1, B_region, 1)
                    mma_emitter.ldscale_fragment(
                        SFA_local_1,
                        SFB_local_1,
                        SFB_rep_local_1,
                        self.SFARegion,
                        self.SFBRegion,
                        ki=1,
                        k_start=self.sf_k_start,
                        sf_a_granularity_k=int(sf_a_granularity_k),
                        sf_b_granularity_k=int(sf_b_granularity_k),
                        sf_layout=sf_layout,
                    )
                    for i in T.unroll(warp_rows):
                        for j in T.unroll(warp_cols):
                            mma_emitter.mma_full_b_atom_with_scale_fragments(
                                A_local_0,
                                B_local_0,
                                C_buf,
                                SFA_local_0,
                                SFB_local_0,
                                SFB_rep_local_0,
                                i,
                                j,
                            )

                    mma_emitter.ldmatrix_a(A_local_0, A_region, 2)
                    mma_emitter.ldmatrix_b(B_local_0, B_region, 2)
                    mma_emitter.ldscale_fragment(
                        SFA_local_0,
                        SFB_local_0,
                        SFB_rep_local_0,
                        self.SFARegion,
                        self.SFBRegion,
                        ki=2,
                        k_start=self.sf_k_start,
                        sf_a_granularity_k=int(sf_a_granularity_k),
                        sf_b_granularity_k=int(sf_b_granularity_k),
                        sf_layout=sf_layout,
                    )
                    for i in T.unroll(warp_rows):
                        for j in T.unroll(warp_cols):
                            mma_emitter.mma_full_b_atom_with_scale_fragments(
                                A_local_1,
                                B_local_1,
                                C_buf,
                                SFA_local_1,
                                SFB_local_1,
                                SFB_rep_local_1,
                                i,
                                j,
                            )

                    mma_emitter.ldmatrix_a(A_local_1, A_region, 3)
                    mma_emitter.ldmatrix_b(B_local_1, B_region, 3)
                    mma_emitter.ldscale_fragment(
                        SFA_local_1,
                        SFB_local_1,
                        SFB_rep_local_1,
                        self.SFARegion,
                        self.SFBRegion,
                        ki=3,
                        k_start=self.sf_k_start,
                        sf_a_granularity_k=int(sf_a_granularity_k),
                        sf_b_granularity_k=int(sf_b_granularity_k),
                        sf_layout=sf_layout,
                    )
                    for i in T.unroll(warp_rows):
                        for j in T.unroll(warp_cols):
                            mma_emitter.mma_full_b_atom_with_scale_fragments(
                                A_local_0,
                                B_local_0,
                                C_buf,
                                SFA_local_0,
                                SFB_local_0,
                                SFB_rep_local_0,
                                i,
                                j,
                            )
                    for i in T.unroll(warp_rows):
                        for j in T.unroll(warp_cols):
                            mma_emitter.mma_full_b_atom_with_scale_fragments(
                                A_local_1,
                                B_local_1,
                                C_buf,
                                SFA_local_1,
                                SFB_local_1,
                                SFB_rep_local_1,
                                i,
                                j,
                            )

                return _Simplify(_gemm_ss_blockscaled_k_static_full_b_atom_scale_stream, inline_let=True)

            if micro_pipeline in (
                "sm120_kblock_fulltile",
                "sm120_kblock_fulltile_selector_probe",
                "sm120_kblock_fulltile_b_owner_probe",
                "sm120_kblock_fulltile_ab_owner_probe",
            ):
                num_k_blocks = int(block_K // micro_size_k)
                if num_k_blocks != 4:
                    raise ValueError(
                        f"micro_pipeline={micro_pipeline!r} currently requires block_K / micro_size_k == 4"
                    )
                if int(warp_rows) != 4 or int(warp_cols) != 2:
                    raise ValueError(
                        f"micro_pipeline={micro_pipeline!r} currently targets the balanced "
                        "128x128 full-tile consumer shape with warp_rows=4 and warp_cols=2"
                    )

                def copy_kblock(A_frag, B_frag, SFA_frag, SFB_frag, SFB_rep_frag, k_block):
                    mma_emitter.ldmatrix_a(A_frag, A_region, k_block)
                    mma_emitter.ldmatrix_b(B_frag, B_region, k_block)
                    if micro_pipeline == "sm120_kblock_fulltile_ab_owner_probe":
                        mma_emitter.ldscale_fragment_ab_owner(
                            SFA_frag,
                            SFB_frag,
                            self.SFARegion,
                            self.SFBRegion,
                            ki=k_block,
                            k_start=self.sf_k_start,
                            sf_a_granularity_k=int(sf_a_granularity_k),
                            sf_b_granularity_k=int(sf_b_granularity_k),
                            sf_layout=sf_layout,
                        )
                    elif micro_pipeline == "sm120_kblock_fulltile_b_owner_probe":
                        mma_emitter.ldscale_fragment_b_owner(
                            SFA_frag,
                            SFB_frag,
                            self.SFARegion,
                            self.SFBRegion,
                            ki=k_block,
                            k_start=self.sf_k_start,
                            sf_a_granularity_k=int(sf_a_granularity_k),
                            sf_b_granularity_k=int(sf_b_granularity_k),
                            sf_layout=sf_layout,
                        )
                    else:
                        mma_emitter.ldscale_fragment(
                            SFA_frag,
                            SFB_frag,
                            SFB_rep_frag,
                            self.SFARegion,
                            self.SFBRegion,
                            ki=k_block,
                            k_start=self.sf_k_start,
                            sf_a_granularity_k=int(sf_a_granularity_k),
                            sf_b_granularity_k=int(sf_b_granularity_k),
                            sf_layout=sf_layout,
                        )

                def gemm_kblock(A_frag, B_frag, SFA_frag, SFB_frag, SFB_rep_frag):
                    if micro_pipeline == "sm120_kblock_fulltile_ab_owner_probe":
                        mma_emitter.mma_with_prefetched_scales_ab_owner(
                            A_frag,
                            B_frag,
                            C_buf,
                            SFA_frag,
                            SFB_frag,
                        )
                    elif micro_pipeline == "sm120_kblock_fulltile_b_owner_probe":
                        mma_emitter.mma_with_prefetched_scales_b_owner(
                            A_frag,
                            B_frag,
                            C_buf,
                            SFA_frag,
                            SFB_frag,
                        )
                    elif micro_pipeline == "sm120_kblock_fulltile_selector_probe":
                        mma_emitter.mma_with_prefetched_scales_selector_probe(
                            A_frag,
                            B_frag,
                            C_buf,
                            SFA_frag,
                            SFB_frag,
                            SFB_rep_frag,
                        )
                    else:
                        mma_emitter.mma_with_scale_fragments(
                            A_frag,
                            B_frag,
                            C_buf,
                            SFA_frag,
                            SFB_frag,
                            SFB_rep_frag,
                        )

                @T.prim_func
                def _gemm_ss_blockscaled_sm120_kblock_fulltile() -> None:
                    A_frag_0 = T.alloc_local((warp_rows * local_size_a), a_dtype)
                    A_frag_1 = T.alloc_local((warp_rows * local_size_a), a_dtype)
                    B_frag_0 = T.alloc_local((warp_cols * local_size_b), b_dtype)
                    B_frag_1 = T.alloc_local((warp_cols * local_size_b), b_dtype)
                    if micro_pipeline == "sm120_kblock_fulltile_ab_owner_probe":
                        SFA_frag_0 = T.alloc_local((2), "uint32")
                        SFA_frag_1 = T.alloc_local((2), "uint32")
                        SFB_frag_0 = T.alloc_local((1), "uint32")
                        SFB_frag_1 = T.alloc_local((1), "uint32")
                        SFB_rep_frag_0 = T.alloc_local((1), "uint32")
                        SFB_rep_frag_1 = T.alloc_local((1), "uint32")
                    elif micro_pipeline == "sm120_kblock_fulltile_b_owner_probe":
                        SFA_frag_0 = T.alloc_local((warp_rows), "uint32")
                        SFA_frag_1 = T.alloc_local((warp_rows), "uint32")
                        SFB_frag_0 = T.alloc_local((1), "uint32")
                        SFB_frag_1 = T.alloc_local((1), "uint32")
                        SFB_rep_frag_0 = T.alloc_local((1), "uint32")
                        SFB_rep_frag_1 = T.alloc_local((1), "uint32")
                    else:
                        SFA_frag_0 = T.alloc_local((warp_rows), "uint32")
                        SFA_frag_1 = T.alloc_local((warp_rows), "uint32")
                        SFB_frag_0 = T.alloc_local((warp_cols), "uint32")
                        SFB_frag_1 = T.alloc_local((warp_cols), "uint32")
                        SFB_rep_frag_0 = T.alloc_local((warp_cols), "uint32")
                        SFB_rep_frag_1 = T.alloc_local((warp_cols), "uint32")
                    if clear_accum:
                        T.clear(C_buf)

                    copy_kblock(A_frag_0, B_frag_0, SFA_frag_0, SFB_frag_0, SFB_rep_frag_0, 0)
                    copy_kblock(A_frag_1, B_frag_1, SFA_frag_1, SFB_frag_1, SFB_rep_frag_1, 1)
                    gemm_kblock(A_frag_0, B_frag_0, SFA_frag_0, SFB_frag_0, SFB_rep_frag_0)
                    copy_kblock(A_frag_0, B_frag_0, SFA_frag_0, SFB_frag_0, SFB_rep_frag_0, 2)
                    gemm_kblock(A_frag_1, B_frag_1, SFA_frag_1, SFB_frag_1, SFB_rep_frag_1)
                    copy_kblock(A_frag_1, B_frag_1, SFA_frag_1, SFB_frag_1, SFB_rep_frag_1, 3)
                    gemm_kblock(A_frag_0, B_frag_0, SFA_frag_0, SFB_frag_0, SFB_rep_frag_0)
                    gemm_kblock(A_frag_1, B_frag_1, SFA_frag_1, SFB_frag_1, SFB_rep_frag_1)

                return _Simplify(_gemm_ss_blockscaled_sm120_kblock_fulltile, inline_let=True)

            if micro_pipeline == "sm120_kblock_fulltile_ab_owner_wide_probe":
                num_k_blocks = int(block_K // micro_size_k)
                if num_k_blocks != 4:
                    raise ValueError(
                        "micro_pipeline='sm120_kblock_fulltile_ab_owner_wide_probe' "
                        "currently requires block_K / micro_size_k == 4"
                    )
                if int(warp_rows) != 4 or int(warp_cols) != 4:
                    raise ValueError(
                        "micro_pipeline='sm120_kblock_fulltile_ab_owner_wide_probe' "
                        "currently targets warp_rows=4, warp_cols=4"
                    )

                def copy_kblock(A_frag, B_frag, SFA_owner_frag, SFB_owner_frag, k_block):
                    mma_emitter.ldmatrix_a(A_frag, A_region, k_block)
                    mma_emitter.ldmatrix_b(B_frag, B_region, k_block)
                    mma_emitter.ldscale_fragment_ab_owner_wide(
                        SFA_owner_frag,
                        SFB_owner_frag,
                        self.SFARegion,
                        self.SFBRegion,
                        ki=k_block,
                        k_start=self.sf_k_start,
                        sf_a_granularity_k=int(sf_a_granularity_k),
                        sf_b_granularity_k=int(sf_b_granularity_k),
                        sf_layout=sf_layout,
                    )

                def gemm_kblock(A_frag, B_frag, SFA_owner_frag, SFB_owner_frag):
                    mma_emitter.mma_with_prefetched_scales_ab_owner_wide(
                        A_frag,
                        B_frag,
                        C_buf,
                        SFA_owner_frag,
                        SFB_owner_frag,
                    )

                @T.prim_func
                def _gemm_ss_blockscaled_sm120_kblock_fulltile_ab_owner_wide() -> None:
                    A_frag_0 = T.alloc_local((warp_rows * local_size_a), a_dtype)
                    A_frag_1 = T.alloc_local((warp_rows * local_size_a), a_dtype)
                    B_frag_0 = T.alloc_local((warp_cols * local_size_b), b_dtype)
                    B_frag_1 = T.alloc_local((warp_cols * local_size_b), b_dtype)
                    SFA_owner_frag_0 = T.alloc_local((2), "uint32")
                    SFA_owner_frag_1 = T.alloc_local((2), "uint32")
                    SFB_owner_frag_0 = T.alloc_local((2), "uint32")
                    SFB_owner_frag_1 = T.alloc_local((2), "uint32")
                    if clear_accum:
                        T.clear(C_buf)

                    copy_kblock(A_frag_0, B_frag_0, SFA_owner_frag_0, SFB_owner_frag_0, 0)
                    copy_kblock(A_frag_1, B_frag_1, SFA_owner_frag_1, SFB_owner_frag_1, 1)
                    gemm_kblock(A_frag_0, B_frag_0, SFA_owner_frag_0, SFB_owner_frag_0)
                    copy_kblock(A_frag_0, B_frag_0, SFA_owner_frag_0, SFB_owner_frag_0, 2)
                    gemm_kblock(A_frag_1, B_frag_1, SFA_owner_frag_1, SFB_owner_frag_1)
                    copy_kblock(A_frag_1, B_frag_1, SFA_owner_frag_1, SFB_owner_frag_1, 3)
                    gemm_kblock(A_frag_0, B_frag_0, SFA_owner_frag_0, SFB_owner_frag_0)
                    gemm_kblock(A_frag_1, B_frag_1, SFA_owner_frag_1, SFB_owner_frag_1)

                return _Simplify(_gemm_ss_blockscaled_sm120_kblock_fulltile_ab_owner_wide, inline_let=True)

            if micro_pipeline in (
                "sm120_backend_kblock_fulltile_ab_owner_wide",
                "sm120_backend_kblock_fulltile_afull_bpanel_owner_wide",
            ):
                num_k_blocks = int(block_K // micro_size_k)
                if num_k_blocks != 4:
                    raise ValueError(
                        f"micro_pipeline={micro_pipeline!r} "
                        "currently requires block_K / micro_size_k == 4"
                    )
                if int(warp_rows) != 4 or int(warp_cols) != 4:
                    raise ValueError(
                        f"micro_pipeline={micro_pipeline!r} "
                        "currently targets warp_rows=4, warp_cols=4"
                    )
                if sf_layout != "cutlass_128x4":
                    raise ValueError(
                        f"micro_pipeline={micro_pipeline!r} "
                        "currently requires sf_layout='cutlass_128x4'"
                    )
                backend_op = (
                    "afull_bpanel_owner_wide"
                    if micro_pipeline == "sm120_backend_kblock_fulltile_afull_bpanel_owner_wide"
                    else "ab_owner_wide"
                )

                @T.prim_func
                def _gemm_ss_blockscaled_sm120_backend_kblock_fulltile_ab_owner_wide() -> None:
                    if clear_accum:
                        T.clear(C_buf)

                    mma_emitter.mma_backend_kblock_fulltile_ab_owner_wide(
                        A_region,
                        B_region,
                        C_buf,
                        self.SFARegion,
                        self.SFBRegion,
                        ki=0,
                        k_start=self.sf_k_start,
                        sf_a_granularity_k=int(sf_a_granularity_k),
                        sf_b_granularity_k=int(sf_b_granularity_k),
                        sf_layout=sf_layout,
                        backend_op=backend_op,
                    )
                    mma_emitter.mma_backend_kblock_fulltile_ab_owner_wide(
                        A_region,
                        B_region,
                        C_buf,
                        self.SFARegion,
                        self.SFBRegion,
                        ki=1,
                        k_start=self.sf_k_start,
                        sf_a_granularity_k=int(sf_a_granularity_k),
                        sf_b_granularity_k=int(sf_b_granularity_k),
                        sf_layout=sf_layout,
                        backend_op=backend_op,
                    )
                    mma_emitter.mma_backend_kblock_fulltile_ab_owner_wide(
                        A_region,
                        B_region,
                        C_buf,
                        self.SFARegion,
                        self.SFBRegion,
                        ki=2,
                        k_start=self.sf_k_start,
                        sf_a_granularity_k=int(sf_a_granularity_k),
                        sf_b_granularity_k=int(sf_b_granularity_k),
                        sf_layout=sf_layout,
                        backend_op=backend_op,
                    )
                    mma_emitter.mma_backend_kblock_fulltile_ab_owner_wide(
                        A_region,
                        B_region,
                        C_buf,
                        self.SFARegion,
                        self.SFBRegion,
                        ki=3,
                        k_start=self.sf_k_start,
                        sf_a_granularity_k=int(sf_a_granularity_k),
                        sf_b_granularity_k=int(sf_b_granularity_k),
                        sf_layout=sf_layout,
                        backend_op=backend_op,
                    )

                return _Simplify(_gemm_ss_blockscaled_sm120_backend_kblock_fulltile_ab_owner_wide, inline_let=True)

            if micro_pipeline == "sm120_backend_kblock_fulltile":
                num_k_blocks = int(block_K // micro_size_k)
                if num_k_blocks != 4:
                    raise ValueError(
                        "micro_pipeline='sm120_backend_kblock_fulltile' currently requires block_K / micro_size_k == 4"
                    )
                if int(warp_rows) != 4 or int(warp_cols) != 4:
                    raise ValueError(
                        "micro_pipeline='sm120_backend_kblock_fulltile' currently targets "
                        "warp_rows=4, warp_cols=4"
                    )

                @T.prim_func
                def _gemm_ss_blockscaled_sm120_backend_kblock_fulltile() -> None:
                    if clear_accum:
                        T.clear(C_buf)

                    mma_emitter.mma_backend_kblock_fulltile(
                        A_region,
                        B_region,
                        C_buf,
                        self.SFARegion,
                        self.SFBRegion,
                        ki=0,
                        k_start=self.sf_k_start,
                        sf_a_granularity_k=int(sf_a_granularity_k),
                        sf_b_granularity_k=int(sf_b_granularity_k),
                        sf_layout=sf_layout,
                    )
                    mma_emitter.mma_backend_kblock_fulltile(
                        A_region,
                        B_region,
                        C_buf,
                        self.SFARegion,
                        self.SFBRegion,
                        ki=1,
                        k_start=self.sf_k_start,
                        sf_a_granularity_k=int(sf_a_granularity_k),
                        sf_b_granularity_k=int(sf_b_granularity_k),
                        sf_layout=sf_layout,
                    )
                    mma_emitter.mma_backend_kblock_fulltile(
                        A_region,
                        B_region,
                        C_buf,
                        self.SFARegion,
                        self.SFBRegion,
                        ki=2,
                        k_start=self.sf_k_start,
                        sf_a_granularity_k=int(sf_a_granularity_k),
                        sf_b_granularity_k=int(sf_b_granularity_k),
                        sf_layout=sf_layout,
                    )
                    mma_emitter.mma_backend_kblock_fulltile(
                        A_region,
                        B_region,
                        C_buf,
                        self.SFARegion,
                        self.SFBRegion,
                        ki=3,
                        k_start=self.sf_k_start,
                        sf_a_granularity_k=int(sf_a_granularity_k),
                        sf_b_granularity_k=int(sf_b_granularity_k),
                        sf_layout=sf_layout,
                    )

                return _Simplify(_gemm_ss_blockscaled_sm120_backend_kblock_fulltile, inline_let=True)

            if micro_pipeline == "sm120_cute_consumer_bridge_skeleton":
                num_k_blocks = int(block_K // micro_size_k)
                if num_k_blocks != 4:
                    raise ValueError(
                        "micro_pipeline='sm120_cute_consumer_bridge_skeleton' currently requires "
                        "block_K / micro_size_k == 4"
                    )
                if int(warp_rows) != 4 or int(warp_cols) != 4:
                    raise ValueError(
                        "micro_pipeline='sm120_cute_consumer_bridge_skeleton' currently targets "
                        "warp_rows=4, warp_cols=4"
                    )

                @T.prim_func
                def _gemm_ss_blockscaled_sm120_cute_consumer_bridge_skeleton() -> None:
                    if clear_accum:
                        T.clear(C_buf)

                    mma_emitter.mma_backend_cute_consumer_bridge(
                        A_region,
                        B_region,
                        C_buf,
                        self.SFARegion,
                        self.SFBRegion,
                        ki=0,
                    )
                    mma_emitter.mma_backend_cute_consumer_bridge(
                        A_region,
                        B_region,
                        C_buf,
                        self.SFARegion,
                        self.SFBRegion,
                        ki=1,
                    )
                    mma_emitter.mma_backend_cute_consumer_bridge(
                        A_region,
                        B_region,
                        C_buf,
                        self.SFARegion,
                        self.SFBRegion,
                        ki=2,
                    )
                    mma_emitter.mma_backend_cute_consumer_bridge(
                        A_region,
                        B_region,
                        C_buf,
                        self.SFARegion,
                        self.SFBRegion,
                        ki=3,
                    )

                return _Simplify(_gemm_ss_blockscaled_sm120_cute_consumer_bridge_skeleton, inline_let=True)

            if micro_pipeline == "sm120_backend_kblock_fulltile_package_pingpong":
                num_k_blocks = int(block_K // micro_size_k)
                if num_k_blocks != 4:
                    raise ValueError(
                        "micro_pipeline='sm120_backend_kblock_fulltile_package_pingpong' currently requires "
                        "block_K / micro_size_k == 4"
                    )
                if int(warp_rows) != 4 or int(warp_cols) != 4:
                    raise ValueError(
                        "micro_pipeline='sm120_backend_kblock_fulltile_package_pingpong' currently targets "
                        "warp_rows=4, warp_cols=4"
                    )
                if sf_layout != "cutlass_128x4":
                    raise ValueError(
                        "micro_pipeline='sm120_backend_kblock_fulltile_package_pingpong' currently requires "
                        "sf_layout='cutlass_128x4'"
                    )

                @T.prim_func
                def _gemm_ss_blockscaled_sm120_backend_package_pingpong() -> None:
                    if clear_accum:
                        T.clear(C_buf)

                    mma_emitter.mma_backend_kblock_fulltile_package_pingpong(
                        A_region,
                        B_region,
                        C_buf,
                        self.SFARegion,
                        self.SFBRegion,
                        sf_layout=sf_layout,
                    )

                return _Simplify(_gemm_ss_blockscaled_sm120_backend_package_pingpong, inline_let=True)

            if micro_pipeline == "sm120_pkg_atom_neutral":
                num_k_blocks = int(block_K // micro_size_k)
                if num_k_blocks != 4:
                    raise ValueError(
                        "micro_pipeline='sm120_pkg_atom_neutral' currently requires block_K / micro_size_k == 4"
                    )

                @T.prim_func
                def _gemm_ss_blockscaled_sm120_pkg_atom_neutral() -> None:
                    A_local_0 = T.alloc_local((warp_rows * local_size_a), a_dtype)
                    A_local_1 = T.alloc_local((warp_rows * local_size_a), a_dtype)
                    B_local_0 = T.alloc_local((warp_cols * local_size_b), b_dtype)
                    B_local_1 = T.alloc_local((warp_cols * local_size_b), b_dtype)
                    SFA_local_0 = T.alloc_local((warp_rows), "uint32")
                    SFA_local_1 = T.alloc_local((warp_rows), "uint32")
                    SFB_local_0 = T.alloc_local((warp_cols), "uint32")
                    SFB_local_1 = T.alloc_local((warp_cols), "uint32")
                    SFB_rep_local_0 = T.alloc_local((warp_cols), "uint32")
                    SFB_rep_local_1 = T.alloc_local((warp_cols), "uint32")
                    if clear_accum:
                        T.clear(C_buf)

                    pkg0 = SM120BlockScaledOperandPackage(
                        mma_emitter,
                        A_local_0,
                        B_local_0,
                        SFA_local_0,
                        SFB_local_0,
                        SFB_rep_local_0,
                        A_region,
                        B_region,
                        C_buf,
                        self.SFARegion,
                        self.SFBRegion,
                        self.sf_k_start,
                        int(sf_a_granularity_k),
                        int(sf_b_granularity_k),
                        sf_layout,
                    )
                    pkg1 = SM120BlockScaledOperandPackage(
                        mma_emitter,
                        A_local_1,
                        B_local_1,
                        SFA_local_1,
                        SFB_local_1,
                        SFB_rep_local_1,
                        A_region,
                        B_region,
                        C_buf,
                        self.SFARegion,
                        self.SFBRegion,
                        self.sf_k_start,
                        int(sf_a_granularity_k),
                        int(sf_b_granularity_k),
                        sf_layout,
                    )

                    pkg0.copy_kblock(0)
                    pkg1.copy_kblock(1)
                    for i in T.unroll(warp_rows):
                        for j in T.unroll(warp_cols):
                            pkg0.gemm_atom(i, j)
                    pkg0.copy_kblock(2)
                    for i in T.unroll(warp_rows):
                        for j in T.unroll(warp_cols):
                            pkg1.gemm_atom(i, j)
                    pkg1.copy_kblock(3)
                    for i in T.unroll(warp_rows):
                        for j in T.unroll(warp_cols):
                            pkg0.gemm_atom(i, j)
                    for i in T.unroll(warp_rows):
                        for j in T.unroll(warp_cols):
                            pkg1.gemm_atom(i, j)

                return _Simplify(_gemm_ss_blockscaled_sm120_pkg_atom_neutral, inline_let=True)

            if micro_pipeline == "k_static_scale_prefetch":
                num_k_blocks = int(block_K // micro_size_k)
                if num_k_blocks != 4:
                    raise ValueError(
                        f"micro_pipeline='{micro_pipeline}' currently requires block_K / micro_size_k == 4"
                    )

                @T.prim_func
                def _gemm_ss_blockscaled_k_static_scale_prefetch() -> None:
                    A_local_0 = T.alloc_local((warp_rows * local_size_a), a_dtype)
                    A_local_1 = T.alloc_local((warp_rows * local_size_a), a_dtype)
                    B_local_0 = T.alloc_local((warp_cols * local_size_b), b_dtype)
                    B_local_1 = T.alloc_local((warp_cols * local_size_b), b_dtype)
                    SFA_local_0 = T.alloc_local((warp_rows), "uint32")
                    SFA_local_1 = T.alloc_local((warp_rows), "uint32")
                    SFB_local_0 = T.alloc_local((warp_cols), "uint32")
                    SFB_local_1 = T.alloc_local((warp_cols), "uint32")
                    SFB_rep_local_0 = T.alloc_local((warp_cols), "uint32")
                    SFB_rep_local_1 = T.alloc_local((warp_cols), "uint32")
                    if clear_accum:
                        T.clear(C_buf)

                    pkg0 = SM120BlockScaledOperandPackage(
                        mma_emitter,
                        A_local_0,
                        B_local_0,
                        SFA_local_0,
                        SFB_local_0,
                        SFB_rep_local_0,
                        A_region,
                        B_region,
                        C_buf,
                        self.SFARegion,
                        self.SFBRegion,
                        self.sf_k_start,
                        int(sf_a_granularity_k),
                        int(sf_b_granularity_k),
                        sf_layout,
                    )
                    pkg1 = SM120BlockScaledOperandPackage(
                        mma_emitter,
                        A_local_1,
                        B_local_1,
                        SFA_local_1,
                        SFB_local_1,
                        SFB_rep_local_1,
                        A_region,
                        B_region,
                        C_buf,
                        self.SFARegion,
                        self.SFBRegion,
                        self.sf_k_start,
                        int(sf_a_granularity_k),
                        int(sf_b_granularity_k),
                        sf_layout,
                    )

                    pkg0.copy_kblock(0)
                    pkg1.copy_kblock(1)
                    pkg0.gemm_kblock()
                    pkg0.copy_kblock(2)
                    pkg1.gemm_kblock()
                    pkg1.copy_kblock(3)
                    pkg0.gemm_kblock()
                    pkg1.gemm_kblock()

                return _Simplify(_gemm_ss_blockscaled_k_static_scale_prefetch, inline_let=True)

            if micro_pipeline in ("k_static_scale_stream", "k_static_scale_stream_cutlass_order"):
                num_k_blocks = int(block_K // micro_size_k)
                if num_k_blocks != 4:
                    raise ValueError(
                        f"micro_pipeline={micro_pipeline!r} currently requires block_K / micro_size_k == 4"
                    )
                issue_mma = (
                    mma_emitter.mma_with_prefetched_scales_cutlass_order
                    if micro_pipeline == "k_static_scale_stream_cutlass_order"
                    else mma_emitter.mma_with_scale_fragments
                )

                @T.prim_func
                def _gemm_ss_blockscaled_k_static_scale_stream() -> None:
                    A_local_0 = T.alloc_local((warp_rows * local_size_a), a_dtype)
                    A_local_1 = T.alloc_local((warp_rows * local_size_a), a_dtype)
                    B_local_0 = T.alloc_local((warp_cols * local_size_b), b_dtype)
                    B_local_1 = T.alloc_local((warp_cols * local_size_b), b_dtype)
                    SFA_local = T.alloc_local((warp_rows), "uint32")
                    SFB_local = T.alloc_local((warp_cols), "uint32")
                    SFB_rep_local = T.alloc_local((warp_cols), "uint32")
                    if clear_accum:
                        T.clear(C_buf)

                    mma_emitter.ldmatrix_a(A_local_0, A_region, 0)
                    mma_emitter.ldmatrix_b(B_local_0, B_region, 0)
                    mma_emitter.ldmatrix_a(A_local_1, A_region, 1)
                    mma_emitter.ldmatrix_b(B_local_1, B_region, 1)
                    mma_emitter.ldscale_fragment(
                        SFA_local,
                        SFB_local,
                        SFB_rep_local,
                        self.SFARegion,
                        self.SFBRegion,
                        ki=0,
                        k_start=self.sf_k_start,
                        sf_a_granularity_k=int(sf_a_granularity_k),
                        sf_b_granularity_k=int(sf_b_granularity_k),
                        sf_layout=sf_layout,
                    )
                    issue_mma(
                        A_local_0,
                        B_local_0,
                        C_buf,
                        SFA_local,
                        SFB_local,
                        SFB_rep_local,
                    )
                    mma_emitter.ldmatrix_a(A_local_0, A_region, 2)
                    mma_emitter.ldmatrix_b(B_local_0, B_region, 2)
                    mma_emitter.ldscale_fragment(
                        SFA_local,
                        SFB_local,
                        SFB_rep_local,
                        self.SFARegion,
                        self.SFBRegion,
                        ki=1,
                        k_start=self.sf_k_start,
                        sf_a_granularity_k=int(sf_a_granularity_k),
                        sf_b_granularity_k=int(sf_b_granularity_k),
                        sf_layout=sf_layout,
                    )
                    issue_mma(
                        A_local_1,
                        B_local_1,
                        C_buf,
                        SFA_local,
                        SFB_local,
                        SFB_rep_local,
                    )
                    mma_emitter.ldmatrix_a(A_local_1, A_region, 3)
                    mma_emitter.ldmatrix_b(B_local_1, B_region, 3)
                    mma_emitter.ldscale_fragment(
                        SFA_local,
                        SFB_local,
                        SFB_rep_local,
                        self.SFARegion,
                        self.SFBRegion,
                        ki=2,
                        k_start=self.sf_k_start,
                        sf_a_granularity_k=int(sf_a_granularity_k),
                        sf_b_granularity_k=int(sf_b_granularity_k),
                        sf_layout=sf_layout,
                    )
                    issue_mma(
                        A_local_0,
                        B_local_0,
                        C_buf,
                        SFA_local,
                        SFB_local,
                        SFB_rep_local,
                    )
                    mma_emitter.ldscale_fragment(
                        SFA_local,
                        SFB_local,
                        SFB_rep_local,
                        self.SFARegion,
                        self.SFBRegion,
                        ki=3,
                        k_start=self.sf_k_start,
                        sf_a_granularity_k=int(sf_a_granularity_k),
                        sf_b_granularity_k=int(sf_b_granularity_k),
                        sf_layout=sf_layout,
                    )
                    issue_mma(
                        A_local_1,
                        B_local_1,
                        C_buf,
                        SFA_local,
                        SFB_local,
                        SFB_rep_local,
                    )

                return _Simplify(_gemm_ss_blockscaled_k_static_scale_stream, inline_let=True)

            if micro_pipeline == "k_static_b_all_prefetch_scale_stream":
                num_k_blocks = int(block_K // micro_size_k)
                if num_k_blocks != 4:
                    raise ValueError(
                        "micro_pipeline='k_static_b_all_prefetch_scale_stream' currently requires "
                        "block_K / micro_size_k == 4"
                    )

                @T.prim_func
                def _gemm_ss_blockscaled_k_static_b_all_prefetch_scale_stream() -> None:
                    A_local_0 = T.alloc_local((warp_rows * local_size_a), a_dtype)
                    A_local_1 = T.alloc_local((warp_rows * local_size_a), a_dtype)
                    B_local_0 = T.alloc_local((warp_cols * local_size_b), b_dtype)
                    B_local_1 = T.alloc_local((warp_cols * local_size_b), b_dtype)
                    B_local_2 = T.alloc_local((warp_cols * local_size_b), b_dtype)
                    B_local_3 = T.alloc_local((warp_cols * local_size_b), b_dtype)
                    SFA_local = T.alloc_local((warp_rows), "uint32")
                    SFB_local = T.alloc_local((warp_cols), "uint32")
                    SFB_rep_local = T.alloc_local((warp_cols), "uint32")
                    if clear_accum:
                        T.clear(C_buf)

                    mma_emitter.ldmatrix_a(A_local_0, A_region, 0)
                    mma_emitter.ldmatrix_b(B_local_0, B_region, 0)
                    mma_emitter.ldmatrix_a(A_local_1, A_region, 1)
                    mma_emitter.ldmatrix_b(B_local_1, B_region, 1)
                    mma_emitter.ldmatrix_b(B_local_2, B_region, 2)
                    mma_emitter.ldmatrix_b(B_local_3, B_region, 3)
                    mma_emitter.ldscale_fragment(
                        SFA_local,
                        SFB_local,
                        SFB_rep_local,
                        self.SFARegion,
                        self.SFBRegion,
                        ki=0,
                        k_start=self.sf_k_start,
                        sf_a_granularity_k=int(sf_a_granularity_k),
                        sf_b_granularity_k=int(sf_b_granularity_k),
                        sf_layout=sf_layout,
                    )
                    mma_emitter.mma_with_scale_fragments(
                        A_local_0,
                        B_local_0,
                        C_buf,
                        SFA_local,
                        SFB_local,
                        SFB_rep_local,
                    )
                    mma_emitter.ldmatrix_a(A_local_0, A_region, 2)
                    mma_emitter.ldscale_fragment(
                        SFA_local,
                        SFB_local,
                        SFB_rep_local,
                        self.SFARegion,
                        self.SFBRegion,
                        ki=1,
                        k_start=self.sf_k_start,
                        sf_a_granularity_k=int(sf_a_granularity_k),
                        sf_b_granularity_k=int(sf_b_granularity_k),
                        sf_layout=sf_layout,
                    )
                    mma_emitter.mma_with_scale_fragments(
                        A_local_1,
                        B_local_1,
                        C_buf,
                        SFA_local,
                        SFB_local,
                        SFB_rep_local,
                    )
                    mma_emitter.ldmatrix_a(A_local_1, A_region, 3)
                    mma_emitter.ldscale_fragment(
                        SFA_local,
                        SFB_local,
                        SFB_rep_local,
                        self.SFARegion,
                        self.SFBRegion,
                        ki=2,
                        k_start=self.sf_k_start,
                        sf_a_granularity_k=int(sf_a_granularity_k),
                        sf_b_granularity_k=int(sf_b_granularity_k),
                        sf_layout=sf_layout,
                    )
                    mma_emitter.mma_with_scale_fragments(
                        A_local_0,
                        B_local_2,
                        C_buf,
                        SFA_local,
                        SFB_local,
                        SFB_rep_local,
                    )
                    mma_emitter.ldscale_fragment(
                        SFA_local,
                        SFB_local,
                        SFB_rep_local,
                        self.SFARegion,
                        self.SFBRegion,
                        ki=3,
                        k_start=self.sf_k_start,
                        sf_a_granularity_k=int(sf_a_granularity_k),
                        sf_b_granularity_k=int(sf_b_granularity_k),
                        sf_layout=sf_layout,
                    )
                    mma_emitter.mma_with_scale_fragments(
                        A_local_1,
                        B_local_3,
                        C_buf,
                        SFA_local,
                        SFB_local,
                        SFB_rep_local,
                    )

                return _Simplify(_gemm_ss_blockscaled_k_static_b_all_prefetch_scale_stream, inline_let=True)

            if micro_pipeline == "k_static":
                num_k_blocks = int(block_K // micro_size_k)
                if num_k_blocks != 4:
                    raise ValueError("micro_pipeline='k_static' currently requires block_K / micro_size_k == 4")

                @T.prim_func
                def _gemm_ss_blockscaled_k_static() -> None:
                    A_local_0 = T.alloc_local((warp_rows * local_size_a), a_dtype)
                    A_local_1 = T.alloc_local((warp_rows * local_size_a), a_dtype)
                    B_local_0 = T.alloc_local((warp_cols * local_size_b), b_dtype)
                    B_local_1 = T.alloc_local((warp_cols * local_size_b), b_dtype)
                    if clear_accum:
                        T.clear(C_buf)

                    mma_emitter.ldmatrix_a(A_local_0, A_region, 0)
                    mma_emitter.ldmatrix_b(B_local_0, B_region, 0)
                    mma_emitter.ldmatrix_a(A_local_1, A_region, 1)
                    mma_emitter.ldmatrix_b(B_local_1, B_region, 1)
                    mma_emitter.mma(
                        A_local_0,
                        B_local_0,
                        C_buf,
                        self.SFARegion,
                        self.SFBRegion,
                        ki=0,
                        k_start=self.sf_k_start,
                        sf_a_granularity_k=int(sf_a_granularity_k),
                        sf_b_granularity_k=int(sf_b_granularity_k),
                        sf_layout=sf_layout,
                    )
                    mma_emitter.ldmatrix_a(A_local_0, A_region, 2)
                    mma_emitter.ldmatrix_b(B_local_0, B_region, 2)
                    mma_emitter.mma(
                        A_local_1,
                        B_local_1,
                        C_buf,
                        self.SFARegion,
                        self.SFBRegion,
                        ki=1,
                        k_start=self.sf_k_start,
                        sf_a_granularity_k=int(sf_a_granularity_k),
                        sf_b_granularity_k=int(sf_b_granularity_k),
                        sf_layout=sf_layout,
                    )
                    mma_emitter.ldmatrix_a(A_local_1, A_region, 3)
                    mma_emitter.ldmatrix_b(B_local_1, B_region, 3)
                    mma_emitter.mma(
                        A_local_0,
                        B_local_0,
                        C_buf,
                        self.SFARegion,
                        self.SFBRegion,
                        ki=2,
                        k_start=self.sf_k_start,
                        sf_a_granularity_k=int(sf_a_granularity_k),
                        sf_b_granularity_k=int(sf_b_granularity_k),
                        sf_layout=sf_layout,
                    )
                    mma_emitter.mma(
                        A_local_1,
                        B_local_1,
                        C_buf,
                        self.SFARegion,
                        self.SFBRegion,
                        ki=3,
                        k_start=self.sf_k_start,
                        sf_a_granularity_k=int(sf_a_granularity_k),
                        sf_b_granularity_k=int(sf_b_granularity_k),
                        sf_layout=sf_layout,
                    )

                return _Simplify(_gemm_ss_blockscaled_k_static, inline_let=True)

            @T.prim_func
            def _gemm_ss_blockscaled() -> None:
                A_local = T.alloc_local((warp_rows * local_size_a), a_dtype)
                B_local = T.alloc_local((warp_cols * local_size_b), b_dtype)
                if clear_accum:
                    T.clear(C_buf)
                for ki in T.serial(0, (block_K // micro_size_k)):
                    mma_emitter.ldmatrix_a(A_local, A_region, ki)
                    mma_emitter.ldmatrix_b(B_local, B_region, ki)
                    mma_emitter.mma(
                        A_local,
                        B_local,
                        C_buf,
                        self.SFARegion,
                        self.SFBRegion,
                        ki=ki,
                        k_start=self.sf_k_start,
                        sf_a_granularity_k=int(sf_a_granularity_k),
                        sf_b_granularity_k=int(sf_b_granularity_k),
                    )

            return _Simplify(_gemm_ss_blockscaled, inline_let=True)

        if self.is_gemm_ss():

            @T.prim_func
            def _gemm_ssr() -> None:
                """
                The inner macro that loads data from shared buffers A_shared and
                B_shared into local fragments, then issues Tensor Core mma ops,
                accumulating into C_local.
                """
                A_local = T.alloc_local((warp_rows * local_size_a), a_dtype)
                B_local = T.alloc_local((warp_cols * local_size_b), b_dtype)
                if clear_accum:
                    T.clear(C_buf)
                for ki in T.serial(0, (block_K // micro_size_k)):
                    # Load A into fragment
                    mma_emitter.ldmatrix_a(
                        A_local,
                        A_region,
                        ki,
                    )

                    # Load B into fragment
                    mma_emitter.ldmatrix_b(
                        B_local,
                        B_region,
                        ki,
                    )

                    # Perform Matrix Multiplication
                    mma_emitter.mma(A_local, B_local, C_buf, ki)

            # Simplify to optimize the index computing
            # Must inline let statements to simplify the analysis
            return _Simplify(_gemm_ssr, inline_let=True)
        elif self.is_gemm_sr():
            assert is_full_region(B_region), "Fragment input B must be a full region"

            @T.prim_func
            def _gemm_srr() -> None:
                """
                The inner macro that loads data from shared buffers A_shared and
                B_shared into local fragments, then issues Tensor Core mma ops,
                accumulating into C_local.
                """
                A_local = T.alloc_local((warp_rows * local_size_a), a_dtype)

                for ki in T.serial(0, (block_K // micro_size_k)):
                    if clear_accum:
                        T.clear(C_buf)
                    # Load A into fragment
                    mma_emitter.ldmatrix_a(
                        A_local,
                        A_region,
                        ki,
                    )

                    # Perform Matrix Multiplication
                    mma_emitter.mma(A_local, B_buf, C_buf, ki)

            # Simplify to optimize the index computing
            # Must inline let statements to simplify the analysis
            # alloc_buffers body
            # insert into parent block
            return _Simplify(_gemm_srr, inline_let=True)
        elif self.is_gemm_rs():
            assert is_full_region(A_region), "Fragment input A must be a full region"

            @T.prim_func
            def _gemm_rsr() -> None:
                """
                The inner macro that loads data from shared buffers A_shared and
                B_shared into local fragments, then issues Tensor Core mma ops,
                accumulating into C_local.
                """
                B_local = T.alloc_local((warp_cols * local_size_b), b_dtype)
                if clear_accum:
                    T.clear(C_buf)
                for ki in T.serial(0, (block_K // micro_size_k)):
                    # Load B into fragment
                    mma_emitter.ldmatrix_b(
                        B_local,
                        B_region,
                        ki,
                    )

                    # Perform Matrix Multiplication
                    mma_emitter.mma(A_buf, B_local, C_buf, ki)

            # Simplify to optimize the index computing
            # Must inline let statements to simplify the analysis
            return _Simplify(_gemm_rsr, inline_let=True)
        elif self.is_gemm_rr():
            assert is_full_region(A_region), "Fragment input A must be a full region"
            assert is_full_region(B_region), "Fragment input B must be a full region"

            @T.prim_func
            def _gemm_rrr() -> None:
                """
                The inner macro that loads data from shared buffers A_shared and
                B_shared into local fragments, then issues Tensor Core mma ops,
                accumulating into C_local.
                """

                for ki in T.serial(0, (block_K // micro_size_k)):
                    # Perform Matrix Multiplication
                    mma_emitter.mma(A_buf, B_buf, C_buf, ki)

            # Simplify to optimize the index computing
            # Must inline let statements to simplify the analysis
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
