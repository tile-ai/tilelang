from __future__ import annotations

from tilelang import language as T
from tilelang.cuda.intrinsics.macro.mma_sm120_macro_generator import (
    TensorCoreIntrinEmitterSM120 as TensorCoreIntrinEmitterBlockScaled,
)
from tilelang.cuda.target import target_is_cuda, target_is_sm120
from tilelang.tileop.gemm_blockscaled.gemm_blockscaled_base import GemmBlockScaledMixin
from tilelang.transform.simplify import _Simplify
from tilelang.utils.language import is_fragment, is_full_region
from tvm import tirx
from tvm.ir import Range
from tvm.target import Target

from .gemm_mma import GemmMMA


GEMM_INST_MMA_BLOCK_SCALED = "cuda.mma.blockscaled"


def _is_explicit_non_sm120_cuda(target: Target) -> bool:
    if not target_is_cuda(target):
        return False
    arch = target.attrs.get("arch", None)
    if arch is None or not str(arch).startswith("sm_"):
        return False
    return not target_is_sm120(target)


class GemmMMASm120BlockScaled(GemmBlockScaledMixin, GemmMMA):
    """SM120 warp-level block-scaled MMA lowering.

    A may be shared or a full fragment; B remains shared. Fragment A and
    fragment scales use row-major packed uint32 scales. Compact K-major
    scales retain the shared/shared ping-pong path, including odd atom grids.
    """

    supported_sf_layouts = ("rowmajor", "blockscaled_chunk_kmajor")

    intrin_emitter_cls = TensorCoreIntrinEmitterBlockScaled

    @staticmethod
    def _validate_target(target: Target) -> None:
        if _is_explicit_non_sm120_cuda(target):
            raise ValueError("T.mma_gemm_blockscaled requires SM120 CUDA target")

    def _validate_operands(self) -> None:
        if not (self.is_gemm_ss() or self.is_gemm_rs()):
            raise ValueError("T.mma_gemm_blockscaled requires shared or fragment A and shared B")
        if self.is_gemm_rs():
            if not is_full_region(self.ARegion):
                raise ValueError("SM120 block-scaled fragment input A must be a full region")
            if self.sf_layout != "rowmajor":
                raise ValueError("SM120 block-scaled fragment A currently requires sf_layout='rowmajor'")

    def _make_mma_emitter(self, target: Target, thread_nums: int, thread_var: tirx.Var | None = None):
        m_warp, n_warp = self.policy.compute_warp_partition(
            self.M,
            self.N,
            thread_nums,
            target,
            GEMM_INST_MMA_BLOCK_SCALED,
        )
        return self.intrin_emitter_cls(
            a_dtype=self.a_dtype,
            b_dtype=self.b_dtype,
            accum_dtype=self.accum_dtype,
            a_transposed=self.trans_A,
            b_transposed=self.trans_B,
            block_row_warps=m_warp,
            block_col_warps=n_warp,
            warp_row_tiles=int(self.M // m_warp),
            warp_col_tiles=int(self.N // n_warp),
            chunk=self.chunk,
            thread_var=thread_var,
            is_blockscaled=True,
        )

    def infer_layout(self, target: Target, thread_nums: int):
        self._validate_target(target)
        self._validate_operands()
        layouts = super().infer_layout(target, thread_nums)
        emitter = self._make_mma_emitter(target, thread_nums)
        for matrix, region, rows in (("A", self.SFARegion, self.M), ("B", self.SFBRegion, self.N)):
            buffer = region.buffer
            if is_fragment(buffer):
                if self.sf_layout != "rowmajor":
                    raise ValueError("SM120 fragment scales currently require sf_layout='rowmajor'")
                if not is_full_region(region) or len(buffer.shape) != 2 or int(buffer.shape[0]) != int(rows):
                    raise ValueError(f"SM120 fragment SF{matrix} must be a full 2D region with {rows} rows")
                scale_layout = emitter.make_scale_load_layout(buffer, matrix)
                # Preserve both operand requirements when SFA and SFB alias.
                if buffer in layouts and not layouts[buffer].is_equal(scale_layout):
                    raise ValueError(
                        f"SM120 block-scaled GEMM cannot reuse fragment buffer '{buffer.name}' as SF{matrix}: "
                        "its operands require different fragment layouts. "
                        "Use separate SFA and SFB fragment buffers, or shared-memory scales."
                    )
                layouts[buffer] = scale_layout
        return layouts

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_index: tirx.PrimExpr,
        mbar_phase_expr: tirx.PrimExpr | None = None,
    ):
        self._validate_target(target)
        self._validate_operands()

        thread_nums = thread_bounds.extent
        local_thread_var = thread_index - thread_bounds.min
        mma_emitter = self._make_mma_emitter(target, thread_nums, thread_var=local_thread_var)

        a_dtype = self.a_dtype
        b_dtype = self.b_dtype
        warp_rows = mma_emitter.warp_rows
        warp_cols = mma_emitter.warp_cols
        local_size_a = mma_emitter.local_size_a
        local_size_b = mma_emitter.local_size_b
        block_K = mma_emitter.chunk
        micro_size_k = mma_emitter.micro_size_k
        A_region = self.ARegion
        B_region = self.BRegion
        C_region = self.CRegion
        C_buf = C_region.buffer
        clear_accum = self.clear_accum

        assert block_K >= micro_size_k, f"block_K ({block_K}) must be >= micro_size_k ({micro_size_k})"
        assert block_K % micro_size_k == 0, f"block_K ({block_K}) must be a multiple of micro_size_k ({micro_size_k})"
        assert is_full_region(C_region), "Fragment output C must be a full region"

        sf_layout = self.sf_layout
        if sf_layout not in self.supported_sf_layouts:
            raise ValueError(f"Unsupported SM120 scale layout: {sf_layout}")
        sf_a_granularity_k = self.sf_a_granularity_k
        sf_b_granularity_k = self.sf_b_granularity_k

        if self.is_gemm_rs():
            A_buf = A_region.buffer

            @T.prim_func
            def _gemm_rs_blockscaled() -> None:
                B_local = T.alloc_local((warp_cols * local_size_b), b_dtype)
                if clear_accum:
                    T.clear(C_buf)
                for ki in T.serial(block_K // micro_size_k):
                    mma_emitter.ldmatrix_b(B_local, B_region, ki)
                    mma_emitter.mma(
                        A_buf,
                        B_local,
                        C_buf,
                        ki,
                        SFA_buf=self.SFARegion,
                        SFB_buf=self.SFBRegion,
                        k_start=self.sf_k_start,
                        sf_a_granularity_k=sf_a_granularity_k,
                        sf_b_granularity_k=sf_b_granularity_k,
                    )

            return _Simplify(_gemm_rs_blockscaled, inline_let=True)

        if sf_layout == "blockscaled_chunk_kmajor":

            @T.prim_func
            def _gemm_ss_blockscaled_kmajor() -> None:
                if clear_accum:
                    T.clear(C_buf)
                mma_emitter.mma_blockscaled_fulltile(
                    A_region,
                    B_region,
                    C_buf,
                    self.SFARegion,
                    self.SFBRegion,
                    sf_layout=sf_layout,
                )

            return _Simplify(_gemm_ss_blockscaled_kmajor, inline_let=True)

        if int(block_K // micro_size_k) == 4:

            @T.prim_func
            def _gemm_ss_blockscaled_static_kblock() -> None:
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
                    sf_a_granularity_k=sf_a_granularity_k,
                    sf_b_granularity_k=sf_b_granularity_k,
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
                    sf_a_granularity_k=sf_a_granularity_k,
                    sf_b_granularity_k=sf_b_granularity_k,
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
                    sf_a_granularity_k=sf_a_granularity_k,
                    sf_b_granularity_k=sf_b_granularity_k,
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
                    sf_a_granularity_k=sf_a_granularity_k,
                    sf_b_granularity_k=sf_b_granularity_k,
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

            return _Simplify(_gemm_ss_blockscaled_static_kblock, inline_let=True)

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
                    ki,
                    SFA_buf=self.SFARegion,
                    SFB_buf=self.SFBRegion,
                    k_start=self.sf_k_start,
                    sf_a_granularity_k=sf_a_granularity_k,
                    sf_b_granularity_k=sf_b_granularity_k,
                )

        return _Simplify(_gemm_ss_blockscaled, inline_let=True)
