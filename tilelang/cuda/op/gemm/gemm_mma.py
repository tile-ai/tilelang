from __future__ import annotations

from tilelang.tileop.gemm.gemm_base import GemmBase
from tilelang.layout import make_swizzled_layout
from tilelang.cuda.intrinsics.macro.mma_macro_generator import TensorCoreIntrinEmitter
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
        emitter_kwargs = dict(
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
        if self.is_blockscaled:
            emitter_kwargs["is_blockscaled"] = True
        emitter = self.intrin_emitter_cls(**emitter_kwargs)
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
        assert block_K % micro_size_k == 0, f"block_K ({block_K}) must be a multiple of micro_size_k ({micro_size_k})"

        assert is_full_region(C_region), "Fragment output C must be a full region"

        if self.is_blockscaled:
            if not self.is_gemm_ss():
                raise ValueError("T.mma_gemm_blockscaled supports shared-memory A/B operands only")
            annotations = getattr(self.gemm_node, "annotations", {})
            sf_a_granularity_k = annotations.get("sf_a_granularity_k")
            sf_b_granularity_k = annotations.get("sf_b_granularity_k")
            sf_layout = annotations.get("sf_layout", "rowmajor")
            if sf_layout not in ("rowmajor", "blockscaled_chunk_kmajor"):
                raise ValueError(f"Unsupported SM120 scale layout: {sf_layout}")
            if sf_a_granularity_k is None or sf_b_granularity_k is None:
                raise ValueError("Block-scaled MMA GEMM requires sf_a_granularity_k and sf_b_granularity_k")

            if sf_layout == "blockscaled_chunk_kmajor":
                num_k_blocks = int(block_K // micro_size_k)
                if num_k_blocks != 4:
                    raise ValueError("SM120 packed-scale blockscaled MMA currently requires block_K / micro_size_k == 4")
                if int(warp_rows) != 4 or int(warp_cols) != 4:
                    raise ValueError(
                        "SM120 packed-scale blockscaled MMA currently requires a 4x4 warp atom grid, "
                        f"got warp_rows={int(warp_rows)}, warp_cols={int(warp_cols)}"
                    )

                @T.prim_func
                def _gemm_ss_blockscaled_packed_scale_package() -> None:
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

                return _Simplify(_gemm_ss_blockscaled_packed_scale_package, inline_let=True)

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

                if clear_accum:
                    T.clear(C_buf)
                for ki in T.serial(0, (block_K // micro_size_k)):
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

                if clear_accum:
                    T.clear(C_buf)
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
