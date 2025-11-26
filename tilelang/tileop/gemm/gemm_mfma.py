from .gemm_base import GemmBase
from tilelang.layout import make_swizzled_layout
from tilelang.intrinsics.mfma_macro_generator import (
    MatrixCoreIntrinEmitter,)
from tilelang.utils.language import is_shared, is_fragment, is_full_region
from tilelang import tvm as tvm
from tvm.target import Target
from tvm import tir
from tilelang import language as T
from tilelang.transform.simplify import _Simplify


class GemmMFMA(GemmBase):

    def infer_layout(self, target: Target, thread_nums: int):
        """
        Infer and return the memory layout mapping for A, B, and C optimized for MFMA-based GEMM on the given target and thread configuration.
        
        Parameters:
            target (Target): Compilation target used to compute warp partitioning and emitter configuration.
            thread_nums (int): Number of threads used to compute warp partitioning.
        
        Returns:
            dict: A mapping from region objects (self.A, self.B, self.C) to layout descriptors. Possible mappings depend on the operand layout combination:
                - ss: A and B use swizzled layouts; C uses the MFMA store layout.
                - sr: A uses a swizzled layout; B uses the MFMA load layout for matrix "B"; C uses the MFMA store layout.
                - rs: A uses the MFMA load layout for matrix "A"; B uses a swizzled layout; C uses the MFMA store layout.
                - rr: A and B use their MFMA load layouts for matrices "A" and "B" respectively; C uses the MFMA store layout.
        
        Raises:
            ValueError: If the GEMM operand combination is not one of ss, sr, rs, or rr.
        """
        m_warp, n_warp = self.policy.compute_warp_partition(self.M, self.N, thread_nums, target,
                                                            False)
        warp_row_tiles = int(self.M // m_warp)
        warp_col_tiles = int(self.N // n_warp)
        mfma_emitter = MatrixCoreIntrinEmitter(
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
        )

        if self.is_gemm_ss():
            return {
                self.A: make_swizzled_layout(self.A),
                self.B: make_swizzled_layout(self.B),
                self.C: mfma_emitter.make_mfma_store_layout(self.C),
            }
        elif self.is_gemm_sr():
            return {
                self.A: make_swizzled_layout(self.A),
                self.B: mfma_emitter.make_mfma_load_layout(self.B, matrix="B"),
                self.C: mfma_emitter.make_mfma_store_layout(self.C),
            }
        elif self.is_gemm_rs():
            return {
                self.A: mfma_emitter.make_mfma_load_layout(self.A, matrix="A"),
                self.B: make_swizzled_layout(self.B),
                self.C: mfma_emitter.make_mfma_store_layout(self.C),
            }
        elif self.is_gemm_rr():
            return {
                self.A: mfma_emitter.make_mfma_load_layout(self.A, matrix="A"),
                self.B: mfma_emitter.make_mfma_load_layout(self.B, matrix="B"),
                self.C: mfma_emitter.make_mfma_store_layout(self.C),
            }
        else:
            raise ValueError(
                f"Unsupported gemm combination, A: {self.A.scope()}, B: {self.B.scope()}")

    def lower(self, layout_map: dict, target: Target, thread_nums: int, thread_var: tir.Var):
        """
        Lower the GEMM configuration into a matrix-core TIR prim_func that performs tiled MFMA-based micro-kernels.
        
        This produces and returns a Simplify-wrapped TIR prim_func implementing the inner GEMM loop for the inferred warp/block tiling and MFMA emitter configuration. The generated prim_func varies by operand layout (ss, sr, rs, rr): it may allocate local fragments for A and/or B (sized with k_pack), load from the provided shared/region buffers as needed, optionally clear the accumulator, iterate over K in steps of micro_size_k * k_pack, and invoke the Matrix Core MFMA intrinsics to accumulate into the C fragment.
        
        Parameters:
            layout_map (dict): Mapping of buffer/layout identifiers to regions used for lowering (unused when regions are carried on the instance but provided for caller context).
            target (Target): Compilation target used to compute warp partitioning and emitter configuration.
            thread_nums (int): Number of threads used to derive warp partitioning.
            thread_var (tir.Var): Thread index variable passed to the MatrixCoreIntrinEmitter for lowering.
        
        Returns:
            tir.PrimFunc: A Simplify-wrapped TIR prim_func that implements the lowered inner GEMM computation for the selected operand layout and MFMA configuration.
        
        Raises:
            ValueError: If the GEMM operand layout combination is not one of 'ss', 'sr', 'rs', or 'rr'.
        """
        m_warp, n_warp = self.policy.compute_warp_partition(self.M, self.N, thread_nums, target,
                                                            False)
        warp_row_tiles = int(self.M // m_warp)
        warp_col_tiles = int(self.N // n_warp)
        mfma_emitter = MatrixCoreIntrinEmitter(
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
            k_pack=self.k_pack,
        )

        in_dtype = self.in_dtype
        warp_rows = mfma_emitter.warp_rows
        warp_cols = mfma_emitter.warp_cols
        local_size_a = mfma_emitter.local_size_a
        local_size_b = mfma_emitter.local_size_b
        block_K = mfma_emitter.chunk
        micro_size_k = mfma_emitter.micro_size_k
        # Use region for shared-memory operands if available
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

        if self.is_gemm_ss():

            @T.prim_func
            def _gemm_ssr() -> None:
                """
                Load tiles of A and B from their shared-memory regions into per-warp local fragments and perform MFMA operations to accumulate into the C_buf local accumulator.
                
                Allocates A_local and B_local sized by warp_rows, warp_cols, local_size_a/local_size_b and self.k_pack. Optionally clears C_buf if clear_accum is set. Iterates over block_K in steps of micro_size_k * self.k_pack; on each iteration it loads the corresponding A and B tiles into the local fragments and issues the mfma emitter to accumulate into C_buf.
                """
                A_local = T.alloc_local((warp_rows * local_size_a * self.k_pack), in_dtype)
                B_local = T.alloc_local((warp_cols * local_size_b * self.k_pack), in_dtype)
                if clear_accum:
                    T.clear(C_buf)
                for ki in T.serial(0, (block_K // (micro_size_k * self.k_pack))):
                    # Load A into fragment
                    mfma_emitter.ldmatrix_a(
                        A_local,
                        A_region,
                        ki,
                    )

                    # Load B into fragment
                    mfma_emitter.ldmatrix_b(
                        B_local,
                        B_region,
                        ki,
                    )

                    # Perform Matrix Multiplication
                    mfma_emitter.mfma(A_local, B_local, C_buf, ki)

            # Simplify to optimize the index computing
            # Must inline let statements to simplify the analysis
            return _Simplify(_gemm_ssr, inline_let=True)
        elif self.is_gemm_sr():
            assert is_full_region(B_region), "Fragment input B must be a full region"

            @T.prim_func
            def _gemm_srr() -> None:
                """
                Load A fragments from shared memory into a local fragment and perform MFMA updates into the accumulator C_buf.
                
                This inner kernel allocates the A_local fragment sized by warp_rows * local_size_a * self.k_pack, iterates over block_K in steps of micro_size_k * self.k_pack, invokes the ldmatrix A loader for each step, and issues the mfma operation to accumulate into C_buf.
                """
                A_local = T.alloc_local((warp_rows * local_size_a * self.k_pack), in_dtype)

                if clear_accum:
                    T.clear(C_buf)

                for ki in T.serial(0, (block_K // (micro_size_k * self.k_pack))):

                    # Load A into fragment
                    mfma_emitter.ldmatrix_a(
                        A_local,
                        A_region,
                        ki,
                    )

                    # Perform Matrix Multiplication
                    mfma_emitter.mfma(A_local, B_buf, C_buf, ki)

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
                Load B tiles from shared memory into a local fragment and run MFMA operations with A_buf to accumulate results into C_buf over K tiles.
                
                Iterates over K in steps of micro_size_k * k_pack, optionally clears the accumulator buffer before the loop, loads B fragments via the MFMA emitter into a local buffer sized by warp_cols, local_size_b, and k_pack, and invokes the emitter's MFMA to update C_buf using A_buf and the loaded B fragment.
                """
                B_local = T.alloc_local((warp_cols * local_size_b * self.k_pack), in_dtype)
                if clear_accum:
                    T.clear(C_buf)
                for ki in T.serial(0, (block_K // (micro_size_k * self.k_pack))):

                    # Load B into fragment
                    mfma_emitter.ldmatrix_b(
                        B_local,
                        B_region,
                        ki,
                    )

                    # Perform Matrix Multiplication
                    mfma_emitter.mfma(A_buf, B_local, C_buf, ki)

            # Simplify to optimize the index computing
            # Must inline let statements to simplify the analysis
            return _Simplify(_gemm_rsr, inline_let=True)
        elif self.is_gemm_rr():
            assert is_full_region(A_region), "Fragment input A must be a full region"
            assert is_full_region(B_region), "Fragment input B must be a full region"

            @T.prim_func
            def _gemm_rsr() -> None:
                """
                Performs tiled MFMA accumulation: iterates over K partitions, loads fragments from shared A/B into local fragments, and applies the Matrix Core MFMA to accumulate into the local C fragment.
                
                Executes one MFMA step per iteration for a total of block_K // (micro_size_k * self.k_pack) iterations by invoking the mfma_emitter to update C_buf from A_buf and B_buf.
                """

                for ki in T.serial(0, (block_K // (micro_size_k * self.k_pack))):
                    # Perform Matrix Multiplication
                    mfma_emitter.mfma(A_buf, B_buf, C_buf, ki)

            # Simplify to optimize the index computing
            # Must inline let statements to simplify the analysis
            return _Simplify(_gemm_rsr, inline_let=True)
        else:
            raise ValueError(
                f"Unsupported gemm combination, A: {self.A.scope()}, B: {self.B.scope()}")

    def is_gemm_ss(self) -> bool:
        return is_shared(self.A) and is_shared(self.B)

    def is_gemm_sr(self) -> bool:
        return is_shared(self.A) and is_fragment(self.B)

    def is_gemm_rs(self) -> bool:
        return is_fragment(self.A) and is_shared(self.B)

    def is_gemm_rr(self) -> bool:
        return is_fragment(self.A) and is_fragment(self.B)