# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from tvm import tir
import tilelang.language as T
from tilelang.utils import is_fragment
from tilelang.primitives.gemm.base import GemmBaseParams
from tilelang.intrinsics.mma_macro_generator import TensorCoreIntrinEmitter


# TODO(lei): Implement GEMM_SR, GEMM_RS, GEMM_RR
@dataclass
class GemmPrimitiveMMA(GemmBaseParams):
    """
    A GEMM (General Matrix Multiply) primitive that uses Tensor Core MMA (Matrix
    Multiply and Accumulate) instructions. Inherits from GemmBaseParams which
    provides basic parameters such as A, B, C buffers and transposition flags.
    """

    def gemm_rrr(
        self,
        A: tir.Buffer,
        B: tir.Buffer,
        C: tir.Buffer,
        mma_emitter: TensorCoreIntrinEmitter,
    ) -> tir.PrimExpr:
        raise NotImplementedError("GEMM_RRR is not implemented yet")

    def gemm_rsr(
        self,
        A: tir.Buffer,
        B: tir.Buffer,
        C: tir.Buffer,
        mma_emitter: TensorCoreIntrinEmitter,
    ) -> tir.PrimExpr:

        in_dtype = self.in_dtype
        warp_cols = mma_emitter.warp_cols
        local_size_b = mma_emitter.local_size_b
        block_K = mma_emitter.chunk
        micro_size_k = mma_emitter.micro_size_k

        # Check if C is a fragment for applying custom layout
        a_is_fragment = is_fragment(A)
        c_is_fragment = is_fragment(C)

        @T.macro
        def _gemm_rsr(A_local: tir.Buffer, B_shared: tir.Buffer, C_local: tir.Buffer) -> None:
            """
            The inner macro that loads data from shared buffers A_shared and
            B_shared into local fragments, then issues Tensor Core mma ops,
            accumulating into C_local.
            """
            B_local = T.alloc_local((warp_cols * local_size_b), in_dtype)

            if a_is_fragment:
                # Annotate layout for A_local if it is a fragment.
                T.annotate_layout({
                    A_local: mma_emitter.make_mma_load_layout(A_local, "A"),
                })
            if c_is_fragment:
                # Annotate layout for C_local if it is a fragment.
                T.annotate_layout({
                    C_local: mma_emitter.make_mma_store_layout(C_local),
                })

            # Make default swizzle layout for shared memory
            # T.annotate_layout({
            #     B_shared: make_mma_swizzle_layout(B_shared),
            # })
            for ki in T.serial(0, (block_K // micro_size_k)):

                # Load B into fragment
                mma_emitter.ldmatrix_b(
                    B_local,
                    B_shared,
                    ki,
                )
                # Perform Matrix Multiplication
                mma_emitter.mma(
                    A_local,
                    B_local,
                    C_local,
                    ki,
                )

        return _gemm_rsr(A, B, C)

    def gemm_srr(
        self,
        A: tir.Buffer,
        B: tir.Buffer,
        C: tir.Buffer,
        mma_emitter: TensorCoreIntrinEmitter,
    ) -> tir.PrimExpr:
        raise NotImplementedError("GEMM_RSR is not implemented yet")

    def gemm_ssr(
        self,
        A: tir.Buffer,
        B: tir.Buffer,
        C: tir.Buffer,
        mma_emitter: TensorCoreIntrinEmitter,
    ) -> tir.PrimExpr:
        """
        Perform a single-step reduction (SSR) GEMM using Tensor Core MMA
        primitives. Loads fragments of A and B from shared memory, multiplies
        them, and accumulates into C.

        Parameters
        ----------
        A : tir.Buffer
            The buffer for matrix A (in shared memory).
        B : tir.Buffer
            The buffer for matrix B (in shared memory).
        C : tir.Buffer
            The buffer for the accumulation results.
        mma_emitter : TensorCoreIntrinEmitter
            A helper object responsible for generating Tensor Core MMA
            instructions (ldmatrix, mma, etc.).

        Returns
        -------
        tir.PrimExpr
            The generated IR expression (macro) representing the GEMM loop.
        """

        in_dtype = self.in_dtype
        warp_rows = mma_emitter.warp_rows
        warp_cols = mma_emitter.warp_cols
        local_size_a = mma_emitter.local_size_a
        local_size_b = mma_emitter.local_size_b
        block_K = mma_emitter.chunk
        micro_size_k = mma_emitter.micro_size_k

        # Check if C is a fragment for applying custom layout
        c_is_fragment = is_fragment(C)

        @T.macro
        def _gemm_ssr(A_shared: tir.Buffer, B_shared: tir.Buffer, C_local: tir.Buffer) -> None:
            """
            The inner macro that loads data from shared buffers A_shared and
            B_shared into local fragments, then issues Tensor Core mma ops,
            accumulating into C_local.
            """
            A_local = T.alloc_local((warp_rows * local_size_a), in_dtype)
            B_local = T.alloc_local((warp_cols * local_size_b), in_dtype)

            if c_is_fragment:
                # Annotate layout for C_local if it is a fragment.
                T.annotate_layout({
                    C_local: mma_emitter.make_mma_store_layout(C_local),
                })

            for ki in T.serial(0, (block_K // micro_size_k)):
                # Load A into fragment
                mma_emitter.ldmatrix_a(
                    A_local,
                    A_shared,
                    ki,
                )

                # Load B into fragment
                mma_emitter.ldmatrix_b(
                    B_local,
                    B_shared,
                    ki,
                )

                # Perform Matrix Multiplication
                mma_emitter.mma(A_local, B_local, C_local)

        return _gemm_ssr(A, B, C)

    def invoke(self) -> tir.PrimExpr:
        """
        Entry point to generate a GEMM SSR (single-step reduction) with Tensor
        Core instructions. Performs the following steps:
            1. Infers block partition parameters if necessary.
            2. Creates a `TensorCoreIntrinEmitter` with the correct data types
               and dimensions.
            3. Invokes the GEMM SSR function to generate the final IR expression.

        Returns
        -------
        tir.PrimExpr
            The generated GEMM IR expression.
        """

        # Infer block partition if necessary
        current_frame = T.KernelLaunchFrame.Current()
        threads = current_frame.get_num_threads()

        self.infer_block_partition(threads)

        A, B, C = self.A, self.B, self.C
        transpose_A, transpose_B = self.transpose_A, self.transpose_B
        block_row_warps, block_col_warps = (
            self.block_row_warps,
            self.block_col_warps,
        )
        warp_row_tiles, warp_col_tiles = (
            self.warp_row_tiles,
            self.warp_col_tiles,
        )
        chunk = self.chunk

        # Check dtypes
        A_dtype, B_dtype, C_dtype = A.dtype, B.dtype, C.dtype
        assert A_dtype == B_dtype, "A and B must have the same dtype"
        in_dtype, accum_dtype = A_dtype, C_dtype

        # Create the MMA emitter
        mma_emitter = TensorCoreIntrinEmitter(
            a_dtype=in_dtype,
            b_dtype=in_dtype,
            accum_dtype=accum_dtype,
            a_transposed=transpose_A,
            b_transposed=transpose_B,
            block_row_warps=block_row_warps,
            block_col_warps=block_col_warps,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=chunk,
        )
        a_is_fragment = is_fragment(A)
        b_is_fragment = is_fragment(B)
        if a_is_fragment and b_is_fragment:
            return self.gemm_rrr(A, B, C, mma_emitter)
        if a_is_fragment:
            return self.gemm_rsr(A, B, C, mma_emitter)
        if b_is_fragment:
            return self.gemm_srr(A, B, C, mma_emitter)
        return self.gemm_ssr(A, B, C, mma_emitter)

    @property
    def in_dtype(self) -> str:
        """
        Returns
        -------
        str
            The input data type for A and B. Assumes both have the same dtype.

        Raises
        ------
        AssertionError
            If A and B do not share the same dtype.
        """
        A_dtype, B_dtype = self.A.dtype, self.B.dtype
        assert A_dtype == B_dtype, "A and B must have the same dtype"
        return self.A.dtype

    @property
    def accum_dtype(self) -> str:
        """
        Returns
        -------
        str
            The accumulation data type for C.
        """
        return self.C.dtype


__all__ = ["GemmPrimitiveMMA"]
