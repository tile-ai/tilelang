from __future__ import annotations

from .gemm_base import GemmBase
from .inst import GemmInst
from tilelang.layout import make_swizzled_layout
from tilelang.intrinsics.mma_macro_generator import (
    TensorCoreIntrinEmitter,
    SubByteTensorCoreMMASpec,
    get_subbyte_tensorcore_mma_spec,
)
from tilelang.utils.language import is_shared, is_fragment, is_full_region
from tilelang import tvm as tvm
from tvm.target import Target
from tvm.ir import Range
from tvm import tir
from tilelang import language as T
from tilelang.transform.simplify import _Simplify


class SubByteGemmOperandAdaptor:
    def __init__(self, mma_spec: SubByteTensorCoreMMASpec):
        self.mma_spec = mma_spec

    def get_storage_dtype(self, matrix: str) -> str:
        return self.mma_spec.get_storage_dtype(matrix)

    def get_packed_chunk(self, logical_chunk: int, matrix: str = "A") -> int:
        logical_chunk = int(logical_chunk)
        return self.mma_spec.pack_extent(logical_chunk, matrix)

    def make_packed_buffer(self, buf: tir.Buffer, matrix: str) -> tir.Buffer:
        shape = list(buf.shape)
        if len(shape) < 2:
            raise ValueError(f"{self.mma_spec.get_logical_dtype(matrix)} T.gemm expects at least 2D operands, but got shape={shape}")
        packed_last_dim = int(shape[-1])
        pack_factor = self.mma_spec.get_pack_factor(matrix)
        if packed_last_dim % pack_factor != 0:
            raise ValueError(
                f"{self.mma_spec.get_logical_dtype(matrix)} T.gemm expects an innermost K extent divisible by "
                f"{pack_factor}, but got {packed_last_dim}"
            )
        shape[-1] = packed_last_dim // pack_factor
        return T.view(buf, tuple(shape), dtype=self.get_storage_dtype(matrix))

    def make_packed_region(self, region: tir.BufferRegion, matrix: str) -> tir.BufferRegion:
        packed_buf = self.make_packed_buffer(region.buffer, matrix)
        pack_factor = self.mma_spec.get_pack_factor(matrix)
        packed_ranges = list(region.region)
        last_range = packed_ranges[-1]
        packed_ranges[-1] = Range.from_min_extent(last_range.min // pack_factor, last_range.extent // pack_factor)
        return tir.BufferRegion(packed_buf, packed_ranges)


class GemmMMA(GemmBase):
    def _get_subbyte_mma_spec(self) -> SubByteTensorCoreMMASpec | None:
        return get_subbyte_tensorcore_mma_spec(self.in_dtype)

    def _get_subbyte_operand_adaptor(self) -> SubByteGemmOperandAdaptor | None:
        mma_spec = self._get_subbyte_mma_spec()
        if mma_spec is None:
            return None
        return SubByteGemmOperandAdaptor(mma_spec)

    def _validate_subbyte_mma_support(self, mma_spec: SubByteTensorCoreMMASpec):
        chunk = int(self.chunk)
        pack_factor_a = mma_spec.get_pack_factor("A")
        pack_factor_b = mma_spec.get_pack_factor("B")
        if not self.is_gemm_ss():
            raise ValueError(f"{self.in_dtype} T.gemm currently only supports shared/shared operands in the subbyte MMA path")
        if self.trans_A or not self.trans_B:
            raise ValueError(
                f"{self.in_dtype} T.gemm currently only supports innermost-K packed layout (transpose_A=False, transpose_B=True)"
            )
        if str(self.accum_dtype) != str(mma_spec.accum_dtype):
            raise ValueError(
                f"{self.in_dtype} T.gemm currently only supports {mma_spec.accum_dtype} accumulation, but got {self.accum_dtype}"
            )
        if chunk % pack_factor_a != 0:
            raise ValueError(f"{self.in_dtype} T.gemm expects the A K tile to be divisible by {pack_factor_a}, but got chunk={chunk}")
        if chunk % pack_factor_b != 0:
            raise ValueError(f"{self.in_dtype} T.gemm expects the B K tile to be divisible by {pack_factor_b}, but got chunk={chunk}")

    def _make_mma_emitter(self, target: Target, thread_nums: int, thread_var: tir.Var | None = None):
        m_warp, n_warp = self.policy.compute_warp_partition(self.M, self.N, thread_nums, target, GemmInst.MMA)
        warp_row_tiles = int(self.M // m_warp)
        warp_col_tiles = int(self.N // n_warp)
        subbyte_mma_spec = self._get_subbyte_mma_spec()
        subbyte_adaptor = self._get_subbyte_operand_adaptor()
        if subbyte_mma_spec is not None and subbyte_adaptor is not None:
            self._validate_subbyte_mma_support(subbyte_mma_spec)
            packed_chunk_a = subbyte_adaptor.get_packed_chunk(self.chunk, matrix="A")
            packed_chunk_b = subbyte_adaptor.get_packed_chunk(self.chunk, matrix="B")
            if packed_chunk_a != packed_chunk_b:
                raise ValueError(
                    f"Subbyte MMA currently expects A/B to use the same packed K tile, but got A={packed_chunk_a}, B={packed_chunk_b}"
                )
            emitter = TensorCoreIntrinEmitter(
                a_dtype=self.in_dtype,
                b_dtype=self.in_dtype,
                accum_dtype=self.accum_dtype,
                a_transposed=self.trans_A,
                b_transposed=self.trans_B,
                block_row_warps=m_warp,
                block_col_warps=n_warp,
                warp_row_tiles=warp_row_tiles,
                warp_col_tiles=warp_col_tiles,
                chunk=packed_chunk_a,
                thread_var=thread_var,
            )
            return emitter, m_warp, n_warp
        emitter = TensorCoreIntrinEmitter(
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
        return emitter, m_warp, n_warp

    def infer_layout(self, target: Target, thread_nums: int):
        mma_emitter, _, _ = self._make_mma_emitter(target, thread_nums)
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
        thread_var: tir.Var,
        mbar_phase_expr: tir.PrimExpr | None = None,
    ):
        thread_nums = thread_bounds.extent
        mma_emitter, _, _ = self._make_mma_emitter(target, thread_nums, thread_var=thread_var)
        subbyte_adaptor = self._get_subbyte_operand_adaptor()

        a_local_dtype = subbyte_adaptor.get_storage_dtype("A") if subbyte_adaptor is not None else self.in_dtype
        b_local_dtype = subbyte_adaptor.get_storage_dtype("B") if subbyte_adaptor is not None else self.in_dtype
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
        if subbyte_adaptor is not None:
            A_region = subbyte_adaptor.make_packed_region(A_region, "A")
            B_region = subbyte_adaptor.make_packed_region(B_region, "B")

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
                The inner macro that loads data from shared buffers A_shared and
                B_shared into local fragments, then issues Tensor Core mma ops,
                accumulating into C_local.
                """
                A_local = T.alloc_local((warp_rows * local_size_a), a_local_dtype)
                B_local = T.alloc_local((warp_cols * local_size_b), b_local_dtype)
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
                    if subbyte_adaptor is not None:
                        mma_emitter.mma(A_local, B_local, C_buf)
                    else:
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
                A_local = T.alloc_local((warp_rows * local_size_a), a_local_dtype)

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
                B_local = T.alloc_local((warp_cols * local_size_b), b_local_dtype)
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
