from __future__ import annotations

from tilelang.tileop.gemm.gemm_base import GemmBase
from tilelang.layout import make_swizzled_layout
from tilelang.cuda.intrinsics.macro.mma_macro_generator import (
    TensorCoreIntrinEmitter,
)
from tilelang.utils.language import is_shared, is_fragment, is_full_region
from tilelang import tvm as tvm
from tvm.target import Target
from tvm.ir import Range
from tvm import tirx
from tilelang import language as T
from tilelang.transform.simplify import _Simplify


GEMM_INST_MMA = "cuda.mma"


class GemmMMA(GemmBase):
    intrin_emitter_cls = TensorCoreIntrinEmitter

    @property
    def allow_f8f6f4_mixed_dtypes(self) -> bool:
        # Let the MMA implementation accept its A8W4 FP8/FP4 path while keeping
        # the exact supported pairs checked in _validate_mma_dtypes().
        return True

    @staticmethod
    def _is_fp8_e4m3(dtype: str) -> bool:
        return str(dtype) in {"float8_e4m3", "float8_e4m3fn", "float8_e4m3fnuz"}

    @staticmethod
    def _is_fp4_e2m1(dtype: str) -> bool:
        return str(dtype) == "float4_e2m1fn"

    @staticmethod
    def _fragment_carrier_dtype(dtype):
        if GemmMMA._is_fp4_e2m1(dtype):
            return T.float4_e2m1_unpacked
        return dtype

    @staticmethod
    def _layout_carrier_buffer(buffer):
        if GemmMMA._is_fp4_e2m1(buffer.dtype):
            return tirx.decl_buffer(
                buffer.shape,
                dtype=T.float4_e2m1_unpacked,
                name=buffer.name,
                scope=buffer.scope(),
            )
        return buffer

    def _validate_mma_dtypes(self):
        a_dtype = str(self.A.dtype)
        b_dtype = str(self.B.dtype)
        if a_dtype == b_dtype:
            return
        # Mixed A8W4 paths are selected only from semantic dtypes. Packed host
        # storage such as uint8 is not treated as an FP4 GEMM dtype.
        mixed_fp8_fp4 = (self._is_fp8_e4m3(a_dtype) and self._is_fp4_e2m1(b_dtype)) or (
            self._is_fp4_e2m1(a_dtype) and self._is_fp8_e4m3(b_dtype)
        )
        if not mixed_fp8_fp4:
            raise AssertionError(f"Unsupported mixed MMA dtypes: A={a_dtype}, B={b_dtype}")

    def _make_mma_emitter(self, target: Target, thread_nums: int, thread_var: tirx.Var | None = None):
        self._validate_mma_dtypes()
        m_warp, n_warp = self.policy.compute_warp_partition(self.M, self.N, thread_nums, target, GEMM_INST_MMA)
        warp_row_tiles = int(self.M // m_warp)
        warp_col_tiles = int(self.N // n_warp)
        emitter = self.intrin_emitter_cls(
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
        if self.chunk % emitter.micro_size_k != 0:
            raise ValueError(
                f"T.gemm K tile ({self.chunk}) must be divisible by MMA instruction K tile "
                f"({emitter.micro_size_k}) for A={self.A.dtype}, B={self.B.dtype}"
            )
        return emitter

    def infer_layout(self, target: Target, thread_nums: int):
        mma_emitter = self._make_mma_emitter(target, thread_nums)
        use_fp4_unpacked_layout = self._is_fp4_e2m1(self.A.dtype) or self._is_fp4_e2m1(self.B.dtype)
        A_layout_buf = self._layout_carrier_buffer(self.A) if use_fp4_unpacked_layout else self.A
        B_layout_buf = self._layout_carrier_buffer(self.B) if use_fp4_unpacked_layout else self.B
        if self.is_gemm_ss():
            return {
                self.A: make_swizzled_layout(A_layout_buf),
                self.B: make_swizzled_layout(B_layout_buf),
                self.C: mma_emitter.make_mma_store_layout(self.C),
            }
        elif self.is_gemm_sr():
            return {
                self.A: make_swizzled_layout(A_layout_buf),
                self.B: mma_emitter.make_mma_load_layout(self.B, matrix="B"),
                self.C: mma_emitter.make_mma_store_layout(self.C),
            }
        elif self.is_gemm_rs():
            return {
                self.A: mma_emitter.make_mma_load_layout(self.A, matrix="A"),
                self.B: make_swizzled_layout(B_layout_buf),
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
        a_fragment_dtype = self._fragment_carrier_dtype(a_dtype)
        b_fragment_dtype = self._fragment_carrier_dtype(b_dtype)
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

        if self.is_gemm_ss():

            @T.prim_func
            def _gemm_ssr() -> None:
                """
                The inner macro that loads data from shared buffers A_shared and
                B_shared into local fragments, then issues Tensor Core mma ops,
                accumulating into C_local.
                """
                A_local = T.alloc_local((warp_rows * local_size_a), a_fragment_dtype)
                B_local = T.alloc_local((warp_cols * local_size_b), b_fragment_dtype)
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
                A_local = T.alloc_local((warp_rows * local_size_a), a_fragment_dtype)

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
                B_local = T.alloc_local((warp_cols * local_size_b), b_fragment_dtype)
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
