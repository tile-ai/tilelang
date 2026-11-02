from __future__ import annotations

from tilelang.tileop.gemm.gemm_base import GemmBase
from tilelang.layout import make_ppu_swizzled_layout, make_swizzled_layout
from tilelang.ppu.intrinsics.macro.mma_macro_generator import (
    PPUTensorCoreIntrinEmitter,
)
from tilelang.utils.language import is_shared, is_fragment, is_full_region
from tvm.target import Target
from tvm.ir import Range
from tvm import arith, tirx, DataType
from tilelang import language as T
from tilelang.transform.simplify import _Simplify

GEMM_INST_MMA_PPU = "ppu.mma"


def _can_load_scale_fragment(region: tirx.BufferRegion, layout_map: dict, k_tiles: int) -> bool:
    """Prove that each b16 matrix-load row is contiguous and 16-byte aligned."""
    buffer = region.buffer
    if not is_shared(buffer) or buffer.data_alignment % 16 != 0 or buffer.data_alignment < 16:
        return False
    # An explicit layout may change both physical adjacency and row alignment.
    if layout_map.get(buffer) is not None:
        return False

    strides = list(buffer.strides)
    if not strides:
        stride = tirx.IntImm("int32", 1)
        for extent in reversed(buffer.shape):
            strides.insert(0, stride)
            stride = stride * extent

    analyzer = arith.Analyzer()
    if not analyzer.can_prove(strides[-1] == 1):
        return False
    offset = buffer.elem_offset + sum(axis.min * stride for axis, stride in zip(region.region, strides))
    # Scale elements are uint16; each lane supplies eight adjacent elements.
    if not analyzer.can_prove(tirx.floormod(offset, 8) == 0):
        return False
    return k_tiles == 1 or analyzer.can_prove(tirx.floormod(strides[-2], 8) == 0)


class PPUGemmMMA(GemmBase):
    intrin_emitter_cls = PPUTensorCoreIntrinEmitter

    @property
    def is_blockscaled(self) -> bool:
        return False

    @property
    def allow_f8f6f4_mixed_dtypes(self) -> bool:
        return True

    def _make_mma_emitter(self, target: Target, thread_nums: int, thread_index: tirx.PrimExpr | None = None):
        m_warp, n_warp = self.policy.compute_warp_partition(self.M, self.N, thread_nums, target, GEMM_INST_MMA_PPU)
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
            thread_index=thread_index,
        )
        arch = str(target.attrs["arch"]) if target and "arch" in target.attrs else "ppu_10"
        try:
            emitter.ppu_arch = int(arch.split("_")[-1].rstrip("af"))
        except ValueError:
            emitter.ppu_arch = 10
        return emitter

    @staticmethod
    def _read_a_from_gemm_c_annotation(gemm_node) -> bool:
        """Read the 'a_from_gemm_c' annotation from gemm_node. Returns False if absent."""
        try:
            annotations = getattr(gemm_node, "annotations", None)
            if not annotations:
                return False
            value = None
            get = getattr(annotations, "get", None)
            if callable(get):
                value = get("a_from_gemm_c")
            elif "a_from_gemm_c" in annotations:
                value = annotations["a_from_gemm_c"]
            if value is None:
                return False
            if isinstance(value, bool):
                return value
            return bool(int(getattr(value, "value", value)))
        except Exception:
            return False

    def _is_ppu0010_fp16_config(self, mma_emitter) -> bool:
        """True if running on PPU0010 with fp16 A/B and fp32 accumulator."""
        return (
            mma_emitter.ppu_arch == 10
            and DataType(self.a_dtype).bits == 16
            and DataType(self.b_dtype).bits == 16
            and DataType(self.accum_dtype).bits == 32
        )

    def _is_chained_rs_gemm(self, ppu_arch: int) -> bool:
        """True if this is a chained RS GEMM (A from prior GEMM C) on PPU0010."""
        return (
            self.is_gemm_rs()
            and ppu_arch == 10
            and DataType(self.a_dtype).bits == 16
            and DataType(self.b_dtype).bits == 16
            and DataType(self.accum_dtype).bits == 32
            and self._read_a_from_gemm_c_annotation(self.gemm_node)
        )

    def infer_layout(self, target: Target, thread_nums: int):
        mma_emitter = self._make_mma_emitter(target, thread_nums)
        if self.is_gemm_ss():
            return {
                self.A: make_swizzled_layout(self.A),
                self.B: make_swizzled_layout(self.B),
                self.C: mma_emitter.make_mma_store_layout(self.C),
            }
        elif self.is_gemm_sr():
            if self._is_ppu0010_fp16_config(mma_emitter):
                return {
                    self.A: make_swizzled_layout(self.A),
                    self.B: mma_emitter.make_mma_load_layout(self.B, matrix="B", is_regB=True),
                    self.C: mma_emitter.make_mma_store_layout(self.C),
                }
            return {
                self.A: make_swizzled_layout(self.A),
                self.B: mma_emitter.make_mma_load_layout(self.B, matrix="B"),
                self.C: mma_emitter.make_mma_store_layout(self.C),
            }
        elif self.is_gemm_rs():
            if self._is_ppu0010_fp16_config(mma_emitter):
                a_from_gemm_c = self._read_a_from_gemm_c_annotation(self.gemm_node)
                return {
                    self.A: mma_emitter.make_mma_load_layout(self.A, matrix="A", a_from_gemm_c=a_from_gemm_c),
                    self.B: make_ppu_swizzled_layout(
                        self.B,
                        k_major=self.trans_B,
                        is_gemm_rs=a_from_gemm_c,
                    ),
                    self.C: mma_emitter.make_mma_store_layout(self.C),
                }
            return {
                self.A: mma_emitter.make_mma_load_layout(self.A, matrix="A"),
                self.B: make_swizzled_layout(self.B),
                self.C: mma_emitter.make_mma_store_layout(self.C),
            }
        elif self.is_gemm_rr():
            if self._is_ppu0010_fp16_config(mma_emitter):
                return {
                    self.A: mma_emitter.make_mma_load_layout(self.A, matrix="A"),
                    self.B: mma_emitter.make_mma_load_layout(self.B, matrix="B", is_regB=True),
                    self.C: mma_emitter.make_mma_store_layout(self.C),
                }
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
        thread_index: tirx.PrimExpr,
        mbar_phase_expr: tirx.PrimExpr | None = None,
    ):
        thread_nums = thread_bounds.extent
        # Emitter lane/warp math uses zero-based ids within the current thread bounds.
        local_thread_index = thread_index - thread_bounds.min
        # Detect chained RS GEMM for this lower call
        arch = str(target.attrs["arch"]) if target and "arch" in target.attrs else "ppu_10"
        try:
            ppu_arch = int(arch.split("_")[-1].rstrip("af"))
        except ValueError:
            ppu_arch = 10
        a_from_gemm_c = self._is_chained_rs_gemm(ppu_arch)
        if a_from_gemm_c and self.trans_B:
            required_layout = make_ppu_swizzled_layout(self.B, k_major=True, is_gemm_rs=True)
            b_layout = layout_map.get(self.B)
            if b_layout is None or not b_layout.is_equal(required_layout):
                raise ValueError(
                    "PPU0010 chained RS GEMM with transpose_B=True does not support the retained B layout. "
                    "Use make_ppu_swizzled_layout(B, k_major=True, is_gemm_rs=True) for this shortcut."
                )
        mma_emitter = self._make_mma_emitter(target, thread_nums, thread_index=local_thread_index)
        # SWZL is a full-warp load; select it from the final inferred layouts.
        if arith.Analyzer().can_prove(thread_bounds.min % 32 == 0):
            mma_emitter.layout_map = layout_map

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
        scale_A_region = self.SFARegion if self.is_blockscaled else None
        scale_B_region = self.SFBRegion if self.is_blockscaled else None

        if self.is_blockscaled:
            if str(a_dtype) != "float4_e2m1fn" or str(b_dtype) != "float4_e2m1fn":
                raise ValueError("PPU runtime block scales are only supported for FP4 x FP4 T.gemm")
            if int(self.K) < micro_size_k or int(self.K) % micro_size_k != 0:
                raise ValueError("PPU MXFP4 runtime scales require the GEMM tile K to be a positive multiple of 64")
            if str(scale_A_region.buffer.dtype) != "uint16" or str(scale_B_region.buffer.dtype) != "uint16":
                raise ValueError("PPU MXFP4 scales must use uint16 packed E8M0 pairs")
            if not scale_A_region.region or not scale_B_region.region:
                raise ValueError("PPU MXFP4 scale regions must have a row or column dimension")
            scale_k_tiles = int(self.K) // micro_size_k
            if len(scale_A_region.region) != len(scale_B_region.region):
                raise ValueError("PPU MXFP4 scale_A and scale_B regions must have the same rank")
            # Software pipelining may prepend one or more stage dimensions to
            # the user-visible scale region.  For a multi-atom K tile, the
            # penultimate dimension remains the K=64 atom index; for K=64 the
            # atom index is implicit and every leading dimension is a prefix.
            if scale_k_tiles > 1:
                if len(scale_A_region.region) < 2:
                    raise ValueError("PPU MXFP4 multi-atom K tiles require two-dimensional scale regions")
                if int(scale_A_region.region[-2].extent) != scale_k_tiles:
                    raise ValueError("PPU MXFP4 scale_A must contain one scale row per K=64 MMA atom")
                if int(scale_B_region.region[-2].extent) != scale_k_tiles:
                    raise ValueError("PPU MXFP4 scale_B must contain one scale row per K=64 MMA atom")
            if int(scale_A_region.region[-1].extent) != int(self.M):
                raise ValueError("PPU MXFP4 scale_A must contain one uint16 per A row in the GEMM tile")
            if int(scale_B_region.region[-1].extent) != int(self.N):
                raise ValueError("PPU MXFP4 scale_B must contain one uint16 per B column in the GEMM tile")

        A_buf = A_region.buffer
        B_buf = B_region.buffer
        C_buf = C_region.buffer

        clear_accum = self.clear_accum

        assert block_K >= micro_size_k, f"block_K ({block_K}) must be >= micro_size_k ({micro_size_k})"
        assert block_K % micro_size_k == 0, f"block_K ({block_K}) must be a multiple of micro_size_k ({micro_size_k})"

        assert is_full_region(C_region), "Fragment output C must be a full region"

        sfa_local_size = (warp_rows + 3) // 4
        sfb_local_size = (warp_cols + 3) // 4
        use_scale_fragments = (
            self.is_blockscaled
            and (warp_rows in (1, 2) or warp_rows % 4 == 0)
            and (warp_cols in (1, 2) or warp_cols % 4 == 0)
            and int(self.M) % 64 == 0
            and int(self.N) % 64 == 0
            and _can_load_scale_fragment(scale_A_region, layout_map, scale_k_tiles)
            and _can_load_scale_fragment(scale_B_region, layout_map, scale_k_tiles)
        )

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
                if use_scale_fragments:
                    SFA_local = T.alloc_local((sfa_local_size,), T.uint32)
                    SFB_local = T.alloc_local((sfb_local_size,), T.uint32)
                if clear_accum:
                    T.clear(C_buf)
                for ki in T.serial(0, (block_K // micro_size_k)):
                    if use_scale_fragments:
                        mma_emitter.load_scale(SFA_local, scale_A_region, ki, operand="A")
                        mma_emitter.load_scale(SFB_local, scale_B_region, ki, operand="B")

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
                    if use_scale_fragments:
                        mma_emitter.mma_blockscaled(
                            A_local,
                            B_local,
                            C_buf,
                            SFA_local,
                            SFB_local,
                            ki,
                        )
                    else:
                        mma_emitter.mma(A_local, B_local, C_buf, ki, scale_A_region, scale_B_region)

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
                if use_scale_fragments:
                    SFA_local = T.alloc_local((sfa_local_size,), T.uint32)
                    SFB_local = T.alloc_local((sfb_local_size,), T.uint32)

                if clear_accum:
                    T.clear(C_buf)
                for ki in T.serial(0, (block_K // micro_size_k)):
                    if use_scale_fragments:
                        mma_emitter.load_scale(SFA_local, scale_A_region, ki, operand="A")
                        mma_emitter.load_scale(SFB_local, scale_B_region, ki, operand="B")

                    # Load A into fragment
                    mma_emitter.ldmatrix_a(
                        A_local,
                        A_region,
                        ki,
                    )

                    # Perform Matrix Multiplication
                    if use_scale_fragments:
                        mma_emitter.mma_blockscaled(
                            A_local,
                            B_buf,
                            C_buf,
                            SFA_local,
                            SFB_local,
                            ki,
                        )
                    else:
                        mma_emitter.mma(A_local, B_buf, C_buf, ki, scale_A_region, scale_B_region)

            # Simplify to optimize the index computing
            # Must inline let statements to simplify the analysis
            # alloc_buffers body
            # insert into parent block
            return _Simplify(_gemm_srr, inline_let=True)
        elif self.is_gemm_rs():
            # PPU: RS with A sourced from a previous GEMM's C fragment needs a
            # dedicated lowering that reads A directly as a fragment buffer
            # and only loads B via ldmatrix.

            assert is_full_region(A_region), "Fragment input A must be a full region"

            @T.prim_func
            def _gemm_rsr() -> None:
                """
                The inner macro that loads data from shared buffers A_shared and
                B_shared into local fragments, then issues Tensor Core mma ops,
                accumulating into C_local.
                """
                B_local = T.alloc_local((warp_cols * local_size_b), b_dtype)
                if use_scale_fragments:
                    SFA_local = T.alloc_local((sfa_local_size,), T.uint32)
                    SFB_local = T.alloc_local((sfb_local_size,), T.uint32)
                if clear_accum:
                    T.clear(C_buf)
                for ki in T.serial(0, (block_K // micro_size_k)):
                    if use_scale_fragments:
                        mma_emitter.load_scale(SFA_local, scale_A_region, ki, operand="A")
                        mma_emitter.load_scale(SFB_local, scale_B_region, ki, operand="B")

                    # Load B into fragment
                    mma_emitter.ldmatrix_b(
                        B_local,
                        B_region,
                        ki,
                        a_from_gemm_c=a_from_gemm_c,
                    )

                    # Perform Matrix Multiplication
                    if use_scale_fragments:
                        mma_emitter.mma_blockscaled(
                            A_buf,
                            B_local,
                            C_buf,
                            SFA_local,
                            SFB_local,
                            ki,
                        )
                    else:
                        mma_emitter.mma(A_buf, B_local, C_buf, ki, scale_A_region, scale_B_region)

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

                if use_scale_fragments:
                    SFA_local = T.alloc_local((sfa_local_size,), T.uint32)
                    SFB_local = T.alloc_local((sfb_local_size,), T.uint32)
                if clear_accum:
                    T.clear(C_buf)
                for ki in T.serial(0, (block_K // micro_size_k)):
                    if use_scale_fragments:
                        mma_emitter.load_scale(SFA_local, scale_A_region, ki, operand="A")
                        mma_emitter.load_scale(SFB_local, scale_B_region, ki, operand="B")

                    # Perform Matrix Multiplication
                    if use_scale_fragments:
                        mma_emitter.mma_blockscaled(
                            A_buf,
                            B_buf,
                            C_buf,
                            SFA_local,
                            SFB_local,
                            ki,
                        )
                    else:
                        mma_emitter.mma(A_buf, B_buf, C_buf, ki, scale_A_region, scale_B_region)

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
