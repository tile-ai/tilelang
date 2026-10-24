from __future__ import annotations

from tilelang import language as T
from tilelang.transform import get_pass_context
from tilelang.tileop.gemm.gemm_base import GemmBase
from tvm import arith, tirx
from tvm.ir import Range
from tvm.target import Target


GEMM_INST_AMX_BF16 = "cpu.amx.bf16"
GEMM_INST_AMX_INT8 = "cpu.amx.int8"


class _GemmAMXBase(GemmBase):
    """Lower a CPU GEMM tile to the AMX helper shipped with C codegen."""

    helper_name: str

    def infer_layout(self, target: Target, thread_nums: int):
        # CPU local buffers have ordinary dense layouts.  The helper performs
        # the AMX B-panel packing explicitly until CPU layout inference can
        # represent the VNNI-style tile layout itself.
        return {}

    def _extra_args(self) -> tuple[int, ...]:
        return ()

    @staticmethod
    def _matrix_view(region, name: str):
        """Return the element offset and physical row stride of a matrix region."""
        buffer = region.buffer
        if len(region.region) < 2:
            raise ValueError(f"AMX GEMM requires {name} to have at least two dimensions")

        analyzer = arith.Analyzer()
        for axis in region.region[:-2]:
            if not analyzer.can_prove_equal(axis.extent, 1):
                raise ValueError(f"AMX GEMM requires leading {name} dimensions to have extent one")

        if len(buffer.strides) > 0:
            if len(buffer.strides) != len(buffer.shape):
                raise ValueError(f"AMX GEMM requires complete strides for {name}")
            strides = list(buffer.strides)
        else:
            strides = [tirx.IntImm(buffer.shape[0].dtype, 1)] * len(buffer.shape)
            for axis in range(len(buffer.shape) - 2, -1, -1):
                strides[axis] = strides[axis + 1] * buffer.shape[axis + 1]

        inner_stride = strides[-1]
        if not analyzer.can_prove_equal(inner_stride, 1):
            raise ValueError(f"AMX GEMM requires a contiguous innermost {name} dimension, but got stride {inner_stride}")

        offset = tirx.IntImm(buffer.shape[0].dtype, 0)
        for axis, stride in zip(region.region, strides):
            offset += axis.min * stride
        row_stride = strides[-2]
        rows = region.region[-2].extent
        cols = region.region[-1].extent
        access_extent = offset + (rows - 1) * row_stride + cols
        return buffer, offset, access_extent, row_stride

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_index: tirx.PrimExpr,
        mbar_phase_expr: tirx.PrimExpr | None = None,
    ):
        M, N, K = self.M, self.N, self.K
        A_buf, a_offset, a_extent, lda = self._matrix_view(self.ARegion, "A")
        B_buf, b_offset, b_extent, ldb = self._matrix_view(self.BRegion, "B")
        C_buf, c_offset, c_extent, ldc = self._matrix_view(self.CRegion, "C")
        helper_name = self.helper_name
        trans_a = int(self.trans_A)
        trans_b = int(self.trans_B)
        extra_args = self._extra_args()
        clear_accum = self.clear_accum

        @T.prim_func
        def _gemm_amx() -> None:
            T.call_extern(
                "void",
                helper_name,
                T.access_ptr(A_buf, "r", extent=a_extent),
                T.access_ptr(B_buf, "r", extent=b_extent),
                T.access_ptr(C_buf, "rw", extent=c_extent),
                M,
                N,
                K,
                lda,
                ldb,
                ldc,
                a_offset,
                b_offset,
                c_offset,
                trans_a,
                trans_b,
                clear_accum,
                *extra_args,
            )

        return _gemm_amx


class GemmAMXBF16(_GemmAMXBase):
    """Intel AMX BF16 GEMM with FP32 accumulation."""

    @property
    def helper_name(self) -> str:
        pass_ctx = get_pass_context()
        if pass_ctx and pass_ctx.config.get("tl.cpu_parallel", False):
            return "tl_amx_bf16_gemm_parallel_offset"
        return "tl_amx_bf16_gemm_offset"


class GemmAMXInt8(_GemmAMXBase):
    """Intel AMX INT8 GEMM with INT32 accumulation."""

    @property
    def helper_name(self) -> str:
        a_kind = "u" if str(self.a_dtype) == "uint8" else "s"
        b_kind = "u" if str(self.b_dtype) == "uint8" else "s"
        helper_name = f"tl_amx_int8_gemm_{a_kind}{b_kind}"
        pass_ctx = get_pass_context()
        if pass_ctx and pass_ctx.config.get("tl.cpu_parallel", False):
            helper_name += "_parallel"
        return helper_name + "_offset"

    def _extra_args(self) -> tuple[int, ...]:
        return (int(str(self.a_dtype) == "uint8"), int(str(self.b_dtype) == "uint8"))
