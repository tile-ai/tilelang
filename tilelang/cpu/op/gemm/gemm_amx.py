from __future__ import annotations

from tilelang import language as T
from tilelang.transform import get_pass_context
from tilelang.tileop.gemm.gemm_base import GemmBase
from tvm import tirx
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

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_index: tirx.PrimExpr,
        mbar_phase_expr: tirx.PrimExpr | None = None,
    ):
        M, N, K = self.M, self.N, self.K
        A_buf = self.ARegion.buffer
        B_buf = self.BRegion.buffer
        C_buf = self.CRegion.buffer

        a0, a1 = (axis.min for axis in self.ARegion.region)
        b0, b1 = (axis.min for axis in self.BRegion.region)
        c0, c1 = (axis.min for axis in self.CRegion.region)

        # The frontend allocates dense local panels in logical source shape.
        # Their physical row strides therefore depend on the transpose flags.
        lda = M if self.trans_A else K
        ldb = K if self.trans_B else N
        ldc = N
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
                T.access_ptr(A_buf[a0, a1], "r", extent=M * K),
                T.access_ptr(B_buf[b0, b1], "r", extent=K * N),
                T.access_ptr(C_buf[c0, c1], "rw", extent=M * N),
                M,
                N,
                K,
                lda,
                ldb,
                ldc,
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
            return "tl_amx_bf16_gemm_parallel"
        return "tl_amx_bf16_gemm"


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
        return helper_name

    def _extra_args(self) -> tuple[int, ...]:
        return (int(str(self.a_dtype) == "uint8"), int(str(self.b_dtype) == "uint8"))
