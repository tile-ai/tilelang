from dataclasses import dataclass
from tilelang import tvm as tvm
from tvm.target import Target
from tvm import tirx
from tvm.ir import Range
from tilelang.utils.language import is_shared, is_fragment
from tilelang.ir import GemmSPWarpPolicy
from tvm.ir.base import Node


@dataclass
class GemmSPBase:
    gemm_sp_node: Node

    def infer_layout(self, target: Target, thread_nums: int):
        raise NotImplementedError("infer_layout is not implemented")

    def lower(self, layout_map: dict, target: Target, thread_bounds: Range, thread_index: tirx.PrimExpr):
        raise NotImplementedError("lower is not implemented")

    def is_gemm_ss(self) -> bool:
        return is_shared(self.A) and is_shared(self.B)

    def is_gemm_sr(self) -> bool:
        return is_shared(self.A) and is_fragment(self.B)

    def is_gemm_rs(self) -> bool:
        return is_fragment(self.A) and is_shared(self.B)

    def is_gemm_rr(self) -> bool:
        return is_fragment(self.A) and is_fragment(self.B)

    @property
    def M(self) -> int:
        return self.gemm_sp_node.M

    @property
    def N(self) -> int:
        return self.gemm_sp_node.N

    @property
    def K(self) -> int:
        return self.gemm_sp_node.K

    @property
    def trans_A(self) -> bool:
        return self.gemm_sp_node.trans_A

    @property
    def trans_B(self) -> bool:
        return self.gemm_sp_node.trans_B

    @property
    def trans_E(self) -> bool:
        return self.gemm_sp_node.trans_E

    @property
    def e_dtype(self) -> str:
        return self.E.dtype

    @property
    def a_dtype(self):
        assert self.A.dtype == self.B.dtype, "A and B must have the same dtype"
        return self.A.dtype

    @property
    def b_dtype(self):
        assert self.A.dtype == self.B.dtype, "A and B must have the same dtype"
        return self.B.dtype

    @property
    def accum_dtype(self) -> str:
        return self.C.dtype

    @property
    def A(self) -> tirx.Buffer:
        return self.gemm_sp_node.A

    @property
    def E(self) -> tirx.Buffer:
        return self.gemm_sp_node.E

    @property
    def B(self) -> tirx.Buffer:
        return self.gemm_sp_node.B

    @property
    def C(self) -> tirx.Buffer:
        return self.gemm_sp_node.C

    @property
    def ARegion(self) -> tirx.PrimExpr:
        return self.gemm_sp_node.aRegion

    @property
    def ERegion(self) -> tirx.PrimExpr:
        return self.gemm_sp_node.eRegion

    @property
    def BRegion(self) -> tirx.PrimExpr:
        return self.gemm_sp_node.bRegion

    @property
    def CRegion(self) -> tirx.PrimExpr:
        return self.gemm_sp_node.cRegion

    @property
    def stride_A(self) -> int:
        return self.gemm_sp_node.stride_A

    @property
    def stride_B(self) -> int:
        return self.gemm_sp_node.stride_B

    @property
    def offset_A(self) -> int:
        return self.gemm_sp_node.offset_A

    @property
    def offset_B(self) -> int:
        return self.gemm_sp_node.offset_B

    @property
    def clear_accum(self) -> bool:
        return self.gemm_sp_node.clear_accum

    @property
    def k_pack(self) -> int:
        return self.gemm_sp_node.k_pack

    @property
    def wg_wait(self) -> int:
        return self.gemm_sp_node.wg_wait

    @property
    def policy(self) -> GemmSPWarpPolicy:
        return self.gemm_sp_node.policy
