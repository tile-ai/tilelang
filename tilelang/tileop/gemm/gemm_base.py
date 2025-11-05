from dataclasses import dataclass
from tilelang import tvm as tvm
from tvm.target import Target
from tvm import tir
from tilelang.utils.language import is_shared, is_fragment
from tilelang.ir import GemmWarpPolicy
from tvm.ir.base import Node
from tvm.ir import PrimExpr


@dataclass
class GemmBase:
    gemm_node: Node

    def infer_layout(self, target: Target, thread_nums: int):
        raise NotImplementedError("infer_layout is not implemented")

    def lower(self, target: Target, thread_nums: int, thread_var: tir.Var):
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
        return self.gemm_node.M

    @property
    def N(self) -> int:
        return self.gemm_node.N

    @property
    def K(self) -> int:
        return self.gemm_node.K

    @property
    def trans_A(self) -> bool:
        return self.gemm_node.trans_A

    @property
    def trans_B(self) -> bool:
        return self.gemm_node.trans_B

    @property
    def in_dtype(self) -> str:
        assert self.A.dtype == self.B.dtype, "A and B must have the same dtype"
        return self.A.dtype

    @property
    def accum_dtype(self) -> str:
        return self.C.dtype

    @property
    def chunk(self) -> int:
        return self.A.shape[-2] if self.trans_A else self.A.shape[-1]

    @property
    def A(self) -> tir.Buffer:
        return self.gemm_node.A

    @property
    def B(self) -> tir.Buffer:
        return self.gemm_node.B

    @property
    def C(self) -> tir.Buffer:
        return self.gemm_node.C

    @property
    def APtr(self) -> tir.PrimExpr:
        return self.gemm_node.APtr

    @property
    def BPtr(self) -> tir.PrimExpr:
        return self.gemm_node.BPtr

    @property
    def CPtr(self) -> tir.PrimExpr:
        return self.gemm_node.CPtr

    @property
    def stride_A(self) -> int:
        return self.gemm_node.stride_A

    @property
    def stride_B(self) -> int:
        return self.gemm_node.stride_B

    @property
    def offset_A(self) -> int:
        return self.gemm_node.offset_A

    @property
    def offset_B(self) -> int:
        return self.gemm_node.offset_B

    @property
    def clear_accum(self) -> PrimExpr:
        return self.gemm_node.clear_accum

    @property
    def k_pack(self) -> int:
        return self.gemm_node.k_pack

    @property
    def wg_wait(self) -> int:
        return self.gemm_node.wg_wait

    @property
    def policy(self) -> GemmWarpPolicy:
        return self.gemm_node.policy

    @property
    def mbarptr(self) -> PrimExpr:
        return getattr(self.gemm_node, "mbarptr", tvm.tir.const(0, "uint32"))

    @property
    def C_coords(self):
        coords = getattr(self.gemm_node, "C_coords", None)
        if coords is None or len(coords) == 0:
            zero = tvm.tir.const(0, "int32")
            return [zero, zero]
        return [coords[i] for i in range(len(coords))]
