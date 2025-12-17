from dataclasses import dataclass
from tilelang import tvm as tvm
from tvm.target import Target
from tvm import tir
from tilelang import language as T
from tilelang.utils.language import is_shared, is_fragment
from tilelang.tileop.base import GemmWarpPolicy
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
        return getattr(self.gemm_node, "m", None)

    @property
    def N(self) -> int:
        return getattr(self.gemm_node, "n", None)

    @property
    def K(self) -> int:
        return getattr(self.gemm_node, "k", None)

    @property
    def trans_A(self) -> bool:
        return getattr(self.gemm_node, "transA", None)

    @property
    def trans_B(self) -> bool:
        return getattr(self.gemm_node, "transB", None)

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
        return getattr(self.gemm_node, "a", None)

    @property
    def B(self) -> tir.Buffer:
        return getattr(self.gemm_node, "b", None)

    @property
    def C(self) -> tir.Buffer:
        return getattr(self.gemm_node, "c", None)

    @property
    def ARegion(self):
        return getattr(self.gemm_node, "aRegion", None)

    @property
    def BRegion(self):
        return getattr(self.gemm_node, "bRegion", None)

    @property
    def CRegion(self):
        return getattr(self.gemm_node, "cRegion", None)

    @property
    def stride_A(self) -> int:
        return getattr(self.gemm_node, "strideA", None)

    @property
    def stride_B(self) -> int:
        return getattr(self.gemm_node, "strideB", None)

    @property
    def offset_A(self) -> int:
        return getattr(self.gemm_node, "offsetA", None)

    @property
    def offset_B(self) -> int:
        return getattr(self.gemm_node, "offsetB", None)

    @property
    def clear_accum(self) -> PrimExpr:
        return getattr(self.gemm_node, "clearAccum", None)

    @property
    def k_pack(self) -> int:
        return getattr(self.gemm_node, "kPack", None)

    @property
    def wg_wait(self) -> int:
        return getattr(self.gemm_node, "wgWait", 0)

    @property
    def policy(self) -> GemmWarpPolicy:
        return getattr(self.gemm_node, "policy", None)

    @property
    def mbarptr(self) -> PrimExpr:
        return getattr(self.gemm_node, "mbarPtr", tvm.tir.const(0, T.uint32))

    @property
    def mbar(self) -> tir.Buffer:
        return getattr(self.gemm_node, "mbar", None)

    @property
    def C_coords(self):
        coords = getattr(self.gemm_node, "cCoords", None)
        if coords is None or len(coords) == 0:
            zero = tvm.tir.const(0, T.int32)
            return [zero, zero]
        return [coords[i] for i in range(len(coords))]

    def get_region_base_offsets(self, region):
        """
        Get the base offset (start index) for each dimension from a BufferRegion.

        For example, if region is A_shared[ko % 2, 0:128, 0:64],
        this returns [ko % 2, 0, 0]

        Args:
            region: BufferRegion object

        Returns:
            List of PrimExpr representing the base offset for each dimension
        """
        if region is None:
            return []
        return [r.min for r in region.region]

    @property
    def A_base_offsets(self):
        """Get base offsets for each dimension of A region"""
        return self.get_region_base_offsets(self.ARegion)

    @property
    def B_base_offsets(self):
        """Get base offsets for each dimension of B region"""
        return self.get_region_base_offsets(self.BRegion)

    @property
    def C_base_offsets(self):
        """Get base offsets for each dimension of C region"""
        return self.get_region_base_offsets(self.CRegion)
