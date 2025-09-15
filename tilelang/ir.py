from tilelang import tvm as tvm
from tvm.ir.base import Node
from tvm.runtime import Scriptable
import tvm.ffi
from tvm.target import Target
from tilelang import _ffi_api


@tvm.ffi.register_object("tl.Fill")
class Fill(Node, Scriptable):
    ...


@tvm.ffi.register_object("tl.AtomicAdd")
class AtomicAdd(Node, Scriptable):
    ...


@tvm.ffi.register_object("tl.Copy")
class Copy(Node, Scriptable):
    ...


@tvm.ffi.register_object("tl.Conv2DIm2Col")
class Conv2DIm2ColOp(Node, Scriptable):
    ...


@tvm.ffi.register_object("tl.GemmWarpPolicy")
class GemmWarpPolicy(Node, Scriptable):
    policy_type: int
    m_warp: int
    n_warp: int

    def compute_warp_partition(self, M: int, N: int, block_size: int, target: Target,
                               is_wgmma: bool):
        _ffi_api.GemmWarpPolicyComputeWarpPartition(self, int(M), int(N), int(block_size), target,
                                                    is_wgmma)
        return self.m_warp, self.n_warp


@tvm.ffi.register_object("tl.Gemm")
class Gemm(Node, Scriptable):
    ...


@tvm.ffi.register_object("tl.GemmSP")
class GemmSP(Node, Scriptable):
    ...


@tvm.ffi.register_object("tl.FinalizeReducerOp")
class FinalizeReducerOp(Node, Scriptable):
    ...


@tvm.ffi.register_object("tl.ParallelOp")
class ParallelOp(Node, Scriptable):
    ...


@tvm.ffi.register_object("tl.ReduceOp")
class ReduceOp(Node, Scriptable):
    ...


@tvm.ffi.register_object("tl.CumSumOp")
class CumSumOp(Node, Scriptable):
    ...


@tvm.ffi.register_object("tl.RegionOp")
class RegionOp(Node, Scriptable):
    ...


@tvm.ffi.register_object("tl.ReduceType")
class ReduceType(Node, Scriptable):
    ...
