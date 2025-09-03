from tilelang import tvm as tvm
from tvm.ir.base import Node
from tvm.runtime import Scriptable
import tvm.ffi


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
    ...


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
