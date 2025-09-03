from tilelang import tvm as tvm
from tvm.ir import Node, Scriptable
import tvm.ffi

@tvm.ffi.register_object("tl.ReduceOp")
class ReduceOp(Node, Scriptable):
    ...

@tvm.ffi.register_object("tl.CumSumOp")
class CumSumOp(Node, Scriptable):
    ...