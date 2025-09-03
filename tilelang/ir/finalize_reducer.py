from tilelang import tvm as tvm
from tvm.ir import Node, Scriptable
import tvm.ffi

@tvm.ffi.register_object("tl.FinalizeReducerOp")
class FinalizeReducerOp(Node, Scriptable):
    ...