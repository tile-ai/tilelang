from tilelang import tvm as tvm
from tvm.ir import Node, Scriptable
import tvm.ffi


@tvm.ffi.register_object("tl.GemmSP")
class GemmSP(Node, Scriptable):
    ...
