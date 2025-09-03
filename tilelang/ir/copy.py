from tilelang import tvm as tvm
from tvm.ir import Node, Scriptable
import tvm.ffi

@tvm.ffi.register_object("tl.Copy")
class Copy(Node, Scriptable):
    ...

@tvm.ffi.register_object("tl.Conv2DIm2Col")
class Conv2DIm2ColOp(Node, Scriptable):
    ...