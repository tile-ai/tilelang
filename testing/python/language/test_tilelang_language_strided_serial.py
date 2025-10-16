import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TVM_PY_PATH = os.path.join(ROOT, "3rdparty", "tvm", "python")
LIB_PATH = os.path.join(ROOT, "build", "tvm")
if TVM_PY_PATH not in sys.path:
    sys.path.insert(0, TVM_PY_PATH)
if LIB_PATH not in sys.path:
    sys.path.insert(0, LIB_PATH)
os.environ.setdefault("TVM_LIBRARY_PATH", LIB_PATH)
os.environ["LD_LIBRARY_PATH"] = (
    LIB_PATH + ":" + os.environ["LD_LIBRARY_PATH"] if "LD_LIBRARY_PATH" in os.environ else LIB_PATH
)

import numpy as np
import tvm
from tvm.script import tir as T


def _collect_first_for(stmt):
    result = None

    def visitor(node):
        nonlocal result
        if result is None and isinstance(node, tvm.tir.For):
            result = node

    tvm.tir.stmt_functor.post_order_visit(stmt, visitor)
    return result


def test_serial_with_step_constant():
    @T.prim_func
    def strided_fill(out: T.Buffer((8,), "float32")):
        for i in T.serial(0, 8, step=2):
            out[i] = T.float32(0)

    loop = _collect_first_for(strided_fill.body)
    assert loop is not None
    analyzer = tvm.arith.Analyzer()
    assert analyzer.simplify(loop.min - T.int32(0)).value == 0
    assert analyzer.simplify(loop.extent - T.int32(8)).value == 0
    assert analyzer.simplify(loop.step - T.int32(2)).value == 0

    build_llvm = tvm.get_global_func("target.build.llvm", allow_missing=True)
    if build_llvm is not None:
        rt_mod = tvm.build(strided_fill, target="llvm")
        buf = tvm.nd.array(np.arange(8, dtype="float32"))
        rt_mod(buf)
        expected = np.arange(8, dtype="float32")
        expected[::2] = 0
        np.testing.assert_array_equal(buf.numpy(), expected)


if __name__ == "__main__":
    test_serial_with_step_constant()
