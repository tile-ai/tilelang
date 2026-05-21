from tilelang import tvm as tvm
import tilelang
import tilelang.testing
from tvm.script import tirx as T


def _run_if_stmt_binding(func):
    mod = tvm.IRModule({"main": func})
    return tilelang.transform.IfStmtBinding()(mod)["main"]


def test_if_stmt_binding_splits_plain_seq_stmt():
    @T.prim_func
    def before(A: T.Buffer((4,), "float32")):
        if A[0] >= T.float32(0):
            A[0] = T.float32(1)
            A[1] = T.float32(2)

    @T.prim_func
    def expected(A: T.Buffer((4,), "float32")):
        if A[0] >= T.float32(0):
            A[0] = T.float32(1)
        if A[0] >= T.float32(0):
            A[1] = T.float32(2)

    after = _run_if_stmt_binding(before)
    tvm.ir.assert_structural_equal(after.body, expected.body, True)


def test_if_stmt_binding_keeps_direct_bind_scope():
    @T.prim_func
    def before(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32")):
        if A[0] >= T.float32(0):
            A[0] = T.float32(1)
            bound = T.bind(A[1] + T.float32(2))
            B[0] = bound
            B[1] = bound + T.float32(1)

    @T.prim_func
    def expected(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32")):
        if A[0] >= T.float32(0):
            A[0] = T.float32(1)
        if A[0] >= T.float32(0):
            bound = T.bind(A[1] + T.float32(2))
            B[0] = bound
            B[1] = bound + T.float32(1)

    after = _run_if_stmt_binding(before)
    tvm.ir.assert_structural_equal(after.body, expected.body, True)


if __name__ == "__main__":
    tilelang.testing.main()
