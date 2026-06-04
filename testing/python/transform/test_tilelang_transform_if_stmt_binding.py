from tilelang import tvm as tvm
import tilelang
import tilelang.testing
from tvm.script import tirx as T


def _run_if_stmt_binding(func, config=None):
    mod = tvm.IRModule({"main": func})
    with tvm.transform.PassContext(config=config or {}):
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


def test_if_stmt_binding_inlines_replayable_bind_by_default():
    @T.prim_func
    def before(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32")):
        if A[0] >= T.float32(0):
            bound = T.bind(A[1] + T.float32(2))
            B[0] = bound
            B[1] = bound + T.float32(1)

    @T.prim_func
    def expected(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32")):
        if A[0] >= T.float32(0):
            B[0] = A[1] + T.float32(2)
        if A[0] >= T.float32(0):
            B[1] = A[1] + T.float32(2) + T.float32(1)

    after = _run_if_stmt_binding(before)
    tvm.ir.assert_structural_equal(after.body, expected.body, True)


def test_if_stmt_binding_does_not_inline_pointer_bind():
    @T.prim_func(check_well_formed=False)
    def before(A: T.Buffer((4,), "float32"), C: T.Buffer((4,), "float32")):
        if A[0] >= T.float32(0):
            idx = T.bind(T.int32(1))
            ptr = T.bind(A.data, type_annotation=A.data)
            B = T.Buffer((4,), "float32", data=ptr)
            C[0] = B[idx]

    @T.prim_func(check_well_formed=False)
    def expected(A: T.Buffer((4,), "float32"), C: T.Buffer((4,), "float32")):
        if A[0] >= T.float32(0):
            ptr = T.bind(A.data, type_annotation=A.data)
            B = T.Buffer((4,), "float32", data=ptr)
            C[0] = B[T.int32(1)]

    after = _run_if_stmt_binding(before)
    tvm.ir.assert_structural_equal(after.body, expected.body, True)


def test_if_stmt_binding_can_disable_replayable_bind_inline():
    @T.prim_func
    def before(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32")):
        if A[0] >= T.float32(0):
            bound = T.bind(A[1] + T.float32(2))
            B[0] = bound
            B[1] = bound + T.float32(1)

    @T.prim_func
    def expected(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32")):
        if A[0] >= T.float32(0):
            bound = T.bind(A[1] + T.float32(2))
            B[0] = bound
            B[1] = bound + T.float32(1)

    config_key = tilelang.PassConfigKey.TL_IF_STMT_BINDING_INLINE_REPLAYABLE_BINDS.value
    after = _run_if_stmt_binding(before, config={config_key: False})
    tvm.ir.assert_structural_equal(after.body, expected.body, True)


if __name__ == "__main__":
    tilelang.testing.main()
