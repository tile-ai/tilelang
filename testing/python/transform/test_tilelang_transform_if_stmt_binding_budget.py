"""Regression test for issue #3700.

`BindIfStmtWithReplayableBindInlining` used to substitute replayable bind values
without an expansion limit, so a guarded chain of `x = x + x` doubled the
expression at every step. Ten statements produced a 2**10-term expression that
made later passes take minutes.

Once the tree-size budget is reached the remaining binds stay in the guarded
lexical scope, so the final store reads the last `x` binding instead of a fully
expanded expression.
"""

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tvm import tirx
from tvm.tirx import stmt_functor


def _program():
    n = T.dynamic("N")

    @T.prim_func
    def kernel(
        A: T.Tensor((128,), "float32"),
        Mask: T.Tensor((128,), "int32"),
        Out: T.Tensor((128,), "float32"),
        Extent: T.Tensor((n,), "float32"),
    ):
        with T.Kernel(1, threads=128):
            acc = T.alloc_fragment((128,), "float32")
            T.clear(acc)
            for i in T.Parallel(128):
                if i < n:
                    far = Mask[i] != 0
                    x = A[i] + T.if_then_else(far, 0.0, 0.5)
                    x = x + x
                    x = x + x
                    x = x + x
                    x = x + x
                    x = x + x
                    x = x + x
                    x = x + x
                    x = x + x
                    x = x + x
                    x = x + x
                    acc[i] += x
            T.copy(acc, Out)

    return kernel


def _nested_program():
    n = T.dynamic("N")

    @T.prim_func
    def kernel(
        A: T.Tensor((128,), "float32"),
        Out: T.Tensor((128,), "float32"),
        Extent: T.Tensor((n,), "float32"),
    ):
        with T.Kernel(1, threads=128):
            acc = T.alloc_fragment((128,), "float32")
            T.clear(acc)
            for i in T.Parallel(128):
                if i < n:
                    x = A[i]
                    x = x + x
                    x = x + x
                    x = x + x
                    x = x + x
                    x = x + x
                    if i > 1:
                        y = x
                        y = y + y
                        y = y + y
                        y = y + y
                        y = y + y
                        y = y + y
                        acc[i] += y
            T.copy(acc, Out)

    return kernel


def test_nested_guards_stay_within_statement_budget():
    func = _nested_program()
    with tvm.transform.PassContext(config={"tl.if_stmt_binding_inline_replayable_binds": True}):
        out = tilelang.transform.IfStmtBinding()(tvm.IRModule({"main": func}))

    binds = [0]

    def count_bind(node):
        if isinstance(node, tirx.Bind):
            binds[0] += 1

    stmt_functor.post_order_visit(out["main"].body, count_bind)
    # Substituting a whole nested statement would exceed the statement budget,
    # so the guard falls back to the legacy binding and keeps its binds in
    # scope instead of expanding them into the store.
    assert binds[0] >= 6


def test_replayable_inlining_stops_at_budget():
    func = _program()
    with tvm.transform.PassContext(config={"tl.if_stmt_binding_inline_replayable_binds": True}):
        out = tilelang.transform.IfStmtBinding()(tvm.IRModule({"main": func}))

    stores = []

    def collect_store(node):
        if isinstance(node, tirx.BufferStore) and node.buffer.name == "acc":
            stores.append(node)

    stmt_functor.post_order_visit(out["main"].body, collect_store)
    assert len(stores) == 1
    value = stores[0].value
    # `acc[i] += x` becomes `acc[i] = acc[i] + x`. Without the budget `x` is
    # expanded into the whole doubling chain, so the right operand is another
    # Add instead of the last kept bind.
    assert isinstance(value, tirx.Add)
    assert isinstance(value.b, tirx.Var)


if __name__ == "__main__":
    tilelang.testing.main()
