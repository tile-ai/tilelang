import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.engine.phase import LowerAndLegalize


def _apply_plan_update(func: tvm.tir.PrimFunc) -> tvm.IRModule:
    target = tvm.target.Target("cuda")
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    with target:
        mod = LowerAndLegalize(mod, target)
        mod = tl.transform.LowerSharedTmem()(mod)
        mod = tl.transform.IfStmtBinding()(mod)
        mod = tl.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    return mod


def _find_block(stmt: tvm.tir.Stmt, name_hint: str) -> tvm.tir.Block:
    blocks = []

    def _visit(node):
        if isinstance(node, tvm.tir.Block) and str(node.name_hint) == name_hint:
            blocks.append(node)

    tvm.tir.stmt_functor.post_order_visit(stmt, _visit)
    assert len(blocks) == 1, f"Expected exactly one block named {name_hint}, got {len(blocks)}"
    return blocks[0]


def _find_first_for(stmt: tvm.tir.Stmt) -> tvm.tir.For:
    loops = []

    def _visit(node):
        if isinstance(node, tvm.tir.For):
            loops.append(node)

    tvm.tir.stmt_functor.post_order_visit(stmt, _visit)
    assert loops, "Expected at least one loop"
    return loops[0]


def test_plan_update_keeps_loop_header_local_var_outside_loop_body():
    @T.prim_func
    def func(x: T.Tensor((256,), "int64")):
        with T.Kernel(256, threads=128):
            a, b = T.alloc_var(T.int), T.alloc_var(T.int)
            T.fill(x[a:b], 0)

    mod = _apply_plan_update(func)
    main = mod["main"]

    tilelang_root = _find_block(main.body, "tilelang_root")
    root_local_vars = {buf.name for buf in tilelang_root.alloc_buffers if buf.scope() == "local.var"}
    assert {"a", "b"} <= root_local_vars

    loop = _find_first_for(main.body)
    loop_body_local_vars = set()

    def _visit_loop_body(node):
        if isinstance(node, tvm.tir.Block):
            for buf in node.alloc_buffers:
                if buf.scope() == "local.var":
                    loop_body_local_vars.add(buf.name)

    tvm.tir.stmt_functor.post_order_visit(loop.body, _visit_loop_body)
    assert "b" not in loop_body_local_vars


if __name__ == "__main__":
    tilelang.testing.main()
