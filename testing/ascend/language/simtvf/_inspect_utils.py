from __future__ import annotations

from tilelang import tvm
from tvm.tirx.stmt_functor import post_order_visit


def collect_simtvf_blocks(func: tvm.tirx.PrimFunc) -> list[tvm.tirx.SBlock]:
    blocks: list[tvm.tirx.SBlock] = []

    def _visit(stmt):
        if isinstance(stmt, tvm.tirx.SBlock) and stmt.name_hint == "SIMT_VF":
            blocks.append(stmt)

    post_order_visit(func.body, _visit)
    return blocks


def print_stage(title: str, mod: tvm.IRModule, func_name: str) -> None:
    print(f"\n===== {title} =====")
    print(mod.script())
    func = mod[func_name]
    blocks = collect_simtvf_blocks(func)
    print(f"[check] simtvf(block has SIMT_VF name_hint) count = {len(blocks)}")
    for idx, block in enumerate(blocks):
        print(f"  block[{idx}] name_hint: {block.name_hint}")


def collect_thread_extent_tags(stmt: tvm.tirx.Stmt) -> list[str]:
    tags: list[str] = []

    def _visit(node):
        if isinstance(node, tvm.tirx.AttrStmt) and node.attr_key == "thread_extent":
            iv = node.node
            if isinstance(iv, tvm.tirx.IterVar):
                tags.append(iv.thread_tag)

    post_order_visit(stmt, _visit)
    return tags


def collect_thread_extent_tags_outside_simtvf(stmt: tvm.tirx.Stmt) -> list[str]:
    tags: list[str] = []

    def _walk(node: tvm.tirx.Stmt, simtvf_depth: int) -> None:
        if isinstance(node, tvm.tirx.SBlock):
            is_simtvf = node.name_hint == "SIMT_VF"
            _walk(node.body, simtvf_depth + (1 if is_simtvf else 0))
            return
        if isinstance(node, tvm.tirx.For):
            _walk(node.body, simtvf_depth)
            return
        if isinstance(node, tvm.tirx.AttrStmt):
            if simtvf_depth == 0 and node.attr_key == "thread_extent" and isinstance(node.node, tvm.tirx.IterVar):
                tags.append(node.node.thread_tag)
            _walk(node.body, simtvf_depth)
            return
        if isinstance(node, tvm.tirx.SeqStmt):
            for s in node.seq:
                _walk(s, simtvf_depth)
            return
        if isinstance(node, tvm.tirx.SBlockRealize):
            _walk(node.block.body, simtvf_depth)
            return
        let_stmt_cls = getattr(tvm.tirx, "LetStmt", None)
        if let_stmt_cls is not None and isinstance(node, let_stmt_cls):
            _walk(node.body, simtvf_depth)
            return
        if isinstance(node, tvm.tirx.IfThenElse):
            _walk(node.then_case, simtvf_depth)
            if node.else_case is not None:
                _walk(node.else_case, simtvf_depth)
            return
        if isinstance(node, tvm.tirx.AssertStmt):
            _walk(node.body, simtvf_depth)
            return

    _walk(stmt, 0)
    return tags


def contains_parallel_for(stmt: tvm.tirx.Stmt) -> bool:
    found = False

    def _visit(node):
        nonlocal found
        if isinstance(node, tvm.tirx.For) and node.kind == tvm.tirx.ForKind.PARALLEL:
            found = True

    post_order_visit(stmt, _visit)
    return found


def find_thread_predicate_hoist_violations(func: tvm.tirx.PrimFunc) -> list[tvm.tirx.PrimExpr]:
    violations: list[tvm.tirx.PrimExpr] = []

    def _is_thread_predicate(expr: tvm.tirx.PrimExpr) -> bool:
        found = False

        def _visit(node):
            nonlocal found
            if isinstance(node, tvm.tirx.Var) and node.name in ("tx", "ty", "tz", "simtvf_tx"):
                found = True

        post_order_visit(expr, _visit)
        return found

    def _walk_stmt(stmt: tvm.tirx.Stmt, simtvf_depth: int) -> None:
        if isinstance(stmt, tvm.tirx.SBlock):
            is_simtvf = stmt.name_hint == "SIMT_VF"
            _walk_stmt(stmt.body, simtvf_depth + (1 if is_simtvf else 0))
            return
        if isinstance(stmt, tvm.tirx.For):
            _walk_stmt(stmt.body, simtvf_depth)
            return
        if isinstance(stmt, tvm.tirx.IfThenElse):
            if simtvf_depth == 0 and _is_thread_predicate(stmt.condition):
                violations.append(stmt.condition)
            _walk_stmt(stmt.then_case, simtvf_depth)
            if stmt.else_case is not None:
                _walk_stmt(stmt.else_case, simtvf_depth)
            return
        if isinstance(stmt, tvm.tirx.SeqStmt):
            for s in stmt.seq:
                _walk_stmt(s, simtvf_depth)
            return
        let_stmt_cls = getattr(tvm.tirx, "LetStmt", None)
        if let_stmt_cls is not None and isinstance(stmt, let_stmt_cls):
            _walk_stmt(stmt.body, simtvf_depth)
            return
        if isinstance(stmt, tvm.tirx.AttrStmt):
            _walk_stmt(stmt.body, simtvf_depth)
            return
        if isinstance(stmt, tvm.tirx.AssertStmt):
            _walk_stmt(stmt.body, simtvf_depth)
            return
        if isinstance(stmt, tvm.tirx.SBlockRealize):
            _walk_stmt(stmt.block.body, simtvf_depth)
            return

    _walk_stmt(func.body, 0)
    return violations
