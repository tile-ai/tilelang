from __future__ import annotations

from tvm import IRModule

import tilelang
from tilelang.transform import PassContext


def should_enable_prelower_semantic_check(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return not pass_ctx.config.get(tilelang.PassConfigKey.TL_DISABLE_PRELOWER_SEMANTIC_CHECK, False)


def should_enable_ast_print(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return bool(pass_ctx and pass_ctx.config.get(tilelang.PassConfigKey.TL_AST_PRINT_ENABLE, False))


def PreLowerSemanticCheck(mod: IRModule) -> None:
    """Run backend-independent validation before lowering."""

    if not should_enable_prelower_semantic_check():
        return

    if should_enable_ast_print():
        tilelang.analysis.ASTPrinter()(mod)
    tilelang.analysis.NestedLoopChecker()(mod)
    tilelang.analysis.FragmentLoopChecker()(mod)
