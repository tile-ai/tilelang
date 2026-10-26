"""Identity-aware traversal and functional rewriting of native TIRx IR."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import tvm_ffi
from tvm_tirx import ir, tirx
from .structural import structural_visit, structural_mutate

IRNode = ir.Expr | ir.Stmt
IRRoot = IRNode | tirx.Function | ir.IRModule


def _check_root(root: IRRoot) -> None:
    if not isinstance(root, (ir.Expr, ir.Stmt, tirx.Function, ir.IRModule)):
        raise TypeError(f"Expected a TIRx expression, statement, function, or module, got {type(root).__name__}")


def _skip_type(node, visitor):
    return None


def post_order_visit(root: IRRoot, fvisit: Callable[[IRNode], object]) -> None:
    """Visit each expression or statement once, after its children.

    Identity is the FFI handle, not the transient Python wrapper's identity.
    Type metadata is not part of statement/expression traversal.
    """
    _check_root(root)
    seen: set[int] = set()

    def visit(node, visitor):
        identity = node.__chandle__()
        if identity in seen:
            return
        seen.add(identity)
        visitor.default_visit(node)
        fvisit(node)

    structural_visit(root, [((ir.Expr, ir.Stmt), visit), (ir.Type, _skip_type)])


def pre_order_visit(root: IRRoot, fvisit: Callable[[IRNode], bool]) -> None:
    """Visit each node once before its children; a false callback prunes it."""
    _check_root(root)
    seen: set[int] = set()

    def visit(node, visitor):
        identity = node.__chandle__()
        if identity in seen:
            return
        seen.add(identity)
        if fvisit(node):
            visitor.default_visit(node)

    structural_visit(root, [((ir.Expr, ir.Stmt), visit), (ir.Type, _skip_type)])


def _checked_replacement(original: IRNode, replacement: IRNode) -> IRNode:
    expected = ir.Stmt if isinstance(original, ir.Stmt) else ir.Expr
    if not isinstance(replacement, expected):
        raise TypeError(f"A {type(original).__name__} rewrite must return {expected.__name__}, not {type(replacement).__name__}")
    return replacement


def ir_transform(
    root: IRRoot,
    preorder: Callable[[IRNode], IRNode | None] | None,
    postorder: Callable[[IRNode], IRNode | None] | None,
    only_enable: Sequence[str] | None = None,
) -> IRRoot:
    """Rewrite IR with the historical pre/descend/post contract.

    A non-None preorder result is final: neither its children nor the postorder
    callback are visited. Returning None requests normal traversal. Filters use
    native type keys, such as ``ir.For`` and ``ir.TensorStore``.
    """
    _check_root(root)
    enabled = set(only_enable or ())
    for type_key in enabled:
        if tvm_ffi.core._object_type_key_to_index(type_key) is None:
            raise ValueError(f"Unknown TIRx node type: {type_key}")

    def mutate(node, engine):
        selected = not enabled or type(node).__tvm_ffi_type_info__.type_key in enabled
        if selected and preorder is not None:
            replacement = preorder(node)
            if replacement is not None:
                return _checked_replacement(node, replacement)
        rewritten = engine.default_mutate(node)
        if selected and postorder is not None:
            replacement = postorder(rewritten)
            if replacement is not None:
                return _checked_replacement(rewritten, replacement)
        return rewritten

    return structural_mutate(root, ((ir.Expr, ir.Stmt), mutate))


def substitute(root: IRRoot, value_map: Mapping[ir.Var, ir.Expr | int | float | bool]) -> IRRoot:
    """Simultaneously substitute variables without mutating the input IR.

    Replacements must have the original type. Definitions can be renamed to
    another variable but cannot be replaced by an arbitrary expression. To
    instantiate a loop iteration, substitute the loop body rather than its
    binding statement.
    """
    _check_root(root)
    replacements: dict[int, ir.Expr] = {}
    for variable, replacement in value_map.items():
        if not isinstance(variable, ir.Var):
            raise TypeError("Substitution keys must be native TIRx variables")
        if not isinstance(replacement, ir.Expr):
            if not isinstance(variable.ty, ir.PrimType):
                raise TypeError("A non-primitive variable requires a typed expression replacement")
            replacement = ir.const(replacement, variable.ty)
        if not tvm_ffi.structural_equal(variable.ty, replacement.ty):
            raise TypeError(f"Substitution changes the type of {variable.name}: {variable.ty} -> {replacement.ty}")
        replacements[variable.__chandle__()] = replacement

    def mutate(variable, engine):
        replacement = replacements.get(variable.__chandle__())
        if replacement is None:
            return engine.default_mutate(variable)
        if engine.def_region_kind() != tvm_ffi.DefRegionKind.NONE and not isinstance(replacement, ir.Var):
            raise ValueError(f"Cannot replace variable definition {variable.name} with a non-variable; substitute its body instead")
        return replacement

    return structural_mutate(root, (ir.Var, mutate))
