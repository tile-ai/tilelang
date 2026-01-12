from tvm import tir
from tvm.tir import PyStmtExprVisitor, PrimFunc, Stmt

from tvm.tir.transform import prim_func_pass


_seq_field_key = "seq"
_then_field_key = "then_case"
_else_field_key = "else_case"
_child_fields = ["body", "block", _seq_field_key, _then_field_key, _else_field_key]
_ignore_fields = ["span"]

_stmt_line_limit = 140
_middle_connector = "├── "
_last_connector = "└── "

_normal_indent = " " * 4
_seq_middle_indent = "|" + " " * 3


@tir.functor.visitor
class _ASTPrintVisitor(PyStmtExprVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.indent: list[str] = []

    def print_with_clip(self, s: str) -> None:
        if len(s) > _stmt_line_limit:
            s = s[:_stmt_line_limit] + "..."
        print("".join(self.indent) + s)

    def print_stmt_brief(self, stmt: Stmt, prefix: str) -> None:
        # stmt_script = repr(stmt).splitlines()[0].split("  ")[0].strip()
        self.print_with_clip(prefix + f"{stmt.__class__.__name__}")

    def visit_stmt(self, stmt: Stmt) -> None:
        anno = stmt.__annotations__  # field_key_name -> field_key_type
        field_keys = anno.keys()
        # Filter out private/built-in fields.
        normal_field_keys = [
            key for key in field_keys if not key.startswith("_") and key not in _child_fields and key not in _ignore_fields
        ]
        child_field_keys = [key for key in field_keys if key in _child_fields]

        for idx, key in enumerate(normal_field_keys):
            value = getattr(stmt, key, None)
            # Try to get its script representation.
            value = repr(value)
            # If has child fields, we enforce child fields to be last two fields.
            # So all other fields are not last.
            is_last = idx == len(normal_field_keys) - 1 and len(child_field_keys) == 0
            # Add tree-like connector
            connector = _last_connector if is_last else _middle_connector
            self.print_with_clip(connector + f"{key}({anno[key]}): {value}")

        # Handle child fields
        # Here we have three cases:
        # 1. SeqStmt, which has a list of child stmts.
        # 2. IfThenElse w/ else condition, which has 2 child stmts.
        # 3. Other stmts like For/Block, which has 1 child stmt.
        if len(child_field_keys) == 2:
            # Special output format for IfThenElse
            try:
                then_child = getattr(stmt, _then_field_key)
                else_child = getattr(stmt, _else_field_key)
            except Exception as e:
                raise ValueError(
                    "Unexpected error when printing AST: The node has two child fields but it violates IfElseNode structure."
                ) from e
            # Some IfElseNodes have no else branch, but they keep the else field and set the value to None.
            has_else_branch = else_child is not None
            # Visit then
            prefix = (_middle_connector if has_else_branch else _last_connector) + f"{_then_field_key}(Stmt): "
            self.print_stmt_brief(then_child, prefix)
            self.indent.append(_seq_middle_indent)
            self.visit_stmt(then_child)
            self.indent.pop()
            # Visit else
            prefix = _last_connector + f"{_else_field_key}(Optional[Stmt]): "
            self.print_stmt_brief(else_child, prefix)
            if has_else_branch:
                self.indent.append(_normal_indent)
                self.visit_stmt(else_child)
                self.indent.pop()
        elif len(child_field_keys) == 1:
            child_field_name = child_field_keys[0]
            child = getattr(stmt, child_field_name)

            # Special output format for SeqStmt
            if child_field_name == _seq_field_key:
                for i, child_node in enumerate(child):
                    is_last_child = i == len(child) - 1
                    prefix = (_last_connector if is_last_child else _middle_connector) + f"{_seq_field_key}{i}(Stmt): "
                    self.print_stmt_brief(child_node, prefix)
                    self.indent.append(_normal_indent if is_last_child else _seq_middle_indent)
                    self.visit_stmt(child_node)
                    self.indent.pop()
            else:
                # Other cases with only 1 child stmt
                prefix = _last_connector + f"{child_field_name}(Stmt): "
                self.print_stmt_brief(child, prefix)
                self.indent.append(_normal_indent)
                self.visit_stmt(child)
                self.indent.pop()
        else:
            assert len(child_field_keys) == 0, "Unexpected error when printing AST: Got 3 or more child field keys."


def ASTPrinter():
    """
    A visitor pass that renders the TileLang AST hierarchy in a visual tree format.

    Comparing with TL script, this printer is more suitable for debugging
    and understanding the internal structure of TensorIR, like the class structure of
    each node and their connections.

    This printer generates a human-readable, tree-structured representation of the
    Abstract Syntax Tree (AST). It uses ASCII/Unicode connectors to visualize
    parent-child relationships, making it easier to inspect nested structures
    (e.g., loops, blocks, scopes) and verify compiler transformations.
    """

    def pass_fn(func: PrimFunc, mod, ctx) -> PrimFunc:
        print(f"PrimFunc(params={func.params}, ret_type={func.ret_type}, buffer_map={func.buffer_map}, attrs={func.attrs})")
        func_body_prefix = _last_connector + "body="
        visitor = _ASTPrintVisitor()
        visitor.print_stmt_brief(func.body, func_body_prefix)
        visitor.indent.append(_normal_indent)
        visitor.visit_stmt(func.body)
        return func

    return prim_func_pass(pass_fn, opt_level=0)
