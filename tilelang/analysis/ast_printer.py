from tvm import tir
from tvm.tir import PyStmtExprVisitor, PrimFunc, Stmt

from tvm.tir.transform import prim_func_pass


_child_fields = ["body", "block", "seq"]

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
        stmt_script = repr(stmt).splitlines()[0].split("  ")[0].strip()
        self.print_with_clip(prefix + f"{stmt.__class__.__name__}: " + stmt_script)

    def visit_stmt(self, stmt: Stmt) -> None:
        child_field_name: str = ""

        field_keys = stmt.__class__.__dict__.keys()
        # Filter out private/built-in fields.
        field_keys = [key for key in field_keys if not key.startswith("_")]

        for idx, key in enumerate(field_keys):
            # For child fields, we'll handle them specially below instead of printing them in current line.
            if key in _child_fields:
                child_field_name = key
                continue

            value = getattr(stmt, key, None)
            if value is None:
                continue
            # Try to get its script representation.
            value = repr(value)

            is_last_child = idx == len(field_keys) - 1 and not child_field_name
            # Add tree-like connector
            connector = _last_connector if is_last_child else _middle_connector

            # Every member
            self.print_with_clip(connector + f"{key}: {value}")

        # Handle child fields
        if child_field_name and hasattr(stmt, child_field_name):
            child = getattr(stmt, child_field_name)

            if child_field_name != "seq":
                prefix = _last_connector + f"{child_field_name}: "
                self.print_stmt_brief(child, prefix)
                self.indent.append(_normal_indent)
                self.visit_stmt(child)
                self.indent.pop()
            else:
                # Special output format for SeqStmt
                for i, child_node in enumerate(child):
                    is_last_child = i == len(child) - 1
                    prefix = (_last_connector if is_last_child else _middle_connector) + f"seq{i}: "
                    self.print_stmt_brief(child_node, prefix)
                    self.indent.append(_normal_indent if is_last_child else _seq_middle_indent)
                    self.visit_stmt(child_node)
                    self.indent.pop()


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
        visitor.visit_stmt(func.body)
        visitor.indent.append(_normal_indent)
        return func

    return prim_func_pass(pass_fn, opt_level=0)
