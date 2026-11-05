"""Print the reflected native statement hierarchy without legacy Buffer fields."""

from tvm_tirx import ir, tirx


_stmt_line_limit = 140
_middle_connector = "├── "
_last_connector = "└── "
_normal_indent = " " * 4
_seq_middle_indent = "|" + " " * 3


def _fields(node):
    """Read fields from the loaded native type, including inherited fields."""
    info = type(node).__tvm_ffi_type_info__
    seen = {"loc"}
    while info is not None:
        for field in info.fields:
            if field.name not in seen:
                seen.add(field.name)
                yield field.name, getattr(node, field.name)
        info = info.parent_type_info


class _ASTPrintVisitor:
    def __init__(self):
        self.indent = []

    def print_with_clip(self, text):
        if len(text) > _stmt_line_limit:
            text = text[:_stmt_line_limit] + "..."
        print("".join(self.indent) + text)

    def print_stmt_brief(self, statement, prefix):
        self.print_with_clip(prefix + type(statement).__name__)

    def visit_stmt(self, statement):
        fields = list(_fields(statement))
        for index, (key, value) in enumerate(fields):
            last = index == len(fields) - 1
            connector = _last_connector if last else _middle_connector
            indent = _normal_indent if last else _seq_middle_indent
            if isinstance(value, ir.Stmt):
                self.print_stmt_brief(value, connector + key + ": ")
                self.indent.append(indent)
                self.visit_stmt(value)
                self.indent.pop()
            elif key == "seq":
                if not value:
                    self.print_with_clip(connector + "seq: []")
                for seq_index, child in enumerate(value):
                    child_last = last and seq_index == len(value) - 1
                    self.print_stmt_brief(child, (_last_connector if child_last else _middle_connector) + f"seq[{seq_index}]: ")
                    self.indent.append(_normal_indent if child_last else _seq_middle_indent)
                    self.visit_stmt(child)
                    self.indent.pop()
            else:
                self.print_with_clip(connector + f"{key}({type(value).__name__}): {value!r}")


def ASTPrinter():
    """Render Function, SeqStmt, Bind, RegionStmt and nested control flow as a tree."""

    def pass_fn(func, _mod, _ctx):
        print(f"Function(params={func.params}, ret_type={func.ret_type}, attrs={func.attrs})")
        if not func.attrs.get("declaration", False):
            visitor = _ASTPrintVisitor()
            visitor.print_stmt_brief(func.body, _last_connector + "body: ")
            visitor.indent.append(_normal_indent)
            visitor.visit_stmt(func.body)
        return func

    return tirx.transform.function_pass(pass_fn, opt_level=0, name="tl.ASTPrinter")
