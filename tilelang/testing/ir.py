"""Small, exact queries for compiler tests, without running any passes.

Counts describe unique static IR nodes within each visited function, following
TVM's post-order visitor. They do not describe dynamic execution counts or order.
"""

from __future__ import annotations

from tilelang import tvm
from tvm import tirx


def _bodies(root, function):
    if function is not None and not isinstance(function, str):
        raise TypeError("function must be a global function name or None")
    if isinstance(root, tvm.IRModule):
        functions = {var.name_hint: func for var, func in root.functions.items()}
        if function is not None:
            if function not in functions:
                raise ValueError(f"No function {function!r} in the IRModule")
            func = functions[function]
            if not isinstance(func, tirx.PrimFunc):
                raise TypeError(f"Function {function!r} is not a PrimFunc")
            return [(function, func.body)]
        return [(name, func.body) for name, func in sorted(functions.items()) if isinstance(func, tirx.PrimFunc)]
    if function is not None:
        raise ValueError("function can only be selected on an IRModule")
    if isinstance(root, tirx.PrimFunc):
        name = str(root.attrs["global_symbol"]) if root.attrs and "global_symbol" in root.attrs else "<PrimFunc>"
        return [(name, root.body)]
    if isinstance(root, (tirx.Stmt, tvm.ir.PrimExpr)):
        return [("<root>", root)]
    raise TypeError(f"Expected IRModule, PrimFunc, Stmt or PrimExpr, got {type(root).__name__}")


def _collect(body, node_type):
    nodes = []

    def visit(node):
        if isinstance(node, node_type):
            nodes.append(node)

    tirx.stmt_functor.post_order_visit(body, visit)
    return nodes


def _resolve_op(op):
    if isinstance(op, str):
        # Deliberately let Op.get fail for unregistered names, even for count=0.
        return tvm.ir.Op.get(op)
    if not isinstance(op, tvm.ir.Op):
        raise TypeError("op must be a registered Op or its full name")
    return op


def collect_nodes(root, node_type, *, function: str | None = None) -> list:
    """Collect nodes of a TVM object type from bodies or an expression.

    A module query visits all PrimFunc bodies unless ``function`` selects one
    by global name. Other module functions are skipped. Function signatures and
    module metadata are not traversed. Shared nodes are visited once per body.
    """
    if not isinstance(node_type, type) or not issubclass(node_type, tvm.runtime.Object):
        raise TypeError("node_type must be a TVM object type")
    return [node for _, body in _bodies(root, function) for node in _collect(body, node_type)]


def collect_calls(root, *, op: str | tvm.ir.Op, function: str | None = None) -> list[tirx.Call]:
    """Collect calls to one exact registered Op; never match name fragments.

    GlobalVar and other non-Op callees do not match. An unregistered name raises
    the Op registry's error instead of silently producing an empty result.
    """
    target = _resolve_op(op)
    return [call for call in collect_nodes(root, tirx.Call, function=function) if call.op.same_as(target)]


def assert_call_count(root, *, op: str | tvm.ir.Op, count: int, function: str | None = None) -> None:
    """Assert a static call count, showing function ownership on failure."""
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("count must be a non-negative integer")
    target = _resolve_op(op)
    calls_by_function = [(name, _collect(body, tirx.Call)) for name, body in _bodies(root, function)]
    matches = [(name, call) for name, calls in calls_by_function for call in calls if call.op.same_as(target)]
    if len(matches) == count:
        return
    scope = ", ".join(name for name, _ in calls_by_function) or "<no PrimFuncs>"
    # Include other callees when the expected call is absent, but keep failures
    # bounded even for large generated functions.
    nearby = matches or [(name, call) for name, calls in calls_by_function for call in calls]
    details = [f"  {name}: {str(call)[:400]}" for name, call in nearby[:8]]
    if len(nearby) > 8:
        details.append(f"  ... {len(nearby) - 8} more call nodes")
    if not details:
        details.append("  <no call nodes>")
    raise AssertionError(f"Expected {count} call(s) to {target.name} in {scope}, found {len(matches)}\n" + "\n".join(details))


def collect_extern_calls(root, *, symbol: str, function: str | None = None) -> list[tirx.Call]:
    """Collect tirx.call_extern calls with an exact StringImm symbol argument.

    Other call operators, missing arguments and non-string symbols do not match.
    This function does not interpret prefixes, templates or regular expressions.
    """
    if not isinstance(symbol, str):
        raise TypeError("symbol must be a string")
    return [
        call
        for call in collect_calls(root, op="tirx.call_extern", function=function)
        if call.args and isinstance(call.args[0], tirx.StringImm) and call.args[0].value == symbol
    ]
