"""Small, exact queries for compiler tests, without running any passes.

Counts describe unique static IR nodes within each visited function, following
TVM's post-order visitor. They do not describe dynamic execution counts or order.
Call queries automatically reuse an index of the most recently queried body.
"""

from __future__ import annotations

from functools import lru_cache

from tilelang import tvm
from tvm import tirx


def _bodies(root, function, *, with_names=True):
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
        name = "<PrimFunc>"
        if with_names and root.attrs and "global_symbol" in root.attrs:
            name = str(root.attrs["global_symbol"])
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


@lru_cache(maxsize=1)
def _call_index(body):
    """Index one immutable body; keep function ownership outside the cache."""
    calls = _collect(body, tirx.Call)
    by_op: dict[tvm.ir.Op, list[tirx.Call]] = {}
    for call in calls:
        op = call.op
        if isinstance(op, tvm.ir.Op):
            # TVM Op keys compare by identity, even across Python handles.
            by_op.setdefault(op, []).append(call)
    return calls, by_op


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
    Consecutive queries of the same body share an internal LRU index. A new
    body rebuilds the index, including after replacing a module function.
    The returned list can be changed without affecting subsequent queries.
    """
    target = _resolve_op(op)
    return [call for _, body in _bodies(root, function, with_names=False) for call in _call_index(body)[1].get(target, ())]


def assert_call_count(root, *, op: str | tvm.ir.Op, count: int, function: str | None = None) -> None:
    """Assert a static call count, showing function ownership on failure.

    Reuse the same internal body index as collect_calls. Function selection
    is resolved anew on every query; diagnostic ownership is read on failure.
    """
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("count must be a non-negative integer")
    target = _resolve_op(op)
    calls_by_function = []
    matches = []
    for index, (_, body) in enumerate(_bodies(root, function, with_names=False)):
        calls, by_op = _call_index(body)
        calls_by_function.append(calls)
        matches.extend((index, call) for call in by_op.get(target, ()))
    if len(matches) == count:
        return
    names = [name for name, _ in _bodies(root, function)]
    scope = ", ".join(names) or "<no PrimFuncs>"
    # Include other callees when the expected call is absent, but keep failures
    # bounded even for large generated functions.
    nearby = matches or [(index, call) for index, calls in enumerate(calls_by_function) for call in calls]
    details = [f"  {names[index]}: {str(call)[:400]}" for index, call in nearby[:8]]
    if len(nearby) > 8:
        details.append(f"  ... {len(nearby) - 8} more call nodes")
    if not details:
        details.append("  <no call nodes>")
    raise AssertionError(f"Expected {count} call(s) to {target.name} in {scope}, found {len(matches)}\n" + "\n".join(details))


def collect_extern_calls(root, *, symbol: str, function: str | None = None) -> list[tirx.Call]:
    """Collect tirx.call_extern calls with an exact StringImm symbol argument.

    Other call operators, missing arguments and non-string symbols do not match.
    This function does not interpret prefixes, templates or regular expressions.
    Reuse the same internal body index through collect_calls.
    """
    if not isinstance(symbol, str):
        raise TypeError("symbol must be a string")
    return [
        call
        for call in collect_calls(root, op="tirx.call_extern", function=function)
        if call.args and isinstance(call.args[0], tirx.StringImm) and call.args[0].value == symbol
    ]
