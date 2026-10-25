import os
import subprocess
import sys

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

from tilelang._compat import PyStmtExprMutator, PyStmtExprVisitor, stmt_functor
from tilelang._compat.functor import mutator, visitor


def make_function():
    tensor = ir.Var("tensor", tirx.TensorType([8], "float32", layout="default"))
    index = ir.Var("index", "int32")
    store = ir.TensorStore(tensor, [index], ir.TensorLoad(tensor, [index]) + ir.const(1.0, "float32"))
    loop = ir.For(index, 0, 8, 1, ir.ForKind.DEFAULT, store)
    return tirx.Function([tensor], loop, ret_type=ir.PrimType("void")), tensor, index


def test_post_order_visits_each_handle_once():
    variable = ir.Var("variable", "int32")
    shared = variable + 1
    expression = shared * shared
    visited = []
    stmt_functor.post_order_visit(expression, visited.append)
    handles = [node.__chandle__() for node in visited]
    assert len(handles) == len(set(handles))
    assert handles.index(variable.__chandle__()) < handles.index(shared.__chandle__())
    assert visited[-1].same_as(expression)


def test_pre_order_prunes_and_deduplicates():
    variable = ir.Var("variable", "int32")
    shared = variable + 1
    expression = shared * shared
    visited = []

    def visit(node):
        visited.append(node)
        return not node.same_as(shared)

    stmt_functor.pre_order_visit(expression, visit)
    assert [node.__chandle__() for node in visited] == [expression.__chandle__(), shared.__chandle__()]


def test_traversal_accepts_module_and_ignores_types():
    function, _, _ = make_function()
    visited = []
    stmt_functor.post_order_visit(ir.IRModule({"main": function}), visited.append)
    assert any(isinstance(node, ir.TensorStore) for node in visited)
    assert any(isinstance(node, ir.TensorLoad) for node in visited)
    assert all(isinstance(node, (ir.Expr, ir.Stmt)) for node in visited)


def test_preorder_replacement_is_final():
    variable = ir.Var("variable", "int32")
    expression = variable + 1
    visited = []
    result = stmt_functor.ir_transform(expression, lambda node: ir.const(9) if node.same_as(expression) else None, visited.append)
    assert result.value == 9
    assert visited == []


def test_postorder_rewrite_is_immutable():
    variable = ir.Var("variable", "int32")
    expression = variable + 1

    def rewrite(node):
        if isinstance(node, ir.IntImm):
            return ir.IntImm(node.ty, node.value + 1)
        return None

    result = stmt_functor.ir_transform(expression, None, rewrite)
    assert expression.b.value == 1
    assert result.b.value == 2
    assert result.a.same_as(variable)


def test_transform_filter():
    variable = ir.Var("variable", "int32")
    visited = []
    expression = variable + 1
    result = stmt_functor.ir_transform(expression, None, visited.append, ["ir.IntImm"])
    assert result.same_as(expression)
    assert len(visited) == 1
    assert isinstance(visited[0], ir.IntImm)


def test_substitution_is_simultaneous():
    source = ir.Var("source", "int32")
    destination = ir.Var("destination", "int32")
    result = stmt_functor.substitute(source + destination, {source: destination, destination: source})
    assert result.a.same_as(destination)
    assert result.b.same_as(source)


def test_substitution_uses_identity_not_name():
    first = ir.Var("same_name", "int32")
    second = ir.Var("same_name", "int32")
    result = stmt_functor.substitute(first + second, {first: 7})
    assert result.a.value == 7
    assert result.b.same_as(second)


def test_substitution_preserves_input_function():
    function, tensor, index = make_function()
    replacement = ir.Var("replacement", tensor.ty)
    rewritten = stmt_functor.substitute(function, {tensor: replacement})
    assert function.params[0].same_as(tensor)
    assert rewritten.params[0].same_as(replacement)
    loop = rewritten.body.seq[0]
    assert loop.var.same_as(index)
    assert loop.body.seq[0].dest.same_as(replacement)


def test_substitution_rejects_type_change():
    variable = ir.Var("variable", "int32")
    with pytest.raises(TypeError, match="changes the type"):
        stmt_functor.substitute(variable + 1, {variable: ir.const(2, "int64")})


def test_substitution_rejects_nonvariable_definition():
    function, _, index = make_function()
    with pytest.raises(ValueError, match="variable definition"):
        stmt_functor.substitute(function, {index: 3})


def test_hook_visitor_calls_super_and_can_be_reused():
    @visitor
    class CollectLoads(PyStmtExprVisitor):
        def __init__(self):
            super().__init__()
            self.loads = []
            self.variables = []

        def visit_tensor_load_(self, node):
            self.loads.append(node)
            return super().visit_tensor_load_(node)

        def visit_var_(self, node):
            self.variables.append(node)

    function, _, _ = make_function()
    collector = CollectLoads()
    collector.visit(function)
    assert len(collector.loads) == 1
    assert collector.variables
    collector.visit(function)
    assert len(collector.loads) == 2


def test_hook_mutator_preserves_structure():
    @mutator
    class RewriteLoad(PyStmtExprMutator):
        def visit_tensor_load_(self, node):
            rewritten = super().visit_tensor_load_(node)
            return rewritten + ir.const(2.0, "float32")

    function, _, _ = make_function()
    result = RewriteLoad().visit(function)
    assert not tvm_ffi.structural_equal(result, function)
    assert tvm_ffi.structural_equal(PyStmtExprMutator().visit(function), function)


def test_compat_does_not_install_global_tvm_alias():
    environment = dict(os.environ, TILELANG_LIGHT_IMPORT="1")
    subprocess.run(
        [sys.executable, "-c", "import sys; import tilelang._compat; assert 'tvm' not in sys.modules"],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
