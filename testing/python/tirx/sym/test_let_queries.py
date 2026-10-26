from tilelang._compat import ir, sym


def test_let_cast_is_normalized_before_hiding_narrowing_casts():
    bound = ir.Var("bound", "int64")
    value = ir.const(2500000000, "int64")
    expression = ir.prim.Let(bound, value, ir.cast(ir.cast(bound, "int8"), "int64") + bound)
    result = sym.Analyzer().simplify(expression)
    assert int(result) == 2500000000
    assert result.ty == ir.PrimType("int64")


def test_let_definition_is_not_replaced_by_analyzer_expression_bindings():
    index = ir.Var("index", "int32")
    bound = ir.Var("bound", "int32")
    analyzer = sym.Analyzer()
    analyzer.bind(bound, index % 8)
    expression = ir.prim.Let(bound, index % 8, bound < 4)
    ir.assert_structural_equal(analyzer.simplify(expression), analyzer.simplify(index % 8 < 4))


def test_let_query_bindings_do_not_escape_their_scope():
    bound = ir.Var("bound", "int32")
    analyzer = sym.Analyzer()
    analyzer.bind(bound, ir.const(5))
    assert int(analyzer.simplify(ir.prim.Let(bound, ir.const(9), bound))) == 9
    assert int(analyzer.simplify(bound)) == 5


def test_let_simplification_preserves_effectful_values():
    bound = ir.Var("bound", "int32")
    effect = ir.Call("tirx.call_extern", [ir.StringImm("effect")], ty="int32")
    expression = ir.prim.Let(bound, effect, ir.const(7))
    ir.assert_structural_equal(sym.Analyzer().simplify(expression), expression)
