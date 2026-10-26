import pytest
import tvm_tirx

from tilelang._compat import ir, stmt_functor, sym
from tvm_tirx.ir import prim


def test_nested_opaque_casts_do_not_escape_simplification():
    source = ir.Var("source", "int8")
    argument = ir.cast(ir.cast(source, "uint8"), "uint32")
    call = ir.Call("tirx.call_pure_extern", [ir.StringImm("__popc"), argument], ty="int32")
    expression = ir.cast(ir.cast(call, "int8"), "int32")
    simplified = sym.Analyzer().simplify(expression)
    variables = []
    stmt_functor.post_order_visit(simplified, lambda node: variables.append(node) if isinstance(node, ir.Var) else None)
    assert variables and all(variable.same_as(source) for variable in variables)


@pytest.mark.parametrize("operation", [prim.BitwiseOr, prim.BitwiseXor])
@pytest.mark.parametrize("left_bound,right_bound", [((0, 3), (0, 3)), ((3, 5), (1, 7)), ((0, 15), (0, 1))])
def test_bitwise_bounds_cover_nonmonotone_operands(operation, left_bound, right_bound):
    left = ir.Var("left", "int64")
    right = ir.Var("right", "int64")
    analyzer = sym.Analyzer()
    analyzer.update(left, sym.ConstIntBound(*left_bound))
    analyzer.update(right, sym.ConstIntBound(*right_bound))
    bounds = analyzer.const_int_bound(operation(left, right))
    assert bounds.min_value == 0
    assert bounds.max_value < sym.ConstIntBound.POS_INF
    for left_value in range(left_bound[0], left_bound[1] + 1):
        for right_value in range(right_bound[0], right_bound[1] + 1):
            result = left_value | right_value if operation is prim.BitwiseOr else left_value ^ right_value
            assert bounds.min_value <= result <= bounds.max_value


@pytest.mark.parametrize("reverse_product", [False, True])
@pytest.mark.parametrize("reverse_sum", [False, True])
def test_symbolic_radix_modulo_discards_complete_rows(reverse_product, reverse_sum):
    row = ir.Var("row", "int32")
    column = ir.Var("column", "int32")
    radix = ir.Var("radix", "int32")
    product = radix * row if reverse_product else row * radix
    offset = column + product if reverse_sum else product + column
    analyzer = sym.Analyzer()
    with analyzer.constraint_scope(radix > 0):
        ir.assert_structural_equal(analyzer.simplify(offset % radix), column % radix)


def test_signed_to_unsigned_cast_bounds_and_modular_stride():
    analyzer = sym.Analyzer()
    source = ir.Var("source", "int8")
    bounded = analyzer.const_int_bound(prim.Cast("uint32", prim.truncmod(source, 3)))
    assert (bounded.min_value, bounded.max_value) == (0, 2**32 - 1)
    modular = analyzer.modular_set(prim.Cast("uint32", source * 3))
    assert (modular.coeff, modular.base) == (1, 0)


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
def test_z3_narrow_cast_is_not_injective(dtype):
    analyzer = sym.Analyzer()
    if not analyzer.is_z3_enabled:
        pytest.skip("TIRx was built without Z3")
    first = ir.Var("first", "int32")
    second = ir.Var("second", "int32")
    analyzer.bind(first, ir.Range(0, 512))
    analyzer.bind(second, ir.Range(0, 512))
    with analyzer.constraint_scope(prim.Cast(dtype, first) == prim.Cast(dtype, second)):
        assert not analyzer.can_prove(first == second, sym.ProofStrength.SYMBOLIC_BOUND)


@pytest.mark.parametrize("dtype,value,expected", [("int8", 255, -1), ("uint8", -1, 255), ("bool", 2, 1), ("bool", -2, 1), ("bool", 0, 0)])
def test_z3_cast_matches_runtime_conversion(dtype, value, expected):
    analyzer = sym.Analyzer()
    if not analyzer.is_z3_enabled:
        pytest.skip("TIRx was built without Z3")
    source = ir.Var("source", "int32")
    converted = prim.Cast("int32", prim.Cast(dtype, source))
    with analyzer.constraint_scope(source == value):
        assert analyzer.can_prove(converted == expected, sym.ProofStrength.SYMBOLIC_BOUND)


def test_clone_retains_adapter_without_registering_native_type():
    analyzer = sym.Analyzer()
    source = ir.Var("source", "int32")
    analyzer.bind(source, ir.Range(128, 256))
    cloned = analyzer.clone()
    assert isinstance(cloned, sym.Analyzer)
    assert type(tvm_tirx.sym.Analyzer()) is tvm_tirx.sym.Analyzer
    bounded = cloned.const_int_bound(prim.Cast("int8", source))
    assert (bounded.min_value, bounded.max_value) == (-128, -1)


def test_simplifying_unknown_narrow_cast_does_not_expand_the_ir():
    analyzer = sym.Analyzer()
    source = ir.Var("source", "int32")
    expression = prim.Cast("int8", source)
    simplified = expression
    for _repetition in range(10):
        simplified = analyzer.simplify(simplified)
    ir.assert_structural_equal(simplified, expression)


def test_nested_bitwise_bounds_apply_to_explicit_domains():
    analyzer = sym.Analyzer()
    source = ir.Var("source", "int32")
    expression = (source ^ 7) + 32
    bounded = analyzer.int_set(expression, {source: sym.IntervalSet(0, 15)})
    assert int(bounded.min_value) == 32
    assert int(bounded.max_value) == 47
