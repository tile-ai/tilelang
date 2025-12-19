import tilelang.testing
import tilelang.language as T
from tvm.arith import Analyzer
from tvm.ir.expr import Range
from tvm.tir.expr import Not, Or


def implies(x, y):
    return Or(Not(x), y)


def test_hard_prove():
    a = T.Var("a", T.int32)
    b = T.Var("b", T.int32)
    c = T.Var("c", T.int32)
    d = T.Var("d", T.int32)

    def check_expr(expr):
        analyzer = Analyzer()
        result = analyzer.can_prove(expr, 1)
        if not result:
            smtlib2 = analyzer.get_smtlib2(expr)
            raise AssertionError(f"Failed to prove: {expr}\nSMT-LIB2:\n{smtlib2}")
        # assert result, f"Failed to prove: {expr}"

    @T.macro
    def complex_expr_1():
        return implies(a > 0 and b > 0 and c > 0, ((b - a) // c) * c + a <= b)

    check_expr(complex_expr_1())

    @T.macro
    def complex_expr_2():
        return implies(a < b and b < c and a * d < b * d, b * d < c * d)

    check_expr(complex_expr_2())

    @T.macro
    def complex_expr_3():
        return implies(a >= 0 and a < 128, a // 128 == (a // 64 * 32 + a % 32 // 16 * 8) // 64)

    check_expr(complex_expr_3())

    @T.macro
    def complex_expr_4():
        return implies(
            a >= 0 and a < 128,
            (a % 16 * 64 + a // 64 * 32 + a % 8 // 4 * 32 + (a % 32 // 16 + a % 2) % 2 * 8 + 16 - (a // 64 + a % 8 // 4) // 2 * 64) // 512
            == (a % 16 * 64 + a // 64 * 32 + a % 8 // 4 * 32 + (a % 32 // 16 + a % 2) % 2 * 8 - (a // 64 + a % 8 // 4) // 2 * 64) // 512,
        )

    check_expr(complex_expr_4())


def test_smtlib2():
    import z3

    a = T.Var("a", T.int32)
    b = T.Var("b", T.int32)
    c = T.Var("c", T.int32)

    @T.macro
    def complex_expr_1():
        return implies(a > 0 and b > 0 and c > 0, ((b - a) // c) * c + a <= b)

    e = complex_expr_1()
    analyzer = Analyzer()
    analyzer.set_z3_timeout_ms(1000)
    smtlib2 = analyzer.get_smtlib2(e)

    solver = z3.Solver()
    solver.from_string(smtlib2)
    assert solver.check() == z3.unsat, f"Expected unsat, got {solver.check()}"


def test_bind():
    a = T.Var("a", T.int32)
    b = T.Var("b", T.int32)
    c = T.Var("c", T.int32)

    analyzer = Analyzer()
    analyzer.bind(a, Range(1, 100000))
    analyzer.bind(b, Range(1, 100000))
    analyzer.bind(c, Range(1, 100000))

    expr = ((b - a) // c) * c + a <= b
    smtlib2 = analyzer.get_smtlib2(expr)
    try:
        result = analyzer.can_prove(expr, 1)
        assert result, f"Failed to prove with bindings: {expr}"
    except Exception as e:
        print(smtlib2)
        raise e


def test_divmod():
    analyzer = Analyzer()
    a = T.Var("a", T.int32)

    assert not analyzer.can_prove(a % 2 % -2 - a % 2 == 0)
    assert analyzer.can_prove(a % -2 % 2 - a % 2 == 0)


if __name__ == "__main__":
    tilelang.testing.main()
