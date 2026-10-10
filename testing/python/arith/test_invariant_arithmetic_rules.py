"""Word-level proofs for the conditions used by LowerInvariantArithmetic.

GPU tests exercise the lowering; these checks quantify over complete word
domains rather than relying on sampled inputs or Z3's mathematical Int model.
"""

import pytest
import z3


@pytest.mark.parametrize("bits", [8, 16, 32, 64])
def test_single_correction_word_domain(bits):
    x, d = z3.BitVecs("x d", bits)
    wide_x, wide_d = z3.ZeroExt(1, x), z3.ZeroExt(1, d)
    bounded = z3.ULT(wide_x, wide_d << 1)
    correction = z3.UGE(x, d)
    remainder = z3.If(correction, x - d, x)
    quotient = z3.If(correction, z3.BitVecVal(1, bits + 1), z3.BitVecVal(0, bits + 1))

    solver = z3.Solver()
    solver.set(timeout=10000)
    solver.add(d != 0)
    # Division by two avoids forming 2*d at the original width.
    solver.add(z3.ULT(z3.LShR(x, 1), d) != bounded)
    assert solver.check() == z3.unsat

    solver.reset()
    solver.set(timeout=10000)
    solver.add(d != 0, bounded)
    # The division algorithm uniquely specifies floor quotient/remainder.
    solver.add(z3.Not(z3.And(z3.ULT(remainder, d), wide_x == quotient * wide_d + z3.ZeroExt(1, remainder))))
    assert solver.check() == z3.unsat


@pytest.mark.parametrize("bits", [8, 16, 32, 64])
@pytest.mark.parametrize("op", ["xor", "or"])
def test_bitwise_sum_bound(bits, op):
    x, y = z3.BitVecs("x y", bits)
    value = x ^ y if op == "xor" else x | y
    solver = z3.Solver()
    solver.set(timeout=10000)
    solver.add(z3.UGT(z3.ZeroExt(1, value), z3.ZeroExt(1, x) + z3.ZeroExt(1, y)))
    assert solver.check() == z3.unsat


@pytest.mark.parametrize("bits", [8, 16, 32, 64])
def test_word_bound_hypotheses_are_necessary(bits):
    x, d = z3.BitVecs("x d", bits)
    wide_x, wide_d = z3.ZeroExt(1, x), z3.ZeroExt(1, d)
    solver = z3.Solver()
    solver.add(d != 0, z3.ULT(wide_x, wide_d << 1) != z3.ULT(x, d << 1))
    assert solver.check() == z3.sat

    solver.reset()
    solver.add(x > 0, d > 0, x * d < 0)
    assert solver.check() == z3.sat
