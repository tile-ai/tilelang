"""A larger Z3 budget makes the reduction projection from #3698 provable."""

import pytest

import tilelang.testing
from tilelang import tvm


def test_z3_rlimit_proves_reduction_projection():
    if tvm.arith.Analyzer().get_smtlib2() == "; Z3 Prover is disabled.":
        pytest.skip("Z3 support is disabled")

    def index(x):
        return x % 32 // 16 * 16 + x % 16 // 8 * 2

    for config, expected in [({"arith.z3_rlimit": 10_000}, False), ({}, True)]:
        with tvm.transform.PassContext(config=config):
            analyzer = tvm.arith.Analyzer()
            rv = tvm.tirx.Var("rv", "int32")
            analyzer.bind(rv, tvm.ir.Range(0, 128))
            projected = rv - rv // 32 % 2 * 32 - rv % 8
            # The equality exhausts the old 10,000-unit resource budget.
            assert analyzer.can_prove_equal(index(projected), index(rv)) == expected


if __name__ == "__main__":
    tilelang.testing.main()
