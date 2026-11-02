"""A larger Z3 budget makes the reduction projection from #3698 provable."""

import tilelang.testing
from tilelang import tvm


def test_z3_rlimit_proves_reduction_projection():
    def index(x):
        return x % 32 // 16 * 16 + x % 16 // 8 * 2

    for config, expected in [({}, False), ({"arith.z3_rlimit": 100_000}, True)]:
        with tvm.transform.PassContext(config=config):
            analyzer = tvm.arith.Analyzer()
            rv = tvm.tirx.Var("rv", "int32")
            analyzer.bind(rv, tvm.ir.Range(0, 128))
            projected = rv - rv // 32 % 2 * 32 - rv % 8
            # The equality exhausts the default 10,000-unit resource budget.
            assert analyzer.can_prove_equal(index(projected), index(rv)) == expected


if __name__ == "__main__":
    tilelang.testing.main()
