# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from tvm_tirx import ir, sym
from tvm_tirx.ir import prim
from tvm_tirx.ir import assert_structural_equal
import pytest
import tilelang.testing


def test_simplify_reshape_flattened_index():
    ana = sym.Analyzer()
    constraints = sym.ConstraintContext()

    i0 = ir.Var("i0", "int64")
    i1 = ir.Var("i1", "int64")
    constraints.append(sym.RangeConstraint(i0, ir.Range(0, 8)))
    constraints.append(sym.RangeConstraint(i1, ir.Range(0, 3)))
    ana.sync_constraint_context(constraints)

    i_flattened = i0 * 3 + i1
    assert_structural_equal(
        ana.simplify((i_flattened) // 12 * 12 + (i_flattened) % 12 // 4 * 4 + (i_flattened) % 4),
        i_flattened,
    )


_dtypes = (
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "float32",
    "float64",
)


@pytest.fixture(params=_dtypes)
def dtype(request):
    return request.param


def test_can_prove_self_identity(dtype):
    ana = sym.Analyzer()

    n = ir.Var("n", dtype)
    assert ana.can_prove(n == n)


def test_can_prove_self_equal_to_self(dtype):
    ana = sym.Analyzer()

    n = ir.Var("n", dtype)
    assert ana.can_prove_equal(n, n)


def test_simplify_symbolic_comparison():
    ana = sym.Analyzer()
    constraints = sym.ConstraintContext()

    i0 = ir.Var("i0", "int64")
    i1 = ir.Var("i1", "int64")
    n, m = ir.Var("n", "int64"), ir.Var("m", "int64")
    # Size variables previously carried this fact through their node type.
    constraints.append(sym.PredicateConstraint(n >= 0))
    constraints.append(sym.PredicateConstraint(m >= 0))
    outer = (n + 31) // 32
    constraints.append(sym.RangeConstraint(i0, ir.Range(0, outer)))
    constraints.append(sym.RangeConstraint(i1, ir.Range(0, 32)))
    ana.sync_constraint_context(constraints)
    PS = sym.ProofStrength

    assert ana.can_prove(i0 * 32 + i1 < (n + 31) // 32 * 32, PS.SYMBOLIC_BOUND)
    assert ana.can_prove(i0 * 32 + i1 < (n + 31) // 32 * 32 + m, PS.SYMBOLIC_BOUND)
    assert ana.can_prove(i0 * 32 + i1 + 1 <= (n + 31) // 32 * 32, PS.SYMBOLIC_BOUND)
    assert ana.can_prove((n + 31) // 32 * 32 >= i0 * 32 + i1 + 1, PS.SYMBOLIC_BOUND)
    assert ana.can_prove((n + 31) // 32 * 32 >= i0 * 32 + i1, PS.SYMBOLIC_BOUND)


def test_regression_simplify_inf_recursion():
    ana = sym.Analyzer()
    cond = ir.Var("cond", "int32")

    res = (prim.NE(cond, 0).astype("int8") - prim.NE(cond, 0).astype("int8")).astype("int32") == 0
    # regression in a previous case
    # try compare and int set recursive call can cause infinite loop
    ana.rewrite_simplify(res)


def test_simplify_floor_mod_with_linear_offset():
    """
    Test that the floor_mod is simplified correctly when the offset is linear.
    """
    ana = sym.Analyzer()
    past_decoder_sequence_length = ir.Var("past_decoder_sequence_length", "int64")
    expr1 = (past_decoder_sequence_length + 1) * 64
    divisor1 = (past_decoder_sequence_length + 1) * 32
    assert ana.can_prove_equal(prim.floormod(expr1, divisor1), 0)
    divisor2 = 32 * (past_decoder_sequence_length + 1)
    assert ana.can_prove_equal(prim.floormod(expr1, divisor2), 0)


def test_simplify_float_division():
    # Test for the discussion:
    # https://discuss.tvm.apache.org/t/discuss-is-constant-division-to-multiplication-rewrite-in-tvm-necessary/18615
    ana = sym.Analyzer()
    x = ir.Var("x", "float32")
    ry = x / 27
    # in old version, the division will be rewritten into x * T.float32(1 / 27)
    sy = ana.rewrite_simplify(ry)
    assert_structural_equal(ry, sy)


if __name__ == "__main__":
    tilelang.testing.main()
