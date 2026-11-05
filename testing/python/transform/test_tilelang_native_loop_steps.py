"""Public stepped loops retain Python-range behavior through native lowering."""

import numpy as np
import pytest
from tvm_tirx import ir, tirx, runtime

import tilelang
import tilelang.language as T
from tilelang.testing.packed import compile_packed_function


@pytest.mark.parametrize("bounds", [(1, 8, 3), (8, 0, -3), (1, 8, -2)])
@pytest.mark.parametrize("explicit", [False, True])
def test_public_stepped_loop_numerical(bounds, explicit):
    begin, end, step = bounds

    @T.prim_func
    def kernel(A: T.Tensor((16,), "int32")):
        for index in T.unroll(begin, end, step, explicit=explicit):
            A[index] = index

    original = ir.save_json(kernel)
    transformed = tilelang.transform.UnrollLoop()(ir.IRModule({"main": kernel}))["main"]
    assert ir.save_json(kernel) == original
    assert tirx.analysis.check_well_formed(transformed)
    executable = compile_packed_function(transformed, c_codegen="compute")
    actual = runtime.tensor(np.full(16, -99, dtype="int32"))
    executable(actual)
    expected = np.full(16, -99, dtype="int32")
    for index in range(begin, end, step):
        expected[index] = index
    np.testing.assert_array_equal(actual.numpy(), expected)
