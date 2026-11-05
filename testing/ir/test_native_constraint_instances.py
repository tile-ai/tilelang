"""Distinct loop instances must not share a collector-generated unknown value."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx


@pytest.mark.parametrize("step", [-1, 0, 1, None])
@pytest.mark.parametrize("local_binding", [False, True])
def test_renamed_loop_facts_do_not_prove_unrelated_claim(step, local_binding):
    index = ir.Var("i", "int32")
    stride = ir.Var("stride", "int32") if step is None else ir.const(step)
    body = []
    if local_binding:
        body.append(ir.Bind(ir.Var("j", "int32"), index + 1))
    output = ir.Var("output", tirx.TensorType((1,), "int32"))
    body.append(ir.TensorStore(output, [0], index))
    function = tirx.Function(
        [output, stride] if step is None else [output],
        [ir.For(index, 3, 7, stride, ir.ForKind.DEFAULT, body)],
    )
    assert tirx.analysis.check_well_formed(function)
    # The collector has no constraint on this fresh query variable. In
    # particular, i_left != i_right must not contradict invented shared state.
    prove = tvm_ffi.get_global_func("tl.testing.RenamedLoopInstances")
    # A zero-step loop can revisit the same coordinate, so distinct iterations
    # must not be modeled as distinct coordinates in that case.
    assert not prove(function, index, step != 0)
