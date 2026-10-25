import pytest
from tvm_tirx import ir, tirx

import tilelang.language as T
from tilelang._compat.stmt_functor import post_order_visit


@pytest.mark.parametrize("private", [False, True])
def test_prim_func_native_visibility_and_validation(private):
    @T.prim_func(private=private, check_well_formed=True)
    def function(output: T.Tensor((4,), "int32")):
        for index in T.serial(4):
            output[index] = index

    assert ("global_symbol" in function.attrs) is not private
    assert tirx.analysis.check_well_formed(function)


def test_capture_bearing_lowering_templates_can_opt_out_of_validation():
    captured = ir.Var("captured", "int32")

    @T.prim_func(check_well_formed=False)
    def template(output: T.Tensor((1,), "int32")):
        output[0] = captured

    assert not tirx.analysis.check_well_formed(template, assert_mode=False)
    with pytest.raises(Exception, match="undefined variable captured"):

        @T.prim_func(check_well_formed=True)
        def invalid(output: T.Tensor((1,), "int32")):
            output[0] = captured


@pytest.mark.parametrize("begin", [0, 16])
def test_thread_binding_uses_native_lexical_launch(begin):
    @T.prim_func(check_well_formed=True)
    def kernel(output: T.Tensor((64,), "int32")):
        for thread in T.thread_binding(begin, 64, thread="threadIdx.x"):
            output[thread] = thread

    launches = []
    post_order_visit(
        kernel, lambda node: launches.append(node) if isinstance(node, ir.RegionStmt) and node.op.name == "tirx.launch_thread" else None
    )
    assert len(launches) == 1
    assert int(launches[0].args[1]) == 64 - begin
