import pytest

import tilelang.language as language
from tilelang._compat import ir
from tvm_tirx.script.ir_builder import IRBuilder
from tvm_tirx.tirx.script import ir_builder


@pytest.mark.parametrize("thread_tag", ["blockIdx.x", "threadIdx.x"])
def test_launch_thread_keeps_metadata_in_a_tilelang_scope(thread_tag):
    with IRBuilder() as builder, ir_builder.function(), language.launch_thread(thread_tag, 32, attrs={"tl.launch_hint": 7}):
        ir_builder.evaluate(0)
    function = builder.get()
    launch = function.body.seq[0]
    assert launch.op.name == "tirx.launch_thread"
    assert not launch.attrs
    scope = launch.body.seq[0]
    assert scope.op.name == "tl.scope"
    assert scope.attrs["tl.launch_hint"] == 7
    assert ir.check_well_formed(function)
    ir.assert_structural_equal(function, ir.load_json(ir.save_json(function)))


def test_native_launch_thread_still_rejects_foreign_attributes():
    with (
        pytest.raises((TypeError, ValueError), match="attrs|attributes"),
        IRBuilder(),
        ir_builder.launch_thread("threadIdx.x", 32, attrs={"tl.launch_hint": 7}),
    ):
        ir_builder.evaluate(0)
