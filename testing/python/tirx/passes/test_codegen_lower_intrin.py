import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

from tilelang._compat.tensor import TensorView
from tilelang.transform import _tirx_passes

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


@pytest.mark.parametrize("backend", ["cuda", "ascend"])
def test_codegen_uses_signed_loop_intrinsic_lowering(backend, request):
    if backend == "ascend":
        request.getfixturevalue("ascend_codegen")
        target = Target({"kind": "ascend", "keys": ["ascend"]})
    else:
        target = Target({"kind": "cuda", "arch": "sm_80"})
    pointer = ir.Var("output", ir.PtrType("int32", scope="global"))
    output = TensorView.create("view", [16], "int32")
    index = ir.Var("index", "int32")
    loop = ir.For(index, 9, -7, -2, ir.ForKind.DEFAULT, output.store(index + 7, ir.prim.floordiv(index, 3)))
    function = tirx.Function([pointer], [output.declare(pointer), loop], ret_type=ir.TupleType([]))
    function = function.with_attr("global_symbol", "signed_loop").with_attr("calling_conv", 2).with_attr("target", target)
    module = ir.IRModule({"main": function})
    assert tirx.analysis.check_well_formed(module)
    build = tvm_ffi.get_global_func(f"target.build.tilelang_{backend}_without_compile")
    expected = build(_tirx_passes.LowerIntrin()(module), target).inspect_source()
    actual = build(module, target).inspect_source()
    assert actual == expected
