import pytest
from tvm_tirx import ir, target, tirx

from tilelang.transform import _tirx_passes

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


@pytest.mark.parametrize("kind,calling_convention", [("c", 1), ("cuda", 2)])
def test_lowered_functions_keep_their_execution_target(kind, calling_convention):
    function = tirx.Function([], []).with_attr("target", target.Target(kind))
    function = function.with_attr("global_symbol", "lowered")
    function = function.with_attr("calling_conv", calling_convention)
    module = ir.IRModule({"main": function})
    result = _tirx_passes.BindTarget(target.Target("cuda", host="c"))(module)
    ir.assert_structural_equal(module, result)
    assert result["main"].attrs["target"].host is None


def test_device_kernel_symbols_are_not_host_entry_candidates():
    host = tirx.Function([], []).with_attr("global_symbol", "host")
    kernel = tirx.Function([], []).with_attr("global_symbol", "kernel")
    kernel = kernel.with_attr("calling_conv", 2)
    result = _tirx_passes.AnnotateEntryFunc()(ir.IRModule({"host": host, "kernel": kernel}))
    assert result["host"].attrs["tirx.is_entry_func"]
    assert not result["kernel"].attrs.get("tirx.is_entry_func", False)
