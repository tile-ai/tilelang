import pytest

from tvm_tirx import tirx
from tvm_tirx.target import Target

from tilelang._compat import ir, stmt_functor
from tilelang._compat.tensor import TensorView
from tilelang.transform import _tirx_passes


@pytest.mark.parametrize("backend", ["c", "llvm"])
def test_scalar_only_host_allocation_has_a_cpu_device_context(backend, compile_lowered_cpu):
    size = ir.Var("size", "int32")
    scratch = TensorView.create("scratch", [size], "int32")
    function = tirx.Function(
        [size],
        [scratch.allocate(), scratch.store(size - 1, size + 7), ir.Return(scratch.load(size - 1))],
        ret_type=ir.PrimType("int32"),
    ).with_attr("global_symbol", "main")
    executable = compile_lowered_cpu(function, backend)
    assert executable(17) == 24
    assert executable(2048) == 2055


def test_explicit_host_allocation_context_is_not_replaced(native_tirx_passes):
    size = ir.Var("size", "int32")
    scratch = TensorView.create("scratch", [size], "int32")
    scope = ir.RegionStmt("tirx.device_context", [ir.const(1), ir.const(3)], [], {}, [scratch.allocate(), scratch.store(0, 7)])
    function = tirx.Function([size], scope).with_attr("target", Target("c"))
    lowered = _tirx_passes.LowerTVMBuiltin()(ir.IRModule({"main": function}))
    allocations = []
    stmt_functor.post_order_visit(
        lowered,
        lambda node: allocations.append(node)
        if isinstance(node, ir.Call) and isinstance(node.op, ir.Op) and node.op.name == "tirx.alloc_workspace"
        else None,
    )
    assert len(allocations) == 1
    assert int(allocations[0].args[1]) == 3
