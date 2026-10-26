import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, tirx
from tvm_tirx.ir import prim
from tvm_tirx.target import Target

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView

pytestmark = pytest.mark.usefixtures("native_layouts", "native_operator_helpers")


@pytest.mark.parametrize("lanes", [3, 4, 8, 16])
@pytest.mark.parametrize("explicit", [False, True])
def test_rocm_async_copy_preserves_native_types_and_visibility(lanes, explicit):
    source = TensorView.create("source", [lanes], "uint8")
    shared = TensorView.create("shared", [lanes], "uint8", scope="shared", data_alignment=16)
    condition = ir.Var("condition", "bool")
    ramp = prim.Ramp(0, 1, lanes)
    value = prim.if_then_else(condition, source.load(ramp), prim.Broadcast(ir.const(0, "uint8"), lanes))
    function = tirx.Function([source.tensor, condition], [shared.allocate(), shared.store(ramp, value)])
    module = ir.IRModule({"main": function})
    lowered = tvm_ffi.get_global_func("tl.rocm.transform.InjectROCmAsyncCopy")(explicit)(module)
    calls = []
    stmt_functor.post_order_visit(lowered, lambda node: calls.append(node) if isinstance(node, ir.Call) else None)
    copies = [call for call in calls if call.op.name == "tl.ptx_cp_async"]
    assert len(copies) == int(lanes != 3)
    if copies:
        assert copies[0].args[0].ty.scope.name == "shared"
        assert copies[0].args[1].ty.scope.name == "global"
        assert int(copies[0].args[2]) == lanes
        assert copies[0].args[3].same_as(condition)
    names = [call.op.name for call in calls]
    assert ("tirx.ptx.cp_async_commit_group" in names) == (not explicit and bool(copies))
    assert ("tirx.ptx.cp_async_wait_group" in names) == (not explicit and bool(copies))
    assert tirx.analysis.check_well_formed(lowered)


@pytest.mark.parametrize("size", [31, 64, 133])
def test_rocm_simt_copy_uses_portable_native_ir(size):
    source = TensorView.create("source", [size], "float32")
    output = TensorView.create("output", [size], "float32")
    operation = tvm_ffi.get_global_func("tl.ParseOperator")(ir.Call("tl.tileop.copy", [source.tensor, output.tensor]), {})
    target = Target({"kind": "rocm", "mcpu": "gfx942"})
    thread = ir.Var("thread", "int32")
    with target:
        body = tvm_ffi.get_global_func("tl.testing.LowerOperator")(operation, target, ir.Range(0, 64), thread, {}, {})
    function = tirx.Function([source.tensor, output.tensor], ir.For(thread, 0, 64, 1, ir.ForKind.DEFAULT, body)).with_attr(
        "global_symbol", "rocm_copy"
    )
    assert tirx.analysis.check_well_formed(function)
    executable = tirx.compile(function, target="c")
    expected = np.arange(size, dtype="float32")
    actual = runtime.empty((size,), "float32")
    executable(runtime.tensor(expected), actual)
    np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "int8", "uint8"])
@pytest.mark.parametrize("architecture", ["gfx942", "gfx950", "gfx1100"])
def test_hip_source_generation_uses_native_dependency(dtype, architecture):
    import tilelang
    import tilelang.language as T

    @T.prim_func
    def copy(source: T.Tensor((128,), dtype), output: T.Tensor((128,), dtype)):
        with T.Kernel(1, threads=64):
            shared = T.alloc_shared((128,), dtype)
            T.copy(source, shared)
            T.copy(shared, output)

    artifact = tilelang.lower(copy, target=Target({"kind": "rocm", "mcpu": architecture}))
    source = artifact.kernel_source
    assert 'extern "C" __global__' in source
    assert "threadIdx.x" in source
    assert "#include <hip/hip_runtime.h>" in source
    assert "__shared__" in source


@pytest.mark.parametrize("architecture", ["gfx942", "gfx950"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("explicit_async", [False, True])
def test_original_hip_gemm_and_async_copy_lower_to_specialized_source(architecture, dtype, explicit_async):
    import tilelang
    from testing.python.amd.test_tilelang_gfx950_copy_async import _matmul_kernel, _matmul_kernel_async

    arguments = (256, 256, 256, 128, 128, 32, False, True, dtype, "float32", "float32")
    if explicit_async:
        function = _matmul_kernel_async(*arguments, coalesced_width=8)
    else:
        function = _matmul_kernel(*arguments, num_stages=3)
    artifact = tilelang.lower(function, target=Target({"kind": "rocm", "mcpu": architecture}))
    source = artifact.kernel_source
    assert "__builtin_amdgcn_mfma_" in source or "tl::gemm" in source
    assert "__shared__" in source
    if explicit_async:
        assert "cp_async_gs<16>" in source
