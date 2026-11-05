"""Absent optional inputs only supply placeholders for proven dead ABI slots."""

import numpy as np
import pytest
import torch
from tvm_tirx import ir, tirx

import tilelang.language as T
import tilelang.testing
from tilelang.engine.param import KernelParam
from tilelang.jit.abi import ShapeBindings
from tilelang.jit.adapter.abi import SourceABIPlan, source_abi_plan


def _function(parameters, statements=()):
    return tirx.Function(parameters, ir.SeqStmt(list(statements)), ir.TupleType([])).with_attr("global_symbol", "main")


def _tensor(name, shape):
    return ir.Var(name, tirx.TensorType(shape, "float32"))


def _adapter(function, backend="cutedsl", outputs=()):
    if backend == "cutedsl":
        from tilelang.jit.adapter.cutedsl.adapter import CuTeDSLKernelAdapter as adapter_type
    else:
        pytest.importorskip("cuda.bindings.nvrtc")
        from tilelang.jit.adapter.nvrtc.adapter import NVRTCKernelAdapter as adapter_type

    adapter = adapter_type.__new__(adapter_type)
    adapter.ir_module = ir.IRModule({"main": function})
    adapter.params = [KernelParam.from_var(parameter) for parameter in function.params]
    adapter.result_idx = list(outputs)
    adapter._initialize_native_arguments()
    return adapter


@pytest.mark.parametrize("kind", ["shape", "stride"])
def test_dead_optional_slot_preserves_zero_only_under_its_policy(kind):
    n = T.dynamic("n")
    annotation = T.Tensor((n,), "float32") if kind == "shape" else T.StridedTensor((4,), (n,), "float32")
    parameter = ir.Var("optional", annotation.as_type())
    plan = SourceABIPlan(_function([parameter]))
    bindings = plan.bind_arguments([None])
    assert len(plan.implicit_symbols) == 1
    assert plan.resolve_implicit_value(n, bindings, require_live_shape=False) == 0
    assert bindings.try_evaluate(n) is None  # A placeholder is not a mathematical binding.
    if kind == "shape":
        with pytest.raises(TypeError, match="no live tensor source"):
            plan.resolve_implicit_value(n, bindings)
    else:
        assert plan.resolve_implicit_value(n, bindings) == 0


@pytest.mark.parametrize("access", ["scalar", "tensor_type", "function_attr", "return_type"])
def test_body_and_tensor_type_uses_never_receive_a_placeholder(access):
    n = T.dynamic("n")
    parameter = _tensor("optional", (n,))
    if access == "function_attr":
        function = _function([parameter]).with_attr("opaque_metadata", n)
    elif access == "return_type":
        function = tirx.Function([parameter], ir.SeqStmt([]), ir.TupleType([parameter.ty]))
    else:
        expression = n if access == "scalar" else ir.TensorLoad(parameter, [ir.const(0)])
        function = _function([parameter], [ir.Evaluate(expression)])
    plan = SourceABIPlan(function)
    with pytest.raises(TypeError, match="no live tensor source"):
        plan.resolve_implicit_value(n, plan.bind_arguments([None]), require_live_shape=False)


@pytest.mark.parametrize("direct_carrier", [False, True])
def test_unresolved_compound_dimensions_do_not_get_a_zero_witness(direct_carrier):
    n = T.dynamic("n")
    parameters = [_tensor("compound", (n * n,))]
    if direct_carrier:
        parameters.append(_tensor("direct", (n,)))
    plan = SourceABIPlan(_function(parameters))
    assert n in plan.implicit_symbols
    with pytest.raises(TypeError, match="no live tensor source"):
        plan.resolve_implicit_value(n, plan.bind_arguments([None] * len(parameters)), require_live_shape=False)


def test_unresolved_offset_and_explicit_scalar_remain_required():
    offset = T.dynamic("offset")
    tensor = ir.Var("optional", T.Buffer((4,), "float32", elem_offset=offset).as_type())
    offset_plan = SourceABIPlan(_function([tensor]))
    assert offset in offset_plan.implicit_symbols
    with pytest.raises(TypeError, match="no live tensor source"):
        offset_plan.resolve_implicit_value(offset, offset_plan.bind_arguments([None]), require_live_shape=False)

    n = T.dynamic("n")
    scalar_plan = SourceABIPlan(_function([_tensor("optional", (n,)), n]))
    assert n not in scalar_plan.implicit_symbols
    with pytest.raises(TypeError, match="no live tensor source"):
        scalar_plan.resolve_implicit_value(n, scalar_plan.bind_arguments([None, None]), require_live_shape=False)
    assert scalar_plan.resolve_implicit_value(n, scalar_plan.bind_arguments([None, 9])) == 9


def test_output_allocation_stays_strict_after_dead_argument_resolution():
    n = T.dynamic("n")
    plan = SourceABIPlan(_function([_tensor("input", (n,)), _tensor("output", (n,))]))
    bindings = plan.bind_arguments([None, None])
    assert plan.resolve_implicit_value(n, bindings, require_live_shape=False) == 0
    with pytest.raises(ValueError, match="Cannot resolve"):
        plan.output_shape(1, bindings)


@pytest.mark.parametrize("restored", [False, True])
def test_later_live_inputs_bind_by_identity_across_json_roundtrip(restored):
    first, second = T.dynamic("same_name"), T.dynamic("same_name")
    function = _function([_tensor("absent", (first,)), _tensor("live", (second,))])
    if restored:
        function = ir.load_json(ir.save_json(function))
    plan = SourceABIPlan(function)
    first, second = plan.implicit_symbols
    bindings = plan.bind_arguments([None, np.empty(7, dtype="float32")])
    assert plan.resolve_implicit_value(first, bindings, require_live_shape=False) == 0
    assert plan.resolve_implicit_value(second, bindings) == 7
    assert bindings.try_evaluate(first) is None


def test_later_affine_shape_can_supply_a_real_value_without_fallback():
    n = T.dynamic("n")
    plan = SourceABIPlan(_function([_tensor("absent", (n,)), _tensor("affine", (2 * n + 1,))]))
    bindings = plan.bind_arguments([None, np.empty(15, dtype="float32")])
    assert plan.resolve_implicit_value(n, bindings) == 7
    with pytest.raises(ValueError, match="divisible"):
        plan.bind_arguments([None, np.empty(14, dtype="float32")])


def test_equal_shape_constraints_and_fresh_calls_are_not_relaxed():
    n = T.dynamic("n")
    plan = SourceABIPlan(_function([_tensor("a", (n,)), _tensor("b", (n,))]))
    for size in [7, 0, 11]:
        bindings = plan.bind_arguments([None, np.empty(size, dtype="float32")])
        assert plan.resolve_implicit_value(n, bindings) == size
    with pytest.raises(ValueError, match="conflicts"):
        plan.bind_arguments([np.empty(7, dtype="float32"), np.empty(8, dtype="float32")])


def test_cached_plan_rechecks_liveness_after_function_identity_changes():
    n = T.dynamic("n")
    function = _function([_tensor("optional", (n,))])
    adapter = _adapter(function)
    old = source_abi_plan(adapter)
    assert old.resolve_implicit_value(n, ShapeBindings(), require_live_shape=False) == 0
    adapter.ir_module = ir.IRModule({"main": function.with_body(ir.SeqStmt([ir.Evaluate(n)]))})
    current = source_abi_plan(adapter)
    assert current is not old
    with pytest.raises(TypeError, match="no live tensor source"):
        current.resolve_implicit_value(n, ShapeBindings(), require_live_shape=False)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("backend", ["cutedsl", "nvrtc"])
@pytest.mark.parametrize("optional_kind", ["shape", "stride"])
def test_real_argument_wrapper_preserves_optional_abi_and_stream(backend, optional_kind):
    n = T.dynamic("n")
    optional = T.Tensor((n,), "float32") if optional_kind == "shape" else T.StridedTensor((4,), (n,), "float32")
    function = _function([_tensor("source", (7,)), ir.Var("optional", optional.as_type())])
    adapter = _adapter(function, backend)
    forwarded = []
    adapter._forward_from_prebuild_lib = lambda *args, **kwargs: forwarded.append((args, kwargs))
    data = torch.empty(7, device="cuda")
    stream = torch.cuda.Stream()
    returned = adapter._wrap_forward_from_prebuild_lib(data, None, stream=stream.cuda_stream)
    assert returned == []
    assert len(forwarded) == 1
    assert forwarded[0][0][0] is data
    assert forwarded[0][0][1] is None
    assert forwarded[0][0][2:] == (0,)
    assert forwarded[0][1]["stream"] == stream.cuda_stream
    assert forwarded[0][1]["device_id"] == data.device.index
    stream.synchronize()


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("backend", ["cutedsl", "nvrtc"])
def test_real_wrapper_never_allocates_a_result_using_optional_zero(backend):
    n = T.dynamic("n")
    function = _function([_tensor("optional", (n,)), _tensor("result", (n,))])
    adapter = _adapter(function, backend, outputs=(1,))
    forwarded = []
    adapter._forward_from_prebuild_lib = lambda *args, **kwargs: forwarded.append((args, kwargs))
    with pytest.raises(ValueError, match="Cannot resolve.*shape"):
        adapter._wrap_forward_from_prebuild_lib(None)
    assert forwarded == []


def _optional_add_program():
    n, missing = T.dynamic("n"), T.dynamic("missing")

    @T.prim_func
    def main(
        source: T.Tensor((n,), "float32"),
        optional: T.Tensor((missing,), "float32"),
        output: T.Tensor((n,), "float32"),
    ):
        with T.Kernel(T.ceildiv(n, 64), threads=64) as (block,):
            for lane in T.Parallel(64):
                index = block * 64 + lane
                if index < n:
                    output[index] = source[index] + 1.0

    return main


@tilelang.testing.requires_cuda
def test_optional_cutedsl_actual_execution_and_cached_reload(tmp_path, monkeypatch):
    from tilelang.jit.adapter.cutedsl.adapter import CuTeDSLKernelAdapter
    from tilelang.jit.adapter.cutedsl.checks import check_cutedsl_available
    from tilelang.jit.adapter.cutedsl import _compiler

    check_cutedsl_available()
    major, minor = torch.cuda.get_device_capability()
    target = {"kind": "cutedsl", "arch": f"sm_{major}{minor}"}
    function = _optional_add_program()
    kernel = tilelang.compile(function, out_idx=2, target=target, execution_backend="cutedsl")
    stream = torch.cuda.Stream()
    for size in (7, 133):
        with torch.cuda.stream(stream):
            source = torch.arange(size, dtype=torch.float32, device="cuda")
            result = kernel(source, None, stream=stream.cuda_stream)
        stream.synchronize()
        torch.testing.assert_close(result, source + 1.0, rtol=0, atol=0)

    module = tmp_path / "kernel.py"
    module.write_text(kernel.adapter.get_generated_module_source())
    kernel.adapter.lib_generator.runtime.export_artifacts(tmp_path)
    restored_function = ir.load_json(ir.save_json(function))
    restored = CuTeDSLKernelAdapter.from_database(
        params=[KernelParam.from_var(parameter) for parameter in restored_function.params],
        result_idx=[2],
        target=target,
        func_or_mod=restored_function,
        host_kernel_source=kernel.get_host_source(),
        device_kernel_source=kernel.get_kernel_source(),
        kernel_lib_path=str(module),
    )

    def no_recompile(*args, **kwargs):
        raise AssertionError("A persisted native CuTeDSL cubin must reload without recompilation")

    monkeypatch.setattr(_compiler, "compile_kernel", no_recompile)
    with torch.cuda.stream(stream):
        source = torch.arange(29, dtype=torch.float32, device="cuda")
        result = restored(source, None, stream=stream.cuda_stream)
    stream.synchronize()
    torch.testing.assert_close(result, source + 1.0, rtol=0, atol=0)
    assert result.shape == (29,)
    assert restored.lib_generator.runtime._entries
