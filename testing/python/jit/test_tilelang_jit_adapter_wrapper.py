import pytest
import torch
import tilelang.testing

from tilelang import tvm
from tilelang.jit.adapter.shape import OutputShapeResolver
from tilelang.jit.adapter.wrapper import TLCPUSourceWrapper


def test_adapter_source_wrapper_requires_lowered_modules():
    with pytest.raises(ValueError, match="missing: device_mod, host_mod"):
        TLCPUSourceWrapper(
            scheduled_ir_module=tvm.IRModule(),
            source="",
            target=tvm.target.Target("c"),
        )


def test_output_shape_expressions():
    tir = tvm.tirx
    n, stride, k = (tir.Var(name, "int32") for name in ("n", "stride", "k"))
    source = tir.decl_buffer((n,), "float32", name="A", strides=(stride,))
    output = tir.decl_buffer((n * 2, (n + k + 3) // 4, stride + 1), "float32", name="C")
    program = tir.PrimFunc([output.data, source.data, k], tir.Evaluate(0), buffer_map={output.data: output, source.data: source})
    for func in (program, tvm.ir.load_json(tvm.ir.save_json(program))):
        resolver = OutputShapeResolver(func, [0])
        assert resolver(0, [None, torch.empty_strided((7,), (5,)), 3]) == [14, 3, 6]
        with pytest.raises(ValueError, match="output C dimension 0.*no live input"):
            resolver(0, [None, None, 3])


@pytest.mark.parametrize("case", ["negative", "unresolved", "zero_divisor"])
def test_output_shape_reports_invalid_expression(case):
    tir = tvm.tirx
    n = tir.Var("n", "int32")
    expressions = {
        "negative": (n - 8, "negative extent -1"),
        "unresolved": (tir.call_extern("int32", "unresolved_dim", n), "did not resolve to an integer"),
        "zero_divisor": (n // (n - 7), "Divide by zero"),
    }
    expr, reason = expressions[case]
    output = tir.decl_buffer((expr,), "float32", name="C")
    func = tir.PrimFunc([n, output.data], tir.Evaluate(0), buffer_map={output.data: output})
    with pytest.raises(ValueError, match="output C dimension 0") as error:
        OutputShapeResolver(func, [1])(1, [7, None])
    assert reason in str(error.value)


@pytest.mark.parametrize("value", [None, 1.5, torch.tensor(7)])
def test_output_shape_rejects_noninteger_scalar(value):
    tir = tvm.tirx
    n = tir.Var("n", "int32")
    output = tir.decl_buffer((n * 2,), "float32", name="C")
    func = tir.PrimFunc([n, output.data], tir.Evaluate(0), buffer_map={output.data: output})
    with pytest.raises(ValueError, match="output C dimension 0"):
        OutputShapeResolver(func, [1])(1, [value, None])


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("cooperative", [False, True])
def test_cuda_host_symbol_identity(cooperative):
    from tilelang.jit.adapter.wrapper import TLCUDASourceWrapper

    tir = tvm.tirx
    left, right = tir.Var("n", "int32"), tir.Var("n", "int32")
    wrapper = TLCUDASourceWrapper.__new__(TLCUDASourceWrapper)
    wrapper.mod = tvm.IRModule({"main": tir.PrimFunc([left, right], tir.Evaluate(0))})
    functions, launches = {}, {}
    for name, operands in [("first", [right, left]), ("second", [left, right])]:
        x, y = tir.Var("n", "int32"), tir.Var("n", "int32")
        functions[name] = tir.PrimFunc([x, y], tir.Evaluate(0))
        launches[name] = {"function_params": operands, "grid_info": [y, 1, 1], "block_info": [1, 1, 1], "dynamic_smem_buf": 0}
    wrapper.device_mod = tvm.IRModule(functions)
    wrapper.use_cooperative_groups = dict.fromkeys(functions, cooperative)
    wrapper.cluster_dims = dict.fromkeys(functions)
    wrapper.l2_persistent_map = {}
    wrapper.tma_descriptor_args = None
    code = "__global__ void first(int n, int n_1); __global__ void second(int n, int n_1); void bodies() {}"
    source = wrapper.create_dispatch_func(code, launches)
    assert "int n, int _tl_arg_1" in source
    if cooperative:
        assert "first_args[] = {(void*)&_tl_arg_1, (void*)&n}" in source
        assert "second_args[] = {(void*)&n, (void*)&_tl_arg_1}" in source
        assert "(void*)first, dim3((n), 1, 1)" in source
        assert "(void*)second, dim3((_tl_arg_1), 1, 1)" in source
    else:
        first_source, second_source = source.split('TILELANG_CHECK_LAST_ERROR("first");')
        assert "config.gridDim = dim3((n), 1, 1)" in first_source
        assert "config.gridDim = dim3((_tl_arg_1), 1, 1)" in second_source
        assert "cudaLaunchKernelEx(&config, first, _tl_arg_1, n)" in source
        assert "cudaLaunchKernelEx(&config, second, n, _tl_arg_1)" in source


if __name__ == "__main__":
    pytest.main([__file__])
