import os

import pytest


pytestmark = pytest.mark.skipif(os.environ.get("TILELANG_LIGHT_IMPORT") == "1", reason="Host adapters require the full frontend")


@pytest.mark.parametrize("explicit_layout", [False, True])
@pytest.mark.parametrize("dtype,element_bytes", [("uint8", 1), ("float16", 2), ("float32", 4)])
def test_native_host_address_of_uses_layout_without_reading_memory(explicit_layout, dtype, element_bytes):
    from tvm_tirx import ir, tirx

    from tilelang._compat.tensor import TensorView
    from tilelang.jit.adapter.nvrtc.runtime import NativeHostProgram

    layout = {"strides": [24, 2], "elem_offset": 5} if explicit_layout else {}
    tensor = TensorView.create("tensor", [8, 16], dtype, **layout)
    pointer = ir.Var("pointer", ir.PtrType(dtype))
    stream = ir.Var("stream", ir.PtrType("void"))
    function = tirx.Function(
        [pointer, stream],
        [
            ir.Bind(tensor.tensor, ir.Call("tirx.decl_tensor", [pointer], ty_args=[tensor.tensor.ty])),
            ir.Return(ir.Call("tirx.address_of", [tensor.load((2, 3))])),
        ],
        ret_type=pointer.ty,
    )
    program = NativeHostProgram(ir.save_json(function))
    element_offset = 2 * 24 + 3 * 2 + 5 if explicit_layout else 2 * 16 + 3
    assert program({}, [4096]) == 4096 + element_offset * element_bytes


@pytest.mark.parametrize("begin,end,step,expected", [(0, 8, 2, 6), (7, -1, -2, 1), (0, 8, -1, 42), (7, 0, 2, 42)])
def test_nvrtc_host_interpreter_respects_signed_loop_steps(begin, end, step, expected):
    from tvm_tirx import ir, tirx

    from tilelang.jit.adapter.nvrtc.runtime import NativeHostProgram

    start = ir.Var("start", "int32")
    stop = ir.Var("stop", "int32")
    stride = ir.Var("stride", "int32")
    stream = ir.Var("stream", ir.PtrType("void"))
    index = ir.Var("index", "int32")
    body = ir.For(
        index,
        start,
        stop,
        stride,
        ir.ForKind.DEFAULT,
        ir.If(index + stride > stop if step < 0 else index + stride < stop, ir.Continue(), ir.Return(index)),
    )
    function = tirx.Function([start, stop, stride, stream], [body, ir.Return(42)], ret_type=ir.PrimType("int32"))
    assert tirx.analysis.check_well_formed(function)
    program = NativeHostProgram(ir.save_json(function))
    assert program({}, [begin, end, step]) == expected
    with pytest.raises(ValueError, match="zero step"):
        program({}, [begin, end, 0])


@pytest.mark.parametrize("backend", ["nvrtc", "cutedsl"])
@pytest.mark.parametrize("explicit_stream", [None, 0, 123])
def test_cuda_adapters_use_target_kind_and_preserve_explicit_stream(monkeypatch, backend, explicit_stream):
    import importlib
    from types import SimpleNamespace

    from tvm_tirx.target import Target

    module = importlib.import_module(f"tilelang.jit.adapter.{backend}.adapter")
    adapter_class = module.NVRTCKernelAdapter if backend == "nvrtc" else module.CuTeDSLKernelAdapter
    adapter = adapter_class.__new__(adapter_class)
    adapter.params = [None]
    adapter.result_idx = []
    adapter.dynamic_symbolic_map = {}
    adapter.dynamic_symbolic_order = []
    adapter.target = Target({"kind": "cuda", "arch": "sm_90"})

    class Tensor:
        is_cuda = True
        device = SimpleNamespace(index=0)

    monkeypatch.setattr(module.torch, "Tensor", Tensor)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=987))
    launches = []
    adapter._forward_from_prebuild_lib = lambda *arguments, **kwargs: launches.append(kwargs)
    adapter._wrap_forward_from_prebuild_lib(Tensor(), stream=explicit_stream)
    assert launches[0]["stream"] == (987 if explicit_stream is None else explicit_stream)
