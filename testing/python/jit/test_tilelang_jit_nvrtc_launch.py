import ast

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.backend.module import create_backend_context
from tilelang.engine.lower import device_codegen_without_compile, extrac_params, get_device_call, get_host_call
from tilelang.jit.adapter.nvrtc import is_nvrtc_available
from tilelang.jit.adapter.nvrtc.adapter import NVRTCKernelAdapter
from tilelang.jit.adapter.nvrtc.host import collect_host_launches
from tilelang.jit.adapter.nvrtc.wrapper import TLNVRTCSourceWrapper


def _lower_host_launch(func, simplify):
    # Exercise the adapter boundary before arithmetic simplification inlines host bindings.
    major, minor = torch.cuda.get_device_capability()
    context = create_backend_context({"kind": "cuda", "arch": f"sm_{major}{minor}"}, "c", "nvrtc")
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    for transform in (
        tvm.tirx.transform.BindTarget(context.target),
        tilelang.transform.MaterializeKernelLaunch(),
        tilelang.transform.LowerOpaqueBlock(),
        tilelang.transform.AnnotateDeviceRegions(),
        tilelang.transform.SplitHostDevice(),
        tvm.tirx.transform.AnnotateEntryFunc(),
        tilelang.transform.MakePackedAPI(),
    ):
        mod = transform(mod)
    if simplify:
        mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LowerDeviceKernelLaunch()(mod)
    host = tvm.tirx.transform.Filter(get_host_call())(mod)
    device = tvm.tirx.transform.Filter(get_device_call())(mod)
    source = device_codegen_without_compile(device, context).inspect_source()
    return NVRTCKernelAdapter(extrac_params(func), [], context.target, func, host_mod=host, device_mod=device, device_kernel_source=source)


@tilelang.testing.requires_cuda
@pytest.mark.skipif(not is_nvrtc_available, reason="Requires cuda-python")
@pytest.mark.parametrize("simplify", [False, True])
def test_nvrtc_host_scalar_arguments(simplify):
    @T.prim_func
    def main(A: T.Tensor((128,), "int32"), B: T.Tensor((128,), "int32"), n: T.int32):
        p = T.bind(n * 3 + 1)
        q = T.bind(p * 2 + 5)
        with T.Kernel(1, threads=128):
            i = T.get_thread_binding()
            B[i] = A[i] + p + q

    kernel = _lower_host_launch(main, simplify)
    a = torch.arange(128, dtype=torch.int32, device="cuda")
    b = torch.empty_like(a)
    for n in (7, 13, 0, -3, 7):
        kernel.func(a, b, n)
        torch.testing.assert_close(b, a + 9 * n + 8)


@tilelang.testing.requires_cuda
@pytest.mark.skipif(not is_nvrtc_available, reason="Requires cuda-python")
@pytest.mark.parametrize("simplify", [False, True])
def test_nvrtc_conditional_host_preparation(simplify):
    @T.prim_func
    def main(A: T.Tensor((128,), "int32"), B: T.Tensor((128,), "int32"), n: T.int32):
        if n > 0:
            p = T.bind(120 // n)
            with T.Kernel(1, threads=128):
                i = T.get_thread_binding()
                B[i] = A[i] + p

    kernel = _lower_host_launch(main, simplify)
    a = torch.arange(128, dtype=torch.int32, device="cuda")
    b = torch.empty_like(a)
    for n in (7, 0, -3, 13):
        b.fill_(-999)
        kernel.func(a, b, n)
        expected = a + 120 // n if n > 0 else torch.full_like(a, -999)
        torch.testing.assert_close(b, expected)


@tilelang.testing.requires_cuda
@pytest.mark.skipif(not is_nvrtc_available, reason="Requires cuda-python")
def test_nvrtc_host_integer_cast_and_division():
    @T.prim_func
    def main(B: T.Tensor((128,), "int32"), n: T.int32):
        p = T.bind(T.Cast("int8", n))
        q = T.bind(p // 3)
        r = T.bind(p % 3)
        with T.Kernel(1, threads=128):
            i = T.get_thread_binding()
            B[i] = q * 10 + r

    kernel = _lower_host_launch(main, False)
    b = torch.empty(128, dtype=torch.int32, device="cuda")
    for n in (-7, 127, 128, 255):
        p = (n + 128) % 256 - 128
        kernel.func(b, n)
        torch.testing.assert_close(b, torch.full_like(b, (p // 3) * 10 + p % 3))


def test_collect_repeated_host_launches():
    @T.prim_func
    def host(n: T.int32):
        T.call_packed("kernel", n)
        if n > 0:
            p = T.bind(n + 1)
            T.call_packed("kernel", p)
        T.call_packed("kernel", n)

    calls = collect_host_launches(host.body, {"kernel": 1})
    assert len(calls) == 3
    assert not calls[0].conditions and len(calls[1].conditions) == 1
    assert not calls[2].conditions
    p = calls[1].arguments[0]
    assert p in calls[1].bindings and p not in calls[2].bindings
    assert calls[0].arguments[0].same_as(calls[2].arguments[0])


def test_reject_looped_host_launches():
    @T.prim_func
    def host(n: T.int32):
        for i in T.serial(n):
            T.call_packed("kernel", i)

    with pytest.raises(ValueError, match="inside host loops"):
        collect_host_launches(host.body, {"kernel": 1})


@tilelang.testing.requires_cuda
def test_nvrtc_wrapper_preserves_tma_descriptor_arguments():
    from tilelang.cuda import language as T
    from tilelang.backend.module import create_backend_context
    from tilelang.engine.lower import lower_with_context

    @T.prim_func
    def copy(A: T.Tensor((128, 128), "float16"), B: T.Tensor((128, 128), "float16")):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((128, 64), "float16")
            T.copy(A[0:128, 0:64], shared, prefer_instruction="tma")
            T.copy(shared, B[0:128, 0:64], prefer_instruction="sync")

    context = create_backend_context({"kind": "cuda", "arch": "sm_90"}, "c", "nvrtc")
    with context.target:
        artifact = lower_with_context(copy, context)
    wrapper = TLNVRTCSourceWrapper(
        tvm.IRModule({"copy": copy}),
        artifact.kernel_source,
        context.target,
        device_mod=artifact.device_mod,
        host_mod=artifact.host_mod,
    )
    assert "cuTensorMapEncodeTiled(" in wrapper.host_func
    tree = ast.parse(wrapper.host_func)
    payloads = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "arg_values" for target in node.targets)
    ]
    assert len(payloads) == 1
    device_func = next(iter(artifact.device_mod.functions.values()))
    assert len(payloads[0].elts) == len(device_func.params)
