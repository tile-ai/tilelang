"""Exercise real Dynamo capture, Ascend kernels, and Torch-owned output lifetime."""

import pytest
import torch

import tilelang.testing
from tilelang.graph import GraphConfig, make_backend
from tilelang.graph.frontend import fx_to_relax
from tilelang.graph.fusion import kernel_count, make_fusion_candidate, validate_runner
from tilelang.graph.runtime import build_runner


_BACKENDS = [pytest.param("ascend", "tvm_ffi", id="ascendc"), pytest.param("pto", "pto", id="pto", marks=pytest.mark.pto)]


def _config(target, execution_backend, **options):
    return GraphConfig(target=target, execution_backend=execution_backend, strict=True, **options)


def _capture_kernels(monkeypatch):
    import tilelang.graph.runtime as runtime

    compile_kernel = runtime._compile_kernel
    kernels = []

    def capture(func, config):
        kernel = compile_kernel(func, config)
        kernels.append(kernel)
        return kernel

    monkeypatch.setattr(runtime, "_compile_kernel", capture)
    return kernels


def _assert_kernel_backends(kernels, execution_backend):
    kernels = tuple(kernels)
    assert kernels
    if execution_backend == "pto":
        from tilelang.jit.adapter.pto import PTOKernelAdapter

        expected_adapter = PTOKernelAdapter
    else:
        from tilelang.jit.adapter.tvm_ffi import TVMFFIKernelAdapter

        expected_adapter = TVMFFIKernelAdapter
    for kernel in kernels:
        assert kernel.execution_backend == execution_backend
        assert isinstance(kernel.adapter, expected_adapter)


def _model(x, bias):
    return torch.relu(x + bias) * bias


def _capture_backend(config):
    reports = []
    backend = make_backend(config)

    def capture(gm, inputs):
        runner = backend(gm, inputs)
        reports.append(runner.graph_report)
        return runner

    return capture, reports


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("target,execution_backend", _BACKENDS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_torch_compile_uses_ascend_kernels(dtype, target, execution_backend, monkeypatch):
    torch._dynamo.reset()
    kernels = _capture_kernels(monkeypatch)
    backend, reports = _capture_backend(_config(target, execution_backend))
    compiled = torch.compile(_model, backend=backend, fullgraph=True)
    x = torch.randn(7, 65, device="npu", dtype=dtype)
    bias = torch.randn(65, device="npu", dtype=dtype)

    actual = compiled(x, bias)
    torch.testing.assert_close(actual, _model(x, bias))
    assert reports and reports[-1].partitions
    assert all(partition.compiled and partition.compiled_calls == 1 for partition in reports[-1].partitions)
    assert sum(partition.kernel_count for partition in reports[-1].partitions) == 3
    assert all(partition.fallback_calls == 0 for partition in reports[-1].partitions)
    assert all(partition.execution_backend == execution_backend for partition in reports[-1].partitions)
    _assert_kernel_backends(kernels, execution_backend)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("target,execution_backend", _BACKENDS)
def test_graph_preserves_current_stream_and_previous_outputs(target, execution_backend, monkeypatch):
    torch._dynamo.reset()
    kernels = _capture_kernels(monkeypatch)
    backend, reports = _capture_backend(_config(target, execution_backend))
    compiled = torch.compile(_model, backend=backend, fullgraph=True)
    x = torch.ones(65536, device="npu")
    bias = torch.ones_like(x)
    previous = compiled(x, bias)
    saved = previous.clone()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        x.fill_(3)
        bias.fill_(2)
        current = compiled(x, bias)
    torch.npu.current_stream().wait_stream(stream)
    torch.testing.assert_close(previous, saved)
    torch.testing.assert_close(current, torch.full_like(x, 10))
    assert all(partition.compiled_calls == 2 and not partition.fallback_calls for partition in reports[-1].partitions)
    _assert_kernel_backends(kernels, execution_backend)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("target,execution_backend", _BACKENDS)
def test_unsupported_pure_operator_stays_in_torch_between_partitions(target, execution_backend, monkeypatch):
    def model(x, bias):
        return torch.relu(x + bias) + torch.sin(bias)

    torch._dynamo.reset()
    kernels = _capture_kernels(monkeypatch)
    backend, reports = _capture_backend(_config(target, execution_backend))
    compiled = torch.compile(model, backend=backend, fullgraph=True)
    x, bias = (torch.randn(65, device="npu") for _ in range(2))
    torch.testing.assert_close(compiled(x, bias), model(x, bias))
    assert reports[-1].partitions
    assert all(partition.compiled for partition in reports[-1].partitions)
    _assert_kernel_backends(kernels, execution_backend)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("target,execution_backend", _BACKENDS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_fused_vector_chain_keeps_no_gm_intermediate(dtype, target, execution_backend):
    x = torch.randn(7, 65, device="npu", dtype=dtype)
    bias = torch.randn(65, device="npu", dtype=dtype)
    mod = fx_to_relax(torch.fx.symbolic_trace(_model), [x, bias])
    fused = make_fusion_candidate(mod)
    runner = build_runner(fused, _config(target, execution_backend))

    validate_runner(runner, _model, [x, bias])
    assert kernel_count(mod) == 3 and runner.kernel_count == 1
    _assert_kernel_backends(runner.kernels.values(), execution_backend)
    for kernel in runner.kernels.values():
        func = kernel.prim_func
        buffers = list(func.buffer_map.values())
        assert len(buffers) == 3  # Two graph inputs and the final output.


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("target,execution_backend", _BACKENDS)
@pytest.mark.parametrize(
    "case,dtype",
    [
        ("sum", torch.float16),
        ("mean", torch.bfloat16),
        ("amax", torch.float32),
        ("softmax", torch.float16),
        ("softmax", torch.bfloat16),
        ("softmax", torch.float32),
        ("rmsnorm", torch.float16),
        ("rmsnorm", torch.bfloat16),
        ("gemm", torch.float16),
        ("gemm", torch.bfloat16),
    ],
)
def test_torch_compile_operator_graphs(case, dtype, target, execution_backend, monkeypatch):
    def model(x, weight):
        if case == "sum":
            return x.sum(-1)
        if case == "mean":
            return x.mean(-1)
        if case == "amax":
            return x.amax(-1)
        if case == "softmax":
            return x.softmax(-1)
        if case == "rmsnorm":
            value = x.float()
            value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-5)
            return value.to(dtype) * weight
        return x @ weight

    shape = (32, 64) if case == "gemm" else (7, 128)
    weight_shape = (64, 64) if case == "gemm" else (128,)
    inputs = [torch.randn(shape, device="npu", dtype=dtype), torch.randn(weight_shape, device="npu", dtype=dtype)]
    torch._dynamo.reset()
    kernels = _capture_kernels(monkeypatch)
    backend, reports = _capture_backend(_config(target, execution_backend))
    compiled = torch.compile(model, backend=backend, fullgraph=True)
    actual = compiled(*inputs)
    torch.testing.assert_close(actual, model(*inputs), rtol=2e-2, atol=2e-2)
    assert reports[-1].partitions
    assert all(partition.compiled and not partition.fallback_calls for partition in reports[-1].partitions)
    _assert_kernel_backends(kernels, execution_backend)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("target,execution_backend", _BACKENDS)
def test_fused_cube_vector_graph_has_one_kernel(target, execution_backend):
    def model(x, weight, bias):
        return torch.relu(x @ weight + bias)

    dtype = torch.bfloat16
    inputs = [
        torch.randn(32, 64, device="npu", dtype=dtype),
        torch.randn(64, 64, device="npu", dtype=dtype),
        torch.randn(64, device="npu", dtype=dtype),
    ]
    mod = fx_to_relax(torch.fx.symbolic_trace(model), inputs)
    fused = make_fusion_candidate(mod)
    runner = build_runner(fused, _config(target, execution_backend))
    validate_runner(runner, model, inputs)
    assert runner.kernel_count == 1
    _assert_kernel_backends(runner.kernels.values(), execution_backend)
    (kernel,) = runner.kernels.values()
    assert ("pto.mte_l0c_ub(" if target == "pto" else "asc_copy_l0c2ub") in kernel.get_kernel_source()


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("target,execution_backend", _BACKENDS)
def test_profiled_torch_compile_records_fusion_decision(target, execution_backend, monkeypatch):
    torch._dynamo.reset()
    kernels = _capture_kernels(monkeypatch)
    backend, reports = _capture_backend(_config(target, execution_backend, fusion="profile"))
    compiled = torch.compile(_model, backend=backend, fullgraph=True)
    x = torch.randn(65536, device="npu")
    bias = torch.randn_like(x)
    torch.testing.assert_close(compiled(x, bias), _model(x, bias))
    (partition,) = reports[-1].partitions
    assert partition.compiled and not partition.fallback_calls
    assert partition.fusion_decision is not None
    assert partition.kernel_count == (1 if partition.fusion_decision.accepted else 3)
    _assert_kernel_backends(kernels, execution_backend)


@tilelang.testing.requires_ascend
@pytest.mark.pto
def test_public_torch_compile_options_select_pto_adapter(monkeypatch):
    torch._dynamo.reset()
    kernels = _capture_kernels(monkeypatch)
    compiled = torch.compile(
        _model,
        backend="tilelang_ascend",
        fullgraph=True,
        options={"target": "pto", "execution_backend": "pto", "strict": True},
    )
    x = torch.randn(7, 65, device="npu")
    bias = torch.randn(65, device="npu")

    actual = compiled(x, bias)
    torch.testing.assert_close(actual, _model(x, bias))
    assert len(kernels) == 3
    _assert_kernel_backends(kernels, "pto")
