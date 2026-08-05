import pytest
import torch
from torch import fx

import tilelang
import tilelang.testing
from tilelang import tvm
from tilelang.graph import backend_config, cache
from tilelang.graph.backend import tilelang_backend
from tilelang.graph.vm_build import VMRunner


def _identity_graph():
    return fx.symbolic_trace(lambda x: x + 1)


def _reset_graph_compiler():
    backend_config.reset()
    cache.clear_cache()
    torch._dynamo.reset()


def test_graph_cache_key_includes_options_and_scalar_values():
    graph = _identity_graph()
    tensor = torch.empty((4, 8))

    default_key = cache.graph_cache_key(graph, [tensor, 1], (False,))
    different_scalar_key = cache.graph_cache_key(graph, [tensor, 2], (False,))
    different_option_key = cache.graph_cache_key(graph, [tensor, 1], (True,))

    assert len({default_key, different_scalar_key, different_option_key}) == 3


@tilelang.testing.requires_cuda
def test_graph_backend_rejects_noncontiguous_inputs_before_compile():
    tensor = torch.randn((8, 16), device="cuda").transpose(0, 1)

    with pytest.raises(ValueError, match="contiguous CUDA tensors"):
        tilelang_backend(_identity_graph(), [tensor])


@tilelang.testing.requires_cuda
def test_graph_runner_rejects_noncontiguous_inputs():
    device = torch.device("cuda", torch.cuda.current_device())
    runner = object.__new__(VMRunner)
    runner._torch_device = device
    tensor = torch.randn((8, 16), device=device).transpose(0, 1)

    with pytest.raises(ValueError, match="contiguous CUDA tensors"):
        runner(tensor)


@tilelang.testing.requires_cuda
def test_torch_compile_graph_backend_preserves_cuda_device():
    _reset_graph_compiler()

    device_index = 1 if torch.cuda.device_count() > 1 else 0
    device = torch.device("cuda", device_index)
    x = torch.randn((32, 64), device=device, dtype=torch.float16)
    y = torch.randn_like(x)

    compiled = torch.compile(lambda lhs, rhs: torch.exp(lhs) + rhs, backend="tilelang")
    actual = compiled(x, y)

    assert actual.device == device
    torch.testing.assert_close(actual, torch.exp(x) + y, atol=2e-3, rtol=2e-3)


@tilelang.testing.requires_cuda
def test_torch_compile_graph_backend_schedules_matmul():
    _reset_graph_compiler()
    torch.manual_seed(0)
    device = torch.device("cuda", torch.cuda.current_device())
    lhs = torch.randn((64, 128), device=device, dtype=torch.float16)
    rhs = torch.randn((128, 96), device=device, dtype=torch.float16)

    compiled = torch.compile(lambda x, y: x @ y, backend="tilelang")
    actual = compiled(lhs, rhs)

    torch.testing.assert_close(actual, lhs @ rhs, atol=1e-2, rtol=1e-2)


@tilelang.testing.requires_cuda
def test_torch_compile_graph_backend_schedules_reduction():
    _reset_graph_compiler()
    torch.manual_seed(0)
    device = torch.device("cuda", torch.cuda.current_device())
    x = torch.randn((64, 256), device=device)
    weight = torch.randn((256,), device=device)

    def rms_norm(inp, scale):
        mean_square = inp.pow(2).mean(dim=-1, keepdim=True)
        return inp * torch.rsqrt(mean_square + 1e-5) * scale

    compiled = torch.compile(rms_norm, backend="tilelang")
    actual = compiled(x, weight)

    torch.testing.assert_close(actual, rms_norm(x, weight), atol=1e-2, rtol=1e-2)


@tilelang.testing.requires_cuda
def test_graph_fallback_registration_is_namespaced_per_graph():
    _reset_graph_compiler()
    backend_config.extern_dispatch = lambda node: getattr(node.target, "__name__", "") == "clamp"
    device = torch.device("cuda", torch.cuda.current_device())
    x = torch.linspace(-1, 1, 32, device=device)

    try:
        clamp_low = torch.compile(lambda value: torch.clamp(value, min=-0.25), backend="tilelang")
        clamp_high = torch.compile(lambda value: torch.clamp(value, min=0.5), backend="tilelang")

        expected_low = torch.clamp(x, min=-0.25)
        torch.testing.assert_close(clamp_low(x), expected_low)
        torch.testing.assert_close(clamp_high(x), torch.clamp(x, min=0.5))
        torch.testing.assert_close(clamp_low(x), expected_low)
    finally:
        backend_config.reset()


if __name__ == "__main__":
    tvm.testing.main()
