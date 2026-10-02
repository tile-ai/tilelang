from types import SimpleNamespace

import pytest
import torch

from tilelang.jit.adapter.cutedsl.adapter import CuTeDSLKernelAdapter
from tilelang.jit.adapter.nvrtc.adapter import NVRTCKernelAdapter


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize("adapter_type", [CuTeDSLKernelAdapter, NVRTCKernelAdapter])
@pytest.mark.parametrize("tensor_device", [0, 1])
def test_implicit_stream_belongs_to_tensor_device(adapter_type, tensor_device):
    # Run the real wrapper with native Torch streams; intercept only the final backend call.
    calls = []
    adapter = object.__new__(adapter_type)
    adapter.params = [None]
    adapter.result_idx = []
    adapter.target = SimpleNamespace(kind=SimpleNamespace(name="cuda"))
    adapter.dynamic_symbolic_order = []
    adapter.dynamic_symbolic_map = {}
    adapter._forward_from_prebuild_lib = lambda *args, **kwargs: calls.append(kwargs)

    with torch.cuda.device(0):
        tensor = torch.empty((1,), device=f"cuda:{tensor_device}")
        current_stream = torch.cuda.Stream(device=0)
        tensor_stream = torch.cuda.Stream(device=tensor_device)
        with torch.cuda.stream(current_stream), torch.cuda.stream(tensor_stream), torch.cuda.device(0):
            adapter._wrap_forward_from_prebuild_lib(tensor)
            assert calls[-1]["stream"] == tensor_stream.cuda_stream
            if adapter_type is CuTeDSLKernelAdapter:
                assert calls[-1]["device_id"] == tensor_device
            adapter._wrap_forward_from_prebuild_lib(tensor, stream=tensor_stream.cuda_stream)
            assert calls[-1]["stream"] == tensor_stream.cuda_stream
