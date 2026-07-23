import torch
import pytest

import tilelang
import tilelang.language as T
from tilelang.engine.lower import extrac_params
from tilelang.jit.adapter.nvrtc import is_nvrtc_available

if not is_nvrtc_available:
    pytest.skip("cuda-python is required to import the NVRTC adapter", allow_module_level=True)

from tilelang.jit.adapter.nvrtc.adapter import NVRTCKernelAdapter


def _make_host_only_adapter(program):
    adapter = NVRTCKernelAdapter.__new__(NVRTCKernelAdapter)
    adapter.ir_module = tilelang.tvm.IRModule({program.attrs["global_symbol"]: program})
    adapter.params = extrac_params(program)
    adapter.result_idx = []
    adapter.param_dtypes = [param.torch_dtype() for param in adapter.params]
    adapter.param_shapes = [list(param.shape) for param in adapter.params]
    adapter.dynamic_symbolic_map = adapter._process_dynamic_symbolic()
    adapter.target = "cuda"
    return adapter


def test_nvrtc_adapter_forwards_scalar_primfunc_parameters():
    @T.prim_func
    def main(A: T.Tensor((8,), T.float32), offset: T.int32):
        T.evaluate(0)

    adapter = _make_host_only_adapter(main)
    forwarded = []
    adapter._forward_from_prebuild_lib = lambda *args, stream: forwarded.append((args, stream))

    tensor = torch.empty(8)
    adapter._wrap_forward_from_prebuild_lib(tensor, 3)

    assert len(forwarded) == 1
    args, stream = forwarded[0]
    assert args[0] is tensor
    assert args[1:] == (3,)
    assert stream == 0


def test_nvrtc_adapter_forwards_dynamic_strides_after_dynamic_shapes():
    length = T.dynamic("length")
    stride = T.dynamic("stride")

    @T.prim_func
    def main(A: T.StridedTensor[(length,), (stride,), T.float32]):
        T.evaluate(0)

    adapter = _make_host_only_adapter(main)
    forwarded = []
    adapter._forward_from_prebuild_lib = lambda *args, stream: forwarded.append((args, stream))

    tensor = torch.empty_strided((7,), (3,))
    adapter._wrap_forward_from_prebuild_lib(tensor)

    assert len(forwarded) == 1
    args, stream = forwarded[0]
    assert args[0] is tensor
    assert args[1:] == (7, 3)
    assert stream == 0
