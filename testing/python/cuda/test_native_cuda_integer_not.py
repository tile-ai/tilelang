"""CUDA vector inversion preserves the bits of each integer lane."""

import numpy as np
import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"])
@pytest.mark.parametrize("lanes", [1, 2, 4])
def test_integer_bitwise_not_keeps_lane_width_and_signedness(dtype, lanes, tmp_path):
    @T.prim_func
    def kernel(source: T.Tensor((lanes,), dtype), output: T.Tensor((lanes,), dtype)):
        with T.Kernel(1, threads=32):
            if T.get_thread_binding() == 0:
                if lanes == 1:
                    output[0] = ~source[0]
                else:
                    output[T.Ramp(0, 1, lanes)] = ~source[T.Ramp(0, 1, lanes)]

    compiled = tilelang.compile(kernel, target="cuda", execution_backend="tvm_ffi")
    (tmp_path / "kernel.cu").write_text(compiled.get_kernel_source())
    bits = np.dtype(dtype).itemsize * 8
    # Interpret exact bit patterns as the requested signed/unsigned dtype.
    # Include both the sign bit and all-ones without out-of-range Python casts.
    patterns = [0, (1 << bits) - 1, 1 << (bits - 1), 1]
    for start in (0, 1):
        raw = np.array([patterns[(start + lane) % len(patterns)] for lane in range(lanes)], dtype=f"uint{bits}")
        source = raw.view(dtype)
        expected = np.bitwise_not(source)
        source_device = torch.from_numpy(source.copy()).cuda()
        actual_device = torch.empty_like(source_device)
        compiled(source_device, actual_device)
        torch.cuda.synchronize()
        np.testing.assert_array_equal(actual_device.cpu().numpy(), expected)
