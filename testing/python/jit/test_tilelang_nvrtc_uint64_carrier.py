"""Preserve full-range uint64 words across the native NVRTC launch boundary."""

import copy
import json

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


@pytest.fixture(scope="module")
def kernel():
    @T.prim_func
    def copy_words(B: T.Tensor((3,), "uint64"), first: T.uint64, bias: T.int32, last: T.uint64):
        with T.Kernel(1, threads=1):
            B[0] = first >> T.uint64(63)
            B[1] = first + T.uint64(bias)
            B[2] = last

    return tilelang.compile(copy_words, target="cuda", execution_backend="nvrtc")


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("first,bias,last", [(0, -1, (1 << 64) - 1), ((1 << 63) - 1, 1, 0), (1 << 63, -1, 1 << 63), ((1 << 64) - 1, 1, 17)])
def test_nvrtc_mixed_words_keep_unsigned_device_arithmetic(kernel, first, bias, last, tmp_path):
    output = torch.empty(3, dtype=torch.uint64, device="cuda")
    kernel(output, first, bias, last)
    assert output.cpu().tolist() == [first >> 63, (first + bias) % (1 << 64), last]
    specs = kernel.adapter.pymodule._kernel_specs
    assert all(spec["arg_types"].count("uint64") == 2 for spec in specs.values())
    assert "uint64_t" in kernel.get_kernel_source()
    (tmp_path / "device.cu").write_text(kernel.get_kernel_source())
    (tmp_path / "specs.json").write_text(json.dumps(specs, indent=2))


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("position", [0, 2])
@pytest.mark.parametrize("invalid,error", [(-1, ValueError), (1 << 64, ValueError), (1.5, TypeError)])
def test_nvrtc_unsigned_public_input_errors_precede_launch(kernel, position, invalid, error):
    output = torch.full((3,), 23, dtype=torch.uint64, device="cuda")
    values = [0, 1, 0]
    values[position] = invalid
    with pytest.raises(error):
        kernel(output, *values)
    assert output.cpu().tolist() == [23, 23, 23]


@tilelang.testing.requires_cuda
def test_nvrtc_reloading_unsigned_specs_does_not_change_their_contract(kernel):
    from tilelang.jit.adapter.nvrtc.libgen import NVRTCLibraryGenerator

    specs = copy.deepcopy(kernel.adapter.pymodule._kernel_specs)
    restored = NVRTCLibraryGenerator(kernel.adapter.target)
    restored.load_lib(kernel.adapter.libpath)
    assert restored.pymodule._kernel_specs == specs
    assert kernel.adapter.pymodule._kernel_specs == specs
    output = torch.empty(3, dtype=torch.uint64, device="cuda")
    stream = torch.cuda.current_stream()
    restored.pymodule.call(
        restored.kernels, output, (1 << 64) - 1, 1, 1 << 63, stream=stream.cuda_stream, device_id=torch.cuda.current_device()
    )
    assert output.cpu().tolist() == [1, 0, 1 << 63]
