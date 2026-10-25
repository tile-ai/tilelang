import ctypes
import shutil

import pytest
from tvm_tirx import runtime
from tvm_tirx.target import Target

from tilelang._compat.tensor import TensorView
from tilelang.contrib import cc, nvcc


@pytest.mark.parametrize("architecture,expected", [("sm_80", "8.0"), ("sm_90a", "9.0"), ("sm_100a", "10.0"), ("sm_120", "12.0")])
def test_nvcc_uses_native_target_scope(architecture, expected):
    target = Target({"kind": "cuda", "arch": architecture})
    assert nvcc.get_target_compute_version(target) == expected
    with target:
        assert nvcc.get_target_compute_version() == expected


def test_native_host_compiler_artifact_executes(tmp_path):
    if not cc.get_cc():
        pytest.skip("A host C++ compiler is required")
    source = tmp_path / "add.cc"
    source.write_text('extern "C" int add_integers(int left, int right) { return left + right; }\n')
    artifact = tmp_path / f"add.{cc.create_shared.output_format}"
    cc.create_shared(str(artifact), [str(source)])
    library = ctypes.CDLL(str(artifact))
    library.add_integers.argtypes = [ctypes.c_int, ctypes.c_int]
    library.add_integers.restype = ctypes.c_int
    assert library.add_integers(17, 25) == 42


def test_nvcc_compiles_with_native_pass_context():
    if not shutil.which("nvcc"):
        pytest.skip("CUDA nvcc is required")
    code = 'extern "C" __global__ void write_value(float* output) { output[threadIdx.x] = 7.0f; }'
    with Target({"kind": "cuda", "arch": "sm_90"}):
        result = nvcc.compile_cuda(code, target_format="ptx")
    assert isinstance(result, bytearray)
    assert ".entry write_value" in result.decode("utf-8")


def test_sparse_metadata_layout_with_native_tensor(native_layouts):
    tensor = TensorView.create("metadata", [64, 8], "uint8", scope="shared")
    result = native_layouts.make_cutlass_metadata_layout(tensor, mma_dtype="float16", arch="9.0", block_k=64)
    assert isinstance(result, native_layouts.Layout)
    assert tuple(int(extent) for extent in result.get_input_shape()) == (64, 8)


def test_nvcc_detects_native_cuda_runtime():
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("A CUDA device is required")
    assert nvcc.get_target_compute_version() == device.compute_version
