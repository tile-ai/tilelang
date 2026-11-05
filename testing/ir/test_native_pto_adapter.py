"""PTO native host emission and its real packed-storage Cython calling ABI.

Host source tests use the actual PTODSL emitter. Runtime tests call a compiled
C library on CPU to isolate pointer/shape/stride/stream contracts; they do not
claim NPU code compilation or execution.
"""

import ctypes
import subprocess

import pytest
import torch
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target
from tvm_tirx.tirx.layout import S

from tilelang.engine.param import dump_kernel_params, load_kernel_params
from tilelang.jit.adapter.abi import SourceABIPlan
from tilelang.jit.adapter.pto.adapter import _accepted_storage_dtypes, _validate_packed_layout
from tilelang.jit.adapter.pto.wrapper import TLPTOSourceWrapper
from tilelang_pto_wrapper import PTOKernelWrapper
from test_native_source_launch import native_launch_case


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "float8_e4m3fn", "float4_e2m1fn"])
@pytest.mark.parametrize("roundtrip", [False, True])
def test_pto_device_signature_and_native_host_control_flow(dtype, roundtrip, tmp_path):
    function, host, device = native_launch_case(dtype=dtype)
    target = Target({"kind": "ascend", "keys": ["pto"]})
    source = tvm_ffi.get_global_func("target.build.tilelang_pto_without_compile")(device, target).inspect_source("py")
    if roundtrip:
        function, host, device = (ir.load_json(ir.save_json(value)) for value in (function, host, device))
    wrapper = TLPTOSourceWrapper(ir.IRModule({"main": function}), source, target, device_mod=device, host_mod=host)
    assert wrapper.pto_kernel_names == ["compute_kernel"]
    assert 'extern "C" __global__ AICORE void compute_kernel(__gm__ ' in wrapper.lib_code
    assert wrapper.lib_code.count("compute_kernel<<<") == 1
    assert "if (" in wrapper.lib_code and "for (" in wrapper.lib_code and "!= 1" in wrapper.lib_code
    assert "TVM_FFI_SAFE_CALL_BEGIN" in wrapper.lib_code and "aclrtGetLastError" in wrapper.lib_code
    if dtype == "float8_e4m3fn":
        assert "float8_e4m3_t" in wrapper.lib_code and "fp8_e4_t" not in wrapper.lib_code
    if dtype == "float4_e2m1fn":
        assert "float4_e2m1x2_t" in wrapper.lib_code
    (tmp_path / "kernel.ptodsl.py").write_text(source)
    (tmp_path / "launch.cpp").write_text(wrapper.lib_code)


@pytest.mark.parametrize(
    "source,error",
    [
        ("def other(a, b, i): pass", "Cannot find PTODSL"),
        ("def compute_kernel(a): pass", "signature disagrees"),
        ("def compute_kernel(a, *, b, i): pass", "positional arguments"),
        ("def compute_kernel(a, b, i): pass\ndef compute_kernel(a, b, i): pass", "Duplicate PTODSL"),
    ],
)
def test_pto_device_source_mismatch_is_rejected(source, error):
    function, host, device = native_launch_case()
    with pytest.raises(RuntimeError, match=error):
        TLPTOSourceWrapper(
            ir.IRModule({"main": function}), source, Target({"kind": "ascend", "keys": ["pto"]}), device_mod=device, host_mod=host
        )


@pytest.fixture(scope="module")
def packed_library(tmp_path_factory):
    directory = tmp_path_factory.mktemp("native-pto-c-abi")
    source = directory / "oracle.c"
    source.write_text(r"""
#include <stdint.h>
static uintptr_t observed_stream;
static int64_t calls;
int call(const uint8_t* input, uint8_t* output, int64_t n, int64_t stride, void* stream) {
  observed_stream = (uintptr_t)stream;
  ++calls;
  if (!input || !output) return 0;
  if (n < 0 || n % 2 || stride % 2) return -1;
  for (int i = 0; i < 3; ++i)
    for (int64_t j = 0; j < n / 2; ++j)
      output[i * (n / 2) + j] = input[i * (stride / 2) + j];
  return 0;
}
const char* get_last_error(void) { return "packed storage is not aligned"; }
uintptr_t get_stream(void) { return observed_stream; }
int64_t get_calls(void) { return calls; }
""")
    library = directory / "oracle.so"
    subprocess.run(["cc", "-shared", "-fPIC", "-O2", str(source), "-o", str(library)], check=True)
    result = ctypes.CDLL(str(library))
    result.get_last_error.restype = ctypes.c_char_p
    result.get_stream.restype = ctypes.c_size_t
    result.get_calls.restype = ctypes.c_int64
    return result


def packed_function():
    n, stride = ir.Var("dimension", "int64"), ir.Var("dimension", "int64")
    lhs = ir.Var("same", tirx.TensorType([3, n], "float4_e2m1fn", layout=tirx.TileLayout(S[[3, n] : [stride, 1]]), data_alignment=1))
    rhs = ir.Var("same", tirx.TensorType([3, n], "float4_e2m1fn", layout="default", data_alignment=1))
    return tirx.Function([lhs, rhs], [], ir.TupleType([]))


@pytest.mark.parametrize("roundtrip", [False, True])
@pytest.mark.parametrize("size", [0, 1, 5, 17])
def test_pto_packed_shape_strides_offsets_and_stream_use_native_identity(packed_library, roundtrip, size):
    function = packed_function()
    if roundtrip:
        function = ir.load_json(ir.save_json(function))
    plan = SourceABIPlan(function)
    _validate_packed_layout(plan)
    parameters = load_kernel_params(dump_kernel_params(list(plan.parameters)))
    wrapper = PTOKernelWrapper([1], parameters, packed_library, lambda: torch.device("cpu"), lambda _: 47)
    wrapper.set_source_abi_plan(plan)
    wrapper.set_buffer_dtype_map(
        {tensor.variable: (index, _accepted_storage_dtypes(tensor.dtype)) for index, tensor in plan.tensors.items()}
    )
    stride = size + 3
    storage = torch.arange(3 * stride + 4, dtype=torch.int8)
    values = storage[3:].as_strided((3, size), (stride, 1))
    result = wrapper.forward([values], stream=83)
    assert result.shape == values.shape
    torch.testing.assert_close(result.view(torch.uint8), values.view(torch.uint8), rtol=0, atol=0)
    assert packed_library.get_stream() == 83
    assert len(plan.implicit_symbols) == 2
    assert plan.implicit_symbols[0].name == plan.implicit_symbols[1].name
    assert not plan.implicit_symbols[0].same_as(plan.implicit_symbols[1])


def test_pto_dtype_negative_never_launches(packed_library):
    plan = SourceABIPlan(packed_function())
    wrapper = PTOKernelWrapper([1], list(plan.parameters), packed_library, lambda: torch.device("cpu"), lambda _: 0)
    wrapper.set_source_abi_plan(plan)
    wrapper.set_buffer_dtype_map({plan.function.params[0]: (0, (torch.int8,))})
    count = packed_library.get_calls()
    with pytest.raises(ValueError, match="dtype mismatch"):
        wrapper.forward([torch.empty((3, 5), dtype=torch.float32)], stream=0)
    assert packed_library.get_calls() == count


@pytest.mark.parametrize(
    "shape,strides,error",
    [
        ([3, 7], [8, 1], "innermost dimension"),
        ([3, 8], [9, 1], "stride of packed"),
        ([3, 8], [16, 2], "innermost stride"),
    ],
)
def test_pto_unrepresentable_packed_layout_is_rejected(shape, strides, error):
    tensor = ir.Var("A", tirx.TensorType(shape, "float4_e2m1fn", layout=tirx.TileLayout(S[shape:strides])))
    with pytest.raises(ValueError, match=error):
        _validate_packed_layout(SourceABIPlan(tirx.Function([tensor], [], ir.TupleType([]))))
