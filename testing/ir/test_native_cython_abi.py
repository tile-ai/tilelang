"""Exercise the compiled Cython wrapper against a real C ABI and native IR."""

import ctypes
import subprocess

import pytest
import torch
from tvm_tirx import ir, tirx
from tvm_tirx.tirx.layout import S

from tilelang.engine.param import KernelParam, dump_kernel_params, load_kernel_params
from tilelang.jit.adapter.abi import source_abi_plan
from tilelang.jit.adapter.cython.adapter import CythonKernelAdapter, CythonKernelWrapper


def tensor(name, shape, strides=None):
    layout = "default" if strides is None else tirx.TileLayout(S[shape:strides])
    return ir.Var(name, tirx.TensorType(shape, "float32", layout=layout))


def compile_library(tmp_path, source):
    source_path = tmp_path / "probe.c"
    library_path = tmp_path / "probe.so"
    source_path.write_text("#include <stdint.h>\n" + source + '\nconst char* get_last_error(void) {return "probe failed";}\n')
    subprocess.run(["cc", "-shared", "-fPIC", "-O2", str(source_path), "-o", str(library_path)], check=True)
    library = ctypes.CDLL(str(library_path))
    library.get_last_error.restype = ctypes.c_char_p
    return library


def wrapper(function, result_idx, library, *, metadata_roundtrip=False):
    adapter = CythonKernelAdapter.__new__(CythonKernelAdapter)
    adapter.ir_module = ir.IRModule({"main": function})
    adapter.result_idx = result_idx
    from tvm_tirx.target import Target

    adapter.target = Target({"kind": "c"})
    parameters = [KernelParam.from_var(value) for value in function.params]
    if metadata_roundtrip:
        parameters = load_kernel_params(dump_kernel_params(parameters))
    plan = source_abi_plan(adapter)
    parameters = plan.validate_parameter_metadata(parameters)
    result = CythonKernelWrapper(result_idx, parameters, library, lambda: torch.device("cpu"), lambda device: 0)
    result.set_source_abi_plan(plan)
    result.set_dynamic_symbolic_map(adapter._process_dynamic_symbolic())
    result.set_dynamic_symbolic_sources(adapter._process_dynamic_symbolic_sources())
    result.set_ptr_map(adapter._process_ptr_map())
    result.set_buffer_dtype_map(adapter._process_buffer_dtype())
    result.set_buffer_device_map(adapter._process_buffer_device())
    shapes, strides, contiguous = adapter._process_static_buffer_infos()
    result.set_static_shape_map(shapes)
    result.set_static_strides_map(strides)
    result.set_static_contiguous_list(contiguous)
    return result


@pytest.mark.parametrize("roundtrip", [False, True])
def test_output_before_input_and_late_scalar_execute(tmp_path, roundtrip):
    n = ir.Var("n", "int64")
    function = tirx.Function([tensor("out", [n * 2 + 1]), tensor("in", [n]), n], [])
    library = compile_library(
        tmp_path,
        """
    int call(float* out, const float* input, int64_t n, void* stream) {
        for (int64_t i = 0; i < n*2+1; ++i) out[i] = (float)(i + 9);
        return 0;
    }
    """,
    )
    runtime = wrapper(function, [0], library, metadata_roundtrip=roundtrip)
    for extent in (1, 7, 257):
        result = runtime.forward([torch.empty(extent), extent], stream=0)
        torch.testing.assert_close(result, torch.arange(2 * extent + 1, dtype=torch.float32) + 9, rtol=0, atol=0)
    with pytest.raises(ValueError, match="conflicts|expected"):
        runtime.forward([torch.empty(7), 8], stream=0)


@pytest.mark.parametrize("roundtrip", [False, True])
def test_same_name_shape_identity_and_noncontiguous_stride(tmp_path, roundtrip):
    n, other_n = ir.Var("n", "int32"), ir.Var("n", "int64")
    stride = ir.Var("stride", "int64")
    function = tirx.Function([tensor("out", [n, other_n]), tensor("A", [n]), tensor("B", [other_n, 2], [stride, 1])], [])
    library = compile_library(
        tmp_path,
        """
    int call(float* out, const float* a, const float* b, int32_t n, int64_t m, int64_t stride, void* stream) {
        for (int i = 0; i < n; ++i) for (int64_t j = 0; j < m; ++j) out[i*m+j] = a[i]+b[j*stride];
        return 0;
    }
    """,
    )
    runtime = wrapper(function, [0], library, metadata_roundtrip=roundtrip)
    a = torch.arange(3, dtype=torch.float32)
    b = torch.arange(28, dtype=torch.float32).reshape(7, 4)[:, :2]
    result = runtime.forward([a, b], stream=0)
    torch.testing.assert_close(result, a[:, None] + b[:, 0][None, :], rtol=0, atol=0)


@pytest.mark.parametrize("present", [(True, True), (True, False), (False, True), (False, False)])
def test_nullable_shape_stride_carriers_have_actual_c_abi(tmp_path, present):
    n = ir.Var("n", "int64")
    function = tirx.Function([tensor("A", [2, 3], [n, 1]), tensor("B", [n])], [])
    library = compile_library(
        tmp_path,
        """
    static int64_t observed;
    int call(float* a, float* b, int64_t size, void* stream) { observed = size; return 0; }
    int64_t get_observed(void) {return observed;}
    """,
    )
    library.get_observed.restype = ctypes.c_int64
    runtime = wrapper(function, [], library)
    a = torch.empty_strided((2, 3), (4, 1)) if present[0] else None
    b = torch.empty(4) if present[1] else None
    runtime.forward([a, b], stream=0)
    assert library.get_observed() == (4 if any(present) else 0)


def test_pointer_and_unsigned_scalar_parameters_keep_call_types(tmp_path):
    function = tirx.Function([ir.Var("ptr", ir.PtrType("uint8")), ir.Var("amount", "uint64")], [])
    library = compile_library(
        tmp_path,
        """
    int call(void* pointer, uint64_t amount, void* stream) {
        return (uintptr_t)pointer == (uintptr_t)0x1234 && amount == UINT64_C(0xffffffff12345678) ? 0 : -1;
    }
    """,
    )
    runtime = wrapper(function, [], library)
    assert runtime.forward([0x1234, 0xFFFFFFFF12345678], stream=0) == []
