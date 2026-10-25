import ctypes

import pytest
import torch

import tilelang
import tilelang.cpu.language as Cpu
import tilelang.language as T


@pytest.mark.parametrize(
    "target,execution_backend",
    [("c", "tvm_ffi"), ("llvm", "tvm_ffi"), ("cuda", "tvm_ffi"), ("cuda", "cython"), ("cuda", "nvrtc")],
)
@pytest.mark.parametrize("argument_kind", ["tensor", "integer"])
def test_raw_pointer_arguments_use_tensor_data_and_storage_offset(target, execution_backend, argument_kind):
    if target == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware is required")
    language = T if target == "cuda" else Cpu
    launch = {"threads": 32} if target == "cuda" else {}

    @language.prim_func
    def increment(source_pointer: language.ptr, output: language.Tensor((17,), "float32")):
        source = language.make_tensor(source_pointer, (17,), "float32")
        with language.Kernel(1, **launch):
            for index in language.Parallel(17):
                output[index] = source[index] + 1

    kernel = tilelang.compile(increment, target=target, execution_backend=execution_backend)
    source = torch.arange(40, dtype=torch.float32, device="cuda" if target == "cuda" else "cpu")[5:22]
    output = torch.empty_like(source)
    argument = source if argument_kind == "tensor" else source.data_ptr()
    kernel(argument, output)
    torch.testing.assert_close(output, source + 1)
    if execution_backend == "tvm_ffi":
        output.zero_()
        kernel(ctypes.c_void_p(source.data_ptr()), output)
        torch.testing.assert_close(output, source + 1)
