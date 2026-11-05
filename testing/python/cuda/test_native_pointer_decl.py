"""Native pointer declarations preserve device addresses through TileLang JIT."""

import ctypes

import pytest
import torch
from tvm_tirx import ir, tirx

import tilelang
import tilelang.cuda.language as T
from tilelang.testing.ir import collect_calls


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("backend", ["tvm_ffi", "cython"])
def test_native_pointer_dynamic_tensor_and_scalar_cuda(backend):
    extent = T.dynamic("extent", "int64")

    @T.prim_func
    def kernel(pointer: T.Ptr("int32"), output: T.TensorType((extent,), "int32", align=4), bias: ir.PrimType("int32")):
        source = T.decl_tensor(pointer, ty_args=[T.TensorType(output.shape, "int32", align=4, layout="default")])
        with T.Kernel(T.ceildiv(output.shape[0], 64), threads=64) as block:
            for lane in T.Parallel(64):
                index = block * 64 + lane
                if index < output.shape[0]:
                    output[index] = source[index] + bias

    function = ir.load_json(ir.save_json(kernel))
    assert tirx.analysis.check_well_formed(function)
    assert isinstance(function.params[0].ty, ir.PtrType)
    (declaration,) = collect_calls(function, op="tirx.decl_tensor")
    assert declaration.args[0].same_as(function.params[0])
    assert not collect_calls(function, op="tl.tensor_descriptor_ptr")
    compiled = tilelang.compile(function, target="cuda", target_host="c", execution_backend=backend)
    for count, bias in [(1, -7), (65, 3), (129, -11)]:
        # The passed pointer already includes the input slice origin. No
        # DLTensor offset may be added to this address a second time.
        source = torch.arange(count + 3, device="cuda", dtype=torch.int32)[3:]
        storage = torch.full((count + 5,), -999, device="cuda", dtype=torch.int32)
        output = storage[2 : 2 + count]
        pointer = ctypes.c_void_p(source.data_ptr()) if backend == "tvm_ffi" else source.data_ptr()
        compiled(pointer, output, bias)
        torch.cuda.synchronize()
        torch.testing.assert_close(output, source + bias, rtol=0, atol=0)
        assert torch.all(storage[:2] == -999)
        assert torch.all(storage[2 + count :] == -999)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_native_pointer_decl_tensor_feeds_shared_copy():
    @T.prim_func
    def kernel(pointer: T.Ptr("float32"), output: T.TensorType((64,), "float32")):
        source = T.decl_tensor(pointer, ty_args=[T.Tensor((64,), "float32")])
        with T.Kernel(1, threads=64):
            shared = T.alloc_shared((64,), "float32")
            T.copy(source, shared)
            T.copy(shared, output)

    assert tirx.analysis.check_well_formed(kernel)
    compiled = tilelang.compile(kernel, target="cuda", target_host="c", execution_backend="tvm_ffi")
    source = torch.arange(64, device="cuda", dtype=torch.float32) - 31
    output = torch.full_like(source, -999)
    compiled(ctypes.c_void_p(source.data_ptr()), output)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, source, rtol=0, atol=0)
