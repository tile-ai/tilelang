import pytest
import tvm_ffi

from tvm_tirx import ir, tirx


def _module():
    output = ir.Var("output", ir.PtrType(ir.PrimType("float32")))
    function = tirx.Function([output], []).with_attr("global_symbol", "source_kernel")
    return ir.IRModule({"source_kernel": function})


def test_hip_source_module_without_device_sdk():
    source = 'extern "C" __global__ void source_kernel(float* output) { output[0] = 1.0f; }'
    module = tvm_ffi.get_global_func("tl.codegen.rocm_from_source")(_module(), source, None)
    assert module.inspect_source("hip") == source
    assert module.inspect_source() == source
    with pytest.raises(RuntimeError, match="ROCm runtime"):
        module["source_kernel"]


def test_hip_source_module_rejects_empty_compilation_result():
    with pytest.raises(ValueError, match="empty binary"):
        tvm_ffi.get_global_func("tl.codegen.rocm_from_source")(_module(), "", b"")
