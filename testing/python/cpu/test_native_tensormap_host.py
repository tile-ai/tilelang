"""The C host consumes native TensorMapType storage without a CUDA SDK header."""

from pathlib import Path

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import IRModule, codegen, ir, runtime, target, tirx


@pytest.mark.parametrize("backend", ["c", "c_host"])
@pytest.mark.parametrize("count", [1, 2, 3])
def test_host_tensormap_alignment_and_payload(backend, count, tmp_path):
    # A real C helper probes the pointer alignment and the last byte of every
    # 128-byte record. This is a host storage contract, independent of CUDA
    # descriptor encoding and GPU execution.
    source = r"""
#include <stdint.h>
extern "C" int64_t inspect_tensormap(void* raw, int32_t count) {
  unsigned char* data = (unsigned char*)raw;
  if (((uintptr_t)raw) % 64 != 0) return -1;
  for (int32_t i = 0; i < count; ++i) data[i * 128 + 127] = (unsigned char)(i + 1);
  int64_t result = 0;
  for (int32_t i = 0; i < count; ++i) result += data[i * 128 + 127];
  return result;
}
"""
    descriptor = tirx.stack_alloca("tensormap", count)
    pointer = ir.Var("descriptor", descriptor.ty)
    output = ir.Var("output", tirx.TensorType((1,), "int64", data_alignment=8))
    body = [
        ir.Bind(pointer, descriptor),
        ir.TensorStore(
            output,
            [0],
            tirx.call_extern(
                "inspect_tensormap",
                ir.Call("tirx.reinterpret", [pointer], ty=ir.PtrType(ir.PrimType("void"))),
                ir.IntImm("int32", count),
                ty="int64",
            ),
        ),
    ]
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")(
        {
            "node": None,
            "key": "pragma_import_c",
            "value": source,
        }
    )
    function = tirx.Function(
        [output], [ir.RegionStmt("tl.attr", [payload], [], {"schema_version": 1, "annotation_schema": schema}, body)]
    ).with_attr("global_symbol", "main")
    cpu = target.Target("c", host="c")
    # Use TileLang's frontend-scope lowering for the imported source while
    # exercising the target's ordinary native ABI and host lowering.
    from tvm_tirx.tirx.compilation_pipeline import tirx_pipeline

    lowering, host, _ = tirx_pipeline(target=cpu)
    module = host()(lowering(IRModule({"main": function})))
    built = tvm_ffi.get_global_func("target.build.tilelang_" + backend)(module, cpu)
    text = built.inspect_source()
    (tmp_path / "host.cc").write_text(text)
    assert "TLTensorMap" in text and "CUtensorMap" not in text
    assert "cuda.h" not in text
    assert "unsigned char data[128]" in text
    library = tmp_path / "host.so"
    root = Path(__file__).resolve().parents[3]
    codegen.Executable(built).export_library(library, cc="c++", options=["-std=c++17", "-fopenmp", f"-I{root / 'src'}"])
    loaded = runtime.load_module(str(library))
    result = runtime.tensor(np.zeros(1, dtype="int64"))
    loaded(result)
    np.testing.assert_array_equal(result.numpy(), np.array([count * (count + 1) // 2], dtype="int64"))


@pytest.mark.parametrize("backend", ["c", "c_host"])
def test_host_tensormap_parameter_is_a_pointer_and_can_be_called(backend, tmp_path):
    import ctypes

    descriptor = ir.Var("descriptor", ir.PtrType(tirx.TensorMapType()))
    byte_pointer = ir.Call("tirx.reinterpret", [descriptor], ty=ir.PtrType("uint8"))
    storage = ir.Var("storage", tirx.TensorType((128,), "uint8", data_alignment=64))
    helper = tirx.Function(
        [descriptor],
        [
            ir.Bind(storage, ir.Call("tirx.decl_tensor", [byte_pointer], ty_args=[storage.ty])),
            ir.TensorStore(storage, [127], ir.IntImm("uint8", 42)),
        ],
    ).with_attr("global_symbol", "host_helper")
    cpu = target.Target("c", host="c")
    built = tvm_ffi.get_global_func("target.build.tilelang_" + backend)(IRModule({"host_helper": helper}), cpu)
    source = built.inspect_source()
    (tmp_path / "helper.cc").write_text(source)
    assert "CUtensorMap" not in source and "__grid_constant__" not in source
    assert "TLTensorMap* descriptor" in source or "TLTensorMap * descriptor" in source
    library = tmp_path / "helper.so"
    root = Path(__file__).resolve().parents[3]
    codegen.Executable(built).export_library(library, cc="c++", options=["-std=c++17", "-fopenmp", f"-I{root / 'src'}"])
    # This is an ordinary native helper, so use its C ABI rather than treating
    # it as a packed FFI entry. The alignment and last byte are observable.
    loaded = ctypes.CDLL(str(library))
    call = loaded.host_helper
    call.argtypes, call.restype = [ctypes.c_void_p], None
    backing = (ctypes.c_uint8 * 192)()
    address = (ctypes.addressof(backing) + 63) // 64 * 64
    call(address)
    assert ctypes.c_uint8.from_address(address + 127).value == 42
