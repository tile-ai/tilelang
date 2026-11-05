"""A source host keeps raw tensor contracts without weakening native packing."""

import pytest
import torch
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.codegen import Executable

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.backend.module import create_backend_context
from tilelang.engine.lower import lower_to_host_device_ir, lower_with_context
from tilelang.jit.adapter.source_host import source_host_functions


def _context(backend):
    if backend == "nvrtc":
        pytest.importorskip("cuda.bindings.nvrtc")
    return create_backend_context({"kind": "cuda", "arch": "sm_100a"}, "c", backend)


def _copy(offset=0):
    @T.prim_func
    def main(A: T.Buffer((8,), "int4", elem_offset=offset), B: T.Tensor((8,), "int32")):
        with T.Kernel(1, threads=32):
            for index in T.Parallel(8):
                B[index] = T.cast(A[index], "int32")

    return main


@pytest.mark.parametrize("backend", ["nvrtc", "tvm_ffi"])
@pytest.mark.parametrize("host_codegen", [False, True])
def test_resolved_execution_overrides_incoming_source_host_flag(backend, host_codegen):
    function = _copy()
    module = ir.IRModule({"main": function}).with_attr("tl.source_host_only", True)
    original = ir.save_json(module)
    host, _, _, _, _ = lower_to_host_device_ir(module, _context(backend), enable_host_codegen=host_codegen)
    source_only = backend == "nvrtc" and not host_codegen
    assert bool(host.attrs["tl.source_host_only"]) == source_only
    public = host["main"]
    assert any(isinstance(parameter.ty, tirx.TensorType) for parameter in public.params) == source_only
    assert int(host.attrs["tl.source_host_schema"]) == 2
    assert len(source_host_functions(host)) == 1
    assert ir.save_json(module) == original


def test_direct_lowering_defaults_to_a_native_packable_host():
    host, _, _, _, _ = lower_to_host_device_ir(_copy(), _context("nvrtc"))
    assert not host.attrs["tl.source_host_only"]
    assert all(not isinstance(parameter.ty, tirx.TensorType) for parameter in host["main"].params)


@pytest.mark.parametrize("backend,host_codegen", [("nvrtc", True), ("tvm_ffi", False), ("tvm_ffi", True)])
def test_native_packing_still_rejects_unrepresentable_nibble_offset(backend, host_codegen):
    module = ir.IRModule({"main": _copy(1)}).with_attr("tl.source_host_only", True)
    with pytest.raises(RuntimeError, match="unmet assertion.*byte_offset"):
        lower_with_context(
            module,
            _context(backend),
            enable_host_codegen=host_codegen,
            enable_device_compile=False,
        )


@tilelang.testing.requires_cuda
def test_nvrtc_context_can_explicitly_request_real_native_host_codegen(tmp_path):
    artifact = lower_with_context(_copy(), _context("nvrtc"), enable_host_codegen=True, enable_device_compile=True)
    assert not artifact.host_mod.attrs["tl.source_host_only"]
    assert artifact.rt_mod is not None
    assert all(not isinstance(parameter.ty, tirx.TensorType) for parameter in artifact.host_mod["main"].params)
    # A C source module becomes callable only after compilation and reload.
    library = tmp_path / "native-host.so"
    Executable(artifact.rt_mod).export_library(str(library))
    runtime = tvm_ffi.load_module(str(library))
    source = torch.tensor([0xF1, 0x27, 0x80, 0xD3], dtype=torch.uint8, device="cuda").view(torch.int8)
    output = torch.empty(8, dtype=torch.int32, device="cuda")
    runtime["main"](source, output)
    expected = torch.tensor([1, -1, 7, 2, 0, -8, 3, -3], dtype=torch.int32, device="cuda")
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_source_only_keeps_user_host_assertions_and_real_body_execution(tmp_path):
    @T.prim_func
    def main(A: T.Tensor((8,), "float32"), B: T.Tensor((8,), "float32"), bound: T.int32):
        assert bound == 8, "NVRTC host guard preserved"
        with T.Kernel(1, threads=32):
            for index in T.Parallel(8):
                B[index] = A[index] + T.float32(1)

    # The shared builder's default dynamic assertion kind is RuntimeError;
    # a Python compile-time assertion is a different operation.
    assert isinstance(main.body.seq[0], ir.AssertStmt)
    assert main.body.seq[0].error_kind.value == "RuntimeError"
    kernel = tilelang.compile(main, out_idx=1, target="cuda", execution_backend="nvrtc")
    source = torch.arange(8, dtype=torch.float32, device="cuda")
    text = kernel.adapter.get_host_source()
    (tmp_path / "host.py").write_text(text)
    assert "NVRTC host guard preserved" in text
    torch.testing.assert_close(kernel(source, 8), source + 1, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="NVRTC host guard preserved"):
        kernel(source, -1)
