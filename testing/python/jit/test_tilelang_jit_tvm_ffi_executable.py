import threading
from types import SimpleNamespace

import pytest
import torch

from tilelang import tvm
from tilelang.jit.abi import prepare_tvm_ffi_callee_allocated_outputs
from tilelang.jit.adapter.tvm_ffi import TVMFFIKernelAdapter

tirx = tvm.tirx


class _FakeKernelParam:
    shape = [1]
    dtype = SimpleNamespace(bits=32, lanes=1)

    @staticmethod
    def torch_dtype():
        return torch.float32


class _TestAdapter(TVMFFIKernelAdapter):
    @property
    def prim_func(self):
        return self._test_prim_func


def _make_adapter():
    adapter = _TestAdapter.__new__(_TestAdapter)
    tir_param = object()
    adapter.params = [_FakeKernelParam()]
    adapter.result_idx = []
    adapter._test_prim_func = SimpleNamespace(
        params=[tir_param],
        buffer_map={tir_param: SimpleNamespace(dtype="float32")},
    )
    adapter._process_dynamic_symbolic = lambda: {}
    adapter.dynamic_symbolic_map = {}
    adapter._ffi_callee_allocated_output_abi = False
    adapter.executable = None
    adapter._executable_lock = threading.Lock()

    created = []

    def make_executable():
        def executable(*args):
            return None

        created.append(executable)
        return executable

    adapter._make_executable = make_executable
    return adapter, created


def test_cold_compiled_dispatch_does_not_probe_cuda(monkeypatch):
    adapter, created = _make_adapter()
    func = adapter._convert_torch_func()
    tensor = torch.empty(1)
    cuda_probe_count = 0

    def counted_is_available():
        nonlocal cuda_probe_count
        cuda_probe_count += 1
        return False

    monkeypatch.setattr(torch.cuda, "is_available", counted_is_available)

    for _ in range(3):
        func(tensor)

    assert cuda_probe_count == 0
    assert len(created) == 1
    assert adapter.executable is created[0]


def test_executable_is_initialized_once_and_reused():
    adapter, created = _make_adapter()

    executable = adapter._get_executable()

    assert adapter._get_executable() is executable
    assert adapter.get_exportable_executable() is executable
    assert adapter.executable is executable
    assert len(created) == 1


def test_preloaded_executable_is_reused():
    adapter, created = _make_adapter()

    def preloaded_executable(*args):
        return None

    adapter.executable = preloaded_executable

    assert adapter._get_executable() is preloaded_executable
    assert adapter.get_exportable_executable() is preloaded_executable
    assert created == []


def test_callee_allocated_output_dispatch_uses_single_main_entry():
    adapter, _ = _make_adapter()
    adapter.params = [_FakeKernelParam(), _FakeKernelParam()]
    adapter.result_idx = [1]
    adapter._ffi_callee_allocated_output_abi = True
    calls = []
    expected = object()

    def main(*args):
        calls.append(args)
        return expected

    adapter.executable = main
    tensor = torch.empty(1)

    assert adapter._convert_torch_func()(tensor) is expected
    assert calls == [(tensor, tensor)]


def test_subbyte_output_uses_callee_allocated_abi():
    adapter, _ = _make_adapter()
    adapter.params = [SimpleNamespace(dtype=SimpleNamespace(bits=4))]
    adapter.result_idx = [0]
    adapter.target = tvm.target.Target("cuda", host="c")

    assert adapter._uses_ffi_callee_allocated_output_abi()


def test_manual_out_idx_is_exposed_to_tvm_ffi_lowering_without_mutating_source():
    input_param = tirx.Var("input", "handle")
    output_param = tirx.Var("output", "handle")
    source = tirx.PrimFunc([input_param, output_param], tirx.Evaluate(0))

    prepared, output_indices = prepare_tvm_ffi_callee_allocated_outputs(source, -1)

    assert source.attrs is None or "tilelang_out_idx" not in source.attrs
    assert list(prepared.attrs["tilelang_out_idx"]) == [-1]
    assert output_indices == [-1]

    attributed = source.with_attr("tilelang_out_idx", [-1])
    prepared, output_indices = prepare_tvm_ffi_callee_allocated_outputs(attributed, [1])
    assert prepared.same_as(attributed)
    assert output_indices == [-1]

    with pytest.raises(ValueError, match="does not match"):
        prepare_tvm_ffi_callee_allocated_outputs(attributed, [0])
