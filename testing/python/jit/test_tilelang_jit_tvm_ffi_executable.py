import threading
from types import SimpleNamespace

import pytest
import torch

from tilelang.jit.adapter.tvm_ffi import TVMFFIKernelAdapter


class _FakeKernelParam:
    shape = [1]
    dtype = SimpleNamespace(bits=32, lanes=1)

    @staticmethod
    def torch_dtype():
        return torch.float32


class _FakeInt4KernelParam:
    dtype = SimpleNamespace(bits=4, lanes=1)

    def __init__(self, logical_elements):
        self.shape = [logical_elements]

    @staticmethod
    def torch_dtype():
        return torch.int8


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


@pytest.mark.parametrize(
    ("logical_elements", "expected_storage_elements"),
    [
        (2, 1),
        (3, 2),
        (4, 2),
    ],
)
def test_subbyte_output_storage_rounds_up_at_byte_boundary(logical_elements, expected_storage_elements):
    adapter, _ = _make_adapter()
    adapter.params = [_FakeInt4KernelParam(logical_elements)]
    adapter.result_idx = [0]
    adapter.executable = lambda output: None

    output = adapter._convert_torch_func()()

    assert output.dtype == torch.int8
    assert output.shape == (expected_storage_elements,)
