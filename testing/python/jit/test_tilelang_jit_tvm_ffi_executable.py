import threading
from types import SimpleNamespace

import pytest
import torch

from tilelang import tvm
from tilelang.engine.param import KernelParam
from tilelang.jit.adapter.tvm_ffi import (
    TVMFFIKernelAdapter,
    _export_rocm_narrow_float_as_tvm_view,
    _rocm_narrow_float_param_mask,
)


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
    ("torch_dtype_name", "tilelang_dtype", "physical_shape", "logical_shape", "storage_dtype"),
    [
        ("float8_e4m3fnuz", "float8_e4m3", (2, 4), (2, 4), "float8_e4m3fnuz"),
        ("float4_e2m1fn_x2", "float4_e2m1fn", (2, 2), (2, 4), "float4_e2m1fnx2"),
    ],
)
def test_rocm_narrow_float_uses_zero_copy_tvm_view(
    monkeypatch,
    torch_dtype_name,
    tilelang_dtype,
    physical_shape,
    logical_shape,
    storage_dtype,
):
    torch_dtype = getattr(torch, torch_dtype_name, None)
    if torch_dtype is None:
        pytest.skip(f"PyTorch does not provide {torch_dtype_name}")

    monkeypatch.setattr(torch.version, "hip", "test")
    param = KernelParam(tvm.DataType(tilelang_dtype), list(logical_shape))
    tensor = torch.empty(physical_shape, dtype=torch_dtype)

    assert _rocm_narrow_float_param_mask([param]) == (True,)
    tvm_view = _export_rocm_narrow_float_as_tvm_view(tensor, param)

    assert tuple(tvm_view.shape) == physical_shape
    assert str(tvm_view.dtype) == storage_dtype
    byte_view = tvm_view._create_view(physical_shape, dtype="int8")
    round_trip = torch.utils.dlpack.from_dlpack(byte_view)
    assert round_trip.data_ptr() == tensor.data_ptr()


def test_rocm_narrow_float_rejects_non_contiguous_tensor(monkeypatch):
    torch_dtype = getattr(torch, "float8_e4m3fnuz", None)
    if torch_dtype is None:
        pytest.skip("PyTorch does not provide float8_e4m3fnuz")

    monkeypatch.setattr(torch.version, "hip", "test")
    tensor = torch.empty((2, 4), dtype=torch_dtype).transpose(0, 1)
    param = KernelParam(tvm.DataType("float8_e4m3"), list(tensor.shape))

    assert not tensor.is_contiguous()
    with pytest.raises(ValueError, match="requires a contiguous tensor"):
        _export_rocm_narrow_float_as_tvm_view(tensor, param)


def test_non_rocm_does_not_prepare_narrow_float_runtime_args(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    param = KernelParam(tvm.DataType("float8_e4m3"), [2, 4])

    assert _rocm_narrow_float_param_mask([param]) is None
