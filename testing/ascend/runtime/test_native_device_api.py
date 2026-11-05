"""Native Ascend shim ABI tests; no NPU or CANN execution is simulated."""

from concurrent.futures import ThreadPoolExecutor
import threading

import pytest
import tvm_ffi
from tvm_tirx import runtime


def _probe(name):
    return tvm_ffi.get_global_func("tl.testing.AscendDeviceAPI." + name)


def _device(index=0):
    return runtime.Device(12, index)


def _set_stream(index, handle):
    tvm_ffi.get_global_func("runtime.DeviceSetStream")(_device(index), handle)


@pytest.fixture
def preserve_streams():
    before = {index: _probe("GetStream")(index) for index in (0, 7)}
    yield before
    for index, handle in before.items():
        _set_stream(index, handle)


def test_factory_returns_one_native_instance():
    factory = tvm_ffi.get_global_func("device_api.ext_dev")
    assert factory() is not None
    assert _probe("Address")(0) != 0
    assert _probe("Address")(0) == _probe("Address")(7)


@pytest.mark.parametrize("index", [0, 7])
def test_registered_shim_uses_named_attribute_contract(index):
    # This retains the old shim's capability marker, not a hardware probe.
    query = tvm_ffi.get_global_func("runtime.GetDeviceAttr")
    assert query(12, index, "exist") is True
    assert query(12, index, "unsupported_optional_attribute") is None
    assert _device(index).exist is True


@pytest.mark.parametrize("shape", [(4,), (0,), ()])
def test_owning_tensor_allocation_rejects_borrowed_storage(shape):
    with pytest.raises(RuntimeError, match=r"does not allocate.*torch_npu"):
        runtime.empty(shape, "float32", _device())


@pytest.mark.parametrize("size", [0, 1, 128])
def test_workspace_c_abi_preserves_allocation_error(size):
    with pytest.raises(RuntimeError, match=r"does not allocate.*torch_npu"):
        _probe("AllocWorkspace")(size)


@pytest.mark.parametrize("through_c_api", [False, True])
def test_workspace_free_cannot_take_ownership(through_c_api):
    with pytest.raises(RuntimeError, match=r"does not free.*torch_npu"):
        _probe("FreeWorkspace")(through_c_api)


@pytest.mark.parametrize("from_type,to_type", [(12, 1), (1, 12), (12, 12)])
def test_copy_rejection_preserves_buffer_and_byte_offset(from_type, to_type):
    with pytest.raises(RuntimeError, match=r"does not implement CopyDataFromTo.*torch_npu"):
        _probe("Copy")(from_type, to_type)


def test_no_current_device_is_invented():
    # This fixed FFI maps unregistered error kinds to Python RuntimeError and
    # retains the original kind in the attached native error object.
    with pytest.raises(RuntimeError, match="GetDeviceIndex") as error:
        _probe("GetDeviceIndex")()
    assert error.value.__tvm_ffi_error__.kind == "NotImplementedError"


def test_invalid_device_index_rejected_by_runtime_c_abi():
    with pytest.raises(ValueError, match="nonnegative device index"):
        _probe("SetDevice")(-1)


def test_shim_does_not_take_device_or_stream_ownership(preserve_streams):
    stream = 0xA517
    _set_stream(0, stream)
    _probe("SetDevice")(0)
    _probe("SetDevice")(7)
    tvm_ffi.get_global_func("runtime.DeviceStreamSync")(_device(), stream)
    tvm_ffi.get_global_func("runtime.DeviceStreamFree")(_device(), stream)
    assert _probe("GetStream")(0) == stream
    assert _probe("EnvStream")(0) == stream


def test_stream_state_is_shared_with_fixed_ffi_and_separate_per_device(preserve_streams):
    for index, handle in ((0, 0x5011), (7, 0x5077)):
        _set_stream(index, handle)
        assert _probe("GetStream")(index) == handle
        assert _probe("EnvStream")(index) == handle
    _set_stream(7, 0)
    assert _probe("GetStream")(7) == 0
    assert _probe("GetStream")(0) == 0x5011


def test_borrowed_stream_state_is_thread_local(preserve_streams):
    _set_stream(0, 0x1100)
    barrier = threading.Barrier(3)

    def worker(handle):
        before = _probe("GetStream")(0)
        try:
            _set_stream(0, handle)
            barrier.wait(timeout=10)
            assert _probe("GetStream")(0) == handle
            assert _probe("EnvStream")(0) == handle
            return before
        finally:
            _set_stream(0, before)

    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(worker, value) for value in (0x2200, 0x3300)]
        barrier.wait(timeout=10)
        assert _probe("GetStream")(0) == 0x1100
        assert [future.result() for future in futures] == [0, 0]
