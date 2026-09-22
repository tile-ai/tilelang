import runpy
from unittest.mock import Mock

import pytest
import torch

from tilelang.profiler import bench, torch_bench
from tilelang.utils import device as device_utils


@pytest.mark.parametrize(
    "cuda_available, mps_available, expected_backend, expected_device",
    [
        pytest.param(True, False, "cuda", 3, id="cuda"),
        pytest.param(False, True, "mps", "mps:0", id="metal"),
        pytest.param(True, True, "cuda", 3, id="cuda-before-metal"),
        pytest.param(False, False, None, None, id="cpu"),
    ],
)
def test_device_compatibility(monkeypatch, cuda_available, mps_available, expected_backend, expected_device):
    cuda_event = Mock()
    mps_event = Mock()
    cuda_sync = Mock()
    mps_sync = Mock()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps_available)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(torch.cuda, "Event", cuda_event)
    monkeypatch.setattr(torch.mps, "Event", mps_event)
    monkeypatch.setattr(torch.cuda, "synchronize", cuda_sync)
    monkeypatch.setattr(torch.mps, "synchronize", mps_sync)

    namespace = runpy.run_path(device_utils.__file__)

    assert namespace["IS_CUDA"] is cuda_available
    assert namespace["IS_MPS"] is mps_available
    assert namespace["Event"] is {"cuda": cuda_event, "mps": mps_event, None: None}[expected_backend]
    assert namespace["get_current_device"]() == expected_device
    cuda_event.assert_not_called()
    mps_event.assert_not_called()
    if expected_backend is None:
        with pytest.raises(RuntimeError, match="No device is available"):
            namespace["device_synchronize"]()
    else:
        namespace["device_synchronize"]()
    if expected_backend == "cuda":
        cuda_sync.assert_called_once_with()
        mps_sync.assert_not_called()
    elif expected_backend == "mps":
        mps_sync.assert_called_once_with()
        cuda_sync.assert_not_called()
    else:
        cuda_sync.assert_not_called()
        mps_sync.assert_not_called()


@pytest.mark.parametrize(
    "device, backend",
    [
        pytest.param(1, "cuda", id="cuda-index"),
        pytest.param(torch.device("cuda:1"), "cuda", id="cuda-device"),
        pytest.param(torch.device("cuda"), "cuda", id="cuda-current-device"),
        pytest.param("cuda:1", "cuda", id="cuda-string"),
        pytest.param(torch.device("mps"), "mps", id="metal-device"),
        pytest.param(torch.device("mps:0"), "mps", id="metal-indexed-device"),
        pytest.param("mps", "mps", id="metal-string"),
    ],
)
def test_device_synchronize_explicit_device(monkeypatch, device, backend):
    cuda_sync = Mock()
    mps_sync = Mock()
    monkeypatch.setattr(device_utils, "IS_CUDA", backend != "cuda")
    monkeypatch.setattr(device_utils, "IS_MPS", backend != "mps")
    monkeypatch.setattr(torch.cuda, "synchronize", cuda_sync)
    monkeypatch.setattr(torch.mps, "synchronize", mps_sync)

    device_utils.device_synchronize(device)

    if backend == "cuda":
        cuda_sync.assert_called_once_with(device)
        mps_sync.assert_not_called()
    else:
        mps_sync.assert_called_once_with()
        cuda_sync.assert_not_called()


@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("meta")])
def test_device_synchronize_rejects_unsupported_devices(monkeypatch, device):
    cuda_sync = Mock()
    mps_sync = Mock()
    monkeypatch.setattr(torch.cuda, "synchronize", cuda_sync)
    monkeypatch.setattr(torch.mps, "synchronize", mps_sync)

    with pytest.raises(ValueError, match="only supports CUDA/HIP or MPS"):
        device_utils.device_synchronize(device)
    cuda_sync.assert_not_called()
    mps_sync.assert_not_called()


def test_profiler_uses_shared_device_helpers():
    assert bench.Event is device_utils.Event
    assert torch_bench.Event is device_utils.Event
    assert bench._cuda_synchronize is device_utils.device_synchronize
    assert torch_bench._cuda_synchronize is device_utils.device_synchronize
