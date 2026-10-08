import pytest
import torch

from tilelang import profiler


@pytest.mark.parametrize("method", ["assert_allclose", "manual_assert_close"])
def test_profiler_validation_does_not_synchronize_cuda_for_cpu(monkeypatch, method):
    def unexpected(*args, **kwargs):
        raise AssertionError("CPU validation must not call CUDA synchronization")

    monkeypatch.setattr(torch.cuda, "synchronize", unexpected)
    instance = profiler.Profiler([], [], profiler.TensorSupplyType.Auto, lambda x: x + 1)
    inputs = [torch.tensor([1.0])]

    if method == "assert_allclose":
        instance.assert_allclose(lambda x: x + 1, input_tensors=inputs)
    else:
        instance.manual_assert_close(
            lambda x: x + 1,
            input_tensors=inputs,
            manual_check_prog=lambda actual, expected: torch.testing.assert_close(actual[0], expected[0]),
        )


def test_profiler_validation_synchronizes_each_accelerator_device_once(monkeypatch):
    class FakeTensor:
        def __init__(self, device):
            self.device = torch.device(device)

    calls = []
    monkeypatch.setattr(profiler.torch, "Tensor", FakeTensor)
    monkeypatch.setattr(profiler, "device_synchronize", calls.append)

    profiler._synchronize_tensor_devices(
        [FakeTensor("cuda:1"), FakeTensor("cuda:1")],
        (FakeTensor("mps"), {"output": FakeTensor("cpu")}),
    )

    assert calls == [torch.device("cuda:1"), torch.device("mps")]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires an MPS runtime")
def test_profiler_validation_synchronizes_mps(monkeypatch):
    calls = []

    def synchronize(device):
        calls.append(device)
        torch.mps.synchronize()

    monkeypatch.setattr(profiler, "device_synchronize", synchronize)
    instance = profiler.Profiler([], [], profiler.TensorSupplyType.Auto, lambda x: x + 1)
    inputs = [torch.tensor([1.0], device="mps")]

    instance.assert_allclose(lambda x: x + 1, input_tensors=inputs)

    assert calls == [torch.device("mps", 0), torch.device("mps", 0)]
