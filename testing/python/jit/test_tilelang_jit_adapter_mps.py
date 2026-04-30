"""Focused tests for JIT adapter device selection without CUDA."""

from types import SimpleNamespace

import torch

from tilelang.jit.adapter.base import BaseKernelAdapter


def test_current_device_functor_prefers_mps_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    if getattr(torch.backends, "mps", None) is None:
        monkeypatch.setattr(torch.backends, "mps", SimpleNamespace(is_available=lambda: True), raising=False)
    else:
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    device_functor = BaseKernelAdapter.get_current_device_functor()

    assert device_functor() == torch.device("mps")


def test_current_device_functor_falls_back_to_cpu_without_cuda_or_mps(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    if getattr(torch.backends, "mps", None) is None:
        monkeypatch.setattr(torch.backends, "mps", SimpleNamespace(is_available=lambda: False), raising=False)
    else:
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    device_functor = BaseKernelAdapter.get_current_device_functor()

    assert device_functor() == torch.device("cpu")
