"""Tests for CUDA capability queries that do not require a CUDA device."""

from types import SimpleNamespace

import pytest

from tilelang import tvm
from tilelang.contrib import nvcc


@pytest.mark.parametrize(
    ("arch", "expected"),
    [("sm_61", False), ("sm_70", True), ("sm_90a", True), ("sm_100a", True), ("sm_120f", True)],
)
def test_tensorcore_explicit_target(monkeypatch, arch, expected):
    monkeypatch.setattr(tvm, "cuda", lambda _: SimpleNamespace(exist=False))

    assert nvcc.have_tensorcore(target=tvm.target.Target({"kind": "cuda", "arch": arch})) is expected


@pytest.mark.parametrize(("arch", "device_version", "expected"), [("sm_61", "9.0", False), ("sm_90a", "6.1", True)])
def test_tensorcore_target_precedes_local_device(monkeypatch, arch, device_version, expected):
    monkeypatch.setattr(tvm, "cuda", lambda _: SimpleNamespace(exist=True, compute_version=device_version))

    assert nvcc.have_tensorcore(target=tvm.target.Target({"kind": "cuda", "arch": arch})) is expected


def test_tensorcore_current_target(monkeypatch):
    monkeypatch.setattr(tvm, "cuda", lambda _: SimpleNamespace(exist=False))

    with tvm.target.Target({"kind": "cuda", "arch": "sm_80"}):
        assert nvcc.have_tensorcore()


@pytest.mark.parametrize(("compute_version", "expected"), [("6.1", False), ("7.0", True)])
def test_tensorcore_explicit_compute_version(monkeypatch, compute_version, expected):
    def unexpected_device_query(_):
        pytest.fail("An explicit compute version must not query the local device")

    monkeypatch.setattr(tvm, "cuda", unexpected_device_query)

    assert nvcc.have_tensorcore(compute_version, target=tvm.target.Target({"kind": "cuda", "arch": "sm_90a"})) is expected


@pytest.mark.parametrize(("device_version", "expected"), [("6.1", False), ("8.0", True)])
def test_tensorcore_local_device_fallback(monkeypatch, device_version, expected):
    monkeypatch.setattr(tvm, "cuda", lambda _: SimpleNamespace(exist=True, compute_version=device_version))

    assert nvcc.have_tensorcore() is expected


def test_tensorcore_without_architecture_warns(monkeypatch):
    monkeypatch.setattr(tvm, "cuda", lambda _: SimpleNamespace(exist=False))

    with pytest.warns(UserWarning, match="no CUDA architecture specified"):
        assert not nvcc.have_tensorcore()
