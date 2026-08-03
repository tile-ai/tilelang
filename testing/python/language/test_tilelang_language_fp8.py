from types import SimpleNamespace

import pytest
import torch

import tilelang.language as T
from tilelang.language.fp8 import determine_fp8_type


@pytest.mark.parametrize(
    ("gcn_arch", "expected_e4m3", "expected_e5m2"),
    [
        ("gfx942", T.float8_e4m3fnuz, T.float8_e5m2fnuz),
        ("gfx950", T.float8_e4m3fn, T.float8_e5m2),
        ("gfx1100", T.float8_e4m3fn, T.float8_e5m2),
        ("gfx1201", T.float8_e4m3fn, T.float8_e5m2),
    ],
)
def test_rocm_fp8_type_matches_hip_template_variant(monkeypatch, gcn_arch, expected_e4m3, expected_e5m2):
    monkeypatch.setattr(torch.version, "hip", "test")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)

    def get_device_properties(device):
        assert device == 3
        return SimpleNamespace(gcnArchName=gcn_arch)

    monkeypatch.setattr(torch.cuda, "get_device_properties", get_device_properties)

    assert determine_fp8_type("e4m3") == expected_e4m3
    assert determine_fp8_type("e5m2") == expected_e5m2


def test_generic_fp8_torch_dtype_uses_current_rocm_arch(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "test")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(gcnArchName="gfx1100"))

    assert T.float8_e4m3.as_torch() == torch.float8_e4m3fn
    assert T.float8_e5m2.as_torch() == torch.float8_e5m2
