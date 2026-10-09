from types import SimpleNamespace

import torch

from tilelang import language as T
from tilelang.language.fp8 import determine_fp8_type, determine_torch_fp8_type


def test_determine_fp8_type_uses_requested_rocm_device(monkeypatch):
    """Select FNUZ or OCP FP8 from the requested device, not fixed device 0."""
    properties = {
        0: SimpleNamespace(gcnArchName="gfx942:sramecc+:xnack-"),
        1: SimpleNamespace(gcnArchName="gfx950:sramecc+:xnack-"),
    }
    requested = []

    def get_device_properties(device):
        requested.append(device)
        return properties[device]

    monkeypatch.setattr(torch.version, "hip", "7.2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", get_device_properties)

    assert determine_fp8_type(device=1) == T.float8_e4m3fn
    assert determine_fp8_type(device=0) == T.float8_e4m3fnuz
    assert determine_torch_fp8_type(device=1) == torch.float8_e4m3fn
    assert requested == [1, 0, 1]
