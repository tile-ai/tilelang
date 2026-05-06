import pytest

from tilelang import tvm as tvm
from tvm.target import Target

import tilelang.utils.target as target_utils
from tilelang.utils.target import (
    determine_target,
    normalize_rocm_arch,
    rocm_warp_size_for_arch,
    target_get_mcpu,
    target_get_rdna_generation,
    target_get_warp_size,
    target_is_cdna,
    target_is_rdna,
)


def test_normalize_rocm_arch_strips_feature_suffix():
    assert normalize_rocm_arch("gfx1151:sramecc+:xnack-") == "gfx1151"
    assert normalize_rocm_arch("gfx942") == "gfx942"
    assert normalize_rocm_arch("") is None
    assert normalize_rocm_arch("sm_90") is None
    assert rocm_warp_size_for_arch("gfx1151") == 32
    assert rocm_warp_size_for_arch("gfx1030") == 32
    assert rocm_warp_size_for_arch("gfx1200") == 32
    assert rocm_warp_size_for_arch("gfx942") == 64


def test_target_mcpu_helpers():
    target = Target("hip -mcpu=gfx1151:sramecc+:xnack-")
    assert target_get_mcpu(target) == "gfx1151"


def test_determine_target_adds_rdna_thread_warp_size():
    target = determine_target("hip -mcpu=gfx1151", return_object=True)
    assert target_get_mcpu(target) == "gfx1151"
    assert int(target.attrs["thread_warp_size"]) == 32


def test_determine_target_adds_known_gfx12_thread_warp_size():
    target = determine_target("hip -mcpu=gfx1200", return_object=True)
    assert target_get_mcpu(target) == "gfx1200"
    assert int(target.attrs["thread_warp_size"]) == 32


def test_auto_target_prefers_rocm_pytorch_over_cuda_toolkit(monkeypatch):
    monkeypatch.setattr(target_utils.torch.version, "hip", "test", raising=False)
    monkeypatch.setattr(target_utils, "check_hip_availability", lambda: True)
    monkeypatch.setattr(target_utils, "check_cuda_availability", lambda: True)
    monkeypatch.setattr(target_utils, "_detect_torch_rocm_arch", lambda: "gfx1151")

    target = determine_target("auto", return_object=True)
    assert target.kind.name == "hip"
    assert target_get_mcpu(target) == "gfx1151"
    assert int(target.attrs["thread_warp_size"]) == 32


def test_rdna_gfx1151_target_classification():
    target = Target("hip -mcpu=gfx1151")
    assert target_is_rdna(target)
    assert not target_is_cdna(target)
    assert target_get_rdna_generation(target) == 11
    assert target_get_warp_size(target) == 32


def test_carver_routes_rdna_without_instantiating_device(monkeypatch):
    import torch

    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch, "mps"):
        monkeypatch.setattr(torch.mps, "is_available", lambda: False, raising=False)

    import tilelang.carver.arch as arch_mod

    def fake_rdna(target):
        return ("rdna", target)

    monkeypatch.setattr(arch_mod, "RDNA", fake_rdna)
    arch = arch_mod.get_arch(Target("hip -mcpu=gfx1151"))
    assert arch[0] == "rdna"
    assert target_get_mcpu(arch[1]) == "gfx1151"


def test_carver_rejects_unsupported_rdna_generations(monkeypatch):
    import tilelang.carver.arch as arch_mod

    def fake_cdna(target):
        return ("cdna", target)

    monkeypatch.setattr(arch_mod, "CDNA", fake_cdna)
    with pytest.raises(ValueError, match="gfx11 targets only"):
        arch_mod.get_arch(Target("hip -mcpu=gfx1200"))


def test_rdna_device_model_rejects_gfx12_before_device_probe():
    from tilelang.carver.arch.rdna import RDNA

    with pytest.raises(ValueError, match="gfx11 targets only"):
        RDNA(Target("hip -mcpu=gfx1200"))


def test_rdna_tensor_instruction_lookup_is_generation_aware():
    from tilelang.carver.arch.rdna import RDNA

    arch = RDNA.__new__(RDNA)
    arch.rdna_generation = 11
    assert arch.get_avaliable_tensorintrin_shapes() == [[16, 16]]
    assert isinstance(arch.available_tensor_instructions, list)

    arch.rdna_generation = 12
    with pytest.raises(ValueError, match="Unsupported RDNA generation"):
        arch.get_avaliable_tensorintrin_shapes()
