import pytest


import tilelang.testing
from tvm_tirx.target import Target

from tilelang.backend.target import determine_target
from tilelang.rocm.target import (
    normalize_rocm_arch,
    rocm_warp_size_for_arch,
    target_get_mcpu,
    target_get_rdna_generation,
    target_get_warp_size,
    target_is_cdna,
    target_is_rdna,
)


def _hip_target(mcpu: str) -> Target:
    return Target({"kind": "hip", "mcpu": mcpu})


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
    target = _hip_target("gfx1151:sramecc+:xnack-")
    assert target_get_mcpu(target) == "gfx1151"


def test_determine_target_adds_rdna_thread_warp_size():
    target = determine_target({"kind": "hip", "mcpu": "gfx1151"}, return_object=True)
    assert target_get_mcpu(target) == "gfx1151"
    assert int(target.attrs["thread_warp_size"]) == 32


def test_determine_target_adds_known_gfx12_thread_warp_size():
    target = determine_target({"kind": "hip", "mcpu": "gfx1200"}, return_object=True)
    assert target_get_mcpu(target) == "gfx1200"
    assert int(target.attrs["thread_warp_size"]) == 32


def test_determine_target_rejects_legacy_option_string():
    with pytest.raises(AssertionError, match="Pass target options as a dict"):
        determine_target("hip -mcpu=gfx1151", return_object=True)


def test_auto_target_uses_registered_detector_order_for_rocm():
    import tilelang.backend.target as target_registry

    old_detectors = dict(target_registry._TARGET_DETECTORS)
    try:
        target_registry._TARGET_DETECTORS.clear()
        target_registry.register_target_detector(
            "unit-hip",
            lambda: Target({"kind": "hip", "mcpu": "gfx1151"}),
            override=True,
        )
        target_registry.register_target_detector(
            "unit-cuda",
            lambda: Target({"kind": "cuda", "arch": "sm_90"}),
            override=True,
        )

        target = determine_target("auto", return_object=True)
        assert target.kind.name == "hip"
        assert target_get_mcpu(target) == "gfx1151"
        assert int(target.attrs["thread_warp_size"]) == 32
    finally:
        target_registry._TARGET_DETECTORS.clear()
        target_registry._TARGET_DETECTORS.update(old_detectors)


def test_rdna_gfx1151_target_classification():
    target = _hip_target("gfx1151")
    assert target_is_rdna(target)
    assert not target_is_cdna(target)
    assert target_get_rdna_generation(target) == 11
    assert target_get_warp_size(target) == 32


if __name__ == "__main__":
    tilelang.testing.main()
