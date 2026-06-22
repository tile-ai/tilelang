from __future__ import annotations

from platform import mac_ver

from tvm.target import Target

from tilelang.backend.target import register_target_detector


def _target_ffi_api():
    from tilelang import _ffi_api

    return _ffi_api


def check_metal_availability() -> bool:
    mac_release, _, arch = mac_ver()
    if not mac_release:
        return False
    # todo: check torch version?
    return arch == "arm64"


def _detect_metal_target() -> str | None:
    if check_metal_availability():
        return "metal"
    return None


def target_is_metal(target: Target) -> bool:
    return _target_ffi_api().TargetIsMetal(target)


register_target_detector("metal", _detect_metal_target, override=True)
