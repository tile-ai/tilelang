from __future__ import annotations

from platform import mac_ver

from tilelang.backend.target import register_target_detector


def _detect_metal_target() -> str | None:
    mac_release, _, arch = mac_ver()
    if mac_release and arch == "arm64":
        return "metal"
    return None


register_target_detector("metal", _detect_metal_target, priority=10, override=True)
