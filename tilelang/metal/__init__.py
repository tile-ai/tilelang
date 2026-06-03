from platform import mac_ver

from tilelang.backend.target import register_target_kind

from . import intrinsics  # noqa: F401
from . import op  # noqa: F401
from . import pipeline  # noqa: F401
from . import transform  # noqa: F401


def _detect_metal_target():
    mac_release, _, arch = mac_ver()
    if mac_release and arch == "arm64":
        return "metal"
    return None


register_target_kind("metal", tvm_kind="metal", detect=_detect_metal_target, priority=10, override=True)
