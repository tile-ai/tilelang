from __future__ import annotations

from tvm.target import Target

from tilelang.backend.target import register_target_detector
from tilelang.backend.target import ROCM_MTRIPLE, normalize_rocm_arch, rocm_warp_size_for_arch


def _detect_rocm_arch() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return normalize_rocm_arch(getattr(torch.cuda.get_device_properties(0), "gcnArchName", None))
    except Exception:
        pass
    return None


def _target_from_arch(arch: str | None) -> Target | str:
    if arch is None:
        return "hip"
    target_dict: dict[str, object] = {
        "kind": "hip",
        "mcpu": arch,
        "mtriple": ROCM_MTRIPLE,
    }
    warp_size = rocm_warp_size_for_arch(arch)
    if warp_size is not None:
        target_dict["thread_warp_size"] = warp_size
    return Target(target_dict)


def _detect_rocm_target() -> Target | str | None:
    try:
        from tvm.contrib import rocm

        rocm.find_rocm_path()
    except Exception:
        return None

    return _target_from_arch(_detect_rocm_arch())


register_target_detector("hip", _detect_rocm_target, priority=80, override=True)
