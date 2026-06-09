from __future__ import annotations

from tvm.target import Target

from tilelang.backend.target import TargetLike, register_target_detector, register_target_normalizer

ROCM_MTRIPLE = "amdgcn-amd-amdhsa-hcc"


def _target_ffi_api():
    from tilelang import _ffi_api

    return _ffi_api


def normalize_rocm_arch(arch: str | None) -> str | None:
    if arch is None:
        return None
    normalized = str(arch).strip().split(":", maxsplit=1)[0]
    return normalized if normalized.startswith("gfx") else None


def target_get_mcpu(target: str | Target | None) -> str | None:
    if target is None:
        return None
    if isinstance(target, str):
        target = Target(target)
    return normalize_rocm_arch(target.attrs.get("mcpu"))


def rocm_warp_size_for_arch(arch: str | None) -> int | None:
    if arch is None:
        return None
    if arch.startswith("gfx9"):
        return 64
    if arch.startswith(("gfx10", "gfx11", "gfx12")):
        return 32
    return None


def with_rocm_target_attrs(target: Target) -> Target:
    if target.kind.name != "hip":
        return target
    arch = target_get_mcpu(target)
    if arch is None:
        return target

    target_dict = dict(target.export())
    target_dict.setdefault("mtriple", ROCM_MTRIPLE)
    warp_size = rocm_warp_size_for_arch(arch)
    if warp_size is not None:
        target_dict["thread_warp_size"] = warp_size
    else:
        target_dict.pop("thread_warp_size", None)
    return Target(target_dict)


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


def check_hip_availability() -> bool:
    """
    Check if HIP (ROCm) is available on the system by locating the ROCm path.
    Returns:
        bool: True if HIP is available, False otherwise.
    """
    try:
        from tvm.contrib import rocm

        rocm.find_rocm_path()
        return True
    except Exception:
        return False


def _detect_rocm_target() -> Target | str | None:
    if not check_hip_availability():
        return None

    return _target_from_arch(_detect_rocm_arch())


def normalize_rocm_target(target: TargetLike) -> Target | None:
    if isinstance(target, Target):
        parsed_target = target
    elif isinstance(target, dict):
        if target.get("kind") != "hip":
            return None
        try:
            parsed_target = Target(target)
        except Exception:
            return None
    else:
        return None

    if parsed_target.kind.name != "hip":
        return None
    if target_get_mcpu(parsed_target) is None:
        return parsed_target if isinstance(target, Target) else None
    return with_rocm_target_attrs(parsed_target)


def target_is_hip(target: Target) -> bool:
    return _target_ffi_api().TargetIsRocm(target)


def target_is_cdna(target: Target) -> bool:
    return _target_ffi_api().TargetIsCDNA(target)


def target_is_rdna(target: Target) -> bool:
    return _target_ffi_api().TargetIsRDNA(target)


def target_is_gfx950(target: Target) -> bool:
    return _target_ffi_api().TargetIsGfx950(target)


def target_get_warp_size(target: Target) -> int:
    return _target_ffi_api().TargetRocmGetWarpSize(target)


def target_get_rdna_generation(target: Target) -> int:
    return _target_ffi_api().TargetGetRDNAGeneration(target)


register_target_detector("hip", _detect_rocm_target, override=True)
register_target_normalizer("hip", normalize_rocm_target, override=True)
