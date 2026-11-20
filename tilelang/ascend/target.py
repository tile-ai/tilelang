from __future__ import annotations

from tvm.target import Target

from tilelang.backend.target import (
    TargetLike,
    register_target_detector,
    register_target_execution_normalizer,
    register_target_normalizer,
)


def _target_ffi_api():
    from tilelang import _ffi_api

    return _ffi_api


def _query_device_aiv_per_aic() -> int | None:
    """Query the device AIC:AIV ratio from hardware only.

    Does NOT read any environment variable or user override — this is the
    ground truth used by frontend launch construction to validate user requests. Returns None when no current NPU can be queried.
    """
    try:
        import torch

        props = torch.npu.get_device_properties(torch.npu.current_device())
        cube = getattr(props, "cube_core_num", 0)
        vector = getattr(props, "vector_core_num", 0)
        if cube > 0 and vector > 0 and vector % cube == 0 and vector // cube in (1, 2):
            return vector // cube
    except Exception:
        pass
    return None


def resolve_aiv_count(requested: int | None = None) -> int:
    """Resolve a frontend-validated request and check device capacity."""
    limit = _query_device_aiv_per_aic()
    if requested is not None and limit is not None and requested > limit:
        raise ValueError(f"AIV group count={requested} exceeds the current device capability {limit}")
    device_count = limit if limit is not None else 2
    return requested if requested is not None else device_count


def _make_ascend_target(target_dict: dict | None = None) -> Target:
    target_dict = dict(target_dict or {})
    target_dict["kind"] = "ascend"
    return Target(target_dict)


def _with_pto_key(target: Target) -> Target:
    target_dict = dict(target.export())
    # The native kind supplies Ascend semantics; "pto" selects PTO codegen,
    # while "ascend" preserves the target kind's default feature key.
    target_dict["keys"] = list(dict.fromkeys(["pto", *target_dict.get("keys", ()), "ascend"]))
    return Target(target_dict)


def _make_pto_target() -> Target:
    return _with_pto_key(_make_ascend_target())


def target_is_ascend(target: Target) -> bool:
    """Return whether *target* uses the Ascend architecture."""
    return _target_ffi_api().TargetIsAscend(target)


def target_is_pto(target: Target) -> bool:
    """Return whether *target* selects PTO within the Ascend architecture."""
    return target_is_ascend(target) and "pto" in target.keys


def target_is_plain_ascend(target: Target) -> bool:
    """Return whether *target* selects AscendC rather than PTO codegen."""
    return target_is_ascend(target) and "pto" not in target.keys


def check_ascend_availability() -> bool:
    try:
        import torch

        return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception:
        return False


def _detect_ascend_target() -> Target | None:
    if check_ascend_availability():
        return _make_ascend_target()
    return None


def normalize_ascend_target(target: TargetLike) -> Target | None:
    if not isinstance(target, str) or target.strip() != "ascend":
        return None

    try:
        return _make_ascend_target()
    except Exception:
        return None


def normalize_pto_target(target: TargetLike) -> Target | None:
    if not isinstance(target, str) or target.strip() != "pto":
        return None

    try:
        return _make_pto_target()
    except Exception:
        return None


def normalize_pto_execution_target(target: Target, execution_backend: str | None) -> Target | None:
    """Select the PTO target variant for an explicit PTO execution backend."""

    if execution_backend is None or str(execution_backend).lower() != "pto":
        return None
    if not target_is_ascend(target):
        return None
    if target_is_pto(target):
        return target
    return _with_pto_key(target)


def normalize_asc_target(target: TargetLike) -> Target | None:
    """Accept ``asc`` as the concise name for the AscendC backend."""
    if isinstance(target, str) and target.strip() == "asc":
        return normalize_ascend_target("ascend")
    return None


register_target_detector("ascend", _detect_ascend_target, override=True)
register_target_normalizer("ascend", normalize_ascend_target, override=True)
register_target_normalizer("asc", normalize_asc_target, override=True)
register_target_normalizer("pto", normalize_pto_target, override=True)
register_target_execution_normalizer("pto", normalize_pto_execution_target, override=True)
