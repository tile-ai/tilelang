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



def _normalize_target_name(target: TargetLike) -> str | None:
    """One normalization rule for every textual target/backend name.

    The normalizers below used to mix ``strip()`` (exact case) with ``lower()``
    (case-insensitive), so e.g. ``" PTO "`` fell through to the AscendC
    default while the execution normalizer accepted it.
    """
    if not isinstance(target, str):
        return None
    return target.strip().lower()

def normalize_ascend_target(target: TargetLike) -> Target | None:
    if _normalize_target_name(target) != "ascend":
        return None
    # Do not swallow Target construction errors: returning None here reports a
    # valid Ascend target as "not Ascend" and silently falls back to another
    # backend, hiding the real failure.
    return _make_ascend_target()


def normalize_pto_target(target: TargetLike) -> Target | None:
    if _normalize_target_name(target) != "pto":
        return None
    return _make_pto_target()


def normalize_pto_execution_target(target: Target, execution_backend: str | None) -> Target | None:
    """Select the PTO target variant for an explicit PTO execution backend."""

    if _normalize_target_name(execution_backend) != "pto":
        return None
    if not target_is_ascend(target):
        return None
    if target_is_pto(target):
        return target
    return _with_pto_key(target)


def normalize_asc_target(target: TargetLike) -> Target | None:
    """Accept ``asc`` as the concise name for the AscendC backend."""
    if _normalize_target_name(target) == "asc":
        return normalize_ascend_target("ascend")
    return None


register_target_detector("ascend", _detect_ascend_target, override=True)
register_target_normalizer("ascend", normalize_ascend_target, override=True)
register_target_normalizer("asc", normalize_asc_target, override=True)
register_target_normalizer("pto", normalize_pto_target, override=True)
register_target_execution_normalizer("pto", normalize_pto_execution_target, override=True)
