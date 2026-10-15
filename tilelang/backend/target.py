from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

from tvm.target import Target

TargetConfig = dict[str, object]
TargetInput = str | Mapping[str, object] | Target
TargetLike = str | TargetConfig | Target
TargetDetector = Callable[[], TargetInput | None]
TargetNormalizer = Callable[[TargetLike], TargetInput | None]


@dataclass(frozen=True, slots=True)
class TargetDetectorSpec:
    name: str
    detect: TargetDetector


@dataclass(frozen=True, slots=True)
class TargetNormalizerSpec:
    name: str
    normalize: TargetNormalizer


_TARGET_DETECTORS: dict[str, TargetDetectorSpec] = {}
_TARGET_NORMALIZERS: dict[str, TargetNormalizerSpec] = {}


def register_target_detector(
    name: str,
    detect: TargetDetector,
    *,
    override: bool = False,
) -> TargetDetectorSpec:
    if name in _TARGET_DETECTORS and not override:
        raise ValueError(f"Target detector {name!r} is already registered")
    spec = TargetDetectorSpec(name=name, detect=detect)
    _TARGET_DETECTORS[name] = spec
    return spec


def register_target_normalizer(
    name: str,
    normalize: TargetNormalizer,
    *,
    override: bool = False,
) -> TargetNormalizerSpec:
    if name in _TARGET_NORMALIZERS and not override:
        raise ValueError(f"Target normalizer {name!r} is already registered")
    spec = TargetNormalizerSpec(name=name, normalize=normalize)
    _TARGET_NORMALIZERS[name] = spec
    return spec


def _normalize_registered_target(target: TargetLike) -> TargetInput | None:
    for spec in _TARGET_NORMALIZERS.values():
        normalized = spec.normalize(target)
        if normalized is not None:
            return normalized
    return None


def auto_detect_target() -> TargetInput:
    errors: list[str] = []
    for spec in _TARGET_DETECTORS.values():
        try:
            detected = spec.detect()
        except Exception as err:
            errors.append(f"{spec.name}: {err}")
            continue
        if detected is not None:
            return detected

    details = f" Tried: {', '.join(errors)}." if errors else ""
    raise ValueError(f"No registered target detector found an available target.{details}")


def list_target_detectors() -> tuple[str, ...]:
    return tuple(_TARGET_DETECTORS)


def _validate_manual_target(target: TargetLike) -> TargetInput:
    normalized = _normalize_registered_target(target)
    if normalized is not None:
        return normalized

    if isinstance(target, Target):
        return target
    if isinstance(target, dict):
        try:
            Target(target)
        except Exception as err:
            raise AssertionError(f"Target {target} is not supported. Pass a valid target config dict.") from err
        return target
    if isinstance(target, str):
        normalized_target = target.strip()
        if not normalized_target:
            raise AssertionError(f"Target {target} is not supported")
        try:
            Target(normalized_target)
        except Exception as err:
            raise AssertionError(
                f"Target {target} is not supported. Pass target options as a dict when the target needs attributes."
            ) from err
        return normalized_target
    raise AssertionError(f"Target {target} is not supported")


def _finalize_target(target: TargetInput, *, return_object: bool) -> str | Mapping[str, object] | Target:
    if isinstance(target, (str, dict, Target)):
        normalized = _normalize_registered_target(target)
        if normalized is not None:
            target = normalized
    if return_object and not isinstance(target, Target):
        return Target(target)
    return target


def determine_target(target: TargetLike | Literal["auto"] = "auto", return_object: bool = False) -> str | Mapping[str, object] | Target:
    """
    Determine and validate the target for compilation.

    Target detection and normalization is provided by registered target modules.
    """
    if target == "auto":
        current_target = Target.current(allow_none=True)
        return_var = current_target if current_target is not None else auto_detect_target()
    else:
        return_var = _validate_manual_target(target)

    return _finalize_target(return_var, return_object=return_object)
