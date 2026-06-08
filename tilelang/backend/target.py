from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module

from tvm.target import Target

TargetInput = str | Mapping[str, object] | Target
TargetDetector = Callable[[], TargetInput | None]


@dataclass(frozen=True, slots=True)
class TargetDetectorSpec:
    name: str
    detect: TargetDetector
    priority: int = 0


_TARGET_DETECTORS: dict[str, TargetDetectorSpec] = {}
_LAZY_TARGET_DETECTORS: dict[str, str] = {}
_LOADED_TARGET_DETECTORS: set[str] = set()


def register_target_detector(
    name: str,
    detect: TargetDetector,
    *,
    priority: int = 0,
    override: bool = False,
) -> TargetDetectorSpec:
    if name in _TARGET_DETECTORS and not override:
        raise ValueError(f"Target detector {name!r} is already registered")
    spec = TargetDetectorSpec(name=name, detect=detect, priority=priority)
    _TARGET_DETECTORS[name] = spec
    return spec


def register_lazy_target_detector(name: str, import_path: str) -> None:
    _LAZY_TARGET_DETECTORS[name] = import_path


def _ensure_target_detectors_loaded() -> list[str]:
    errors: list[str] = []
    for name, import_path in tuple(_LAZY_TARGET_DETECTORS.items()):
        if name in _LOADED_TARGET_DETECTORS:
            continue
        try:
            import_module(import_path)
        except Exception as err:
            errors.append(f"{name}: {err}")
        finally:
            _LOADED_TARGET_DETECTORS.add(name)
    return errors


def auto_detect_target() -> TargetInput:
    errors = _ensure_target_detectors_loaded()
    detectors = sorted(_TARGET_DETECTORS.values(), key=lambda spec: spec.priority, reverse=True)
    for spec in detectors:
        try:
            detected = spec.detect()
        except Exception as err:
            errors.append(f"{spec.name}: {err}")
            continue
        if detected is not None:
            return detected

    details = f" Tried: {', '.join(errors)}." if errors else ""
    raise ValueError(f"No CUDA or HIP or MPS available on this system.{details}")


def list_target_detectors() -> tuple[str, ...]:
    _ensure_target_detectors_loaded()
    return tuple(sorted(_TARGET_DETECTORS))
