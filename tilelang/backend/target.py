from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import shlex
from types import MappingProxyType
from typing import Any

from tvm.target import Target as TVMTarget

TargetInput = str | Mapping[str, object] | TVMTarget
TargetNormalizer = Callable[["Target"], str | Mapping[str, object] | TVMTarget]
TargetPresetResolver = Callable[["Target"], TargetInput]
TargetDetector = Callable[[], TargetInput | None]

_TARGET_KINDS: dict[str, TargetKind] = {}
_TARGET_PRESETS: dict[str, TargetPresetResolver] = {}
_EXECUTION_BACKEND_ALIASES = {
    "dlpack": "tvm_ffi",
}


def _canon_execution_backend(name: str | None) -> str | None:
    if name is None:
        return None
    key = str(name).lower()
    return _EXECUTION_BACKEND_ALIASES.get(key, key)


def _normalize_name(name: str, *, field_name: str = "target") -> str:
    normalized = str(name).strip()
    if not normalized:
        raise ValueError(f"{field_name} name must not be empty")
    return normalized


def _normalize_keys(keys: str | Sequence[object] | None) -> tuple[str, ...]:
    if keys is None:
        return ()
    if isinstance(keys, str):
        raw_keys: Sequence[object] = keys.split(",") if "," in keys else (keys,)
    else:
        raw_keys = keys
    return tuple(dict.fromkeys(str(key) for key in raw_keys if str(key)))


def _parse_option_value(value: str) -> str | bool | list[str]:
    if "," in value:
        return [item for item in value.split(",") if item]
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    return value


def _parse_target_string(target: str) -> tuple[str, dict[str, object], tuple[str, ...], TargetInput | None]:
    tokens = shlex.split(target.strip())
    if not tokens:
        raise ValueError("Target name must not be empty")

    kind = tokens[0]
    attrs: dict[str, object] = {}
    keys: tuple[str, ...] = ()
    host: TargetInput | None = None
    for token in tokens[1:]:
        if not token.startswith("-"):
            raise ValueError(f"Unsupported target option {token!r}; use -key=value syntax")
        option = token.lstrip("-")
        key, has_value, raw_value = option.partition("=")
        key = key.replace("-", "_")
        value: object = _parse_option_value(raw_value) if has_value else True
        if key == "keys":
            keys = _normalize_keys(value if isinstance(value, (str, list, tuple)) else str(value))
        elif key == "host":
            host = str(value)
        else:
            attrs[key] = value
    return kind, attrs, keys, host


@dataclass(frozen=True, slots=True)
class TargetKind:
    """TileLang-owned target kind metadata.

    A TargetKind can be registered by a hardware package without adding a TVM
    TargetKind. The normalizer is the boundary that maps TileLang's target
    intent into the TVM target input required by the existing lowering stack.
    """

    name: str
    tvm_kind: str | None = None
    default_attrs: Mapping[str, object] = field(default_factory=dict)
    default_keys: tuple[str, ...] = ()
    normalize: TargetNormalizer | None = None
    detect: TargetDetector | None = None
    priority: int = 0
    description: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalize_name(self.name, field_name="TargetKind"))
        object.__setattr__(self, "default_attrs", MappingProxyType(dict(self.default_attrs)))
        object.__setattr__(self, "default_keys", _normalize_keys(self.default_keys))

    def to_tvm_target_input(self, target: Target) -> str | Mapping[str, object] | TVMTarget:
        if self.normalize is not None:
            return self.normalize(target)

        kind = self.tvm_kind or self.name
        config = {**dict(self.default_attrs), **dict(target.attrs)}
        keys = _normalize_keys((*self.default_keys, *target.keys))
        if not config and not keys and target.host is None and kind == target.kind:
            return kind

        config["kind"] = kind
        if keys:
            config["keys"] = list(keys)
        if target.host is not None:
            config["host"] = _target_input_to_tvm(target.host)
        return config


class Target:
    """TileLang target spec independent from TVM TargetKind registration."""

    _tilelang_target_marker = True

    def __init__(
        self,
        target: TargetInput | Target = "auto",
        *,
        execution_backend: str | None = None,
        keys: str | Sequence[object] | None = None,
        host: TargetInput | None = None,
        **attrs: Any,
    ) -> None:
        parsed_kind, parsed_attrs, parsed_keys, parsed_host, parsed_execution_backend = self._parse_target(target)
        if host is not None:
            parsed_host = host
        if execution_backend is None:
            execution_backend = parsed_execution_backend

        parsed_attrs.update(attrs)
        self.kind = parsed_kind
        self.attrs = MappingProxyType(parsed_attrs)
        self.keys = _normalize_keys((*parsed_keys, *_normalize_keys(keys)))
        self.host = parsed_host
        self.execution_backend = _canon_execution_backend(execution_backend)

    @classmethod
    def _parse_target(cls, target: TargetInput | Target) -> tuple[str, dict[str, object], tuple[str, ...], TargetInput | None, str | None]:
        if is_tilelang_target(target):
            assert isinstance(target, Target)
            return target.kind, dict(target.attrs), target.keys, target.host, target.execution_backend

        if isinstance(target, TVMTarget):
            return cls._parse_target(target.export())

        if isinstance(target, Mapping):
            config = dict(target)
            if "kind" not in config:
                raise ValueError("Target config dict must contain a 'kind' field")
            kind = _normalize_name(str(config.pop("kind")))
            keys = _normalize_keys(config.pop("keys", None))
            host = config.pop("host", None)
            execution_backend = config.pop("execution_backend", None)
            return kind, config, keys, host, _canon_execution_backend(execution_backend)  # type: ignore[arg-type]

        if isinstance(target, str):
            kind, parsed_attrs, parsed_keys, parsed_host = _parse_target_string(target)
            return _normalize_name(kind), parsed_attrs, parsed_keys, parsed_host, None

        raise TypeError(f"Unsupported target type: {type(target)!r}")

    def to_config(self) -> dict[str, object]:
        config = {"kind": self.kind, **dict(self.attrs)}
        if self.keys:
            config["keys"] = list(self.keys)
        if self.host is not None:
            config["host"] = self.host.to_config() if is_tilelang_target(self.host) else self.host
        if self.execution_backend is not None:
            config["execution_backend"] = self.execution_backend
        return config

    def to_tvm_target_input(self) -> str | Mapping[str, object] | TVMTarget:
        if self.kind == "auto":
            return _target_input_to_tvm(_detect_auto_target())

        resolver = _TARGET_PRESETS.get(self.kind)
        if resolver is not None:
            return _target_input_to_tvm(resolver(self))

        target_kind = _TARGET_KINDS.get(self.kind)
        if target_kind is not None:
            return target_kind.to_tvm_target_input(self)

        config = dict(self.attrs)
        config["kind"] = self.kind
        if self.keys:
            config["keys"] = list(self.keys)
        if self.host is not None:
            config["host"] = _target_input_to_tvm(self.host)
        return config

    def to_tvm_target(self) -> TVMTarget:
        target_input = self.to_tvm_target_input()
        target = target_input if isinstance(target_input, TVMTarget) else TVMTarget(target_input)
        try:
            from tilelang.utils.target import with_rocm_target_attrs

            return with_rocm_target_attrs(target)
        except Exception:
            return target

    def __repr__(self) -> str:
        args = [repr(self.kind)]
        if self.execution_backend is not None:
            args.append(f"execution_backend={self.execution_backend!r}")
        if self.keys:
            args.append(f"keys={list(self.keys)!r}")
        if self.host is not None:
            args.append(f"host={self.host!r}")
        args.extend(f"{key}={value!r}" for key, value in self.attrs.items())
        return f"tilelang.Target({', '.join(args)})"


def _target_input_to_tvm(target: TargetInput | Target) -> str | Mapping[str, object] | TVMTarget:
    if is_tilelang_target(target):
        assert isinstance(target, Target)
        return target.to_tvm_target_input()
    if isinstance(target, TVMTarget):
        return target
    if isinstance(target, Mapping):
        return Target(target).to_tvm_target_input()
    if isinstance(target, str):
        return Target(target).to_tvm_target_input()
    raise TypeError(f"Unsupported target type: {type(target)!r}")


def _detect_auto_target() -> TargetInput:
    detectors = [kind for kind in _TARGET_KINDS.values() if kind.detect is not None]
    detectors.sort(key=lambda kind: kind.priority, reverse=True)
    errors: list[str] = []
    for kind in detectors:
        assert kind.detect is not None
        try:
            detected = kind.detect()
        except Exception as err:
            errors.append(f"{kind.name}: {err}")
            continue
        if detected is not None:
            return detected

    details = f" Tried: {', '.join(errors)}." if errors else ""
    raise ValueError(f"No TileLang target detector found an available target.{details}")


def is_tilelang_target(target: object) -> bool:
    return getattr(target, "_tilelang_target_marker", False)


def register_target_kind(kind: TargetKind | str, *, override: bool = False, **kwargs: Any) -> TargetKind:
    target_kind = kind if isinstance(kind, TargetKind) else TargetKind(kind, **kwargs)
    if target_kind.name in _TARGET_KINDS and not override:
        raise ValueError(f"Target kind {target_kind.name!r} is already registered")
    _TARGET_KINDS[target_kind.name] = target_kind
    return target_kind


def get_target_kind(name: str) -> TargetKind:
    normalized = _normalize_name(name, field_name="TargetKind")
    try:
        return _TARGET_KINDS[normalized]
    except KeyError as err:
        raise ValueError(f"Target kind {normalized!r} is not registered") from err


def list_target_kinds() -> tuple[str, ...]:
    return tuple(sorted(_TARGET_KINDS))


def register_target_preset(name: str, resolver: TargetPresetResolver, *, override: bool = False) -> None:
    normalized = _normalize_name(name, field_name="Target preset")
    if normalized in _TARGET_PRESETS and not override:
        raise ValueError(f"Target preset {normalized!r} is already registered")
    _TARGET_PRESETS[normalized] = resolver


def list_target_presets() -> tuple[str, ...]:
    return tuple(sorted(_TARGET_PRESETS))


def resolve_target_execution_backend(
    target: TargetInput | Target | None,
    execution_backend: str | None,
) -> tuple[str | Mapping[str, object] | TVMTarget | None, str | None]:
    requested_backend = _canon_execution_backend(execution_backend)
    if not is_tilelang_target(target):
        return target, requested_backend  # type: ignore[return-value]

    assert isinstance(target, Target)
    embedded_backend = target.execution_backend
    if requested_backend not in (None, "auto") and embedded_backend not in (None, "auto") and requested_backend != embedded_backend:
        raise ValueError(
            f"Conflicting execution backend requests: target has {embedded_backend!r}, compile argument has {requested_backend!r}"
        )
    return target.to_tvm_target_input(), requested_backend or embedded_backend
