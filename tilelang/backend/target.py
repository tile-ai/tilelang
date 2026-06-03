from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import json
import shlex
from types import MappingProxyType
from typing import Any, TypeAlias

TargetInput: TypeAlias = str | Mapping[str, object] | object
TargetNormalizer = Callable[["Target"], TargetInput]
TargetPresetResolver = Callable[["Target"], TargetInput]
TargetDetector = Callable[[], TargetInput | None]
TargetCanonicalizer = Callable[[dict[str, object]], Mapping[str, object]]
TargetOptionSpec: TypeAlias = "TargetOption | type | tuple[type, ...] | Callable[[object], object]"

_TARGET_KINDS: dict[str, TargetKind] = {}
_TARGET_TAGS: dict[str, TargetPresetResolver] = {}
_TARGET_PRESETS = _TARGET_TAGS
_EXECUTION_BACKEND_ALIASES = {
    "dlpack": "tvm_ffi",
}
_MISSING = object()


def _tvm_target_cls():
    from tvm.target import Target as TVMTarget

    return TVMTarget


def _is_tvm_target(target: object) -> bool:
    try:
        return isinstance(target, _tvm_target_cls())
    except Exception:
        return False


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


def _normalize_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = value.split(",") if "," in value else [value]
    elif isinstance(value, Sequence):
        values = value
    else:
        raise TypeError(f"Expected a string list, got {type(value).__name__}")
    return [str(item) for item in values if str(item)]


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


def _validate_type(option: str, value: object, expected_type: type | tuple[type, ...]) -> object:
    if expected_type is int and isinstance(value, bool):
        raise TypeError(f"Option {option!r}: expected type 'int' but got 'bool'")
    if expected_type is bool and not isinstance(value, bool):
        raise TypeError(f"Option {option!r}: expected type 'bool' but got {type(value).__name__!r}")
    if not isinstance(value, expected_type):
        if expected_type is int and isinstance(value, str):
            try:
                return int(value, 0)
            except ValueError as err:
                raise TypeError(f"Option {option!r}: expected type 'int' but got {value!r}") from err
        if expected_type is bool and isinstance(value, str):
            lowered = value.lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
        if expected_type is str:
            return str(value)
        expected = getattr(expected_type, "__name__", str(expected_type))
        raise TypeError(f"Option {option!r}: expected type {expected!r} but got {type(value).__name__!r}")
    return value


@dataclass(frozen=True, slots=True)
class TargetOption:
    """One TileLang target option declaration.

    This is the Python equivalent of TVM's TargetKind `add_attr_option`: it
    validates an option, optionally materializes a default, and can run a small
    custom coercion function before the target kind canonicalizer runs.
    """

    name: str | None = None
    expected_type: type | tuple[type, ...] | None = None
    default: object = _MISSING
    required: bool = False
    validator: Callable[[object], object] | None = None
    description: str | None = None

    def with_name(self, name: str) -> TargetOption:
        if self.name == name:
            return self
        return TargetOption(
            name=name,
            expected_type=self.expected_type,
            default=self.default,
            required=self.required,
            validator=self.validator,
            description=self.description,
        )

    @property
    def has_default(self) -> bool:
        return self.default is not _MISSING

    def resolve(self, value: object) -> object:
        assert self.name is not None
        if self.validator is not None:
            value = self.validator(value)
        if self.expected_type is not None:
            value = _validate_type(self.name, value, self.expected_type)
        return value


def target_option(
    expected_type: type | tuple[type, ...] | None = None,
    *,
    default: object = _MISSING,
    required: bool = False,
    validator: Callable[[object], object] | None = None,
    description: str | None = None,
) -> TargetOption:
    return TargetOption(
        expected_type=expected_type,
        default=default,
        required=required,
        validator=validator,
        description=description,
    )


def _coerce_option(name: str, spec: TargetOptionSpec) -> TargetOption:
    if isinstance(spec, TargetOption):
        return spec.with_name(name)
    if isinstance(spec, type):
        return TargetOption(name=name, expected_type=spec)
    if isinstance(spec, tuple) and spec and all(isinstance(item, type) for item in spec):
        return TargetOption(name=name, expected_type=spec)
    if callable(spec):
        return TargetOption(name=name, validator=spec)
    raise TypeError(f"Unsupported target option declaration for {name!r}: {spec!r}")


_COMMON_OPTIONS: dict[str, TargetOption] = {
    "kind": target_option(str).with_name("kind"),
    "keys": target_option(validator=_normalize_string_list).with_name("keys"),
    "tag": target_option(str).with_name("tag"),
    "device": target_option(str).with_name("device"),
    "model": target_option(str).with_name("model"),
    "libs": target_option(validator=_normalize_string_list).with_name("libs"),
    "host": target_option().with_name("host"),
    "from_device": target_option(int).with_name("from_device"),
    "target_device_type": target_option(int).with_name("target_device_type"),
}


@dataclass(frozen=True, slots=True)
class TargetConfigSchema:
    """TVM ConfigSchema semantics ported to TileLang Python."""

    options: Mapping[str, TargetOption] = field(default_factory=dict)
    canonicalizer: TargetCanonicalizer | None = None
    error_on_unknown: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))

    def resolve(self, config: Mapping[str, object]) -> dict[str, object]:
        saved_features = {key: value for key, value in config.items() if str(key).startswith("feature.")}
        working = {key: value for key, value in config.items() if key not in saved_features}
        result: dict[str, object] = {}

        for name, option in self.options.items():
            if name in working:
                result[name] = option.resolve(working[name])
            elif option.has_default:
                result[name] = option.default
            elif option.required:
                raise ValueError(f"Missing required target option {name!r}")

        if self.error_on_unknown:
            unknown = [name for name in working if name not in self.options]
            if unknown:
                known = ", ".join(repr(name) for name in self.options)
                raise ValueError(f"Unknown target option {unknown[0]!r}. Known options: {known}")
        else:
            for name, value in working.items():
                result.setdefault(name, value)

        if self.canonicalizer is not None:
            result = dict(self.canonicalizer(dict(result)))

        for name, value in saved_features.items():
            result.setdefault(name, value)
        return result


def _make_schema(
    options: Mapping[str, TargetOptionSpec],
    canonicalizer: TargetCanonicalizer | None,
    error_on_unknown: bool | None,
) -> TargetConfigSchema:
    merged_options = dict(_COMMON_OPTIONS)
    merged_options.update({name: _coerce_option(name, spec) for name, spec in options.items()})
    if error_on_unknown is None:
        error_on_unknown = bool(options)
    return TargetConfigSchema(merged_options, canonicalizer=canonicalizer, error_on_unknown=error_on_unknown)


@dataclass(frozen=True, slots=True)
class TargetKind:
    """TileLang-owned target kind metadata.

    The shape intentionally mirrors TVM TargetKind: a kind has default device
    type/keys, a schema, and an optional canonicalizer.  Unlike TVM, registering
    a TileLang TargetKind is pure Python and does not require the kind to exist
    in TVM.  `tvm_kind` and `normalize` are only used by the current legacy
    TVM adapter boundary.
    """

    name: str
    tvm_kind: str | None = None
    default_device_type: int | None = None
    default_attrs: Mapping[str, object] = field(default_factory=dict)
    default_keys: tuple[str, ...] = ()
    options: Mapping[str, TargetOptionSpec] = field(default_factory=dict)
    canonicalizer: TargetCanonicalizer | None = None
    error_on_unknown: bool | None = None
    normalize: TargetNormalizer | None = None
    detect: TargetDetector | None = None
    priority: int = 0
    description: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalize_name(self.name, field_name="TargetKind"))
        object.__setattr__(self, "default_attrs", MappingProxyType(dict(self.default_attrs)))
        object.__setattr__(self, "default_keys", _normalize_keys(self.default_keys))
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))

    @property
    def schema(self) -> TargetConfigSchema:
        return _make_schema(self.options, self.canonicalizer, self.error_on_unknown)

    def export_config(self, target: Target) -> dict[str, object]:
        config = {**dict(self.default_attrs), **dict(target.attrs)}
        config["kind"] = self.name
        if target.tag:
            config["tag"] = target.tag
        if target.keys:
            config["keys"] = list(target.keys)
        if target.device is not None:
            config["device"] = target.device
        if target.host is not None:
            config["host"] = target.host

        resolved = self.schema.resolve(config)
        has_keys = "keys" in resolved
        keys = _normalize_keys(resolved.pop("keys", None))
        if not has_keys:
            keys = self.default_keys
        if "device" in resolved:
            keys = _normalize_keys((*keys, resolved["device"]))

        exported: dict[str, object] = {"kind": str(resolved.pop("kind", self.name))}
        if "tag" in resolved:
            exported["tag"] = resolved.pop("tag")
        if keys:
            exported["keys"] = list(keys)
        if "host" in resolved:
            exported["host"] = resolved.pop("host")
        exported.update(resolved)
        return exported

    def to_legacy_tvm_target_input(self, target: Target) -> TargetInput:
        if self.normalize is not None:
            return self.normalize(target)

        config = self.export_config(target)
        config["kind"] = self.tvm_kind or config["kind"]
        if target.host is not None:
            config["host"] = _target_input_to_tvm(target.host)
        if set(config) == {"kind"} and config["kind"] == target.kind:
            return target.kind
        return config

    def to_tvm_target_input(self, target: Target) -> TargetInput:
        return self.to_legacy_tvm_target_input(target)


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
        tag: str | None = None,
        device: str | None = None,
        **attrs: Any,
    ) -> None:
        parsed_kind, parsed_attrs, parsed_keys, parsed_host, parsed_execution_backend, parsed_tag, parsed_device = self._parse_target(
            target
        )
        if host is not None:
            parsed_host = host
        if tag is not None:
            parsed_tag = tag
        if device is not None:
            parsed_device = device
        if execution_backend is None:
            execution_backend = parsed_execution_backend

        parsed_attrs.update(attrs)
        self.kind = parsed_kind
        self.attrs = MappingProxyType(parsed_attrs)
        self.keys = _normalize_keys((*parsed_keys, *_normalize_keys(keys)))
        self.host = parsed_host
        self.tag = parsed_tag
        self.device = parsed_device
        self.execution_backend = _canon_execution_backend(execution_backend)

    @classmethod
    def _parse_target(
        cls,
        target: TargetInput | Target,
    ) -> tuple[str, dict[str, object], tuple[str, ...], TargetInput | None, str | None, str | None, str | None]:
        if is_tilelang_target(target):
            assert isinstance(target, Target)
            return target.kind, dict(target.attrs), target.keys, target.host, target.execution_backend, target.tag, target.device

        if _is_tvm_target(target):
            return cls._parse_target(target.export())  # type: ignore[attr-defined]

        if isinstance(target, Mapping):
            config = dict(target)
            tag = config.pop("tag", None)
            if "kind" not in config:
                if tag is None:
                    raise ValueError("Target config dict must contain a 'kind' field unless a 'tag' field is provided")
                kind = _normalize_name(str(tag))
            else:
                kind = _normalize_name(str(config.pop("kind")))
            keys = _normalize_keys(config.pop("keys", None))
            host = config.pop("host", None)
            execution_backend = config.pop("execution_backend", None)
            device = config.pop("device", None)
            return (
                kind,
                config,
                keys,
                host,
                _canon_execution_backend(execution_backend),  # type: ignore[arg-type]
                str(tag) if tag is not None else None,
                str(device) if device is not None else None,
            )

        if isinstance(target, str):
            stripped = target.strip()
            if stripped.startswith("{"):
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError as err:
                    raise ValueError(f"Failed to parse target JSON config: {err}") from err
                if not isinstance(parsed, Mapping):
                    raise ValueError(f"Target JSON config must be a dict, got: {target}")
                return cls._parse_target(parsed)
            kind, parsed_attrs, parsed_keys, parsed_host = _parse_target_string(target)
            return _normalize_name(kind), parsed_attrs, parsed_keys, parsed_host, None, None, None

        raise TypeError(f"Unsupported target type: {type(target)!r}")

    def to_config(self) -> dict[str, object]:
        config = {"kind": self.kind, **dict(self.attrs)}
        if self.tag is not None:
            config["tag"] = self.tag
        if self.keys:
            config["keys"] = list(self.keys)
        if self.device is not None:
            config["device"] = self.device
        if self.host is not None:
            config["host"] = self.host.to_config() if is_tilelang_target(self.host) else self.host
        if self.execution_backend is not None:
            config["execution_backend"] = self.execution_backend
        return config

    def resolve(self) -> Target:
        if self.kind == "auto":
            return Target(_detect_auto_target()).resolve()

        resolver = _TARGET_PRESETS.get(self.kind)
        if resolver is not None:
            return Target(resolver(self), execution_backend=self.execution_backend).resolve()

        target_kind = _TARGET_KINDS.get(self.kind)
        if target_kind is not None:
            config = target_kind.export_config(self)
            return Target(config, execution_backend=self.execution_backend)

        return self

    def export(self) -> dict[str, object]:
        resolved = self.resolve()
        if resolved is not self:
            config = resolved.to_config()
            config.pop("execution_backend", None)
            return config
        config = dict(self.attrs)
        config["kind"] = self.kind
        if self.tag is not None:
            config["tag"] = self.tag
        if self.keys:
            config["keys"] = list(self.keys)
        if self.device is not None:
            config["device"] = self.device
        if self.host is not None:
            config["host"] = self.host.to_config() if is_tilelang_target(self.host) else self.host
        return config

    def to_legacy_tvm_target_input(self) -> TargetInput:
        if self.kind == "auto":
            return _target_input_to_legacy_tvm(_detect_auto_target())

        resolver = _TARGET_PRESETS.get(self.kind)
        if resolver is not None:
            return _target_input_to_legacy_tvm(resolver(self))

        target_kind = _TARGET_KINDS.get(self.kind)
        if target_kind is not None:
            return target_kind.to_legacy_tvm_target_input(self)

        return self.export()

    def to_legacy_tvm_target(self) -> object:
        target_input = self.to_legacy_tvm_target_input()
        TVMTarget = _tvm_target_cls()
        target = target_input if isinstance(target_input, TVMTarget) else TVMTarget(target_input)
        try:
            from tilelang.utils.target import with_rocm_target_attrs

            return with_rocm_target_attrs(target)
        except Exception:
            return target

    def to_tvm_target_input(self) -> TargetInput:
        return self.to_legacy_tvm_target_input()

    def to_tvm_target(self) -> object:
        return self.to_legacy_tvm_target()

    def __repr__(self) -> str:
        args = [repr(self.kind)]
        if self.execution_backend is not None:
            args.append(f"execution_backend={self.execution_backend!r}")
        if self.tag is not None:
            args.append(f"tag={self.tag!r}")
        if self.keys:
            args.append(f"keys={list(self.keys)!r}")
        if self.device is not None:
            args.append(f"device={self.device!r}")
        if self.host is not None:
            args.append(f"host={self.host!r}")
        args.extend(f"{key}={value!r}" for key, value in self.attrs.items())
        return f"tilelang.Target({', '.join(args)})"


def _target_input_to_legacy_tvm(target: TargetInput | Target) -> TargetInput:
    if is_tilelang_target(target):
        assert isinstance(target, Target)
        return target.to_legacy_tvm_target_input()
    if _is_tvm_target(target):
        return target
    if isinstance(target, Mapping):
        return Target(target).to_legacy_tvm_target_input()
    if isinstance(target, str):
        return Target(target).to_legacy_tvm_target_input()
    raise TypeError(f"Unsupported target type: {type(target)!r}")


def _target_input_to_tvm(target: TargetInput | Target) -> TargetInput:
    return _target_input_to_legacy_tvm(target)


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
    register_target_tag(name, resolver, override=override)


def register_target_tag(name: str, resolver: TargetPresetResolver, *, override: bool = False) -> None:
    normalized = _normalize_name(name, field_name="Target preset")
    if normalized in _TARGET_TAGS and not override:
        raise ValueError(f"Target preset {normalized!r} is already registered")
    _TARGET_TAGS[normalized] = resolver


def list_target_presets() -> tuple[str, ...]:
    return tuple(sorted(_TARGET_PRESETS))


def list_target_tags() -> tuple[str, ...]:
    return tuple(sorted(_TARGET_TAGS))


def resolve_target_execution_backend(
    target: TargetInput | Target | None,
    execution_backend: str | None,
) -> tuple[TargetInput | None, str | None]:
    requested_backend = _canon_execution_backend(execution_backend)
    if not is_tilelang_target(target):
        return target, requested_backend  # type: ignore[return-value]

    assert isinstance(target, Target)
    embedded_backend = target.execution_backend
    if requested_backend not in (None, "auto") and embedded_backend not in (None, "auto") and requested_backend != embedded_backend:
        raise ValueError(
            f"Conflicting execution backend requests: target has {embedded_backend!r}, compile argument has {requested_backend!r}"
        )
    return target.resolve(), requested_backend or embedded_backend
