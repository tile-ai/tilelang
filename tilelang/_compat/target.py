"""Normalize TileLang target inputs for TIRx's typed target schema."""

from __future__ import annotations

import json
import shlex
from collections.abc import Mapping

from tvm_tirx.target import Target, TargetKind


def _option_value(value: str, schema: str):
    if schema == "int":
        return int(value)
    if schema == "bool":
        if value.lower() not in ("true", "false", "1", "0"):
            raise ValueError(f"Expected a boolean target option, got {value!r}")
        return value.lower() in ("true", "1")
    if schema.startswith("Array<"):
        item_schema = schema[len("Array<") : -1]
        return [_option_value(item, item_schema) for item in value.split(",")] if value else []
    if schema.startswith("Map<"):
        return json.loads(value)
    if schema == "target.Target":
        return normalize_target(value)
    return value


def normalize_target(target: str | Mapping[str, object] | Target, *, host=None) -> Target:
    """Return a native Target, preserving legacy CLI options without dropping any.

    CLI forms such as ``cuda -arch=sm_90 -max_num_threads=256`` are converted to
    schema-typed dictionaries before entering TIRx. Unknown keys are errors;
    TileLang pass settings must not be smuggled into target attributes.
    """
    if isinstance(target, Target):
        return target if host is None else Target(target, host=normalize_target(host))
    if isinstance(target, Mapping):
        config = dict(target)
        if config.get("kind") == "hip":
            config["kind"] = "rocm"
        if "host" in config:
            config["host"] = normalize_target(config["host"])
    elif isinstance(target, str):
        text = target.strip()
        if not text:
            raise ValueError("A target cannot be empty")
        if text.startswith("{"):
            config = json.loads(text)
            if not isinstance(config, dict):
                raise ValueError("A target JSON value must be an object")
            return normalize_target(config, host=host)
        tokens = shlex.split(text)
        kind = "rocm" if tokens[0] == "hip" else tokens[0]
        if len(tokens) == 1:
            return Target(kind, host=normalize_target(host) if host is not None else None)
        config = {"kind": kind}
        schema = dict(TargetKind.options_from_name(kind))
        if kind == "cuda":
            schema["code"] = "Array<str>"
        index = 1
        while index < len(tokens):
            option = tokens[index]
            if not option.startswith("-"):
                raise ValueError(f"Expected a target option, got {option!r}")
            key, separator, value = option.lstrip("-").partition("=")
            if not key or key not in schema:
                raise ValueError(f"Unknown {tokens[0]} target option: {key!r}")
            if key in config:
                raise ValueError(f"Duplicate target option: {key!r}")
            if not separator:
                index += 1
                if index == len(tokens):
                    raise ValueError(f"Missing value for target option {key!r}")
                value = tokens[index]
            config[key] = _option_value(value, schema[key])
            index += 1
    else:
        raise TypeError(f"Expected a target string, mapping, or Target, got {type(target).__name__}")
    if config.get("kind") == "cuda" and "code" in config:
        from tilelang.contrib.nvcc import get_target_code_list

        codes = get_target_code_list(config.pop("code"))
        backend_config = dict(config.get("backend_config", {}))
        tilelang_config = dict(backend_config.get("tilelang", {}))
        if "code" in tilelang_config and list(tilelang_config["code"]) != codes:
            raise ValueError("Conflicting CUDA code and backend_config.tilelang.code")
        tilelang_config["code"] = codes
        backend_config["tilelang"] = tilelang_config
        config["backend_config"] = backend_config
    return Target(config, host=normalize_target(host) if host is not None else None)


def native_backend_target(target: Target) -> Target:
    """Remove TileLang-owned toolchain options before native backend resolution."""
    backend_config = dict(target.attrs.get("backend_config", {}))
    if "tilelang" not in backend_config:
        return target
    backend_config.pop("tilelang")
    config = dict(target.export())
    config["backend_config"] = backend_config
    return Target(config, host=target.host)
