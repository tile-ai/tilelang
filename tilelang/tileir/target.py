"""TileIR target normalization, registered with the backend target registry.

A TileIR target is a CUDA target carrying the ``"tileir"`` key. Users select it via
``"tileir -arch=sm_120"`` (CLI style), ``{"kind": "tileir", "arch": "sm_120"}``, or a
``Target`` that already carries the key; all normalize to a CUDA ``Target`` + key so the
rest of the stack treats it as CUDA while the TileIR execution backend claims it.
"""

from __future__ import annotations

import shlex

from tvm.target import Target

from tilelang.backend.target import TargetConfig, TargetLike, register_target_normalizer


def _with_tileir_key(target: Target | str) -> Target:
    if not isinstance(target, Target):
        target = Target(target)
    target_dict = dict(target.export())
    target_dict["keys"] = list(dict.fromkeys([*target_dict.get("keys", ()), "tileir"]))
    return Target(target_dict)


def _parse_cli_style_target(target: str, kind: str) -> TargetConfig | None:
    try:
        parts = shlex.split(target)
    except ValueError as err:
        if target.startswith(kind):
            raise AssertionError(f"TileIR target {target!r} is not supported.") from err
        return None
    if not parts or parts[0] != kind:
        return None

    target_dict: TargetConfig = {"kind": kind}
    index = 1
    while index < len(parts):
        token = parts[index]
        if not token.startswith("-"):
            return None

        option = token.lstrip("-")
        if not option:
            return None
        if "=" in option:
            name, value = option.split("=", 1)
        else:
            index += 1
            if index >= len(parts):
                return None
            name, value = option, parts[index]
        if not name:
            return None
        target_dict[name.replace("-", "_")] = value
        index += 1
    return target_dict


def normalize_tileir_target(target: TargetLike) -> Target | None:
    if isinstance(target, Target):
        if target.kind.name == "cuda" and "tileir" in target.keys:
            return target
        return None

    if isinstance(target, dict):
        if target.get("kind") == "tileir":
            cuda_target = dict(target)
            cuda_target["kind"] = "cuda"
            try:
                return _with_tileir_key(Target(cuda_target))
            except Exception as err:
                raise AssertionError(
                    f"TileIR target {target!r} is not supported. Pass a CUDA architecture, e.g. `{{'kind': 'tileir', 'arch': 'sm_120'}}`."
                ) from err
        try:
            temp_target = Target(target)
        except Exception:
            return None
        if temp_target.kind.name == "cuda" and "tileir" in temp_target.keys:
            return temp_target
        return None

    target_dict = _parse_cli_style_target(target.strip(), "tileir")
    if target_dict is not None:
        cuda_target = dict(target_dict)
        cuda_target["kind"] = "cuda"

        try:
            return _with_tileir_key(Target(cuda_target))
        except Exception as err:
            raise AssertionError(f"TileIR target {target!r} is not supported. Pass target options as `tileir -arch=sm_120`.") from err

    return None


register_target_normalizer("tileir", normalize_tileir_target, override=True)
