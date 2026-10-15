from __future__ import annotations

from platform import mac_ver
import re
import subprocess

from tvm.target import Target

from tilelang.backend.target import TargetLike, register_target_detector, register_target_normalizer


def _target_ffi_api():
    from tilelang import _ffi_api

    return _ffi_api


def check_metal_availability() -> bool:
    mac_release, _, arch = mac_ver()
    if not mac_release:
        return False
    # todo: check torch version?
    return arch == "arm64"


def _parse_major_version(version: str) -> int:
    try:
        return int(version.split(".", 1)[0])
    except (TypeError, ValueError):
        return 0


def _command_stdout(cmd: list[str]) -> str | None:
    try:
        proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def check_metal4_availability() -> bool:
    mac_release, _, arch = mac_ver()
    if arch != "arm64" or _parse_major_version(mac_release) < 26:
        return False

    sdk_version = _command_stdout(["xcrun", "-sdk", "macosx", "--show-sdk-version"])
    if sdk_version is None or _parse_major_version(sdk_version) < 26:
        return False

    cpu_brand = _command_stdout(["sysctl", "-n", "machdep.cpu.brand_string"]) or ""
    match = re.search(r"\bApple M(\d+)\b", cpu_brand)
    return match is not None and int(match.group(1)) >= 5


def _metal_target_config(enable_metal4: bool) -> dict[str, object]:
    keys = ["metal", "gpu"]
    if enable_metal4:
        keys.append("metal4")
    return {"kind": "metal", "keys": keys}


def _detect_metal_target() -> Target | str | None:
    if check_metal_availability():
        return Target(_metal_target_config(check_metal4_availability()))
    return None


def normalize_metal_target(target: TargetLike) -> Target | None:
    if isinstance(target, Target):
        if target.kind.name == "metal":
            return target
        return None
    if isinstance(target, dict):
        if target.get("kind") != "metal":
            return None
        return Target(target)
    if target.strip() != "metal":
        return None
    return Target(_metal_target_config(check_metal4_availability()))


def target_is_metal(target: Target) -> bool:
    return _target_ffi_api().TargetIsMetal(target)


def target_metal_supports_metal4(target: Target) -> bool:
    return _target_ffi_api().TargetMetalSupportsMetal4(target)


register_target_detector("metal", _detect_metal_target, override=True)
register_target_normalizer("metal", normalize_metal_target, override=True)
