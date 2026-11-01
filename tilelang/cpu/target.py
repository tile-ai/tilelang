"""Intel CPU target helpers."""

from __future__ import annotations

import platform

from tvm.target import Target


def amx_target(kind: str = "c") -> Target:
    """Return a CPU target explicitly opting in to Intel AMX lowering.

    AMX is currently emitted by TileLang's C source backend.  Keeping the
    feature explicit makes compiled artifacts safe to cache and prevents an
    AMX binary from being selected merely because the compiler host has AMX.
    """
    if kind != "c":
        raise ValueError("Intel AMX lowering currently requires the TileLang 'c' target")
    return Target({"kind": "c", "keys": ["cpu", "amx"]})


def host_has_amx() -> bool:
    """Best-effort Linux host capability check used by tests and examples."""
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        return False
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            flags = cpuinfo.read().lower()
    except OSError:
        return False
    return all(flag in flags for flag in ("amx_tile", "amx_bf16", "amx_int8", "avx512_vbmi"))
