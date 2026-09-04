"""CUDA include-path discovery for NVRTC compilation."""

from __future__ import annotations

import os.path as osp
import platform
import sys


def _target_include_path(cuda_home: str, machine: str | None = None) -> str:
    machine = machine or platform.machine()
    target_arch = "sbsa-linux" if machine in ("aarch64", "arm64") else "x86_64-linux"
    return osp.join(cuda_home, "targets", target_arch, "include")


def discover_cuda_include_paths(cuda_home: str, machine: str | None = None, system: str | None = None) -> list[str]:
    """Return NVRTC include paths supported by the CUDA layout at ``cuda_home``.

    CUDA pip packages may install headers in a flat ``include`` tree, while
    system toolkits commonly use ``targets/<arch>-linux/include``. Include both
    layouts when present so split or overlaid installations continue to work.
    """
    flat_include = osp.join(cuda_home, "include")
    system = system or sys.platform
    if system.startswith("win32"):
        return [flat_include, osp.join(flat_include, "cccl")]

    target_include = _target_include_path(cuda_home, machine)
    include_paths = []

    for include_path in (flat_include, target_include):
        if osp.isdir(include_path):
            include_paths.append(include_path)
            cccl_include = osp.join(include_path, "cccl")
            if osp.isdir(cccl_include):
                include_paths.append(cccl_include)

    if include_paths:
        return include_paths

    # Preserve the previous Linux search paths when probing an incomplete or
    # not-yet-mounted toolkit. NVRTC will then report the missing headers.
    return [flat_include, target_include, osp.join(target_include, "cccl")]
