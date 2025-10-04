"""Tenstorrent lowering entry point.

This module provides a stub implementation that wires the Tenstorrent target
into TileLang's lowering flow. The real lowering pipeline will be added in
subsequent tickets.
"""

from __future__ import annotations

from typing import List, Optional, Union

from tvm.target import Target

from tilelang import tvm as tvm
from tilelang.engine.param import CompiledArtifact, KernelParam


def lower(
    mod: tvm.IRModule,
    params: Optional[List[KernelParam]],
    target: Union[str, Target],
    target_host: Optional[Union[str, Target]],
    *,
    runtime_only: bool,
    enable_host_codegen: bool,
    enable_device_compile: bool,
) -> CompiledArtifact:
    """Lower the given module for the Tenstorrent backend.

    The concrete lowering pipeline will be filled in by later workstreams. For now
    we return a placeholder ``CompiledArtifact`` so that callers can exercise the
    Tenstorrent code path without tripping assertions in the shared engine code.
    """

    # TODO(tenstorrent): replace placeholder once the TT lowering pipeline is implemented.
    _ = (target, target_host, runtime_only, enable_host_codegen, enable_device_compile)

    host_mod = tvm.IRModule()
    device_mod = tvm.IRModule()
    kernel_source = "// Tenstorrent backend lowering not yet implemented\n"
    return CompiledArtifact(host_mod, device_mod, params or [], kernel_source)
