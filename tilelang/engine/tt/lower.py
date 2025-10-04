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

    This is a stub implementation. It validates the target and then raises
    NotImplementedError, since the actual lowering pipeline is not yet implemented.
    The concrete lowering pipeline will be implemented in future workstreams.

    Args:
        mod: The TVM IRModule to lower (unused in stub)
        params: Optional list of kernel parameters (unused in stub)
        target: The target (should be Tenstorrent target)
        target_host: Optional host target (unused in stub)
        runtime_only: Whether to generate runtime-only code (unused in stub)
        enable_host_codegen: Whether to enable host code generation (unused in stub)
        enable_device_compile: Whether to enable device compilation (unused in stub)

    Raises:
        ValueError: If the target is not a Tenstorrent target
        NotImplementedError: This stub implementation always raises this exception
            instead of returning a CompiledArtifact
    """
    from tilelang.engine.lower import get_target_kind
    from tilelang.utils.target import TENSTORRENT_TARGET

    # Unused parameters in this stub implementation - will be used in full implementation
    _ = mod
    _ = params
    _ = target_host
    _ = runtime_only
    _ = enable_host_codegen
    _ = enable_device_compile

    # Validate that we're actually targeting Tenstorrent
    target_kind = get_target_kind(target)
    if target_kind != TENSTORRENT_TARGET:
        raise ValueError(f"Tenstorrent lowering called with invalid target: {target_kind}. "
                         f"Expected: {TENSTORRENT_TARGET}")

    raise NotImplementedError("Tenstorrent backend lowering is not yet implemented. "
                              "This is a stub implementation. The lowering pipeline will be "
                              "added in future workstreams.")
