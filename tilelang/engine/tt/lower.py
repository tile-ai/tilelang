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

    This is a stub implementation that raises NotImplementedError. The concrete
    lowering pipeline will be implemented in future workstreams.

    Args:
        mod: The TVM IRModule to lower
        params: Optional list of kernel parameters
        target: The target (should be Tenstorrent target)
        target_host: Optional host target
        runtime_only: Whether to generate runtime-only code
        enable_host_codegen: Whether to enable host code generation
        enable_device_compile: Whether to enable device compilation

    Returns:
        CompiledArtifact: The compiled artifact

    Raises:
        NotImplementedError: Always raised as this is a stub implementation
    """
    from tilelang.utils.target import TENSTORRENT_TARGET

    # Validate that we're actually targeting Tenstorrent
    target_kind = target.kind.name if isinstance(target, Target) else target
    if target_kind != TENSTORRENT_TARGET:
        raise ValueError(
            f"Tenstorrent lowering called with invalid target: {target_kind}. "
            f"Expected: {TENSTORRENT_TARGET}"
        )

    raise NotImplementedError(
        "Tenstorrent backend lowering is not yet implemented. "
        "This is a stub implementation. The lowering pipeline will be "
        "added in future workstreams."
    )
