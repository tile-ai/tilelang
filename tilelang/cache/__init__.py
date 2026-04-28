"""The cache utils with class and database persistence - Init file"""

from __future__ import annotations

import logging
from typing import Literal
from tvm.target import Target
from tvm.tir import PrimFunc
from tilelang.jit import JITKernel
from tilelang import env


def _build_dispatch_map():
    from tilelang.backend.registry import registered_device_backends

    dispatch = {}
    for backend in registered_device_backends():
        for spec in backend.execution_specs:
            if spec.name in dispatch:
                continue
            dispatch[spec.name] = spec.cache_factory()
    return dispatch


_dispatch_map = _build_dispatch_map()


def cached(
    func: PrimFunc = None,
    out_idx: list[int] = None,
    *args,
    target: str | Target | None = None,
    target_host: str | Target | None = None,
    execution_backend: Literal["auto", "tvm_ffi", "cython", "nvrtc", "torch", "cutedsl"] | None = None,
    verbose: bool | None = None,
    pass_configs: dict | None = None,
    compile_flags: list[str] | str | None = None,
) -> JITKernel:
    """
    Caches and reuses compiled kernels (using KernelCache class).
    """
    # Apply environment variable defaults if parameters are not explicitly set
    # This is the SINGLE source of truth for env var processing
    if target is None:
        target = env.get_default_target()
    if execution_backend is None:
        execution_backend = env.get_default_execution_backend()
    if verbose is None:
        verbose = env.get_default_verbose()

    # Normalize target and resolve execution backend before proceeding
    from tilelang.utils.target import determine_target as _determine_target
    from tilelang.jit.execution_backend import resolve_execution_backend, allowed_backends_for_target

    norm_target = Target(_determine_target(target)) if isinstance(target, str) else target
    requested_backend = execution_backend
    execution_backend = resolve_execution_backend(requested_backend, norm_target)
    if verbose:
        allowed_now = allowed_backends_for_target(norm_target, include_unavailable=False)
        # Avoid duplicate logs when caller already resolved explicitly
        if requested_backend in (None, "auto") or requested_backend != execution_backend:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            logger.info(
                "Execution backend resolved -> '%s' (requested='%s', target='%s', allowed: %s)",
                execution_backend,
                requested_backend,
                norm_target.kind.name,
                ", ".join(sorted(allowed_now)),
            )
    if execution_backend not in _dispatch_map:
        raise ValueError(f'Cannot find support for execution backend "{execution_backend}"')

    kernel_cache = _dispatch_map[execution_backend]
    return kernel_cache.cached(
        func,
        out_idx,
        *args,
        target=norm_target,
        target_host=target_host,
        execution_backend=execution_backend,
        verbose=verbose,
        pass_configs=pass_configs,
        compile_flags=compile_flags,
    )


def clear_cache():
    """
    Disabled helper that previously removed the entire kernel cache.

    Raises:
        RuntimeError: Always raised to warn users to clear the cache manually.
    """
    cache_dir = env.TILELANG_CACHE_DIR
    raise RuntimeError(
        "tilelang.clear_cache() is disabled because deleting the cache directory "
        "is dangerous. If you accept the risk, remove it manually with "
        f"`rm -rf '{cache_dir}'`."
    )


if env.TILELANG_CLEAR_CACHE.lower() in ("1", "true", "yes", "on"):
    clear_cache()
