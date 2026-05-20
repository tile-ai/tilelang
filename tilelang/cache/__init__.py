"""The cache utils with class and database persistence - Init file"""

from __future__ import annotations

import logging
import json
from hashlib import sha256
from typing import TYPE_CHECKING, Literal
from tvm.target import Target
from tvm.tir import PrimFunc
from tilelang.jit import JITKernel
from tilelang import env
from tilelang.jit.adapter.cutedsl.kernel_cache import CuTeDSLKernelCache
from tilelang.jit.adapter.cython.kernel_cache import CythonKernelCache
from tilelang.jit.adapter.nvrtc.kernel_cache import NVRTCKernelCache
from tilelang.jit.adapter.torch.kernel_cache import TorchKernelCache
from tilelang.jit.adapter.kernel_cache import TVMFFIKernelCache

if TYPE_CHECKING:
    from .kernel_cache import KernelCache

# Create a map of singleton instance of KernelCaches
_dispatch_map: dict[str, KernelCache] = {
    "tvm_ffi": TVMFFIKernelCache(),
    "cython": CythonKernelCache(),
    "nvrtc": NVRTCKernelCache(),
    "cutedsl": CuTeDSLKernelCache(),
    "torch": TorchKernelCache(),
}


def _normalize_for_json(value):
    if isinstance(value, dict):
        return {str(k): _normalize_for_json(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _resolve_cache_dispatch(
    target: str | Target | None,
    execution_backend: Literal["auto", "tvm_ffi", "cython", "nvrtc", "torch", "cutedsl"] | None,
    verbose: bool | None,
):
    if target is None:
        target = env.get_default_target()
    if execution_backend is None:
        execution_backend = env.get_default_execution_backend()
    if verbose is None:
        verbose = env.get_default_verbose()

    from tilelang.utils.target import determine_target as _determine_target
    from tilelang.jit.execution_backend import resolve_execution_backend, allowed_backends_for_target

    norm_target = Target(_determine_target(target)) if isinstance(target, str) else target
    requested_backend = execution_backend
    resolved_backend = resolve_execution_backend(requested_backend, norm_target)
    if verbose:
        allowed_now = allowed_backends_for_target(norm_target, include_unavailable=False)
        if requested_backend in (None, "auto") or requested_backend != resolved_backend:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            logger.info(
                "Execution backend resolved -> '%s' (requested='%s', target='%s', allowed: %s)",
                resolved_backend,
                requested_backend,
                norm_target.kind.name,
                ", ".join(sorted(allowed_now)),
            )
    if resolved_backend not in _dispatch_map:
        raise ValueError(f'Cannot find support for execution backend "{resolved_backend}"')
    return _dispatch_map[resolved_backend], norm_target, resolved_backend, verbose


def _make_frontend_cache_key(
    frontend_key_data: dict,
    *,
    target: Target,
    target_host: str | Target | None,
    execution_backend: str,
    out_idx: list[int] | int | None,
    pass_configs: dict | None,
    compile_flags: list[str] | str | None,
) -> str:
    key_data = {
        "frontend": _normalize_for_json(frontend_key_data),
        "out_idx": _normalize_for_json(out_idx),
        "target": str(target),
        "target_host": str(target_host) if target_host else None,
        "execution_backend": execution_backend,
        "pass_configs": _normalize_for_json(pass_configs),
        "compile_flags": _normalize_for_json(compile_flags),
    }
    key_string = json.dumps(key_data, sort_keys=True)
    return sha256(key_string.encode()).hexdigest()


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
    cache, norm_target, execution_backend, verbose = _resolve_cache_dispatch(target, execution_backend, verbose)
    return cache.cached(
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


def load_frontend_cached(
    frontend_key_data: dict,
    *,
    out_idx: list[int] | int | None = None,
    target: str | Target | None = None,
    target_host: str | Target | None = None,
    execution_backend: Literal["auto", "tvm_ffi", "cython", "nvrtc", "torch", "cutedsl"] | None = None,
    verbose: bool | None = None,
    pass_configs: dict | None = None,
    compile_flags: list[str] | str | None = None,
) -> JITKernel | None:
    cache, norm_target, execution_backend, verbose = _resolve_cache_dispatch(target, execution_backend, verbose)
    frontend_key = _make_frontend_cache_key(
        frontend_key_data,
        target=norm_target,
        target_host=target_host,
        execution_backend=execution_backend,
        out_idx=out_idx,
        pass_configs=pass_configs,
        compile_flags=compile_flags,
    )
    return cache.load_frontend_cached(
        frontend_key,
        target=norm_target,
        target_host=target_host,
        out_idx=out_idx,
        execution_backend=execution_backend,
        verbose=verbose,
        pass_configs=pass_configs,
        compile_flags=compile_flags,
    )


def store_frontend_cache(
    frontend_key_data: dict,
    kernel_key: str,
    *,
    out_idx: list[int] | int | None = None,
    target: str | Target | None = None,
    target_host: str | Target | None = None,
    execution_backend: Literal["auto", "tvm_ffi", "cython", "nvrtc", "torch", "cutedsl"] | None = None,
    verbose: bool | None = None,
    pass_configs: dict | None = None,
    compile_flags: list[str] | str | None = None,
) -> None:
    cache, norm_target, execution_backend, verbose = _resolve_cache_dispatch(target, execution_backend, verbose)
    frontend_key = _make_frontend_cache_key(
        frontend_key_data,
        target=norm_target,
        target_host=target_host,
        execution_backend=execution_backend,
        out_idx=out_idx,
        pass_configs=pass_configs,
        compile_flags=compile_flags,
    )
    cache.store_frontend_cache(frontend_key, kernel_key, verbose=verbose)


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
