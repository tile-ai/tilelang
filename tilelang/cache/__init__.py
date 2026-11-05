"""The cache utils with class and database persistence - Init file"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal
from tvm_tirx.target import Target as TVMTarget
from tvm_tirx.tirx import Function
from tilelang.jit import JITKernel
from tilelang import env
from importlib import import_module
import threading

if TYPE_CHECKING:
    from .kernel_cache import KernelCache

TargetLike = str | dict[str, object] | TVMTarget

# Resolve one cache implementation with its matching runtime adapter. Optional
# backend imports must not become prerequisites for compiling a CPU function.
_CACHE_CLASSES = {
    "tvm_ffi": ("tilelang.jit.adapter.kernel_cache", "TVMFFIKernelCache"),
    "cython": ("tilelang.jit.adapter.cython.kernel_cache", "CythonKernelCache"),
    "nvrtc": ("tilelang.jit.adapter.nvrtc.kernel_cache", "NVRTCKernelCache"),
    "cutedsl": ("tilelang.jit.adapter.cutedsl.kernel_cache", "CuTeDSLKernelCache"),
    "pto": ("tilelang.jit.adapter.cython.kernel_cache", "CythonKernelCache"),
    "torch": ("tilelang.jit.adapter.torch.kernel_cache", "TorchKernelCache"),
}
_ASCEND_CACHE_CLASSES = {
    "tvm_ffi": ("tilelang.ascend.kernel_cache", "AscendTVMFFIKernelCache"),
    "cython": ("tilelang.ascend.kernel_cache", "AscendCythonKernelCache"),
}
_dispatch_map: dict[str, KernelCache] = {}
_ascend_dispatch_map: dict[str, KernelCache] = {}
_dispatch_lock = threading.RLock()


def _load_cache(name: str, *, ascend: bool):
    mapping = _ascend_dispatch_map if ascend else _dispatch_map
    classes = _ASCEND_CACHE_CLASSES if ascend else _CACHE_CLASSES
    with _dispatch_lock:
        if name not in mapping:
            if name not in classes:
                raise ValueError(f'Cannot find support for execution backend "{name}"')
            module_name, class_name = classes[name]
            mapping[name] = getattr(import_module(module_name), class_name)()
        return mapping[name]


def _resolve_cache_dispatch(
    target: TargetLike | None,
    target_host: TargetLike | None,
    execution_backend: Literal["auto", "tvm_ffi", "cython", "nvrtc", "torch", "cutedsl", "pto"] | None,
    verbose: bool | None,
):
    if target is None:
        target = env.get_default_target()
    if execution_backend is None:
        execution_backend = env.get_default_execution_backend()
    if verbose is None:
        verbose = env.get_default_verbose()

    from tilelang.backend.module import create_backend_context

    requested_backend = execution_backend
    context = create_backend_context(target, target_host, requested_backend)
    resolved_backend = context.execution_backend.name
    if verbose:
        allowed_now = context.module.allowed_execution_backends(context.target, include_unavailable=False)
        if requested_backend in (None, "auto") or requested_backend != resolved_backend:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            logger.info(
                "Execution backend resolved -> '%s' (requested='%s', target='%s', allowed: %s)",
                resolved_backend,
                requested_backend,
                context.target.kind.name,
                ", ".join(sorted(allowed_now)),
            )
    # Dispatch on the resolved backend module rather than on the target kind.
    cache = _load_cache(resolved_backend, ascend=context.module.name == "ascend")
    return cache, context, verbose


def cached(
    func: Function = None,
    out_idx: list[int] = None,
    *args,
    target: TargetLike | None = None,
    target_host: TargetLike | None = None,
    execution_backend: Literal["auto", "tvm_ffi", "cython", "nvrtc", "torch", "cutedsl", "pto"] | None = None,
    verbose: bool | None = None,
    pass_configs: dict | None = None,
    compile_flags: list[str] | str | None = None,
) -> JITKernel:
    """
    Caches and reuses compiled kernels (using KernelCache class).
    """
    cache, backend_context, verbose = _resolve_cache_dispatch(target, target_host, execution_backend, verbose)
    return cache.cached(
        func,
        out_idx,
        *args,
        backend_context=backend_context,
        verbose=verbose,
        pass_configs=pass_configs,
        compile_flags=compile_flags,
    )
