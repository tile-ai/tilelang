"""Execution backend policy value type."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tvm.target import Target


def normalize_execution_alias(target, execution_backend, pass_configs):
    """Translate retired public execution names before cache-key construction."""
    from tilelang import env

    name = execution_backend if execution_backend is not None else env.get_default_execution_backend()
    name = str(name).lower()
    if name not in ("nvrtc", "cutedsl"):
        return target, execution_backend, pass_configs

    import warnings
    from tilelang.backend.target import determine_target

    target = determine_target(target if target is not None else env.get_default_target(), return_object=True)
    if target.kind.name != "cuda" or (name == "nvrtc" and "cutedsl" in target.keys):
        raise ValueError(f"execution_backend={name!r} requires a compatible CUDA target")
    if name == "cutedsl":
        from tilelang.cuda.target import _with_cutedsl_key

        target = _with_cutedsl_key(target)
        replacement = "target='cutedsl', execution_backend='tvm_ffi'"
    else:
        pass_configs = dict(pass_configs or {})
        if pass_configs.get("tl.cuda_compiler", "nvrtc") != "nvrtc":
            raise ValueError("execution_backend='nvrtc' conflicts with tl.cuda_compiler")
        pass_configs["tl.cuda_compiler"] = "nvrtc"
        replacement = "execution_backend='tvm_ffi', pass_configs={'tl.cuda_compiler': 'nvrtc'}"
    warnings.warn(
        f"execution_backend={name!r} is deprecated; use {replacement}. The legacy wrapper is no longer used.",
        DeprecationWarning,
        stacklevel=3,
    )
    return target, "tvm_ffi", pass_configs


TargetPredicate = Callable[[Target], bool]
AvailabilityCheck = Callable[[], bool]


def _always_available() -> bool:
    return True


@dataclass(frozen=True, slots=True)
class ExecutionBackendSpec:
    name: str
    is_available: AvailabilityCheck = _always_available
    supports_target: TargetPredicate | None = None
    enable_host_codegen: bool = False
    enable_device_compile: bool = False
    # Declares that this backend's host codegen lowers the TVM-FFI
    # callee-allocated-output result slot and that its runtime provides an
    # environment tensor allocator, so kernels with out_idx may allocate
    # their outputs inside the generated host function.
    supports_callee_allocated_outputs: bool = False

    def matches(self, target: Target) -> bool:
        return True if self.supports_target is None else self.supports_target(target)
