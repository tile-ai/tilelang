"""Opt-in static checking of a selected CTA with the external PipeGuard package.

Install the pinned checkout with ``pip install ./3rdparty/pipeguard`` (Python
3.10+). Importing TileLang does not import or require PipeGuard.
"""

from __future__ import annotations

from typing import Any


def _pipeguard():
    try:
        from pipeguard.integration import check_tilelang_ir
        from pipeguard.capture.tilelang_host import host_descriptors
    except ImportError as error:
        raise ImportError(
            "PipeGuard is optional and requires Python 3.10+. From a source checkout, "
            "run `git submodule update --init 3rdparty/pipeguard` followed by "
            "`python -m pip install ./3rdparty/pipeguard`."
        ) from error
    return check_tilelang_ir, host_descriptors


def check(
    kernel: Any,
    *,
    launch_grid: tuple[int, ...] | list[int],
    selected_cta: tuple[int, int, int] = (0, 0, 0),
    target: Any = None,
    pass_configs: dict[str, Any] | None = None,
    scalar_values: dict[str, int] | None = None,
    pointer_table_allocations: dict[str, list[str]] | None = None,
    buffer_values: dict[str, list[int]] | None = None,
    descriptors: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Check a PrimFunc, compiled JITKernel, or CompiledArtifact without launching it.

    ``launch_grid`` and bindings must describe the actual invocation. Only
    ``selected_cta`` is analyzed. A PrimFunc is lowered with ``target`` and
    ``pass_configs``; an existing artifact is checked as compiled, without
    re-lowering. Cached kernels without a retained artifact must be checked
    through their original PrimFunc instead.

    Exact TMA shapes and strides are recovered from the artifact's host IR
    where possible. Explicit descriptors may supplement that evidence, but
    conflicting evidence is rejected. Missing evidence stays UNSUPPORTED.
    ``timeout`` covers capture and checking after lowering/serialization.

    The returned dict contains ``verdict``, ``diagnostics`` and ``scope``.
    This is the experimental PipeGuard 0.2 checker: SAFE is a model result,
    not a numerical or final-device-code guarantee. See the integration guide.
    """
    check_ir, host_descriptors = _pipeguard()
    from tilelang import tvm
    from tilelang.engine.param import CompiledArtifact
    from tilelang.transform.pass_config import normalize_pass_configs

    if isinstance(kernel, tvm.tirx.PrimFunc):
        import tilelang

        if not (kernel.attrs and "global_symbol" in kernel.attrs):
            kernel = kernel.with_attr("global_symbol", "main")
        normalized_target = tvm.target.Target({"kind": "cuda", "arch": "sm_90a"} if target is None else target)
        with normalized_target, tvm.transform.PassContext(opt_level=3, config=normalize_pass_configs(pass_configs)):
            artifact = tilelang.lower(
                kernel,
                target=normalized_target,
                enable_host_codegen=False,
                enable_device_compile=False,
            )
    else:
        artifact = kernel if isinstance(kernel, CompiledArtifact) else getattr(kernel, "artifact", None)
        if not isinstance(artifact, CompiledArtifact):
            raise ValueError("Pass a PrimFunc or a compiled kernel with a retained CompiledArtifact; cached kernels may lack IR.")
        if target is not None or pass_configs is not None:
            raise ValueError("target/pass_configs apply only to a PrimFunc; the compiled artifact is checked unchanged")

    recovered = host_descriptors(artifact.host_mod, scalar_values or {})
    for name, descriptor in (descriptors or {}).items():
        if name in recovered and recovered[name] != descriptor:
            raise ValueError(f"Descriptor {name!r} conflicts with the compiled host IR")
        recovered[name] = descriptor
    return check_ir(
        tvm.ir.save_json(artifact.device_mod),
        source="tilelang.contrib.pipeguard.check",
        launch_grid=launch_grid,
        selected_cta=selected_cta,
        scalar_values=scalar_values,
        pointer_table_allocations=pointer_table_allocations,
        buffer_values=buffer_values,
        descriptors=recovered,
        timeout=timeout,
    )
