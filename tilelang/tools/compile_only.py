"""Compile-only tool: emit inspectable ``lower()`` kernel source without a GPU.

Usage::

    python3 -I -m tilelang.tools.compile_only --output_file out.s example.py

The default target is ``c`` (CPU C). CUDA is optional::

    python3 -I -m tilelang.tools.compile_only --target cuda --output_file out.s example.py

Programmatic API::

    from tilelang.tools.compile_only import compile_kernel_source

    source = compile_kernel_source(func)  # default target c
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

__all__ = [
    "DEFAULT_TARGET",
    "cli_main",
    "compile_kernel_source",
    "discover_prim_func",
    "load_example",
    "resolve_target",
]

# D3: explicit CPU C. Never "auto". Bare "cuda" still runs the CUDA normalizer
# (torch.cuda probe); pin an arch so optional --target cuda does not device-detect.
DEFAULT_TARGET = "c"
_PINNED_CUDA_TARGET: dict[str, str] = {"kind": "cuda", "arch": "sm_80"}
_ERROR_PREFIX = "tilelang compile-only error"


def resolve_target(target: str) -> str | dict[str, str]:
    """Map a CLI/library target string to an explicit lower() target.

    Parameters
    ----------
    target : str
        User-facing target. ``auto`` is rejected. ``cuda`` is pinned to sm_80.

    Returns
    -------
    str or dict
        A TVM target string, or a pinned CUDA target dict.
    """
    normalized = target.strip()
    key = normalized.lower()
    if not normalized or key == "auto":
        raise ValueError("target must be explicit; do not use auto")
    if key == "cuda":
        return dict(_PINNED_CUDA_TARGET)
    return normalized


def load_example(path: str) -> ModuleType:
    """Load ``example.py`` as a module (D2). Do not treat it as a launch script."""
    spec = importlib.util.spec_from_file_location("tilelang_compile_only_input", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load input file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def discover_prim_func(module: ModuleType):
    """Return the first ``@tilelang.jit`` / PrimFunc in module order (D2, 古法: one piece)."""
    from tilelang.jit import JITImpl
    from tilelang.language.eager import PrimFunc

    for obj in vars(module).values():
        if isinstance(obj, JITImpl):
            return obj.get_tir()
    for obj in vars(module).values():
        if isinstance(obj, PrimFunc):
            return obj
    raise RuntimeError("no @tilelang.jit kernel or PrimFunc found")


def compile_kernel_source(func, target: str | dict[str, str] = DEFAULT_TARGET) -> str:
    """Lower ``func`` with ``enable_device_compile=False`` and return ``kernel_source`` (D1).

    Parameters
    ----------
    func : PrimFunc or IRModule
        Kernel from ``JITImpl.get_tir()`` or a PrimFunc. No tensor args.
    target : str or dict, optional
        Explicit target. Strings go through :func:`resolve_target`. Default ``c``.

    Returns
    -------
    str
        Non-empty ``CompiledArtifact.kernel_source`` from existing ``lower()``.
    """
    import tilelang
    from tvm.target import Target

    if isinstance(target, str):
        target = resolve_target(target)
    # Match JITKernel._compile_artifact: LayoutInference reads Target.current().
    resolved = target if isinstance(target, Target) else Target(target)
    with tilelang.transform.PassContext(opt_level=3), resolved:
        artifact = tilelang.lower(func, target=resolved, enable_device_compile=False)
    source = artifact.kernel_source
    if not source or not str(source).strip():
        raise RuntimeError("lower produced empty kernel_source")
    return str(source)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tilelang.tools.compile_only",
        description="Compile a TileLang example to inspectable kernel source without a GPU.",
    )
    parser.add_argument("input_file", help="Compile-only example.py (no kernel launch)")
    parser.add_argument("--output_file", required=True, help="Destination for generated kernel source")
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help="Explicit target (default: c). CUDA is optional: --target cuda",
    )
    return parser


def cli_main(argv: list[str] | None = None) -> int:
    """CE-shaped CLI (D2): ``--output_file`` + example.py. Default target ``c`` (D3)."""
    args = _build_parser().parse_args(argv)

    try:
        module = load_example(args.input_file)
        func = discover_prim_func(module)
        source = compile_kernel_source(func, resolve_target(args.target))
        Path(args.output_file).write_text(source)
    except Exception as exc:
        print(f"{_ERROR_PREFIX}: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
