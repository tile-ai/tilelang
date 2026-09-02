"""OpenMP compile-flag helpers for CPU kernels (``tl.cpu_parallel``).

The kernel shared library must link the *same* libomp copy the host process
already loaded, otherwise macOS dyld raises ``OMP: Error #15`` (multiple
OpenMP runtimes). Detection order (mirrors torch inductor's cpp_builder):

1. ``OMP_PREFIX`` environment variable (a prefix with ``include/`` and
   ``lib/``);
2. an already-imported (or importable) torch's bundled ``libomp.dylib``;
3. the conda prefix of the current interpreter;
4. Homebrew's ``libomp``.

On Linux the compiler runtime ``-fopenmp`` is used directly. When no runtime
can be found the OpenMP flags are dropped with a warning: unknown pragmas
are ignored by the compiler, so the kernel stays correct, just serial.
"""

from __future__ import annotations

import os
import sys
import warnings


class _LibompLocation:
    """A detected libomp location: `prefix` is an install prefix with lib/
    and include/ subdirectories; `torch_lib` is torch's own lib directory
    holding the dylib the host process already loaded (no omp headers)."""

    def __init__(self, kind: str, path: str):
        self.kind = kind
        self.path = path

    @property
    def lib_dir(self) -> str:
        return self.path if self.kind == "torch_lib" else os.path.join(self.path, "lib")

    @property
    def include_dir(self) -> str | None:
        if self.kind == "torch_lib":
            # Torch bundles the dylib but not the headers; pragma-only code
            # does not need <omp.h>, so no -I is added.
            return None
        include = os.path.join(self.path, "include")
        return include if os.path.isdir(include) else None


def _warn_invalid_prefix(prefix: str) -> None:
    warnings.warn(
        f"tl.cpu_parallel: OMP_PREFIX={prefix} has no lib/libomp.dylib; falling back to the next candidate.",
        stacklevel=3,
    )


def _find_libomp() -> _LibompLocation | None:
    """Locate a libomp installation to link against (macOS)."""
    env_prefix = os.environ.get("OMP_PREFIX")
    if env_prefix:
        if os.path.exists(os.path.join(env_prefix, "lib", "libomp.dylib")):
            return _LibompLocation("prefix", env_prefix)
        _warn_invalid_prefix(env_prefix)

    torch_module = sys.modules.get("torch")
    if torch_module is None:
        try:
            import torch as torch_module  # noqa: PLC0415
        except ImportError:
            torch_module = None
    if torch_module is not None:
        torch_lib = os.path.join(os.path.dirname(torch_module.__file__), "lib")
        if os.path.exists(os.path.join(torch_lib, "libomp.dylib")):
            # Link the very copy the host process already loaded, keeping the
            # OpenMP runtime unique.
            return _LibompLocation("torch_lib", torch_lib)

    candidates = [sys.prefix, "/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"]
    for cand in candidates:
        if cand and os.path.exists(os.path.join(cand, "lib", "libomp.dylib")):
            return _LibompLocation("prefix", cand)
    return None


def get_openmp_compile_flags() -> list[str]:
    """Compile flags for OpenMP CPU kernels: ``-O2`` plus OpenMP flags."""
    flags = ["-O2"]

    if sys.platform == "darwin":
        loc = _find_libomp()
        if loc is None:
            warnings.warn(
                "tl.cpu_parallel: no libomp runtime found (tried OMP_PREFIX, "
                "torch, conda prefix, Homebrew); compiling serially. Install "
                "libomp (e.g. `brew install libomp`) to enable OpenMP.",
                stacklevel=2,
            )
            return flags
        flags += ["-Xclang", "-fopenmp", f"-L{loc.lib_dir}", "-lomp", f"-Wl,-rpath,{loc.lib_dir}"]
        if loc.include_dir is not None:
            flags.append(f"-I{loc.include_dir}")
        return flags

    if sys.platform == "win32":
        warnings.warn(
            "tl.cpu_parallel: OpenMP flag injection is not supported on Windows yet; compiling serially.",
            stacklevel=2,
        )
        return flags

    # Linux / other Unix: the compiler runtime carries libomp/libgomp.
    flags += ["-fopenmp"]
    return flags
