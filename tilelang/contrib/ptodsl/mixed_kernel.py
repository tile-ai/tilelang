"""PTODSL compatibility helpers for mixed Cube/Vector kernels."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import Literal

from ptodsl import pto
from ptodsl._tracing import ModuleStyle


MixedKernelRole = Literal["cube", "vector"]


@contextmanager
def mixed_kernel_section(role: MixedKernelRole) -> Iterator[None]:
    """Emit one PTO physical section in the active kernel entry.

    The public context records the physical core while tracing.  Helpers such
    as ``pto.init_core()`` use that context to specialize their Cube and Vector
    implementations, while the emitted section remains a direct child of the
    entry function for ``vpto-split-cv-module``.
    """

    if role not in {"cube", "vector"}:
        raise ValueError(f"Unsupported mixed-kernel role: {role!r}")

    with pto.section(role):
        yield


def finalize_mixed_kernel(kernel):
    """Select PTODSL's flat module layout before the first specialization."""

    compiler = getattr(kernel, "_compiler", None)
    if compiler is None:
        raise TypeError("Expected a PTODSL @pto.jit kernel handle")

    module_spec = compiler._module_spec
    if not module_spec.entry:
        raise ValueError("A mixed PTO kernel must be a launchable entry")
    if module_spec.kernel_kind_explicit:
        raise ValueError("A mixed PTO kernel cannot declare one kernel_kind")
    if module_spec.backend != "vpto":
        raise ValueError("A mixed PTO kernel requires the VPTO backend")

    if module_spec.module_style == ModuleStyle.FLAT_AICORE:
        return kernel
    if kernel.cached_specializations():
        raise RuntimeError("A mixed PTO kernel must select its module layout before compilation")
    if module_spec.module_style != ModuleStyle.BACKEND_PARTITIONED:
        raise ValueError(f"Unsupported initial PTODSL module style: {module_spec.module_style!r}")

    # @pto.jit currently hard-codes BACKEND_PARTITIONED and exposes no
    # module_style argument. KernelModuleSpec is frozen, so replace the spec
    # before tracing; this is the narrow compatibility point to remove once
    # PTODSL offers a public mixed/flat module option.
    compiler._module_spec = replace(
        module_spec,
        module_style=ModuleStyle.FLAT_AICORE,
    )
    return kernel
