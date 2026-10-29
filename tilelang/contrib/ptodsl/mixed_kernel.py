"""Physical sections for mixed Cube/Vector kernels."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

from ptodsl import pto


MixedKernelRole = Literal["cube", "vector"]


@contextmanager
def mixed_kernel_section(role: MixedKernelRole) -> Iterator[None]:
    """Emit one PTO physical section in the active kernel entry.

    The public context records the physical core while tracing.  Helpers such
    as ``pto.init_core()`` use that context to specialize their Cube and Vector
    implementations.  Keep PTODSL's public module layout so mixed entries and
    ordinary entries remain composable through ``pto.merge_jit_modules``.
    """

    if role not in {"cube", "vector"}:
        raise ValueError(f"Unsupported mixed-kernel role: {role!r}")

    with pto.section(role):
        yield
