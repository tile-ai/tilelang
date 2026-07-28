# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Target-dispatched entry point for TileLang default schedule rules.

Each subpackage (``gpu``, ``cpu``, ...) owns a set of rules suited to that
target family and exposes its own ``default_schedule_rules()``.  The
``default_schedule_rules(target)`` function here looks at the target kind and
returns the rule list for that family.
"""

from __future__ import annotations

from tvm.target import Target

_GPU_KINDS = frozenset({"cuda", "rocm", "opencl", "vulkan", "metal", "hip"})
_CPU_KINDS = frozenset({"llvm", "c"})


def _kind_name(target: Target | str) -> str:
    if isinstance(target, Target):
        return target.kind.name
    return str(target)


def default_schedule_rules(target: Target | str):
    """Return the default TileLang schedule rules for ``target``.

    Dispatches to the appropriate subpackage based on ``target.kind.name``.
    Subpackages are imported lazily so an environment missing one family's
    dependencies (e.g. CUDA headers) can still use the other.
    """
    kind = _kind_name(target)
    if kind in _GPU_KINDS:
        from .gpu import default_schedule_rules as _rules

        return _rules()
    if kind in _CPU_KINDS:
        try:
            from .cpu import default_schedule_rules as _rules  # type: ignore[import-not-found]
        except ImportError as e:
            raise NotImplementedError(
                f"Target kind {kind!r} is reserved for future CPU schedule "
                "rules but tilelang.schedule.templates.cpu is not yet "
                "implemented. Contributions welcome."
            ) from e
        return _rules()
    raise ValueError(f"No default schedule rules registered for target kind {kind!r}. Supported kinds: {sorted(_GPU_KINDS | _CPU_KINDS)}.")
