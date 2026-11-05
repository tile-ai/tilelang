from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from importlib import import_module

from tvm_tirx.target import Target


GemmTargetPredicate = Callable[[Target], bool]


@dataclass(frozen=True)
class GemmImplEntry:
    name: str
    inst_name: str
    predicate: GemmTargetPredicate
    impl_class: type | str


_GEMM_IMPLS: list[GemmImplEntry] = []


def register_gemm_impl(
    name: str,
    inst_name: str,
    predicate: GemmTargetPredicate,
    impl_class: type | str,
) -> None:
    """Register a class or a deferred ``module:Class`` implementation.

    Deferring backend imports keeps instruction selection independent of
    unrelated architecture emitters. A selected implementation is still
    imported normally, and import or class errors propagate to its caller.
    """
    entry = GemmImplEntry(name, inst_name, predicate, impl_class)
    for idx, registered in enumerate(_GEMM_IMPLS):
        if registered.name == name:
            _GEMM_IMPLS[idx] = entry
            return
    _GEMM_IMPLS.append(entry)


def resolve_gemm_impl(gemm_inst: str, target: Target) -> type:
    """Resolve the registered implementation class for a GEMM instruction key."""
    matches = [entry for entry in _GEMM_IMPLS if entry.inst_name == gemm_inst and entry.predicate(target)]
    if not matches:
        raise ValueError(f"No GEMM implementation registered for instruction {gemm_inst} and target {target}")
    if len(matches) > 1:
        names = ", ".join(entry.name for entry in matches)
        raise ValueError(f"Multiple GEMM implementations matched instruction {gemm_inst} and target {target}: {names}")
    return load_gemm_impl_class(matches[0].impl_class)


def load_gemm_impl_class(implementation: type | str) -> type:
    """Load a selected implementation without caching over registration overrides."""
    if isinstance(implementation, str):
        module, separator, name = implementation.partition(":")
        if not separator:
            raise ValueError("A deferred GEMM implementation must use module:Class")
        implementation = getattr(import_module(module), name)
    if not isinstance(implementation, type):
        raise TypeError("A GEMM implementation must resolve to a class")
    return implementation
