from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from tvm.target import Target


GemmSPTargetPredicate = Callable[[Target], bool]


@dataclass(frozen=True)
class GemmSPImplEntry:
    name: str
    predicate: GemmSPTargetPredicate
    impl_class: type


_GEMM_SP_IMPLS: list[GemmSPImplEntry] = []


def register_gemm_sp_impl(
    name: str,
    predicate: GemmSPTargetPredicate,
    impl_class: type,
) -> None:
    """Register a backend-specific GEMM_SP Python implementation class."""
    entry = GemmSPImplEntry(name, predicate, impl_class)
    for idx, registered in enumerate(_GEMM_SP_IMPLS):
        if registered.name == name:
            _GEMM_SP_IMPLS[idx] = entry
            return
    _GEMM_SP_IMPLS.append(entry)


def resolve_gemm_sp_impl(target: Target) -> type:
    """Resolve the registered GEMM_SP implementation class for a target."""
    matches = [entry for entry in _GEMM_SP_IMPLS if entry.predicate(target)]
    if not matches:
        raise ValueError(f"No GEMM_SP implementation registered for target {target}")
    if len(matches) > 1:
        names = ", ".join(entry.name for entry in matches)
        raise ValueError(f"Multiple GEMM_SP implementations matched target {target}: {names}")
    return matches[0].impl_class
