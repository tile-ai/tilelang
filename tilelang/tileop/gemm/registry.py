from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from tvm.target import Target


GemmTargetPredicate = Callable[[Target], bool]


@dataclass(frozen=True)
class GemmImplEntry:
    name: str
    inst_name: str
    predicate: GemmTargetPredicate
    impl_class: type


_GEMM_IMPLS: list[GemmImplEntry] = []


def register_gemm_impl(
    name: str,
    inst_name: str,
    predicate: GemmTargetPredicate,
    impl_class: type,
) -> None:
    """Register a backend-specific GEMM implementation class."""
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
    return matches[0].impl_class
