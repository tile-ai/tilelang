"""Shared helpers for preparing TVM-FFI callable ABIs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tvm.tirx import PrimFunc

# Declarative attribute: which parameters are outputs (mirrors out_idx).
OUT_IDX_ATTR = "tilelang_out_idx"
# ABI decision attribute: set only when the resolved execution backend
# declares supports_callee_allocated_outputs. MakePackedAPI and the TVM-FFI
# adapter both read this attribute, so the compile-time decision has a single
# source of truth instead of each side re-deriving it from the target.
CALLEE_ALLOCATED_OUTPUTS_ATTR = "tilelang_callee_allocated_outputs"


def _normalize_output_indices(output_indices: list[int], num_params: int) -> list[int]:
    normalized = []
    for raw_index in output_indices:
        index = int(raw_index)
        if index < 0:
            index += num_params
        if index < 0 or index >= num_params:
            raise ValueError(f"out_idx index {raw_index} is out of range for a function with {num_params} parameters")
        normalized.append(index)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"out_idx contains duplicate tensor indices: {output_indices}")
    return normalized


def _stamp_callee_allocated_decision(func: PrimFunc) -> PrimFunc:
    if func.attrs is not None and CALLEE_ALLOCATED_OUTPUTS_ATTR in func.attrs:
        return func
    return func.with_attr(CALLEE_ALLOCATED_OUTPUTS_ATTR, 1)


def prepare_tvm_ffi_callee_allocated_outputs(
    func: PrimFunc,
    out_idx: list[int] | int | None,
    *,
    supports_callee_allocated_outputs: bool = False,
) -> tuple[PrimFunc, list[int] | None]:
    """Resolve output indices and expose them to TVM-FFI lowering.

    When the resolved execution backend declares
    ``supports_callee_allocated_outputs``, the ABI decision is recorded on the
    derived PrimFunc so lowering and the runtime adapter both honor it.
    Otherwise ``tilelang_out_idx`` stays purely declarative and the kernel
    keeps the caller-preallocated output path.
    """
    requested_indices = None if out_idx is None else ([out_idx] if isinstance(out_idx, int) else list(out_idx))
    attr_indices = None
    if func.attrs is not None and OUT_IDX_ATTR in func.attrs:
        attr_indices = [int(index) for index in func.attrs[OUT_IDX_ATTR]]

    if attr_indices is not None:
        if requested_indices is not None:
            num_params = len(func.params)
            if _normalize_output_indices(requested_indices, num_params) != _normalize_output_indices(attr_indices, num_params):
                raise ValueError("out_idx does not match the PrimFunc's tilelang_out_idx attribute")
        if attr_indices and supports_callee_allocated_outputs:
            func = _stamp_callee_allocated_decision(func)
        return func, attr_indices

    output_indices = requested_indices or []
    if not output_indices:
        return func, None
    _normalize_output_indices(output_indices, len(func.params))
    func = func.with_attr(OUT_IDX_ATTR, output_indices)
    if supports_callee_allocated_outputs:
        func = _stamp_callee_allocated_decision(func)
    return func, output_indices
