"""Shared helpers and version markers for persisted JIT callable ABIs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tvm.tirx import PrimFunc


def get_tvm_ffi_torch_storage_abi_tag() -> str:
    """Describe Torch capabilities that affect output storage lowering."""
    import torch

    fp4_storage = "float4_e2m1fnx2" if hasattr(torch, "float4_e2m1fn_x2") else "int8"
    float8_e4m3_storage = "float8_e4m3fnuz" if torch.version.hip is not None else "float8_e4m3fn"
    return f"fp4={fp4_storage};float8_e4m3={float8_e4m3_storage}"


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


def prepare_tvm_ffi_callee_allocated_outputs(
    func: PrimFunc,
    out_idx: list[int] | int | None,
) -> tuple[PrimFunc, list[int] | None]:
    """Resolve output indices and expose them to TVM-FFI lowering."""
    requested_indices = None if out_idx is None else ([out_idx] if isinstance(out_idx, int) else list(out_idx))
    attr_indices = None
    if func.attrs is not None and "tilelang_out_idx" in func.attrs:
        attr_indices = [int(index) for index in func.attrs["tilelang_out_idx"]]

    if attr_indices is not None:
        if requested_indices is not None:
            num_params = len(func.params)
            if _normalize_output_indices(requested_indices, num_params) != _normalize_output_indices(attr_indices, num_params):
                raise ValueError("out_idx does not match the PrimFunc's tilelang_out_idx attribute")
        return func, attr_indices

    output_indices = requested_indices or []
    if not output_indices:
        return func, None
    _normalize_output_indices(output_indices, len(func.params))
    return func.with_attr("tilelang_out_idx", output_indices), output_indices


# Bump this whenever the callable ABI of generated TVM-FFI host modules
# changes. Version 4 resolves Torch's physical storage dtype while lowering, so
# sub-byte outputs also use the callee-allocated single-main ABI.
TVM_FFI_KERNEL_ABI_VERSION = 4
