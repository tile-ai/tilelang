"""Parameter contracts for device-accessible mapped host memory."""

from __future__ import annotations

from tvm import tirx

HOST_MAPPED_PARAMS_ATTR = "tl.host_mapped_params"


def get_host_mapped_params(func: tirx.PrimFunc) -> tuple[int, ...]:
    """Read and validate parameter indices, including on cache restoration."""
    indices = []
    attrs = getattr(func, "attrs", None)
    for index in attrs.get(HOST_MAPPED_PARAMS_ATTR, []) if attrs else []:
        if not isinstance(index, (int, tirx.IntImm)):
            raise ValueError(f"{HOST_MAPPED_PARAMS_ATTR} must contain parameter indices")
        index = int(index)
        if index < 0 or index >= len(func.params) or func.params[index] not in func.buffer_map:
            raise ValueError(f"Invalid host-mapped buffer parameter index: {index}")
        if func.buffer_map[func.params[index]].scope() != "global":
            raise ValueError("Host-mapped parameters must be global buffers")
        if index not in indices:
            indices.append(index)
    return tuple(indices)


def validate_host_mapped_params(func, target, execution_backend, out_idx=None) -> tuple[int, ...]:
    """Reject unsupported execution and allocation paths before compilation."""
    indices = get_host_mapped_params(func)
    if not indices:
        return indices
    if target.kind.name != "cuda" or execution_backend not in ("cython", "tvm_ffi"):
        raise ValueError("Host-mapped buffers currently require target='cuda' and execution_backend='cython' or 'tvm_ffi'")
    outputs = out_idx if out_idx is not None else func.attrs.get("tilelang_out_idx", [])
    if isinstance(outputs, int):
        outputs = [outputs]
    outputs = {int(i) if int(i) >= 0 else int(i) + len(func.params) for i in outputs}
    if outputs.intersection(indices):
        raise ValueError("Host-mapped buffers must be preallocated and passed explicitly; out_idx/T.empty is not supported for them")
    return indices
