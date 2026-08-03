from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import torch


def determine_fp8_type(fp8_format: Literal["e4m3", "e5m2"] = "e4m3") -> str:
    """
    Select the correct FP8 dtype string for the current platform.
    - CUDA defaults to FP8 E4M3FN / E5M2.
    - ROCm gfx940/gfx941/gfx942 use FNUZ; other supported AMD targets use FN.
    """
    import torch

    from tilelang import language as T

    if fp8_format not in {"e4m3", "e5m2"}:
        raise ValueError(f"Unsupported FP8 format: {fp8_format}")
    if torch.version.hip is None:
        return T.float8_e4m3fn if fp8_format == "e4m3" else T.float8_e5m2
    if not torch.cuda.is_available():
        return T.float8_e4m3fnuz if fp8_format == "e4m3" else T.float8_e5m2fnuz
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    gcn_arch = getattr(props, "gcnArchName", "")
    uses_fnuz = gcn_arch.startswith(("gfx940", "gfx941", "gfx942"))
    if fp8_format == "e4m3":
        return T.float8_e4m3fnuz if uses_fnuz else T.float8_e4m3fn
    return T.float8_e5m2fnuz if uses_fnuz else T.float8_e5m2


def determine_torch_fp8_type(fp8_format: Literal["e4m3", "e5m2"] = "e4m3") -> torch.dtype:
    import torch

    dtype_name = determine_fp8_type(fp8_format)
    torch_dtype = getattr(torch, dtype_name, None)
    if torch_dtype is None:
        raise RuntimeError(f"PyTorch does not expose dtype {dtype_name}")
    return torch_dtype
