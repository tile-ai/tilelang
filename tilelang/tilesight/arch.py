"""Hardware abstraction used by the refactored TileSight model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any


@dataclass(frozen=True)
class HardwareSpec:
    name: str
    compute_capability: str
    sm_count: int
    ddr_bandwidth: float
    l2_bandwidth: float
    l2_capacity: float
    smem_bandwidth: float
    configurable_smem_capacity: float
    register_capacity_per_sm: float
    fp16_tensor_flops: float
    fp32_cuda_flops: float
    fp16_cuda_flops: float
    fp64_cuda_flops: float
    sfu_flops: float = 0.0
    int8_tensor_flops: float | None = None
    fp8_tensor_flops: float | None = None
    max_blocks_per_sm: int = 24
    ddr_max_util: float = 0.9
    l2_max_util: float = 0.9
    smem_max_util: float = 0.9
    compute_max_util: float = 0.9

    @classmethod
    def from_target(cls, target: Any) -> "HardwareSpec":
        arch = _target_arch(target)
        profile = _ARCH_PROFILES.get(arch) or _ARCH_PROFILES.get(_major_arch(arch)) or _ARCH_PROFILES["sm_90"]
        return cls(**profile)

    def tensor_flops_for_dtype(self, dtype: str | None, bytes_per_element: float | None) -> float:
        if bytes_per_element == 1:
            return self.fp8_tensor_flops or self.int8_tensor_flops or self.fp16_tensor_flops
        return self.fp16_tensor_flops

    def cuda_flops_for_dtype(self, dtype: str | None, bytes_per_element: float | None) -> float:
        if bytes_per_element == 8:
            return self.fp64_cuda_flops
        if bytes_per_element == 2:
            return self.fp16_cuda_flops
        return self.fp32_cuda_flops

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, **kwargs)


def _target_arch(target: Any) -> str:
    for attr_name in ("arch", "mcpu"):
        value = getattr(target, attr_name, None)
        if value:
            return str(value)
    attrs = getattr(target, "attrs", None)
    if attrs is not None:
        try:
            value = attrs.get("arch", None)
            if value:
                return str(value)
        except Exception:
            pass
    return "sm_90"


def _major_arch(arch: str) -> str:
    if arch.startswith("sm_"):
        parts = arch.split("_", 1)[1]
        if len(parts) >= 2:
            return f"sm_{parts[:2]}"
    return arch


def _profile(
    name: str,
    cc: str,
    sm_count: int,
    max_freq: float,
    tensor_core_flops_per_sm_cycle: float,
    fp32_cores_per_sm: int,
    ddr_gbps: float,
    l2_gbps: float,
    l2_mib: float,
    smem_bytes_per_sm_cycle: float = 128.0,
    sfu_cores_per_sm: int = 16,
    smem_kib: float = 228.0,
    register_kib: float = 256.0,
    fp8_factor: float | None = None,
) -> dict[str, Any]:
    fp16_tensor = sm_count * max_freq * tensor_core_flops_per_sm_cycle
    fp32_cuda = sm_count * max_freq * fp32_cores_per_sm * 2
    sfu = sm_count * max_freq * sfu_cores_per_sm
    return {
        "name": name,
        "compute_capability": cc,
        "sm_count": sm_count,
        "ddr_bandwidth": ddr_gbps * 1e9,
        "l2_bandwidth": l2_gbps * 1e9,
        "l2_capacity": l2_mib * 1024 * 1024,
        "smem_bandwidth": sm_count * max_freq * smem_bytes_per_sm_cycle,
        "configurable_smem_capacity": smem_kib * 1024,
        "register_capacity_per_sm": register_kib * 1024,
        "fp16_tensor_flops": fp16_tensor,
        "fp32_cuda_flops": fp32_cuda,
        "fp16_cuda_flops": fp32_cuda,
        "fp64_cuda_flops": fp32_cuda * 0.5,
        "sfu_flops": sfu,
        "int8_tensor_flops": fp16_tensor * 2,
        "fp8_tensor_flops": fp16_tensor * fp8_factor if fp8_factor else None,
    }


_H100_PCIE_SPEC = _profile("H100 PCIe spec", "sm_90", 114, 1.42e9, 4 * 1024, 128, 2039, 7598.47, 50)
_H100_PCIE_NCU = _profile("H100 PCIe ncu", "sm_90", 114, 1.06e9, 4 * 1024, 128, 2039, 7598.47, 50)
_H100_SXM = _profile("H100 SXM", "sm_90", 132, 1.83e9, 4 * 1024, 128, 3350, 9784, 50)


_ARCH_PROFILES: dict[str, dict[str, Any]] = {
    "sm_80": _profile("A100", "sm_80", 108, 1.41e9, 4 * 512, 64, 1935, 5288, 40, smem_kib=164),
    "sm_86": _profile("Ampere", "sm_86", 84, 1.70e9, 4 * 512, 128, 936, 3000, 6, smem_kib=100),
    "sm_89": _profile("Ada", "sm_89", 128, 2.52e9, 4 * 512, 128, 1008, 4500, 72, smem_kib=100),
    # TileLang's CUDA target only carries compute capability (sm_90/sm_90a),
    # not the board SKU. Default to the local H100 PCIe NCU-effective profile
    # from TileSight/tilesight/arch/h100_pcie.py::set_to_ncu().
    "sm_90": _H100_PCIE_NCU,
    "sm_90a": _H100_PCIE_NCU,
    "h100_pcie": _H100_PCIE_NCU,
    "h100_pcie_spec": _H100_PCIE_SPEC,
    "h100_sxm": _H100_SXM,
    "sm_100": _profile("B200", "sm_100", 148, 1.965e9, 4 * 2048, 128, 8000, 20161, 126.5, register_kib=512, fp8_factor=2),
    "sm_120": _profile("Blackwell", "sm_120", 148, 1.965e9, 4 * 2048, 128, 8000, 20161, 126.5, register_kib=512, fp8_factor=2),
}
