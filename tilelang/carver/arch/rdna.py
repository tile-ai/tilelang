from __future__ import annotations
from dataclasses import dataclass
import tvm
from tvm.target import Target
from .arch_base import TileDevice
from .cuda import TensorInstruction
from tilelang.utils.target import target_get_mcpu, target_get_rdna_generation

_RDNA_DEFAULT_LDS_SIZE = 64 * 1024
_RDNA_TENSOR_INSTRUCTIONS = {
    11: (TensorInstruction("wmma", [16, 16]),),
}


@dataclass(frozen=True)
class _RDNATuningConfig:
    """Internal Roller tuning heuristics for RDNA WMMA targets.

    Keep this deliberately small and table-driven so exact mcpu overrides stay
    localized here as gfx1100/gfx1151 measurements diverge. These values are
    scheduling defaults, not architectural correctness constraints.
    """

    preferred_warps_per_block: int = 4
    pipeline_stage: int = 1
    reduction_step_by_dtype_bits: tuple[tuple[int, int], ...] = ((16, 32),)

    def reduction_step_for_dtype_bits(self, bits: int) -> int | None:
        return dict(self.reduction_step_by_dtype_bits).get(bits)


_RDNA_DEFAULT_TUNING = _RDNATuningConfig()
_RDNA_TUNING_OVERRIDES: dict[str, _RDNATuningConfig] = {
    # Strix Halo / Radeon 8060S: measured best large FP16 GEMM configs use
    # 128-thread blocks, K tiles starting at 32, and two software pipeline stages.
    "gfx1151": _RDNATuningConfig(
        preferred_warps_per_block=4,
        pipeline_stage=2,
        reduction_step_by_dtype_bits=((16, 32),),
    ),
}


def _get_rdna_tuning_config(mcpu: str | None) -> _RDNATuningConfig:
    normalized = str(mcpu).strip().split(":", maxsplit=1)[0] if mcpu else None
    return _RDNA_TUNING_OVERRIDES.get(normalized, _RDNA_DEFAULT_TUNING)


def _get_tensor_instructions_for_generation(rdna_generation: int) -> tuple[TensorInstruction, ...]:
    try:
        return _RDNA_TENSOR_INSTRUCTIONS[rdna_generation]
    except KeyError as err:
        raise ValueError(f"Unsupported RDNA generation for tensor instructions: {rdna_generation}") from err


def is_rdna_arch(arch: TileDevice) -> bool:
    return isinstance(arch, RDNA)


class RDNA(TileDevice):
    def __init__(self, target: Target | str):
        if isinstance(target, str):
            target = tvm.target.Target(target)
        self.target = target
        self.mcpu = target_get_mcpu(target)
        self.rdna_generation = target_get_rdna_generation(target)
        self.tuning = _get_rdna_tuning_config(self.mcpu)
        if self.rdna_generation != 11:
            arch = self.mcpu or str(target)
            raise ValueError(f"RDNA device model currently supports gfx11 targets only, got {arch}.")
        device = tvm.runtime.rocm(0)
        if not device.exist:
            raise RuntimeError("Cannot find HIP device 0.")
        self.device: tvm.runtime.Device = device
        self.platform: str = "RDNA"

        reported_smem = device.max_shared_memory_per_block
        self.smem_cap = reported_smem if reported_smem > 0 else _RDNA_DEFAULT_LDS_SIZE
        self.compute_max_core = device.multi_processor_count
        self.warp_size = 32
        self.compute_capability = device.compute_version.replace(".", "")
        self.reg_cap: int = 32768
        self.max_smem_usage: int = 2 * self.smem_cap
        self.sm_partition: int = 4
        self.l2_cache_size_bytes: int = getattr(target, "l2_cache_size_bytes", 0)
        self.transaction_size: list[int] = [32, 128]

        # Keep the same units as the existing CUDA/CDNA heuristic. Strix Halo
        # is a UMA part, so use a conservative global-memory score seed.
        self.bandwidth: list[int] = [750, 12080]
        self.available_tensor_instructions: list[TensorInstruction] | None = None

    def get_avaliable_tensorintrin_shapes(self):
        self.available_tensor_instructions = list(_get_tensor_instructions_for_generation(self.rdna_generation))
        return [t.shape for t in self.available_tensor_instructions]

    def __repr__(self):
        return f"RDNA({self.target})"


__all__ = [
    "is_rdna_arch",
    "RDNA",
]
