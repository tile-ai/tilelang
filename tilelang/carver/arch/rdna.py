from __future__ import annotations
import tvm
from tvm.target import Target
from .arch_base import TileDevice
from .cuda import TensorInstruction
from tilelang.utils.target import target_get_mcpu, target_get_rdna_generation

_RDNA_DEFAULT_LDS_SIZE = 64 * 1024


def is_rdna_arch(arch: TileDevice) -> bool:
    return isinstance(arch, RDNA)


class RDNA(TileDevice):
    def __init__(self, target: Target | str):
        if isinstance(target, str):
            target = tvm.target.Target(target)
        self.target = target
        if target_get_rdna_generation(target) != 11:
            arch = target_get_mcpu(target) or str(target)
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
        self.available_tensor_instructions = (TensorInstruction("wmma", [16, 16]),)
        return [t.shape for t in self.available_tensor_instructions]

    def __repr__(self):
        return f"RDNA({self.target})"


__all__ = [
    "is_rdna_arch",
    "RDNA",
]
