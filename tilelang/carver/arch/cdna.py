from __future__ import annotations
import tvm
from tvm.target import Target
from .arch_base import TileDevice

# LDS size per CU for specific AMD GPU architectures (in bytes).
# gfx950 (CDNA4 / MI350): 160 KB — larger than the 64 KB default for gfx942.
_GFX950_LDS_SIZE = 160 * 1024  # 163840 bytes


def is_cdna_arch(arch: TileDevice) -> bool:
    return isinstance(arch, CDNA)


class CDNA(TileDevice):
    def __init__(self, target: Target | str):
        if isinstance(target, str):
            target = tvm.target.Target(target)
        self.target = target
        device = tvm.runtime.rocm(0)
        if not device.exist:
            raise RuntimeError("Cannot find HIP device 0.")
        self.device: tvm.runtime.Device = device
        self.platform: str = "CDNA"

        # TVM runtime should correctly report 160 KB (163840 B) for gfx950; the
        # override is kept as a safety net in case an older driver reports the
        # conservative 64 KB default.
        mcpu = str(target.attrs.get("mcpu", ""))
        reported = device.max_shared_memory_per_block
        if "gfx950" in mcpu and reported < _GFX950_LDS_SIZE:
            self.smem_cap = _GFX950_LDS_SIZE
        else:
            self.smem_cap = reported

        self.compute_max_core = device.multi_processor_count
        self.warp_size = device.warp_size
        self.compute_capability = device.compute_version.replace(".", "")
        self.reg_cap: int = 32768
        self.max_smem_usage: int = 2 * self.smem_cap
        self.sm_partition: int = 4
        self.l2_cache_size_bytes: int = target.l2_cache_size_bytes
        self.transaction_size: list[int] = [32, 128]  # in bytes

        self.bandwidth: list[int] = [1300, 14000]


__all__ = [
    "is_cdna_arch",
    "CDNA",
]
