import tvm
from tvm.target import Target
from .arch_base import TileDevice


def is_cpu_arch(arch: TileDevice) -> bool:
    return isinstance(arch, CPU)


# For LLVM Backend, we do not provide the detailed information of the CPU
# As the LLVM backend do not required tuning, just maintain the consistency
class CPU(TileDevice):
    def __init__(self, target: Target):
        self.target = target
        device = tvm.runtime.cpu(0)
        if not device.exist:
            raise RuntimeError("Cannot find cpu device 0.")
        self.device: tvm.runtime.Device = device
        self.platform: str = "CPU"


__all__ = [
    "is_cpu_arch",
    "CPU",
]
