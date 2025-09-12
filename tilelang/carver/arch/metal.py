
from .arch_base import TileDevice


def is_metal_arch(arch: TileDevice) -> bool:
    return isinstance(arch, METAL)


class METAL(TileDevice):
    pass

__all__ = [
    'is_metal_arch',
    'METAL',
]