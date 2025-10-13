from __future__ import annotations
from tvm.target import Target
from .arch_base import TileDevice


def is_metal_arch(arch: TileDevice) -> bool:
    return isinstance(arch, METAL)


class METAL(TileDevice):
    def __init__(self, target: Target | str):
        if isinstance(target, str):
            target = Target(target)
        self.target = target


__all__ = [
    "is_metal_arch",
    "METAL",
]
