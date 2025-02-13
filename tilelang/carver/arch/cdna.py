# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tvm
from tvm.target import Target
from .arch_base import TileDevice
from typing import List, Union


def is_cdna_arch(arch: TileDevice) -> bool:
    return isinstance(arch, CDNA)


class CDNA(TileDevice):

    def __init__(self, target: Union[Target, str]):
        if isinstance(target, str):
            target = tvm.target.Target(target)
        self.target = target
        device = tvm.runtime.rocm(0)
        if not device.exist:
            raise RuntimeError("Cannot find HIP device 0.")
        self.device: tvm.runtime.Device = device
        self.platform: str = "CDNA"
        self.smem_cap = device.max_shared_memory_per_block
        self.compute_max_core = device.multi_processor_count
        self.warp_size = device.warp_size
        self.compute_capability = device.compute_version.replace(".", "")
        self.reg_cap: int = 32768
        self.max_smem_usage: int = 2 * self.smem_cap
        self.sm_partition: int = 4
        self.l2_cache_size_bytes: int = target.l2_cache_size_bytes
        self.transaction_size: List[int] = [32, 128]  # in bytes

        self.bandwidth: List[int] = [1300, 14000]
