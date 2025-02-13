# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tvm
from tvm.target import Target
from .arch_base import TileDevice
from typing import List, Union


def check_sm_version(arch: str) -> int:
    sm_version = arch.replace("sm_", "")
    return int(sm_version) if sm_version.isdigit() else -1


def is_cuda_arch(arch: TileDevice) -> bool:
    return isinstance(arch, CUDA)


def is_volta_arch(arch: TileDevice) -> bool:
    conditions = [True]
    conditions.append(is_cuda_arch(arch))
    conditions.append(arch.sm_version >= 70)
    conditions.append(arch.sm_version < 80)
    return all(conditions)


def is_ampere_arch(arch: TileDevice) -> bool:
    conditions = [True]
    conditions.append(is_cuda_arch(arch))
    conditions.append(arch.sm_version >= 80 and arch.sm_version < 89)
    return all(conditions)


def is_ada_arch(arch: TileDevice) -> bool:
    conditions = [True]
    conditions.append(is_cuda_arch(arch))
    conditions.append(arch.sm_version == 89)
    return all(conditions)


def is_hopper_arch(arch: TileDevice) -> bool:
    conditions = [True]
    conditions.append(is_cuda_arch(arch))
    conditions.append(arch.sm_version == 90)
    return all(conditions)


def has_mma_support(arch: TileDevice) -> bool:
    conditions = [True]
    conditions.append(is_cuda_arch(arch))
    conditions.append(arch.sm_version >= 80)
    return all(conditions)


volta_tensorcore_supported = [
    ("float16", "float32"),
    ("float16", "float16"),
]
ampere_tensorcore_supported = [
    ("bfloat16", "float32"),
    ("float16", "float32"),
    ("float16", "float16"),
    ("int8", "int32"),
    ("int4", "int32"),
    ("int2", "int32"),
    ("int1", "int32"),
]
ada_tensorcore_supported = [
    ("bfloat16", "float32"),
    ("float16", "float32"),
    ("float16", "float16"),
    ("int8", "int32"),
    ("e5m2_float8", "float32"),
    ("e4m3_float8", "float32"),
]
hopper_tensorcore_supported = ada_tensorcore_supported


# TODO(lei): we should consider the dtype of the input a and b
# instead of assuming both a and b share the same dtype.
# As the tensorcore may supports e4m3_float8 * e5m2_float8
def is_tensorcore_supported_precision(in_dtype: str, accum_dtype: str, arch: TileDevice) -> bool:

    if is_volta_arch(arch):
        return (in_dtype, accum_dtype) in volta_tensorcore_supported
    elif is_ampere_arch(arch):
        return (in_dtype, accum_dtype) in ampere_tensorcore_supported
    elif is_ada_arch(arch):
        return (in_dtype, accum_dtype) in ada_tensorcore_supported
    elif is_hopper_arch(arch):
        return (in_dtype, accum_dtype) in hopper_tensorcore_supported
    else:
        raise ValueError(f"Unsupported architecture: {arch}")


class TensorInstruction(object):

    def __init__(
        self,
        name: str,
        shape: List[int],
    ):
        self.name: str = name
        # only hold the shape of M and N
        self.shape: List[int] = shape


class CUDA(TileDevice):

    def __init__(self, target: Union[Target, str]):
        if isinstance(target, str):
            target = tvm.target.Target(target)
        self.target = target
        self.sm_version = check_sm_version(self.target.arch)
        device = tvm.runtime.cuda(0)
        if not device.exist:
            raise RuntimeError("Cannot find cuda device 0.")
        self.device: tvm.runtime.Device = device
        self.platform: str = "CUDA"
        self.smem_cap = device.max_shared_memory_per_block
        self.compute_max_core = device.multi_processor_count
        self.warp_size = device.warp_size
        self.compute_capability = device.compute_version.replace(".", "")
        self.reg_cap: int = 65536
        self.max_smem_usage: int = 2 * self.smem_cap
        self.sm_partition: int = 4
        self.l2_cache_size_bytes: int = target.l2_cache_size_bytes
        # the number of transaction size in bytes
        self.transaction_size: List[int] = [32, 128]  # in bytes
        # bandwidth in MB/s, will be used for recommend basic tile size
        # TODO(lei): find some way to get the real bandwidth
        # However, the ratio of bandwidth between different devices can
        # be similar. The bandwidth can work for another devices as well.
        self.bandwidth: List[int] = [750, 12080]
        # get the available tensor instructions during runtime to avoid
        # the dependency of the tensor intrinsics registration
        self.available_tensor_instructions: List[TensorInstruction] = None

    def get_avaliable_tensorintrin_shapes(self):
        self.available_tensor_instructions = (
            TensorInstruction("mma", [16, 16]),
            TensorInstruction("wmma", [16, 16]),
        )
        return [t.shape for t in self.available_tensor_instructions]

    def __repr__(self):
        return f"CUDA({self.target})"
