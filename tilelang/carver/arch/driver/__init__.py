# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

from .cuda_driver import (
    get_cuda_device_properties,  # noqa: F401
    get_device_name,  # noqa: F401
    get_shared_memory_per_block,  # noqa: F401
    get_device_attribute,  # noqa: F401
    get_max_dynamic_shared_size_bytes,  # noqa: F401
    get_num_sms,  # noqa: F401
)

from .hip_driver import (
    get_hip_device_properties,  # noqa: F401
    get_hip_device_name,  # noqa: F401
    get_hip_shared_memory_per_block,  # noqa: F401
    get_hip_device_attribute,  # noqa: F401
    get_hip_max_dynamic_shared_size_bytes,  # noqa: F401
    get_hip_num_sms,  # noqa: F401
    get_hip_registers_per_block,  # noqa: F401
)
