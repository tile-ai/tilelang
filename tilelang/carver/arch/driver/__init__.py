from .cuda_driver import (
    get_cuda_device_properties,  # noqa: F401
    get_device_name,  # noqa: F401
    get_shared_memory_per_block,  # noqa: F401
    get_device_attribute,  # noqa: F401
    get_max_dynamic_shared_size_bytes,  # noqa: F401
    get_persisting_l2_cache_max_size,  # noqa: F401
    get_num_sms,  # noqa: F401
    get_registers_per_block,  # noqa: F401
)

from .ascend_driver import (
    get_ascend_device_properties,  # noqa: F401
    get_device_name as get_ascend_device_name,  # noqa: F401
    get_num_cube_cores,  # noqa: F401
    get_num_vector_cores,  # noqa: F401
    get_total_memory as get_ascend_total_memory,  # noqa: F401
    get_l2_cache_size as get_ascend_l2_cache_size,  # noqa: F401
)
