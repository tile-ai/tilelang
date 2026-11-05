"""Public utilities, imported on demand to keep frontend construction independent."""

from importlib import import_module
from .deprecated import deprecated as deprecated

_EXPORTS = {
    "TensorSupplyType": ("tensor", "TensorSupplyType"),
    "torch_assert_close": ("tensor", "torch_assert_close"),
    "is_global": ("language", "is_global"),
    "is_shared": ("language", "is_shared"),
    "is_shared_dynamic": ("language", "is_shared_dynamic"),
    "is_tensor_memory": ("language", "is_tensor_memory"),
    "is_fragment": ("language", "is_fragment"),
    "is_local": ("language", "is_local"),
    "array_reduce": ("language", "array_reduce"),
    "retrieve_stride": ("language", "retrieve_stride"),
    "retrieve_shape": ("language", "retrieve_shape"),
    "retrive_ptr_from_buffer_region": ("language", "retrive_ptr_from_buffer_region"),
    "is_full_region": ("language", "is_full_region"),
    "to_buffer_region": ("language", "to_buffer_region"),
    "get_buffer_region_from_load": ("language", "get_buffer_region_from_load"),
    "get_prim_func_name": ("language", "get_prim_func_name"),
    "side_effect": ("language", "side_effect"),
    "deprecated": ("deprecated", "deprecated"),
    "build_date": ("version", "build_date"),
}

__all__ = tuple(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(f".{module}", __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
