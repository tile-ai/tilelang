from importlib import import_module as _imp

_C = _imp("tilescale_ext._C")

tensor_from_ptr = _C.tensor_from_ptr
_create_tensor = _C._create_tensor
_create_ipc_handle = _C._create_ipc_handle
_sync_ipc_handles = _C._sync_ipc_handles
_get_device_tensor = _C.get_device_tensor

__all__ = [
    "tensor_from_ptr", 
    "_create_tensor", 
    "_create_ipc_handle", 
    "_sync_ipc_handles", 
    "_get_device_tensor",
    "_C",
]
