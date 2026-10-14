# cython: language_level=3
"""Torch NPU stream integration for TVM-FFI execution.

Patches Torch's DLPack Exchange API table so TVM-FFI submits Ascend kernels
to Torch's current NPU stream. Compiled as its own extension
(tilelang_ascend_npu_exchange); the Python-facing surface is
tilelang/ascend/torch_exchange.py.
"""

import torch
from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_New
from libc.stdint cimport int32_t, uint32_t, uintptr_t


ctypedef struct DLPackVersion:
    uint32_t major
    uint32_t minor


ctypedef struct DLPackExchangeAPIHeader:
    DLPackVersion version
    void* prev_api


ctypedef int (*DLPackCurrentWorkStream)(
    int32_t device_type,
    int32_t device_id,
    void** out_stream,
) except -1


ctypedef struct DLPackExchangeAPI:
    DLPackExchangeAPIHeader header
    void* managed_tensor_allocator
    void* managed_tensor_from_py_object_no_sync
    void* managed_tensor_to_py_object_no_sync
    void* dltensor_from_py_object_no_sync
    DLPackCurrentWorkStream current_work_stream


cdef int DLPACK_MAJOR_VERSION = 1
cdef int DLPACK_EXT_DEVICE = 12
cdef const char* DLPACK_EXCHANGE_CAPSULE = "dlpack_exchange_api"
cdef DLPackExchangeAPI* torch_exchange_original_api = NULL
cdef DLPackExchangeAPI torch_exchange_patched_api
cdef object torch_exchange_original_capsule = None
cdef object torch_npu_stream_getter = None


cdef int torch_npu_current_work_stream(
    int32_t device_type,
    int32_t device_id,
    void** out_stream,
) except -1 with gil:
    if device_type == DLPACK_EXT_DEVICE:
        out_stream[0] = <void*><uintptr_t>torch_npu_stream_getter(device_id)
        return 0
    return torch_exchange_original_api.current_work_stream(
        device_type,
        device_id,
        out_stream,
    )


def install_torch_npu_stream_exchange():
    """Route TVM-FFI Ascend submissions to Torch's current NPU stream."""
    global torch_exchange_original_api
    global torch_exchange_original_capsule
    global torch_exchange_patched_api
    global torch_npu_stream_getter

    import torch_npu

    cdef object tensor_type = torch.Tensor
    cdef object capsule
    cdef object patched_capsule
    cdef object stream_getter
    cdef object current_stream
    cdef DLPackExchangeAPI* exchange_api

    if not hasattr(tensor_type, "__dlpack_c_exchange_api__"):
        raise RuntimeError(
            "torch.Tensor does not expose __dlpack_c_exchange_api__; "
            "load TVM-FFI's Torch DLPack extension first"
        )

    capsule = tensor_type.__dlpack_c_exchange_api__
    exchange_api = <DLPackExchangeAPI*>PyCapsule_GetPointer(
        capsule,
        DLPACK_EXCHANGE_CAPSULE,
    )
    if exchange_api == &torch_exchange_patched_api:
        return False
    if torch_exchange_original_api != NULL:
        raise RuntimeError(
            "torch.Tensor.__dlpack_c_exchange_api__ was replaced after "
            "TileLang installed its Torch NPU stream callback"
        )
    if exchange_api.header.version.major != DLPACK_MAJOR_VERSION:
        raise RuntimeError("unsupported DLPack Exchange API major version")
    if (
        exchange_api.managed_tensor_allocator == NULL
        or exchange_api.managed_tensor_from_py_object_no_sync == NULL
        or exchange_api.managed_tensor_to_py_object_no_sync == NULL
        or exchange_api.current_work_stream == NULL
    ):
        raise RuntimeError("incomplete Torch DLPack Exchange API table")

    stream_getter = getattr(
        torch_npu._C,
        "_npu_getCurrentRawStream",
        None,
    )
    if stream_getter is None:
        stream_getter = getattr(
            torch_npu._C,
            "_npu_getCurrentRawStreamNoWait",
            None,
        )
    if stream_getter is None:
        current_stream = torch_npu.npu.current_stream
        stream_getter = lambda device_id: current_stream(device_id).npu_stream

    torch_exchange_patched_api = exchange_api[0]
    torch_exchange_patched_api.current_work_stream = torch_npu_current_work_stream
    patched_capsule = PyCapsule_New(
        &torch_exchange_patched_api,
        DLPACK_EXCHANGE_CAPSULE,
        NULL,
    )

    torch_exchange_original_api = exchange_api
    torch_npu_stream_getter = stream_getter
    try:
        tensor_type.__dlpack_c_exchange_api__ = patched_capsule
    except BaseException:
        torch_exchange_original_api = NULL
        torch_npu_stream_getter = None
        raise

    # The copied callbacks belong to the original table, so retain its capsule
    # for the lifetime of this extension module.
    torch_exchange_original_capsule = capsule
    return True


def is_torch_npu_stream_exchange_installed():
    """Return whether Torch currently points at TileLang's patched table."""
    cdef DLPackExchangeAPI* exchange_api

    if not hasattr(torch.Tensor, "__dlpack_c_exchange_api__"):
        return False
    exchange_api = <DLPackExchangeAPI*>PyCapsule_GetPointer(
        torch.Tensor.__dlpack_c_exchange_api__,
        DLPACK_EXCHANGE_CAPSULE,
    )
    return exchange_api == &torch_exchange_patched_api
