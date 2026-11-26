# cython: language_level=3
"""Torch NPU stream integration for TVM-FFI execution.

Patches Torch's DLPack Exchange API table so TVM-FFI submits Ascend kernels
to Torch's current NPU stream. Compiled as its own extension
(tilelang_ascend_npu_exchange); the Python-facing surface is
tilelang/ascend/torch_exchange.py.
"""

import torch
from cpython.pycapsule cimport (
    PyCapsule_Destructor,
    PyCapsule_GetPointer,
    PyCapsule_IsValid,
    PyCapsule_New,
)
from cpython.ref cimport Py_INCREF, PyObject
from libc.stdint cimport (
    int32_t,
    int64_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
    uintptr_t,
)
from libc.stdlib cimport free, malloc


ctypedef struct DLPackVersion:
    uint32_t major
    uint32_t minor


ctypedef struct DLPackExchangeAPIHeader:
    DLPackVersion version
    void* prev_api


ctypedef struct DLDevice:
    int32_t device_type
    int32_t device_id


ctypedef struct DLDataType:
    uint8_t code
    uint8_t bits
    uint16_t lanes


ctypedef struct DLTensor:
    void* data
    DLDevice device
    int32_t ndim
    DLDataType dtype
    int64_t* shape
    int64_t* strides
    uint64_t byte_offset


ctypedef struct DLManagedTensor:
    DLTensor dl_tensor
    void* manager_ctx
    void (*deleter)(DLManagedTensor* self) noexcept


ctypedef struct DLManagedTensorVersioned:
    # Field order mirrors dlpack.h: dl_tensor is the last member.
    DLPackVersion version
    void* manager_ctx
    void (*deleter)(DLManagedTensorVersioned* self) noexcept
    uint64_t flags
    DLTensor dl_tensor


ctypedef struct TorchNpuLegacyTensor:
    # Legacy wrapper handed to torch_npu, plus the versioned tensor it may
    # eventually own. The forward flag is separate from the pointer so the
    # wrapper never has to be mutated after torch_npu could have freed it.
    DLManagedTensor legacy
    DLManagedTensorVersioned* source
    int forward_source


ctypedef int (*DLPackManagedTensorToPyObjectNoSync)(
    DLManagedTensorVersioned* tensor,
    void** out_py_object,
) except -1


ctypedef int (*DLPackCurrentWorkStream)(
    int32_t device_type,
    int32_t device_id,
    void** out_stream,
) except -1


ctypedef struct DLPackExchangeAPI:
    DLPackExchangeAPIHeader header
    void* managed_tensor_allocator
    void* managed_tensor_from_py_object_no_sync
    DLPackManagedTensorToPyObjectNoSync managed_tensor_to_py_object_no_sync
    void* dltensor_from_py_object_no_sync
    DLPackCurrentWorkStream current_work_stream


cdef int DLPACK_MAJOR_VERSION = 1
cdef int DLPACK_EXT_DEVICE = 12
cdef const char* DLPACK_EXCHANGE_CAPSULE = "dlpack_exchange_api"
cdef const char* DLPACK_LEGACY_CAPSULE = "dltensor"
cdef DLPackExchangeAPI* torch_exchange_original_api = NULL
cdef DLPackExchangeAPI torch_exchange_patched_api
cdef object torch_exchange_original_capsule = None
cdef object torch_npu_stream_getter = None
cdef object torch_npu_from_dlpack = None


cdef void _torch_npu_legacy_release(TorchNpuLegacyTensor* wrapper) noexcept:
    """Drop the wrapper, forwarding the versioned tensor only when armed.

    ``forward_source`` is set after the import succeeds, so a failed import
    leaves the versioned tensor owned by TVM-FFI, which frees it exactly once.
    """
    cdef DLManagedTensorVersioned* source
    cdef int forward
    if wrapper == NULL:
        return
    source = wrapper.source
    forward = wrapper.forward_source
    free(wrapper)
    if forward and source != NULL and source.deleter != NULL:
        source.deleter(source)


cdef void _torch_npu_legacy_deleter(DLManagedTensor* self) noexcept:
    """Release a legacy wrapper and forward ownership to the versioned tensor."""
    if self == NULL:
        return
    _torch_npu_legacy_release(<TorchNpuLegacyTensor*>self.manager_ctx)


cdef void _torch_npu_legacy_capsule_destructor(object pycaps) noexcept:
    """Free the legacy wrapper if torch_npu never consumed the capsule."""
    cdef DLManagedTensor* legacy
    if PyCapsule_IsValid(pycaps, DLPACK_LEGACY_CAPSULE):
        legacy = <DLManagedTensor*>PyCapsule_GetPointer(pycaps, DLPACK_LEGACY_CAPSULE)
        if legacy != NULL and legacy.deleter != NULL:
            legacy.deleter(legacy)


cdef int torch_npu_managed_tensor_to_py_object(
    DLManagedTensorVersioned* tensor,
    void** out_py_object,
) except -1 with gil:
    """Import NPU results through torch_npu instead of Torch's generic from_blob.

    Torch's stock importer rebuilds the tensor with ``at::from_blob``, which
    yields a plain ``StorageImpl`` without torch_npu's ``NPUStorageDesc``.
    Any later ``get_npu_format``/``ConvertType`` call on such a tensor
    dereferences the missing descriptor and segfaults. torch_npu's own
    importer builds a proper NPU storage, so route NPU results through it.
    """
    cdef TorchNpuLegacyTensor* wrapper = NULL
    cdef object capsule
    cdef object py_tensor

    if tensor == NULL or tensor.dl_tensor.device.device_type != DLPACK_EXT_DEVICE:
        return torch_exchange_original_api.managed_tensor_to_py_object_no_sync(
            tensor, out_py_object)

    if torch_npu_from_dlpack is None:
        # The table only overrides this callback when the importer exists, so
        # reaching here means the installation was bypassed. Raising is still
        # the safe choice: Torch's own importer would rebuild the tensor with
        # at::from_blob and silently lose the NPU storage descriptor.
        raise RuntimeError(
            "torch_npu._C._npu_from_dlpack is unavailable; cannot import an NPU "
            "kernel output without losing its NPU storage descriptor"
        )

    wrapper = <TorchNpuLegacyTensor*>malloc(sizeof(TorchNpuLegacyTensor))
    if wrapper == NULL:
        raise MemoryError("failed to allocate DLManagedTensor for the torch_npu import")
    wrapper.legacy.dl_tensor = tensor.dl_tensor
    wrapper.legacy.manager_ctx = <void*>wrapper
    wrapper.legacy.deleter = _torch_npu_legacy_deleter
    wrapper.source = tensor
    wrapper.forward_source = 0

    capsule = PyCapsule_New(
        <void*>&wrapper.legacy,
        DLPACK_LEGACY_CAPSULE,
        <PyCapsule_Destructor>_torch_npu_legacy_capsule_destructor,
    )
    if capsule is None:
        free(wrapper)
        raise MemoryError("failed to create the DLPack capsule for torch_npu")

    try:
        py_tensor = torch_npu_from_dlpack(capsule)
    except BaseException:
        # The import failed, so TVM-FFI still owns the versioned tensor and
        # frees it on error. Drop the wrapper without forwarding. Do not touch
        # it afterwards: torch_npu may already have freed it through the
        # capsule, and the capsule destructor releases it otherwise.
        raise

    # The import succeeded. From here the wrapper belongs to the imported
    # tensor, so arm the forward and let its deleter release the versioned
    # tensor once the tensor dies.
    wrapper.forward_source = 1

    Py_INCREF(py_tensor)
    out_py_object[0] = <void*>(<PyObject*>py_tensor)
    return 0


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
    global torch_npu_from_dlpack

    import torch_npu

    cdef object tensor_type = torch.Tensor
    cdef object capsule
    cdef object patched_capsule
    cdef object stream_getter
    cdef object current_stream
    cdef object from_dlpack
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

    from_dlpack = getattr(torch_npu._C, "_npu_from_dlpack", None)

    torch_exchange_patched_api = exchange_api[0]
    torch_exchange_patched_api.current_work_stream = torch_npu_current_work_stream
    if from_dlpack is not None:
        # Only take over output import when torch_npu can build a proper NPU
        # storage. Without it the callback would have to fall back to Torch's
        # at::from_blob importer, which silently drops NPUStorageDesc. Leaving
        # the original callback in place keeps stream-only kernels working;
        # kernels with callee-allocated NPU outputs are rejected by the adapter.
        torch_exchange_patched_api.managed_tensor_to_py_object_no_sync = (
            torch_npu_managed_tensor_to_py_object
        )
    patched_capsule = PyCapsule_New(
        &torch_exchange_patched_api,
        DLPACK_EXCHANGE_CAPSULE,
        NULL,
    )

    torch_exchange_original_api = exchange_api
    torch_npu_stream_getter = stream_getter
    torch_npu_from_dlpack = from_dlpack
    try:
        tensor_type.__dlpack_c_exchange_api__ = patched_capsule
    except BaseException:
        torch_exchange_original_api = NULL
        torch_npu_stream_getter = None
        torch_npu_from_dlpack = None
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
