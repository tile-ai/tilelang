# cython: language_level=3

import torch
cimport cython
import ctypes
from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_New
from libc.stdint cimport int32_t, int64_t, uint32_t, uintptr_t
from libc.stdlib cimport malloc, free
from tvm import tirx


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


cdef class CythonKernelWrapper:
    # Class attributes to store kernel configuration and library reference
    cdef:
        object dynamic_symbolic_map    # Maps dynamic dimensions to their corresponding tensor indices
        object dynamic_symbolic_sources  # Maps dynamic var names to ALL buffer carriers for cascaded None resolution
        object buffer_device_map       # Maps buffer variables to their corresponding devices
        object buffer_dtype_map        # Maps buffer variables to their corresponding dtypes
        object static_shape_map        # Maps buffer variables to their corresponding static shapes
        object static_strides_map      # Maps buffer variables to their corresponding static strides
        object static_contiguous_list  # A list contains contiguous buffers
        object ptr_map                 # Maps pointer arguments to their corresponding buffer indices
        list result_idx                # Indices of output tensors in the params list
        list params                    # List of parameter specifications (includes both inputs and outputs)
        object lib                     # Reference to the compiled library containing the kernel
        # Add new cache attributes
        list param_dtypes              # Cache for parameter dtypes
        list param_shapes              # Cache for parameter shapes as native Python lists
        object get_current_device

    def __cinit__(self, result_idx, params, lib, target=None):
        # Initialize wrapper with kernel configuration
        self.result_idx = result_idx
        self.params = params
        self.lib = lib
        # Convert TVM types to native Python types during initialization
        # Convert tvm.DataType to torch.dtype for tensor creation
        self.param_dtypes = [param.torch_dtype() for param in params]
        # Convert TVM shape arrays to native Python lists
        self.param_shapes = []
        if hasattr(torch, 'npu') and torch.npu.is_available():
            self.get_current_device = lambda: torch.device('npu', torch.npu.current_device())
        else:
            self.get_current_device = torch.cuda.current_device
        for param in params:
            native_shape = []
            for dim in param.storage_shape(target=target):
                if isinstance(dim, tirx.IntImm):
                    native_shape.append(int(dim))
                elif isinstance(dim, tirx.Var):
                    native_shape.append(dim)  # Keep tirx.Var for dynamic dimensions
                else:
                    native_shape.append(dim)
            self.param_shapes.append(native_shape)

    def set_dynamic_symbolic_map(self, dynamic_symbolic_map):
        self.dynamic_symbolic_map = dynamic_symbolic_map
        return self

    def set_dynamic_symbolic_sources(self, dynamic_symbolic_sources):
        self.dynamic_symbolic_sources = dynamic_symbolic_sources
        return self

    def set_buffer_dtype_map(self, buffer_dtype_map):
        self.buffer_dtype_map = buffer_dtype_map
        return self

    def set_static_shape_map(self, static_shape_map):
        self.static_shape_map = static_shape_map
        return self

    def set_static_strides_map(self, static_strides_map):
        self.static_strides_map = static_strides_map
        return self

    def set_static_contiguous_list(self, static_contiguous_list):
        self.static_contiguous_list = static_contiguous_list
        return self

    def set_ptr_map(self, ptr_map):
        self.ptr_map = ptr_map
        return self

    def set_buffer_device_map(self, buffer_device_map):
        self.buffer_device_map = buffer_device_map
        return self

    cpdef void _check_buffer_device(self, list tensor_list):
        for param, (buffer_idx, device) in self.buffer_device_map.items():
            tensor = tensor_list[buffer_idx]
            if isinstance(tensor, torch.Tensor):
                tensor_device = tensor.device
                device_type_match = device.type == tensor_device.type
                device_index_match = (
                    tensor_device.index is None or
                    device.index is None or
                    tensor_device.index == device.index
                )
                if not (device_type_match and device_index_match):
                    raise ValueError(
                        f"Buffer device mismatch for parameter {param}: "
                        f"expected {device}, got {tensor_device}"
                    )

    cpdef void _check_buffer_dtype(self, list tensor_list):
        for param, (buffer_idx, torch_dtype) in self.buffer_dtype_map.items():
            tensor = tensor_list[buffer_idx]
            if isinstance(tensor, torch.Tensor) and tensor.dtype != torch_dtype:
                raise ValueError(
                    f"Buffer dtype mismatch for parameter {param}: "
                    f"expected {torch_dtype}, got {tensor.dtype}"
                )

    cpdef void _check_static_shape(self, list tensor_list):
        for param, (buffer_idx, shape_list) in self.static_shape_map.items():
            tensor = tensor_list[buffer_idx]
            if not isinstance(tensor, torch.Tensor):
                # otherwise, maybe torch.data_ptr() for T.ptr inputs
                continue

            # Check ndim
            if tensor.dim() != len(shape_list):
                raise ValueError(
                    f"Static shape mismatch for parameter {param}: "
                    f"expected {len(shape_list)} dimensions, "
                    f"got {tensor.dim()}"
                )

            # Check each dimension
            for shape_idx, expected_shape in shape_list:
                actual_shape = tensor.shape[shape_idx]
                if expected_shape != -1 and actual_shape != expected_shape:
                    raise ValueError(
                        f"Static shape mismatch for parameter {param}: "
                        f"expected {expected_shape} at index {shape_idx}, "
                        f"got {actual_shape}"
                    )

    cpdef void _check_static_strides(self, list tensor_list):
        for param, (buffer_idx, strides_list) in self.static_strides_map.items():
            tensor = tensor_list[buffer_idx]
            if not isinstance(tensor, torch.Tensor):
                # otherwise, maybe torch.data_ptr() for T.ptr inputs
                continue
            for stride_idx, expected_stride in strides_list:
                # Ensure the stride index is within the valid range of tensor dimensions
                # (stride_idx should be less than the number of dimensions of the tensor)
                assert stride_idx < tensor.dim(), f"Stride index {stride_idx} out of bounds for tensor with {tensor.dim()} dimensions"
                if tensor.shape[stride_idx] == 1:
                    continue
                actual_stride = tensor.stride(stride_idx)
                if actual_stride != expected_stride:
                    raise ValueError(
                        f"Static stride mismatch for parameter {param}: "
                        f"expected {expected_stride} at index {stride_idx}, "
                        f"got {actual_stride}"
                    )

    cpdef void _check_static_contiguous(self, list tensor_list):
        for buffer_idx, param in self.static_contiguous_list:
            tensor = tensor_list[buffer_idx]
            if not isinstance(tensor, torch.Tensor):
                # otherwise, maybe torch.data_ptr() for T.ptr inputs
                continue
            if not tensor.is_contiguous():
                raise ValueError(f"Expected parameter {param} to be a contiguous tensor")

    cdef object _infer_output_device(self, list inputs):
        for tensor in inputs:
            if isinstance(tensor, torch.Tensor):
                return tensor.device
        if hasattr(torch, 'npu') and torch.npu.is_available():
            return torch.device('npu', torch.npu.current_device())
        return torch.cuda.current_device()

    cpdef forward(self, list inputs, int64_t stream = -1, bint skip_tensor_validation = False):
        # Validate input dimensions and prepare for kernel execution
        cdef int total_params = len(self.params)
        cdef int total_inputs = len(inputs)
        cdef int total_result_idx = len(self.result_idx)
        cdef int total_dynamic_symbolics = len(self.dynamic_symbolic_map)

        # Ensure the number of inputs matches expected parameter count
        if total_params != total_inputs + total_result_idx:
            raise ValueError(
                f"Expected {len(self.params)} inputs, got {len(inputs) + len(self.result_idx)} with {len(inputs)} inputs and {len(self.result_idx)} outputs"
            )

        # Use current device stream if none specified
        if stream == -1:
            if hasattr(torch, 'npu') and torch.npu.is_available():
                # NOTE(chaofan): Use the low-level raw-stream getter (mirrors the CUDA branch
                # below). torch.npu.current_stream() goes through a Python
                # wrapper that internally probes torch.cuda.is_available(), which
                # costs ~150us per call on a CUDA-less NPU host; the _C accessor
                # avoids that and returns the raw stream in <1us.
                try:
                    import torch_npu
                    stream = torch_npu._C._npu_getCurrentRawStream(torch.npu.current_device())
                except (ImportError, AttributeError):
                    stream = torch.npu.current_stream().npu_stream
            elif torch.cuda.is_available():
                try:
                    stream = torch._C._cuda_getCurrentRawStream(torch.cuda.current_device())
                except (ImportError, AttributeError):
                    stream = torch.cuda.current_stream().cuda_stream
            else:
                stream = 0

        cdef int ins_idx = 0
        cdef list tensor_list = []
        device = None

        # Prepare input and output tensors
        for i in range(len(self.params)):
            if i in self.result_idx:
                dtype = self.param_dtypes[i]
                shape = []
                # Now working with native Python list, no FFI calls needed
                for s in self.param_shapes[i]:
                    if isinstance(s, tirx.Var):
                        for key in self.dynamic_symbolic_map:
                            if str(s) == str(key):
                                ref_id, ref_tensor_idx, ref_shape_idx, stride_scale = self.dynamic_symbolic_map[key]
                                if ref_id == 0:
                                    shape.append(tensor_list[ref_tensor_idx].shape[ref_shape_idx])
                                else:
                                    shape.append(tensor_list[ref_tensor_idx].stride(ref_shape_idx) * stride_scale)
                    else:  # Already converted to Python int during initialization
                        shape.append(s)

                if device is None:
                    device = self._infer_output_device(inputs)

                if len(shape) == 0:
                    param_name = self.params[i].name if hasattr(self.params[i], 'name') else f'parameter_{i}'
                    raise ValueError(
                        f"Cannot create output tensor (name={param_name}) - 0-dimensional tensors are not supported. "
                        f"Expected shape: {shape}"
                    )
                tensor = torch.empty(*shape, dtype=dtype, device=device)
            else:
                tensor = inputs[ins_idx]
                ins_idx += 1
            # TODO(chenggang): remove this check or rewrite by ourselves?
            '''
            if isinstance(tensor, torch.Tensor) and tensor._base is not None and not tensor.is_contiguous():
                base_tensor = tensor._base.as_strided(tensor._base.shape, tensor.stride())
                if torch._debug_has_internal_overlap(base_tensor):
                    raise ValueError(f"Cannot use an overlapping tensor"
                                     f"(shape={tensor.shape}, strides={tensor.stride()}, "
                                     f"overlap={torch._debug_has_internal_overlap(base_tensor)}) as the kernel input")
            '''
            tensor_list.append(tensor)

        # Convert tensor pointers to C void pointers for kernel call
        cdef dict dtype_to_ctype = {
            torch.float16: ctypes.c_float,
            torch.float32: ctypes.c_float,
            torch.float64: ctypes.c_double,
            torch.int8: ctypes.c_int8,
            torch.int16: ctypes.c_int16,
            torch.int32: ctypes.c_int32,
            torch.int64: ctypes.c_int64,
            torch.bool: ctypes.c_bool,
        }

        call_args = []
        for i, tensor in enumerate(tensor_list):
            if isinstance(tensor, torch.Tensor):
                call_args.append(ctypes.c_void_p(tensor.data_ptr()))
            elif isinstance(tensor, (int, float, bool)):
                if i in self.ptr_map:
                    call_args.append(ctypes.c_void_p(tensor))
                else:
                    dtype = self.param_dtypes[i]
                    if dtype not in dtype_to_ctype:
                        raise ValueError(f"Unsupported tensor dtype: {dtype}")
                    call_args.append(dtype_to_ctype[dtype](tensor))
            elif tensor is None:
                call_args.append(ctypes.c_void_p(0))
            else:
                raise ValueError(f"Unsupported tensor type: {type(tensor)}")

        # Check buffer device
        if not skip_tensor_validation:
            self._check_buffer_device(tensor_list)
            self._check_buffer_dtype(tensor_list)
            self._check_static_shape(tensor_list)
            self._check_static_strides(tensor_list)
            self._check_static_contiguous(tensor_list)

        # Add dynamic dimension values to kernel arguments
        for var, (ref_id, buffer_idx, shape_idx, stride_scale) in self.dynamic_symbolic_map.items():
            # Cascaded resolution across all carrier buffers to handle None
            var_key = str(var)
            sources = self.dynamic_symbolic_sources.get(var_key, [(buffer_idx, shape_idx, stride_scale)])
            value = 0
            for src_buf_idx, src_dim_idx, src_stride_scale in sources:
                tensor = tensor_list[src_buf_idx]
                if tensor is not None:
                    if ref_id == 0:
                        value = tensor.shape[src_dim_idx]
                    else:
                        value = tensor.stride(src_dim_idx) * src_stride_scale
                    break
            call_args.append(ctypes.c_int64(value))

        # Add CUDA stream to kernel arguments
        call_args.append(ctypes.c_void_p(stream))

        # Execute the kernel
        result = self.lib.call(*call_args)
        if result != 0:
            error_msg = self.lib.get_last_error().decode('utf-8')
            raise RuntimeError(f"Kernel call failed: {error_msg}")

        # Return output tensor(s)
        if len(self.result_idx) == 1:
            return tensor_list[self.result_idx[0]]
        else:
            return [tensor_list[i] for i in self.result_idx]
