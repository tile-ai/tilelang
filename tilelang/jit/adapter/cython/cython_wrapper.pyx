# cython: language_level=3

import torch
cimport cython
import ctypes
from libc.stdint cimport int64_t, uintptr_t
from libc.stdlib cimport malloc, free
from tvm import tirx
from tilelang.jit.adapter.base import output_allocation_stream, stream_device_index


cdef object _cuda_index(object device):
    if isinstance(device, torch.device) and device.type == "cuda":
        return device.index
    return None


cdef class CythonKernelWrapper:
    # Class attributes to store kernel configuration and library reference
    cdef:
        object dynamic_symbolic_map    # Maps dynamic dimensions to their corresponding tensor indices
        object dynamic_symbolic_sources  # Maps dynamic var names to ALL buffer carriers for cascaded None resolution
        object output_scalar_dim_map  # (output index, dimension index) -> scalar argument index
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
        object get_current_stream      # Raw-stream provider injected by the adapter (None = CUDA default)
        object host_mapped_indices
        object get_host_mapped_pointer

    def __cinit__(self, result_idx, params, lib, get_current_device=None, get_current_stream=None, host_mapped_indices=()):
        # Initialize wrapper with kernel configuration
        self.result_idx = result_idx
        self.params = params
        self.lib = lib
        self.host_mapped_indices = frozenset(host_mapped_indices)
        if self.host_mapped_indices:
            from tvm import get_global_func
            self.get_host_mapped_pointer = get_global_func("tl.cuda.get_host_mapped_pointer")
        # Convert TVM types to native Python types during initialization
        # Convert tvm.DataType to torch.dtype for tensor creation
        self.param_dtypes = [param.torch_dtype() for param in params]
        # Convert TVM shape arrays to native Python lists
        self.param_shapes = []
        self.output_scalar_dim_map = {}
        self.get_current_device = get_current_device if get_current_device is not None else torch.cuda.current_device
        self.get_current_stream = get_current_stream
        for param in params:
            native_shape = []
            for dim in param.shape:
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

    def set_output_scalar_dim_map(self, output_scalar_dim_map):
        self.output_scalar_dim_map = output_scalar_dim_map
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
                if self.host_mapped_indices and device.type == "cuda":
                    current_device = self.get_current_device()
                    device = current_device if isinstance(current_device, torch.device) else torch.device("cuda", current_device)
                    device_index_match = tensor_device.index == device.index
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
                if self.host_mapped_indices and tensor.device.type == "cpu":
                    continue
                return tensor.device
        return self.get_current_device()

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

        device = None
        cdef bint caller_stream = stream != -1
        if stream == -1:
            device = self._infer_output_device(inputs)
            if self.get_current_stream is not None:
                stream = self.get_current_stream(device)
            elif torch.cuda.is_available():
                index = _cuda_index(device)
                if index is None:
                    index = torch.cuda.current_device()
                try:
                    stream = torch._C._cuda_getCurrentRawStream(index)
                except (ImportError, AttributeError):
                    stream = torch.cuda.current_stream(index).cuda_stream
            else:
                stream = 0
        # A caller-supplied stream runs the kernel on the device that owns it,
        # so the outputs live there too.
        stream_index = stream_device_index(stream) if caller_stream else None
        if stream_index is not None:
            device = torch.device("cuda", stream_index)
        if device is None:
            # A caller-supplied stream takes the inputs' device too, also when
            # the kernel has no outputs to allocate.
            device = self._infer_output_device(inputs)

        cdef int ins_idx = 0
        cdef list tensor_list = [None] * len(self.params)

        # Inputs are placed first so that a symbolic dimension owned by an input can be
        # resolved even when the output that needs it comes earlier in the signature;
        # the outputs are then allocated below, in parameter order.
        for i in range(len(self.params)):
            if i not in self.result_idx:
                tensor_list[i] = inputs[ins_idx]
                ins_idx += 1

        # Outputs written on a caller-supplied stream are allocated on that stream.
        alloc_stream = None
        if caller_stream and total_result_idx > 0:
            alloc_stream = output_allocation_stream(stream, device)

        # Prepare output tensors
        for i in range(len(self.params)):
            if i in self.result_idx:
                dtype = self.param_dtypes[i]
                shape = []
                # Now working with native Python list, no FFI calls needed
                for dim_index, s in enumerate(self.param_shapes[i]):
                    scalar_index = self.output_scalar_dim_map.get((i, dim_index))
                    if scalar_index is not None:
                        shape.append(tensor_list[scalar_index])
                        continue
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
                if alloc_stream is None:
                    tensor_list[i] = torch.empty(*shape, dtype=dtype, device=device)
                else:
                    with torch.cuda.stream(alloc_stream):
                        tensor_list[i] = torch.empty(*shape, dtype=dtype, device=device)
                # TODO(chenggang): remove this check or rewrite by ourselves?
                '''
                if isinstance(tensor_list[i], torch.Tensor) and tensor_list[i]._base is not None and not tensor_list[i].is_contiguous():
                    base_tensor = tensor_list[i]._base.as_strided(tensor_list[i]._base.shape, tensor_list[i].stride())
                    if torch._debug_has_internal_overlap(base_tensor):
                        raise ValueError(f"Cannot use an overlapping tensor"
                                         f"(shape={tensor_list[i].shape}, strides={tensor_list[i].stride()}, "
                                         f"overlap={torch._debug_has_internal_overlap(base_tensor)}) as the kernel input")
                '''

        # Convert tensor pointers to C void pointers for kernel call
        cdef dict dtype_to_ctype = {
            torch.float16: ctypes.c_float,
            torch.float32: ctypes.c_float,
            torch.float64: ctypes.c_double,
            torch.int8: ctypes.c_int8,
            torch.int16: ctypes.c_int16,
            torch.int32: ctypes.c_int32,
            torch.int64: ctypes.c_int64,
            torch.uint64: ctypes.c_uint64,
            torch.bool: ctypes.c_bool,
        }

        # Validate before querying mappings or constructing launch arguments.
        if not skip_tensor_validation:
            self._check_buffer_device(tensor_list)
            self._check_buffer_dtype(tensor_list)
            self._check_static_shape(tensor_list)
            self._check_static_strides(tensor_list)
            self._check_static_contiguous(tensor_list)

        call_args = []
        for i, tensor in enumerate(tensor_list):
            if i in self.host_mapped_indices:
                if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
                    raise ValueError(f"Host-mapped parameter {i} must be a CPU tensor")
                # Resolve on every eager call. A raw address cache could survive
                # unregister/free/reallocation and incorrectly reuse a mapping.
                pointer = self.get_host_mapped_pointer(ctypes.c_void_p(tensor.data_ptr()))
                call_args.append(pointer)
                continue
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

        # Add dynamic dimension values to kernel arguments
        for var, (ref_id, buffer_idx, shape_idx, stride_scale) in self.dynamic_symbolic_map.items():
            # Cascaded resolution across all carrier buffers to handle None
            var_key = str(var)
            sources = self.dynamic_symbolic_sources.get(var_key, [(ref_id, buffer_idx, shape_idx, stride_scale)])
            value = 0
            for src_ref_id, src_buf_idx, src_dim_idx, src_stride_scale in sources:
                tensor = tensor_list[src_buf_idx]
                if tensor is not None:
                    if src_ref_id == 0:
                        value = tensor.shape[src_dim_idx]
                    else:
                        value = tensor.stride(src_dim_idx) * src_stride_scale
                    break
            call_args.append(ctypes.c_int64(value))

        # Add CUDA stream to kernel arguments
        call_args.append(ctypes.c_void_p(stream))

        # The launcher works on the current device, so make the stream's device
        # current: the inputs' device for a stream picked here, the owning
        # device for a caller-supplied one.
        launch_index = stream_index if stream_index is not None else _cuda_index(device)
        if launch_index is not None and launch_index != torch.cuda.current_device():
            with torch.cuda.device(launch_index):
                result = self.lib.call(*call_args)
        else:
            result = self.lib.call(*call_args)
        if result != 0:
            error_msg = self.lib.get_last_error().decode('utf-8')
            raise RuntimeError(f"Kernel call failed: {error_msg}")

        # Return output tensor(s)
        if len(self.result_idx) == 1:
            return tensor_list[self.result_idx[0]]
        else:
            return [tensor_list[i] for i in self.result_idx]
