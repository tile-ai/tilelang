# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
# cython: language_level=3

import torch
cimport cython
import ctypes
from libc.stdint cimport int64_t, uintptr_t
from libc.stdlib cimport malloc, free
from tvm import tir
from tilelang.utils.tensor import map_torch_type

cdef class CythonKernelWrapper:
    # Class attributes to store kernel configuration and library reference
    cdef:
        object dynamic_symbolic_map  # Maps dynamic dimensions to their corresponding tensor indices
        object buffer_device_map     # Maps buffer variables to their corresponding devices
        object buffer_dtype_map     # Maps buffer variables to their corresponding dtypes
        object static_shape_map     # Maps buffer variables to their corresponding static shapes
        list result_idx             # Indices of output tensors in the params list
        list params                 # List of parameter specifications (includes both inputs and outputs)
        object lib                  # Reference to the compiled library containing the kernel
        # Add new cache attributes
        list param_dtypes    # Cache for parameter dtypes
        list param_shapes    # Cache for parameter shapes as native Python lists

    def __cinit__(self, result_idx, params, lib):
        # Initialize wrapper with kernel configuration
        self.result_idx = result_idx
        self.params = params
        self.lib = lib
        # Convert TVM types to native Python types during initialization
        self.param_dtypes = [param.dtype for param in params]
        # Convert TVM shape arrays to native Python lists
        self.param_shapes = []
        for param in params:
            native_shape = []
            for dim in param.shape:
                if isinstance(dim, tir.IntImm):
                    native_shape.append(int(dim))
                elif isinstance(dim, tir.Var):
                    native_shape.append(dim)  # Keep tir.Var for dynamic dimensions
                else:
                    native_shape.append(dim)
            self.param_shapes.append(native_shape)

    def set_dynamic_symbolic_map(self, dynamic_symbolic_map):
        self.dynamic_symbolic_map = dynamic_symbolic_map
        return self

    def set_buffer_dtype_map(self, buffer_dtype_map):
        self.buffer_dtype_map = buffer_dtype_map
        return self

    def set_static_shape_map(self, static_shape_map):
        self.static_shape_map = static_shape_map
        return self

    def set_buffer_device_map(self, buffer_device_map):
        self.buffer_device_map = buffer_device_map
        return self

    cpdef forward(self, list inputs, int64_t stream = -1):
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

        # Use current CUDA stream if none specified
        if stream == -1: 
            stream = torch.cuda.current_stream().cuda_stream

        cdef int ins_idx = 0
        cdef list tensor_list = []

        # Prepare input and output tensors
        for i in range(len(self.params)):
            if i in self.result_idx:
                dtype = self.param_dtypes[i]
                shape = []
                # Now working with native Python list, no FFI calls needed
                for s in self.param_shapes[i]:
                    if isinstance(s, tir.Var):
                        for key in self.dynamic_symbolic_map:
                            if(str(s) == str(key)):
                                ref_tensor_idx, ref_shape_idx = self.dynamic_symbolic_map[key]
                                shape.append(tensor_list[ref_tensor_idx].shape[ref_shape_idx])
                    else:  # Already converted to Python int during initialization
                        shape.append(s)
                device = inputs[0].device if len(inputs) > 0 else torch.cuda.current_device()
                tensor = torch.empty(*shape, dtype=dtype, device=device)
            else:
                tensor = inputs[ins_idx]
                ins_idx += 1
            tensor_list.append(tensor)
        
        # Convert tensor pointers to C void pointers for kernel call
        call_args = []
        for i in range(len(tensor_list)):
            if isinstance(tensor_list[i], torch.Tensor):
                call_args.append(ctypes.c_void_p(tensor_list[i].data_ptr()))
            elif isinstance(tensor_list[i], int):
                # Dynamic symbolics which are passed as integer arguments
                call_args.append(tensor_list[i])
            else:
                raise ValueError(f"Unsupported tensor type: {type(tensor_list[i])}")

        # Check buffer device
        for param, (buffer_idx, device) in self.buffer_device_map.items():
            tensor_device = tensor_list[buffer_idx].device
            # Compare device types and indices separately to handle both string and torch.device objects            
            if (tensor_device.type != device.type or 
                (tensor_device.index is not None and device.index is not None and tensor_device.index != device.index)):
                raise ValueError(f"Buffer device mismatch for parameter {param}: expected {device}, got {tensor_device}")

        # Check buffer dtype map
        for param, (buffer_idx, torch_dtype) in self.buffer_dtype_map.items():
            if tensor_list[buffer_idx].dtype != torch_dtype:
                raise ValueError(f"Buffer dtype mismatch for parameter {param}: expected {torch_dtype}, got {tensor_list[buffer_idx].dtype}")
        
        # Check static shape map
        for param, (buffer_idx, shape_list) in self.static_shape_map.items():
            for shape_idx, shape in shape_list:
                if tensor_list[buffer_idx].shape[shape_idx] != shape:
                    raise ValueError(f"Static shape mismatch for parameter {param}: expected {shape} at index {shape_idx}, got {tensor_list[buffer_idx].shape}")

        # Add dynamic dimension values to kernel arguments
        for _, (buffer_idx, shape_idx) in self.dynamic_symbolic_map.items():
            call_args.append(tensor_list[buffer_idx].shape[shape_idx])
        
        # Add CUDA stream to kernel arguments
        call_args.append(ctypes.c_void_p(stream))

        # Execute the kernel
        self.lib.call(*call_args)

        # Return output tensor(s)
        if len(self.result_idx) == 1:
            return tensor_list[self.result_idx[0]]
        else:
            return [tensor_list[i] for i in self.result_idx]
    