# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# cython: language_level=3

import torch
cimport cython
import ctypes
from libc.stdint cimport int64_t, uintptr_t
from libc.stdlib cimport malloc, free

cdef class CythonKernelWrapper:
    # Class attributes to store kernel configuration and library reference
    cdef:
        object dynamic_symbolic_map  # Maps dynamic dimensions to their corresponding tensor indices
        list result_idx             # Indices of output tensors in the params list
        list params                 # List of parameter specifications (includes both inputs and outputs)
        object lib                  # Reference to the compiled library containing the kernel

    def __cinit__(self, dynamic_symbolic_map, result_idx, params, lib):
        # Initialize wrapper with kernel configuration
        self.dynamic_symbolic_map = dynamic_symbolic_map
        self.result_idx = result_idx
        self.params = params
        self.lib = lib

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
        cdef list call_args = []

        # Prepare input and output tensors
        for i in range(len(self.params)):
            if i in self.result_idx:
                # Create empty output tensor with specified dtype and shape
                dtype = torch.__getattribute__(str(self.params[i].dtype))
                shape = list(map(int, self.params[i].shape))
                device = inputs[0].device if len(inputs) > 0 else torch.cuda.current_device()
                tensor = torch.empty(*shape, dtype=dtype, device=device)
            else:
                # Use provided input tensor
                tensor = inputs[ins_idx]
                ins_idx += 1
            tensor_list.append(tensor)
        
        # Convert tensor pointers to C void pointers for kernel call
        call_args = [ctypes.c_void_p(tensor_list[i].data_ptr()) for i in range(len(tensor_list))]

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
    