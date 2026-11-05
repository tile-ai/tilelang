# cython: language_level=3
"""PTO raw-source calls with native identities and packed tensor storage."""

import ctypes
import torch
from libc.stdint cimport int64_t


cdef class PTOKernelWrapper:
    cdef:
        list result_idx
        list params
        object lib
        object source_abi_plan
        object buffer_dtype_map
        object buffer_device_map
        object ptr_map
        object get_current_device
        object get_current_stream

    def __cinit__(self, result_idx, params, lib, get_current_device, get_current_stream):
        self.result_idx = result_idx
        self.params = params
        self.lib = lib
        self.get_current_device = get_current_device
        self.get_current_stream = get_current_stream
        self.buffer_dtype_map = {}
        self.buffer_device_map = {}
        self.ptr_map = {}

    def set_source_abi_plan(self, plan):
        self.params = plan.validate_parameter_metadata(self.params)
        self.source_abi_plan = plan
        return self

    def set_buffer_dtype_map(self, value):
        self.buffer_dtype_map = value
        return self

    def set_buffer_device_map(self, value):
        self.buffer_device_map = value
        return self

    def set_ptr_map(self, value):
        self.ptr_map = value
        return self

    cdef object _infer_output_device(self, list inputs):
        for tensor in inputs:
            if isinstance(tensor, torch.Tensor):
                return tensor.device
        return self.get_current_device()

    cpdef forward(self, list inputs, int64_t stream=-1, bint skip_tensor_validation=False):
        if self.source_abi_plan is None:
            raise RuntimeError("PTO runtime requires a native source ABI plan")
        if len(inputs) + len(self.result_idx) != len(self.params):
            raise ValueError(
                f"Expected {len(self.params)} arguments, got {len(inputs)} inputs and "
                f"{len(self.result_idx)} outputs"
            )
        tensor_list = [None] * len(self.params)
        cursor = 0
        for index in range(len(self.params)):
            if index not in self.result_idx:
                tensor_list[index] = inputs[cursor]
                cursor += 1
        plan = self.source_abi_plan
        bindings = plan.bind_arguments(tensor_list, packed_storage=True,
                                       validate=not skip_tensor_validation)
        device = None
        if stream == -1:
            device = self._infer_output_device(inputs)
            stream = self.get_current_stream(device)
        for index in self.result_idx:
            shape = plan.output_shape(index, bindings, packed_storage=True)
            if not shape:
                raise ValueError("Cannot create output tensor - 0-dimensional tensors are not supported")
            if device is None:
                device = self._infer_output_device(inputs)
            tensor_list[index] = torch.empty(shape, dtype=self.params[index].torch_dtype(), device=device)

        if not skip_tensor_validation:
            plan.bind_arguments(tensor_list, packed_storage=True)
            for variable, (index, expected) in self.buffer_device_map.items():
                tensor = tensor_list[index]
                if isinstance(tensor, torch.Tensor):
                    if tensor.device.type != expected.type or (
                        expected.index is not None and tensor.device.index is not None
                        and tensor.device.index != expected.index
                    ):
                        raise ValueError(f"Buffer device mismatch for parameter {variable.name}: expected {expected}, got {tensor.device}")
            for variable, (index, expected) in self.buffer_dtype_map.items():
                tensor = tensor_list[index]
                accepted = expected if isinstance(expected, (tuple, list, set)) else (expected,)
                if isinstance(tensor, torch.Tensor) and tensor.dtype not in accepted:
                    raise ValueError(f"Buffer dtype mismatch for parameter {variable.name}: expected {expected}, got {tensor.dtype}")

        scalar_types = {
            "float16": ctypes.c_float, "float32": ctypes.c_float, "float64": ctypes.c_double,
            "int8": ctypes.c_int8, "uint8": ctypes.c_uint8,
            "int16": ctypes.c_int16, "uint16": ctypes.c_uint16,
            "int32": ctypes.c_int32, "uint32": ctypes.c_uint32,
            "int64": ctypes.c_int64, "uint64": ctypes.c_uint64, "bool": ctypes.c_bool,
        }
        call_args = []
        for index, value in enumerate(tensor_list):
            parameter = self.params[index]
            if isinstance(value, torch.Tensor):
                call_args.append(ctypes.c_void_p(value.data_ptr()))
            elif value is None:
                call_args.append(ctypes.c_void_p(0))
            elif isinstance(value, (int, float, bool)):
                if parameter.kind in ("tensor", "pointer") or index in self.ptr_map:
                    call_args.append(ctypes.c_void_p(value))
                else:
                    constructor = scalar_types.get(str(parameter.dtype))
                    if constructor is None:
                        raise TypeError(f"Unsupported PTO scalar dtype: {parameter.dtype}")
                    call_args.append(constructor(value))
            else:
                raise ValueError(f"Unsupported PTO argument type: {type(value)}")
        for variable in plan.implicit_symbols:
            resolved = bindings.try_evaluate(variable)
            # Preserve the raw nullable ABI when no live carrier supplies a
            # dimension not needed for output allocation. Distinct variables
            # with the same printed name are independent slots.
            value = 0 if resolved is None else resolved
            constructor = scalar_types.get(str(variable.ty.dtype))
            if constructor is None:
                raise TypeError(f"Unsupported PTO dynamic symbol type: {variable.ty}")
            call_args.append(constructor(value))
        call_args.append(ctypes.c_void_p(stream))
        if self.lib.call(*call_args) != 0:
            message = self.lib.get_last_error().decode("utf-8")
            raise RuntimeError(f"Kernel call failed: {message}")
        if len(self.result_idx) == 1:
            return tensor_list[self.result_idx[0]]
        return [tensor_list[index] for index in self.result_idx]
