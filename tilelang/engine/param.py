# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The profiler and convert to torch utils"""

from dataclasses import dataclass
from typing import List, Union, Optional
import torch
from tilelang import tvm as tvm
from tvm.tir import Buffer, IntImm, Var
from tilelang.utils.tensor import map_torch_type


@dataclass
class KernelParam:
    """
    Represents parameters for a kernel operation, storing dtype and shape information.
    Used to describe tensor or scalar parameters in TVM/PyTorch interop.
    """
    dtype: torch.dtype  # PyTorch data type of the parameter
    shape: List[Union[int, Var]]  # List of dimensions, can be integers or TVM variables

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        """
        Creates a KernelParam instance from a TVM Buffer object.
        
        Args:
            buffer: TVM Buffer object containing dtype and shape information
            
        Returns:
            KernelParam instance with converted dtype and shape
            
        Raises:
            ValueError: If dimension type is not supported (not IntImm or Var)
        """
        dtype = map_torch_type(buffer.dtype)
        shape = []
        for s in buffer.shape:
            if isinstance(s, IntImm):
                shape.append(s.value)
            elif isinstance(s, Var):
                shape.append(s)
            else:
                raise ValueError(f"Unsupported dimension type: {type(s)}")
        return cls(dtype, shape)

    @classmethod
    def from_var(cls, var: Var):
        """
        Creates a KernelParam instance from a TVM Variable object.
        Used for scalar parameters.
        
        Args:
            var: TVM Variable object containing dtype information
            
        Returns:
            KernelParam instance representing a scalar (empty shape)
        """
        return cls(var.dtype, [])

    def is_scalar(self) -> bool:
        """
        Checks if the parameter represents a scalar value.
        
        Returns:
            bool: True if parameter has no dimensions (empty shape), False otherwise
        """
        return len(self.shape) == 0

    def is_unsigned(self) -> bool:
        """
        Checks if the parameter represents an unsigned integer type.
        
        Returns:
            bool: True if parameter is an unsigned integer type, False otherwise
        """
        dtype_str = str(self.dtype)
        if dtype_str.startswith("torch."):
            dtype_str = dtype_str[6:]
        return dtype_str.startswith("uint")

    def is_float8(self) -> bool:
        """
        Checks if the parameter represents a float8 type.
        
        Returns:
            bool: True if parameter is a float8 type, False otherwise
        """
        dtype_str = str(self.dtype)
        if dtype_str.startswith("torch."):
            dtype_str = dtype_str[6:]
        return dtype_str.startswith("float8")

    def is_boolean(self) -> bool:
        """
        Checks if the parameter represents a boolean type.
        
        Returns:
            bool: True if parameter is a boolean type, False otherwise
        """
        dtype_str = str(self.dtype)
        return dtype_str[6:] if dtype_str.startswith("torch.") else dtype_str.startswith("bool")


@dataclass
class CompiledArtifact:
    """
    Represents a compiled kernel artifact containing both host and device code.
    Stores all necessary components for kernel execution in the TVM runtime.
    """
    host_mod: tvm.IRModule  # Host-side TVM IR module for managing kernel execution
    device_mod: tvm.IRModule  # Device-side TVM IR module containing the actual kernel code
    params: List[KernelParam]  # List of parameters (tensors/scalars) used by the kernel
    kernel_source: str  # Raw source code of the generated kernel
    rt_mod: Optional[
        tvm.runtime.Module] = None  # Runtime module for execution, may be lazily initialized
