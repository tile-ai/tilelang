# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""The profiler and convert to torch utils"""
from enum import Enum
import torch
from tvm.relay import TensorType
from tvm.runtime import ndarray
from torch.utils.dlpack import to_dlpack


class TensorSupplyType(Enum):
    Integer = 1
    Uniform = 2
    Normal = 3
    Randn = 4
    Zero = 5
    One = 6
    Auto = 7


def map_torch_type(intype: str) -> torch.dtype:
    typemap = {
        'e4m3_float8': torch.float8_e4m3fn,
        'e5m2_float8': torch.float8_e5m2,
    }
    if intype in typemap:
        return typemap[intype]
    else:
        return getattr(torch, intype)


def adapt_torch2tvm(arg):
    float8_dtype_map = {
        torch.float8_e4m3fn: "e4m3_float8",
        torch.float8_e4m3fnuz: "e4m3_float8",
        torch.float8_e5m2: "e5m2_float8",
        torch.float8_e5m2fnuz: "e5m2_float8",
    }
    if isinstance(arg, torch.Tensor):
        if arg.dtype in {
                torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz
        }:
            return ndarray.from_dlpack(to_dlpack(arg.view(torch.int8)))._create_view(
                shape=arg.shape, dtype=float8_dtype_map[arg.dtype])
        return ndarray.from_dlpack(to_dlpack(arg))
    return arg


def get_tensor_supply(supply_type: TensorSupplyType):

    def get_tensor(tensor: TensorType) -> torch.Tensor:
        dtype = map_torch_type(str(tensor.dtype))
        device = torch.cuda.current_device()

        if hasattr(tensor, "shape") and not tensor.shape:
            raise ValueError(
                f"TensorType must have a shape, but got {type(tensor)}, "
                "likely you are trying to generate a random tensor with a dynamic symbolic shape.")

        shape = list(map(int, tensor.shape))
        if supply_type == TensorSupplyType.Auto:
            if dtype == torch.float16 or dtype == torch.float32:
                return torch.empty(*shape, device=device, dtype=dtype).normal_(-1.0, 1.0)
            elif dtype == torch.uint8:
                return torch.randint(0, 2, size=shape, device=device, dtype=dtype)
            else:
                raise NotImplementedError(dtype)

        if dtype == torch.int8 and supply_type in [
                TensorSupplyType.Uniform,
                TensorSupplyType.Normal,
        ]:
            return torch.ones(*shape, device=device, dtype=dtype)

        if supply_type == TensorSupplyType.Integer:
            is_unsigned = tensor.dtype.startswith("uint")
            is_float8 = tensor.dtype.endswith("float8")
            if is_unsigned:
                return torch.randint(low=0, high=3, size=shape, device=device, dtype=dtype)
            elif is_float8:
                return torch.randint(
                    low=-128, high=128, size=shape, device=device, dtype=torch.int8).to(dtype)
            else:
                return torch.randint(low=-2, high=3, size=shape, device=device, dtype=dtype)
        elif supply_type == TensorSupplyType.Uniform:
            return torch.empty(*shape, device=device, dtype=dtype).uniform_(-1.0, 1.0)
        elif supply_type == TensorSupplyType.Normal:
            return torch.empty(*shape, device=device, dtype=dtype).normal_(-1.0, 1.0)
        elif supply_type == TensorSupplyType.Randn:
            return torch.randn(*shape, device=device).to(dtype)
        elif supply_type == TensorSupplyType.Zero:
            return torch.zeros(*shape, device=device, dtype=dtype)
        elif supply_type == TensorSupplyType.One:
            return torch.ones(*shape, device=device, dtype=dtype)
        else:
            raise NotImplementedError(supply_type)

    return get_tensor


def torch_assert_close(
    tensor_a,
    tensor_b,
    rtol=1e-2,
    atol=1e-3,
    max_mismatched_ratio=0.001,
    verbose=False,
):
    """
    Custom function to assert that two tensors are "close enough," allowing a specified
    percentage of mismatched elements.

    Parameters:
    ----------
    tensor_a : torch.Tensor
        The first tensor to compare.
    tensor_b : torch.Tensor
        The second tensor to compare.
    rtol : float, optional
        Relative tolerance for comparison. Default is 1e-2.
    atol : float, optional
        Absolute tolerance for comparison. Default is 1e-3.
    max_mismatched_ratio : float, optional
        Maximum ratio of mismatched elements allowed (relative to the total number of elements).
        Default is 0.001 (0.1% of total elements).

    Raises:
    -------
    AssertionError:
        If the ratio of mismatched elements exceeds `max_mismatched_ratio`.
    """
    import torch

    # Compute the absolute difference between the two tensors
    diff = torch.abs(tensor_a - tensor_b)

    # Compute the maximum allowable difference for each element
    max_diff = atol + rtol * torch.abs(tensor_b)

    # Identify elements where the difference exceeds the maximum allowable difference
    mismatched = diff > max_diff

    # Count the number of mismatched elements
    num_mismatched = mismatched.sum().item()

    # Calculate the total number of elements in the tensor
    total_elements = tensor_a.numel()

    # Compute the allowed mismatched elements based on the ratio
    max_allowed_mismatched = int(total_elements * max_mismatched_ratio)

    # Print debug information about the mismatch
    if verbose:
        print(f"Number of mismatched elements: {num_mismatched} / {total_elements} "
              f"(allowed: {max_allowed_mismatched})")

    # Check if the number of mismatched elements exceeds the allowed threshold
    if num_mismatched > max_allowed_mismatched:
        raise AssertionError(
            f"Too many mismatched elements: {num_mismatched} > {max_allowed_mismatched} "
            f"({max_mismatched_ratio * 100:.2f}% allowed). "
            f"Greatest absolute difference: {diff.max().item()}, "
            f"Greatest relative difference: {(diff / (torch.abs(tensor_b) + 1e-12)).max().item()}.")
    else:
        return True
