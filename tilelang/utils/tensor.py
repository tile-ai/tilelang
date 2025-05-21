# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.

from __future__ import annotations
"""The profiler and convert to torch utils"""
from enum import Enum
import torch
from tvm.runtime import ndarray
from tvm import tir
from torch.utils.dlpack import to_dlpack
import numpy as np


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


def get_tensor_supply(supply_type: TensorSupplyType = TensorSupplyType.Integer):

    from tilelang.engine.param import KernelParam

    def get_tensor(param: KernelParam) -> torch.Tensor:
        dtype: torch.dtype = param.dtype
        device: torch.device = torch.cuda.current_device()

        if hasattr(param, "shape") and not param.shape:
            raise ValueError(
                f"TensorType must have a shape, but got {type(param)}, "
                "likely you are trying to generate a random tensor with a dynamic symbolic shape.")

        # Check if with dynamic symbolic shape
        for shape in param.shape:
            if isinstance(shape, tir.Var):
                raise ValueError(
                    f"TensorType must have a static shape, but got {shape}, "
                    "likely you are trying to generate a random tensor with a dynamic symbolic shape."
                )

        shape = list(map(int, param.shape))
        if supply_type == TensorSupplyType.Auto:
            is_unsigned = param.is_unsigned()
            is_float8 = param.is_float8()
            is_boolean = param.is_boolean()
            if is_unsigned:
                return torch.randint(low=0, high=3, size=shape, device=device, dtype=dtype)
            elif is_float8:
                return torch.randint(
                    low=-128, high=128, size=shape, device=device, dtype=torch.int8).to(dtype)
            elif is_boolean:
                return torch.randint(low=0, high=2, size=shape, device=device, dtype=dtype)
            elif dtype in {torch.float16, torch.float32, torch.bfloat16}:
                return torch.empty(*shape, device=device, dtype=dtype).uniform_(-1.0, 1.0)
            else:
                return torch.randint(low=-2, high=3, size=shape, device=device, dtype=dtype)

        if dtype == torch.int8 and supply_type in [
                TensorSupplyType.Uniform,
                TensorSupplyType.Normal,
        ]:
            return torch.ones(*shape, device=device, dtype=dtype)

        if supply_type == TensorSupplyType.Integer:
            is_unsigned = param.is_unsigned()
            is_float8 = param.is_float8()
            is_boolean = param.is_boolean()
            if is_unsigned:
                return torch.randint(low=0, high=3, size=shape, device=device, dtype=dtype)
            elif is_float8:
                return torch.randint(
                    low=-128, high=128, size=shape, device=device, dtype=torch.int8).to(dtype)
            elif is_boolean:
                return torch.randint(low=0, high=2, size=shape, device=device, dtype=dtype)
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


# Adapted from https://github.com/pytorch/pytorch/blob/main/torch/testing/_comparison.py
def _compare_attributes(
    actual: torch.Tensor,
    expected: torch.Tensor,
    check_device: bool = True,
    check_dtype: bool = True,
    check_layout: bool = True,
    check_stride: bool = False,
) -> None:
    """Checks if the attributes of two tensors match.
    Always checks
    - the :attr:`~torch.Tensor.shape`,
    - whether both inputs are quantized or not,
    - and if they use the same quantization scheme.
    Checks for
    - :attr:`~torch.Tensor.layout`,
    - :meth:`~torch.Tensor.stride`,
    - :attr:`~torch.Tensor.device`, and
    - :attr:`~torch.Tensor.dtype`
    are optional and can be disabled through the corresponding ``check_*`` flag during construction of the pair.
    """

    def raise_mismatch_error(attribute_name: str, actual_value, expected_value):
        raise AssertionError(
            f"The values for attribute '{attribute_name}' do not match: {actual_value} != {expected_value}."
        )

    if actual.shape != expected.shape:
        raise_mismatch_error("shape", actual.shape, expected.shape)
    if actual.is_quantized != expected.is_quantized:
        raise_mismatch_error("is_quantized", actual.is_quantized, expected.is_quantized)
    elif actual.is_quantized and actual.qscheme() != expected.qscheme():
        raise_mismatch_error("qscheme()", actual.qscheme(), expected.qscheme())
    if actual.layout != expected.layout:
        if check_layout:
            raise_mismatch_error("layout", actual.layout, expected.layout)
    elif (actual.layout == torch.strided and check_stride and actual.stride() != expected.stride()):
        raise_mismatch_error("stride()", actual.stride(), expected.stride())
    if check_device and actual.device != expected.device:
        raise_mismatch_error("device", actual.device, expected.device)
    if check_dtype and actual.dtype != expected.dtype:
        raise_mismatch_error("dtype", actual.dtype, expected.dtype)


def _equalize_attributes(actual: torch.Tensor,
                         expected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Equalizes some attributes of two tensors for value comparison.
    If ``actual`` and ``expected`` are ...
    - ... not on the same :attr:`~torch.Tensor.device`, they are moved CPU memory.
    - ... not of the same ``dtype``, they are promoted  to a common ``dtype`` (according to
        :func:`torch.promote_types`).
    - ... not of the same ``layout``, they are converted to strided tensors.
    Args:
        actual (Tensor): Actual tensor.
        expected (Tensor): Expected tensor.
    Returns:
        (Tuple[Tensor, Tensor]): Equalized tensors.
    """
    # The comparison logic uses operators currently not supported by the MPS backends.
    #  See https://github.com/pytorch/pytorch/issues/77144 for details.
    # TODO: Remove this conversion as soon as all operations are supported natively by the MPS backend
    if actual.is_mps or expected.is_mps:  # type: ignore[attr-defined]
        actual = actual.cpu()
        expected = expected.cpu()
    if actual.device != expected.device:
        actual = actual.cpu()
        expected = expected.cpu()
    if actual.dtype != expected.dtype:
        actual_dtype = actual.dtype
        expected_dtype = expected.dtype
        # For uint64, this is not sound in general, which is why promote_types doesn't
        # allow it, but for easy testing, we're unlikely to get confused
        # by large uint64 overflowing into negative int64
        if actual_dtype in [torch.uint64, torch.uint32, torch.uint16]:
            actual_dtype = torch.int64
        if expected_dtype in [torch.uint64, torch.uint32, torch.uint16]:
            expected_dtype = torch.int64
        dtype = torch.promote_types(actual_dtype, expected_dtype)
        actual = actual.to(dtype)
        expected = expected.to(dtype)
    if actual.layout != expected.layout:
        # These checks are needed, since Tensor.to_dense() fails on tensors that are already strided
        actual = actual.to_dense() if actual.layout != torch.strided else actual
        expected = (expected.to_dense() if expected.layout != torch.strided else expected)
    return actual, expected


def torch_assert_close(
    tensor_a,
    tensor_b,
    rtol=1e-2,
    atol=1e-3,
    max_mismatched_ratio=0.001,
    verbose: bool = False,
    equal_nan: bool = True,
    check_device: bool = True,
    check_dtype: bool = True,
    check_layout: bool = True,
    check_stride: bool = False,
    base_name: str = "LHS",
    ref_name: str = "RHS",
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

    _compare_attributes(
        tensor_a,
        tensor_b,
        check_device=check_device,
        check_dtype=check_dtype,
        check_layout=check_layout,
        check_stride=check_stride)
    tensor_a, tensor_b = _equalize_attributes(tensor_a, tensor_b)

    mismatched = ~torch.isclose(tensor_a, tensor_b, rtol=rtol, atol=atol, equal_nan=equal_nan)
    # Compute the absolute difference between the two tensors
    diff = torch.abs(tensor_a - tensor_b)
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

    # If there are mismatched elements, print the first mismatch
    if num_mismatched > 0:
        # Find the first mismatch index
        flat_idx = torch.argmax(mismatched.view(-1).int()).item()
        idx = np.unravel_index(flat_idx, tensor_a.shape)
        idx = [int(i) for i in idx]
        a_val = tensor_a.reshape(-1)[flat_idx].item()
        b_val = tensor_b.reshape(-1)[flat_idx].item()
        abs_diff = abs(a_val - b_val)
        rel_diff = abs_diff / (abs(b_val) + 1e-12)
        mismatch_info = (f"\nFirst mismatch at index {idx}: "
                         f"lhs={a_val:.6f}, rhs={b_val:.6f}, "
                         f"abs_diff={abs_diff:.6f}, rel_diff={rel_diff:.6f}")
    else:
        mismatch_info = ""

    # Modify the exception information
    if num_mismatched > max_allowed_mismatched:
        raise AssertionError(
            f"Too many mismatched elements: {num_mismatched} > {max_allowed_mismatched} "
            f"({max_mismatched_ratio * 100:.2f}% allowed, but get {num_mismatched / total_elements * 100:.2f}%)."
            f"{mismatch_info}"
            f"\nGreatest absolute difference: {diff.max().item()}, "
            f"Greatest relative difference: {(diff / (torch.abs(tensor_b) + 1e-12)).max().item()}"
            f"\n{base_name}: {tensor_a}"
            f"\n{ref_name}: {tensor_b}")
    else:
        return True
