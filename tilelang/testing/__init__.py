# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import sys
import inspect
import pytest
import random
import torch
import numpy as np
from tvm.testing.utils import *


# pytest.main() wrapper to allow running single test file
def main():
    test_file = inspect.getsourcefile(sys._getframe(1))
    sys.exit(pytest.main([test_file] + sys.argv[1:]))


def torch_assert_close(tensor_a,
                       tensor_b,
                       rtol=1e-2,
                       atol=1e-2,
                       max_mismatched_ratio=0.001,
                       verbose=False):
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

    # Assert shapes are the same
    assert tensor_a.shape == tensor_b.shape, f"Tensor shapes must be the same, but got {tensor_a.shape} and {tensor_b.shape}"

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

    # If there are mismatched elements, print the first mismatch
    if num_mismatched > 0:
        # Find the first mismatch index
        flat_idx = torch.argmax(mismatched.view(-1).int()).item()
        idx = np.unravel_index(flat_idx, tensor_a.shape)
        idx = [int(i) for i in idx]
        a_val = tensor_a.view(-1)[flat_idx].item()
        b_val = tensor_b.view(-1)[flat_idx].item()
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
            f"Greatest relative difference: {(diff / (torch.abs(tensor_b) + 1e-12)).max().item()}.")
    else:
        return True


def set_random_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
