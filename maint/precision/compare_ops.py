#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ruff: noqa
"""
Precision comparison tool for CUDA Precise/Fast, Triton, Triton LibDevice, PyTorch, and TileLang operations.
"""

import os
import argparse
import sys
from typing import Dict, Optional, Tuple
import torch
from torch.utils.cpp_extension import load
import triton
import triton.language as tl
from triton.language.extra import libdevice
import tilelang
import tilelang.language as T

from tilelang.contrib import nvcc
from tilelang.utils.target import determine_target

# GPU configuration setup
target = determine_target(return_object=True)
compute_version = nvcc.get_target_compute_version(target)
major, minor = nvcc.parse_compute_version(compute_version)
os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

# Operator enumeration - must match OperatorType in C++
OP_NAMES: Dict[int, str] = {
    0: "div",
    1: "reciprocal",
    2: "exp",
    3: "log",
    4: "sin",
    5: "cos",
    6: "sqrt",
    7: "tanh",
    8: "rsqrt",
    9: "inv_sqrt",
}

# Block sizes for kernels
TRITON_BLOCK_SIZE = 1024
TILELANG_BLOCK_M = 32
TILELANG_BLOCK_N = 32
TILELANG_THREADS = 128


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Precision comparison tool for various CUDA implementations")
    parser.add_argument("--n", type=int, default=1000000, help="Number of elements to test")
    parser.add_argument("--low", type=float, default=-4.0, help="Lower bound for random values")
    parser.add_argument("--high", type=float, default=4.0, help="Upper bound for random values")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    return parser.parse_args()


def initialize_cuda() -> torch.nn.Module:
    """Initialize CUDA and load the custom operators module."""
    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        sys.exit(1)

    return load(
        name="cuda_ops",
        sources=["cuda_ops.cu"],
        extra_cuda_cflags=[],  # No fast_math flags
    )


# Initialize global variables
args = parse_arguments()
torch.manual_seed(args.seed)
mod = initialize_cuda()
device = torch.device("cuda")
n = args.n
low, high = args.low, args.high


# Triton kernels
@triton.jit
def triton_binary_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Standard Triton kernel for binary operations (div)."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    result = x / y  # Division operation
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.jit
def triton_libdevice_binary_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """LibDevice Triton kernel for binary operations (div)."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    result = libdevice.div_rn(x, y)  # Round to nearest
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.jit
def tl_tanh(x):
    """Triton tanh implementation using sigmoid."""
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def triton_unary_kernel(x_ptr, out_ptr, n_elements, op_id: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """Standard Triton kernel for unary operations."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    if op_id == 1:  # reciprocal
        result = 1.0 / x
    elif op_id == 2:  # exp
        result = tl.exp(x)
    elif op_id == 3:  # log
        result = tl.log(x)
    elif op_id == 4:  # sin
        result = tl.sin(x)
    elif op_id == 5:  # cos
        result = tl.cos(x)
    elif op_id == 6:  # sqrt
        result = tl.sqrt(x)
    elif op_id == 7:  # tanh
        result = tl_tanh(x)
    elif op_id == 8:  # rsqrt
        result = tl.rsqrt(x)
    elif op_id == 9:  # inv_sqrt
        result = 1.0 / tl.sqrt(x)
    else:
        result = x  # Default case

    tl.store(out_ptr + offsets, result, mask=mask)


@triton.jit
def triton_libdevice_unary_kernel(x_ptr, out_ptr, n_elements, op_id: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """LibDevice Triton kernel for unary operations."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    if op_id == 1:  # reciprocal
        result = libdevice.rcp_rn(x)
    elif op_id == 2:  # exp
        result = libdevice.exp(x)
    elif op_id == 3:  # log
        result = libdevice.log(x)
    elif op_id == 4:  # sin
        result = libdevice.sin(x)
    elif op_id == 5:  # cos
        result = libdevice.cos(x)
    elif op_id == 6:  # sqrt
        result = libdevice.sqrt_rn(x)  # Round to nearest
    elif op_id == 7:  # tanh
        result = libdevice.tanh(x)
    elif op_id == 8:  # rsqrt
        result = libdevice.rsqrt_rn(x)
    elif op_id == 9:  # inv_sqrt
        result = libdevice.rcp_rn(libdevice.sqrt_rn(x))
    else:
        result = x  # Default case

    tl.store(out_ptr + offsets, result, mask=mask)


# TileLang kernel generators
def make_tilelang_unary_kernel(M: int, N: int, op_id: int, use_fastmath: bool = False):
    """Generate TileLang unary operation kernel."""

    @T.prim_func
    def tilelang_unary_kernel(
        A: T.Tensor((M, N), T.float32),
        B: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(T.ceildiv(N, TILELANG_BLOCK_N), T.ceildiv(M, TILELANG_BLOCK_M), threads=TILELANG_THREADS) as (bx, by):
            for i, j in T.Parallel(TILELANG_BLOCK_M, TILELANG_BLOCK_N):
                row = by * TILELANG_BLOCK_M + i
                col = bx * TILELANG_BLOCK_N + j
                x = A[row, col]

                if op_id == 1:  # reciprocal
                    B[row, col] = 1.0 / x
                elif op_id == 2:  # exp
                    B[row, col] = T.exp(x)
                elif op_id == 3:  # log
                    B[row, col] = T.log(x)
                elif op_id == 4:  # sin
                    B[row, col] = T.sin(x)
                elif op_id == 5:  # cos
                    B[row, col] = T.cos(x)
                elif op_id == 6:  # sqrt
                    B[row, col] = T.sqrt(x)
                elif op_id == 7:  # tanh
                    B[row, col] = T.tanh(x)
                elif op_id == 8:  # rsqrt
                    B[row, col] = T.rsqrt(x)
                elif op_id == 9:  # inv_sqrt
                    B[row, col] = 1.0 / T.sqrt(x)
                else:
                    B[row, col] = x  # Default case

    return tilelang_unary_kernel


def make_tilelang_binary_kernel(M: int, N: int):
    """Generate TileLang binary operation kernel (division)."""

    @T.prim_func
    def tilelang_binary_kernel(
        A: T.Tensor((M, N), T.float32),
        B: T.Tensor((M, N), T.float32),
        C: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(T.ceildiv(N, TILELANG_BLOCK_N), T.ceildiv(M, TILELANG_BLOCK_M), threads=TILELANG_THREADS) as (bx, by):
            for i, j in T.Parallel(TILELANG_BLOCK_M, TILELANG_BLOCK_N):
                row = by * TILELANG_BLOCK_M + i
                col = bx * TILELANG_BLOCK_N + j
                x = A[row, col]
                y = B[row, col]
                C[row, col] = x / y  # Division operation

    return tilelang_binary_kernel


def tilelang_op(x: torch.Tensor, op_id: int, y: Optional[torch.Tensor] = None, use_fastmath: bool = False) -> torch.Tensor:
    """TileLang operation interface."""
    assert x.is_cuda

    # Reshape 1D tensor to 2D for TileLang kernels
    original_shape = x.shape
    if len(x.shape) == 1:
        x = x.view(1, -1)
        if y is not None:
            y = y.view(1, -1)

    M, N = x.shape

    if op_id == 0:  # Division - binary operation
        assert y is not None, "Division operation requires second operand"
        kernel_func = make_tilelang_binary_kernel(M, N)
        kernel = tilelang.compile(
            kernel_func,
            out_idx=[2],
            target="cuda",
            pass_configs={
                tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: use_fastmath,
            },
        )
        out = kernel(x, y)
    else:  # Unary operation
        kernel_func = make_tilelang_unary_kernel(M, N, op_id, use_fastmath)
        kernel = tilelang.compile(
            kernel_func,
            out_idx=[1],
            target="cuda",
            pass_configs={
                tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: use_fastmath,
            },
        )
        out = kernel(x)

    # Restore original shape
    return out.view(original_shape)


def triton_op(x: torch.Tensor, op_id: int, y: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Standard Triton operation interface."""
    assert x.is_cuda
    out = torch.empty_like(x)
    grid = lambda meta: ((x.numel() + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    if op_id == 0:  # Division - binary operation
        assert y is not None, "Division operation requires second operand"
        triton_binary_kernel[grid](x, y, out, x.numel(), BLOCK_SIZE=TRITON_BLOCK_SIZE)
    else:  # Unary operation
        triton_unary_kernel[grid](x, out, x.numel(), op_id, BLOCK_SIZE=TRITON_BLOCK_SIZE)

    return out


def triton_libdevice_op(x: torch.Tensor, op_id: int, y: Optional[torch.Tensor] = None) -> torch.Tensor:
    """LibDevice Triton operation interface."""
    assert x.is_cuda
    out = torch.empty_like(x)
    grid = lambda meta: ((x.numel() + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    if op_id == 0:  # Division - binary operation
        assert y is not None, "Division operation requires second operand"
        triton_libdevice_binary_kernel[grid](x, y, out, x.numel(), BLOCK_SIZE=TRITON_BLOCK_SIZE)
    else:  # Unary operation
        triton_libdevice_unary_kernel[grid](x, out, x.numel(), op_id, BLOCK_SIZE=TRITON_BLOCK_SIZE)

    return out


def get_pytorch_reference(x: torch.Tensor, op_id: int, y: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Get PyTorch reference implementation for the given operation."""
    if op_id == 0:
        assert y is not None, "Division requires second operand"
        return x / y
    elif op_id == 1:
        return torch.reciprocal(x)
    elif op_id == 2:
        return torch.exp(x)
    elif op_id == 3:
        return torch.log(x)
    elif op_id == 4:
        return torch.sin(x)
    elif op_id == 5:
        return torch.cos(x)
    elif op_id == 6:
        return torch.sqrt(x)
    elif op_id == 7:
        return torch.tanh(x)
    elif op_id == 8:
        return torch.rsqrt(x)
    elif op_id == 9:
        return 1 / torch.sqrt(x)
    else:
        raise ValueError(f"Unknown op_id: {op_id}")


def summarize_error(tag: str, output: Optional[torch.Tensor], reference: torch.Tensor) -> None:
    """Summarize and print error statistics for an implementation."""
    if output is None:
        print(f"{tag:<32} FAILED")
        return

    # Convert results to double precision for error calculation
    output_double = output.double()
    reference_double = reference.double() if reference.dtype != torch.float64 else reference

    abs_err = (output_double - reference_double).abs()
    rel_err = abs_err / (reference_double.abs().clamp_min(1e-30))
    print(
        f"{tag:<32} max abs: {abs_err.max():.3e}, mean abs: {abs_err.mean():.3e}, "
        f"max rel: {rel_err.max():.3e}, mean rel: {rel_err.mean():.3e}"
    )


# Precision comparison function
def compare(op_id: int, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> None:
    name = OP_NAMES[op_id]
    print(f"\n=== {name} ===")

    # Create double precision version of input data as reference standard
    x_double = x.double()
    y_double = y.double() if y is not None else None

    # Double CUDA Precise as golden standard
    ref_double = torch.empty_like(x_double)
    mod.launch_double_precise_operator(x_double, y_double, ref_double, op_id)

    # CUDA Precise (FP32)
    ref_float = torch.empty_like(x)
    mod.launch_precise_operator(x, y, ref_float, op_id)

    # CUDA Fast
    result_fast = torch.empty_like(ref_float)
    mod.launch_fast_operator(x, y, result_fast, op_id)

    # PyTorch reference
    torch_ref = get_pytorch_reference(x, op_id, y)

    # Test implementations with error handling
    implementations = [
        ("Standard Triton", lambda: triton_op(x, op_id, y)),
        ("LibDevice Triton", lambda: triton_libdevice_op(x, op_id, y)),
        ("TileLang Standard", lambda: tilelang_op(x, op_id, y, use_fastmath=False)),
        ("TileLang Fastmath", lambda: tilelang_op(x, op_id, y, use_fastmath=True)),
    ]

    results = {}
    for name, impl_func in implementations:
        try:
            results[name] = impl_func()
        except Exception as e:
            print(f"{name} failed: {e}")
            results[name] = None

    # Print comparison header
    print(f"{'Implementation':<32} {'Max Abs Error':<19} {'Mean Abs Error':<20} {'Max Rel Error':<19} {'Mean Rel Error'}")
    print("-" * 90)

    # Compare all implementations against double precision reference
    comparisons = [
        ("FP32 Precise vs Double", ref_float),
        ("Triton LibDevice vs Double", results.get("LibDevice Triton")),
        ("TileLang vs Double", results.get("TileLang Standard")),
        ("PyTorch vs Double", torch_ref),
        ("Triton vs Double", results.get("Standard Triton")),
        ("TileLang Fastmath vs Double", results.get("TileLang Fastmath")),
        ("CUDA Fast vs Double", result_fast),
    ]

    for tag, output in comparisons:
        summarize_error(tag, output, ref_double)


def generate_test_data(op_id: int, n: int, device: torch.device, low: float, high: float) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Generate appropriate test data for each operation."""
    if op_id == 0:  # Division
        x = torch.empty(n, device=device).uniform_(low, high)
        y = torch.empty(n, device=device).uniform_(1e-3, high)  # Avoid division by zero
        return x, y
    elif op_id in (3, 6):  # log and sqrt need positive inputs
        x = torch.empty(n, device=device).uniform_(1e-3, high)
        return x, None
    elif op_id in (8, 9):  # rsqrt and inv_sqrt need positive inputs (use consistent data)
        x = torch.empty(n, device=device).uniform_(1e-3, high)
        return x, None
    elif op_id == 1:  # reciprocal - avoid values close to zero
        x = torch.empty(n, device=device).uniform_(1e-3, high)
        return x, None
    else:  # General case
        x = torch.empty(n, device=device).uniform_(low, high)
        return x, None


def main() -> None:
    """Main execution function."""
    print("Precision comparison between CUDA Precise/Fast, Triton, Triton LibDevice, PyTorch, and TileLang")
    print("=" * 90)

    for op_id in range(len(OP_NAMES)):
        try:
            x, y = generate_test_data(op_id, n, device, low, high)
            compare(op_id, x, y)
        except Exception as e:
            print(f"Error in {OP_NAMES[op_id]}: {e}")
            continue


if __name__ == "__main__":
    main()
