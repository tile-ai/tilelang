"""Tests for subtype (subbyte dtype like 4bit int or fp) shape and stride bindings ."""

import torch
import tilelang
import tilelang.testing
import tilelang.language as T


@tilelang.lazy_jit
def basic_shape_kernel(x):
    m = T.dynamic("m")
    x: T.Tensor[(m, 16), T.float4_e2m1fn]

    with T.Kernel(1, threads=32):
        pass


@tilelang.lazy_jit
def strided_kernel(x):
    m = T.dynamic("m")
    s = T.dynamic("s")
    x: T.StridedTensor[[m, 16], [s, 1], T.float4_e2m1fn]

    with T.Kernel(1, threads=32):
        pass


@tilelang.testing.requires_cuda
def test_subtype_basic_shape_binding():
    """Test that symbolic shape variables are correctly bound for subtype buffers.

    For fp4 (4 bits), pack_factor = 8 / 4 = 2.
    Logical shape [m, 16] corresponds to runtime shape [m, 8].
    The symbolic variable 'm' should be bound from runtime_shape[0].
    """
    # Runtime shape [4, 8] -> Logical shape [4, 16] for fp4
    t = torch.randint(0, 256, (4, 8), dtype=torch.uint8, device="cuda")
    basic_shape_kernel(t)


@tilelang.testing.requires_cuda
def test_subtype_stride_binding():
    """Test that symbolic stride variables are correctly bound for subtype buffers.

    For fp4, the stride relationship is:
    - Last dimension: logical_stride = runtime_stride
    - Other dimensions: logical_stride = runtime_stride * pack_factor

    With pack_factor = 2:
    - Runtime stride [8, 1] -> Logical stride [16, 1]
    """
    # Contiguous tensor: runtime stride [8, 1] -> logical stride [16, 1]
    t = torch.randint(0, 256, (4, 8), dtype=torch.uint8, device="cuda")
    strided_kernel(t)


@tilelang.testing.requires_cuda
def test_subtype_noncontiguous_tensor():
    """Test subtype with non-contiguous (strided) tensor.

    Create a tensor with stride [16, 1] (by slicing every other row).
    This corresponds to logical stride [32, 1] for fp4.
    """
    # Create a larger tensor and slice to get non-contiguous strides
    t_large = torch.randint(0, 256, (8, 8), dtype=torch.uint8, device="cuda")
    # Slice every other row: shape [4, 8] but stride [16, 1]
    t_noncontig = t_large[::2, :]
    assert t_noncontig.shape == (4, 8)
    assert t_noncontig.stride() == (16, 1)

    strided_kernel(t_noncontig)


@tilelang.testing.requires_cuda
def test_subtype_different_m_values():
    """Test subtype binding with different values of symbolic variable m."""
    for m in [1, 2, 4, 8, 16, 32]:
        # Runtime shape [m, 8] -> Logical shape [m, 16] for fp4
        t = torch.randint(0, 256, (m, 8), dtype=torch.uint8, device="cuda")
        basic_shape_kernel(t)


@tilelang.testing.requires_cuda
def test_subtype_different_strides():
    """Test subtype stride binding with different stride values."""
    # Test with different non-contiguous strides
    for stride_multiplier in [1, 2, 4]:
        # Create tensor with specific stride pattern
        t_large = torch.randint(0, 256, (4 * stride_multiplier, 8), dtype=torch.uint8, device="cuda")
        # Slice to get stride [8 * stride_multiplier, 1]
        t_strided = t_large[::stride_multiplier, :]
        assert t_strided.shape == (4, 8)
        assert t_strided.stride() == (8 * stride_multiplier, 1)

        strided_kernel(t_strided)


@tilelang.lazy_jit
def symbolic_last_dim_kernel(x):
    """Kernel with symbolic variable in the last dimension."""
    n = T.dynamic("n")
    x: T.Tensor[(4, n), T.float4_e2m1fn]

    with T.Kernel(1, threads=32):
        pass


@tilelang.lazy_jit
def symbolic_last_dim_strided_kernel(x):
    """Kernel with symbolic variable in both shape and stride of last dimension."""
    n = T.dynamic("n")
    s = T.dynamic("s")
    x: T.StridedTensor[[4, n], [s, 1], T.float4_e2m1fn]

    with T.Kernel(1, threads=32):
        pass


@tilelang.lazy_jit
def shared_symbolic_kernel(x, y):
    """Kernel with shared symbolic variable across multiple buffers."""
    m = T.dynamic("m")
    x: T.Tensor[(m, 16), T.float4_e2m1fn]
    y: T.Tensor[(m * 4, 16), T.float4_e2m1fn]

    with T.Kernel(1, threads=32):
        pass


@tilelang.lazy_jit
def shared_symbolic_strided_kernel(x, y):
    """Kernel with shared symbolic variable in strides."""
    m = T.dynamic("m")
    s = T.dynamic("s")
    x: T.StridedTensor[[m, 16], [s, 1], T.float4_e2m1fn]
    y: T.StridedTensor[[m * 2, 16], [s, 1], T.float4_e2m1fn]

    with T.Kernel(1, threads=32):
        pass


@tilelang.lazy_jit
def complex_expr_kernel(x, y):
    """Kernel with complex expressions involving symbolic variables."""
    m = T.dynamic("m")
    n = T.dynamic("n")
    x: T.Tensor[(m, n * 2), T.float4_e2m1fn]
    y: T.Tensor[(m * 2, n), T.float4_e2m1fn]

    with T.Kernel(1, threads=32):
        pass


@tilelang.testing.requires_cuda
def test_subtype_symbolic_last_dim():
    """Test symbolic variable in the last dimension.

    For fp4, the last dimension has pack_factor applied:
    Logical shape [4, n] with n=32 corresponds to runtime shape [4, 16].
    So n = runtime_shape[1] * pack_factor = 16 * 2 = 32.
    """
    # Runtime shape [4, 16] -> Logical shape [4, 32] for fp4
    t = torch.randint(0, 256, (4, 16), dtype=torch.uint8, device="cuda")
    symbolic_last_dim_kernel(t)


@tilelang.testing.requires_cuda
def test_subtype_symbolic_last_dim_various_sizes():
    """Test symbolic last dimension with various sizes."""
    for n_runtime in [4, 8, 16, 32]:
        # Logical n = runtime_n * 2 (pack_factor for fp4)
        t = torch.randint(0, 256, (4, n_runtime), dtype=torch.uint8, device="cuda")
        symbolic_last_dim_kernel(t)


@tilelang.testing.requires_cuda
def test_subtype_symbolic_last_dim_strided():
    """Test symbolic variable in last dimension with strides.

    Note: For subtype (packed storage), the last dimension stride must be 1
    since elements are packed together. Column slicing doesn't make sense
    for packed types.
    """
    # Contiguous tensor
    t = torch.randint(0, 256, (4, 16), dtype=torch.uint8, device="cuda")
    symbolic_last_dim_strided_kernel(t)

    # Non-contiguous tensor (row slicing only, last dim stride stays 1)
    t_large = torch.randint(0, 256, (8, 16), dtype=torch.uint8, device="cuda")
    t_strided = t_large[::2, :]  # shape [4, 16], stride [32, 1]
    assert t_strided.shape == (4, 16)
    assert t_strided.stride() == (32, 1)
    symbolic_last_dim_strided_kernel(t_strided)


@tilelang.testing.requires_cuda
def test_subtype_shared_symbolic():
    """Test shared symbolic variable across multiple buffers.

    x has shape (m, 16), y has shape (m*4, 16).
    For fp4 with pack_factor=2:
    - x runtime shape (m, 8)
    - y runtime shape (m*4, 8)

    If m=2:
    - x runtime: (2, 8), logical: (2, 16)
    - y runtime: (8, 8), logical: (8, 16)
    """
    for m in [1, 2, 4, 8]:
        x = torch.randint(0, 256, (m, 8), dtype=torch.uint8, device="cuda")
        y = torch.randint(0, 256, (m * 4, 8), dtype=torch.uint8, device="cuda")
        shared_symbolic_kernel(x, y)


@tilelang.testing.requires_cuda
def test_subtype_shared_symbolic_strided():
    """Test shared symbolic variable in strides across multiple buffers.

    x has shape (m, 16) with stride (s, 1)
    y has shape (m*2, 16) with stride (s, 1)
    """
    for m in [2, 4, 8]:
        # Create contiguous tensors
        x = torch.randint(0, 256, (m, 8), dtype=torch.uint8, device="cuda")
        y = torch.randint(0, 256, (m * 2, 8), dtype=torch.uint8, device="cuda")
        shared_symbolic_strided_kernel(x, y)


@tilelang.testing.requires_cuda
def test_subtype_shared_symbolic_strided_noncontig():
    """Test shared symbolic stride with non-contiguous tensors."""
    # Create non-contiguous tensors with same stride pattern
    x_large = torch.randint(0, 256, (8, 8), dtype=torch.uint8, device="cuda")
    y_large = torch.randint(0, 256, (16, 8), dtype=torch.uint8, device="cuda")

    # Slice to get stride [16, 1] for both
    x = x_large[::2, :]  # shape (4, 8), stride (16, 1)
    y = y_large[::2, :]  # shape (8, 8), stride (16, 1)

    assert x.shape == (4, 8)
    assert y.shape == (8, 8)
    assert x.stride() == (16, 1)
    assert y.stride() == (16, 1)

    shared_symbolic_strided_kernel(x, y)


@tilelang.testing.requires_cuda
def test_subtype_complex_expressions():
    """Test complex expressions with symbolic variables.

    x has shape (m, n*2), y has shape (m*2, n).
    For fp4 with pack_factor=2:
    - x logical (m, n*2) -> runtime (m, n)
    - y logical (m*2, n) -> runtime (m*2, n/2)
    """
    # m=4, n=16: x logical (4, 32), y logical (8, 16)
    # x runtime (4, 16), y runtime (8, 8)
    m, n = 4, 16
    x = torch.randint(0, 256, (m, n), dtype=torch.uint8, device="cuda")
    y = torch.randint(0, 256, (m * 2, n // 2), dtype=torch.uint8, device="cuda")
    complex_expr_kernel(x, y)


@tilelang.testing.requires_cuda
def test_subtype_complex_expressions_various():
    """Test complex expressions with various m, n values."""
    for m, n in [(2, 8), (4, 16), (8, 32)]:
        # x logical (m, n*2) -> runtime (m, n)
        # y logical (m*2, n) -> runtime (m*2, n/2)
        x = torch.randint(0, 256, (m, n), dtype=torch.uint8, device="cuda")
        y = torch.randint(0, 256, (m * 2, n // 2), dtype=torch.uint8, device="cuda")
        complex_expr_kernel(x, y)


if __name__ == "__main__":
    tilelang.testing.main()
