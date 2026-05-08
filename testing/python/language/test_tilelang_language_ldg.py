"""Tests for load_global_32, load_global_64, load_global_128, load_global_256 intrinsics codegen using eager jit style."""

import tilelang
import tilelang.language as T
import tilelang.testing
import torch


@tilelang.testing.requires_cuda
def test_ldg32_codegen():
    """Test that ldg32 generates tl::load_global_32 in CUDA source."""

    @tilelang.jit
    def ldg32_kernel(X, Y):
        N = T.const("N")
        X: T.Tensor[[N], T.float32]
        Y: T.Tensor[[N], T.float32]

        with T.Kernel(N, threads=32) as pid:
            Y[pid] = T.reinterpret(T.ldg32(X[pid]), T.float32)

    X = torch.randn(128, dtype=torch.float32, device="cuda")
    Y = torch.empty(128, dtype=torch.float32, device="cuda")

    ldg32_kernel(X, Y)
    src = ldg32_kernel.get_kernel_source(N=128)
    print("=== ldg32 codegen ===")
    print(src)
    # Verify codegen
    assert "load_global_32" in src, "Expected load_global_32 call in generated CUDA source"

    # Verify correctness
    Y_ref = X
    torch.testing.assert_close(Y, Y_ref, atol=1e-5, rtol=1e-5)


@tilelang.testing.requires_cuda
def test_ldg64_codegen():
    """Test that ldg64 generates tl::load_global_64 in CUDA source."""

    @tilelang.jit
    def ldg64_kernel(X, Y):
        N = T.const("N")
        X: T.Tensor[[N], T.float32]
        Y: T.Tensor[[N], T.float32]

        with T.Kernel(N // 2, threads=32) as pid:
            Y[pid * 2 : pid * 2 + 2] = T.reinterpret(T.ldg64(X[pid * 2 : pid * 2 + 2]), T.float32x2)

    X = torch.randn(128, dtype=torch.float32, device="cuda")
    Y = torch.empty(128, dtype=torch.float32, device="cuda")

    ldg64_kernel(X, Y)

    # Verify codegen
    src = ldg64_kernel.get_kernel_source(N=128)
    print("=== ldg64 codegen ===")
    print(src)
    assert "load_global_64" in src, "Expected load_global_64 call in generated CUDA source"

    # Verify correctness
    Y_ref = X
    torch.testing.assert_close(Y, Y_ref, atol=1e-5, rtol=1e-5)


@tilelang.testing.requires_cuda
def test_ldg128_codegen():
    """Test that ldg128 generates tl::load_global_128 in CUDA source."""

    @tilelang.jit
    def ldg128_kernel(X, Y):
        N = T.const("N")
        X: T.Tensor[[N], T.float32]
        Y: T.Tensor[[N], T.float32]

        with T.Kernel(N // 4, threads=32) as pid:
            Y[pid * 4 : pid * 4 + 4] = T.reinterpret(T.ldg128(X[pid * 4 : pid * 4 + 4]), T.float32x4)

    X = torch.randn(128, dtype=torch.float32, device="cuda")
    Y = torch.empty(128, dtype=torch.float32, device="cuda")

    ldg128_kernel(X, Y)

    # Verify codegen
    src = ldg128_kernel.get_kernel_source(N=128)
    print("=== ldg128 codegen ===")
    print(src)
    assert "load_global_128" in src, "Expected load_global_128 call in generated CUDA source"

    # Verify correctness
    Y_ref = X
    torch.testing.assert_close(Y, Y_ref, atol=1e-5, rtol=1e-5)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
def test_ldg256_codegen():
    """Test that ldg256 generates tl::load_global_256 in CUDA source."""

    @tilelang.jit
    def ldg256_kernel(X, Y):
        N = T.const("N")
        X: T.Tensor[[N], T.float32]
        Y: T.Tensor[[N], T.float32]

        with T.Kernel(N // 8, threads=32) as pid:
            Y[pid * 8 : pid * 8 + 8] = T.reinterpret(T.ldg256(X[pid * 8 : pid * 8 + 8]), T.float32x8)

    X = torch.randn(256, dtype=torch.float32, device="cuda")
    Y = torch.empty(256, dtype=torch.float32, device="cuda")

    ldg256_kernel(X, Y)

    # Verify codegen
    src = ldg256_kernel.get_kernel_source(N=256)
    print("=== ldg256 codegen ===")
    print(src)
    assert "load_global_256" in src, "Expected load_global_256 call in generated CUDA source"

    # Verify correctness
    Y_ref = X
    torch.testing.assert_close(Y, Y_ref, atol=1e-5, rtol=1e-5)


@tilelang.testing.requires_cuda
def test_ldg32_predicated_codegen():
    """Test that ldg32 with predicate generates tl::load_global_32_conditional(ptr, pred) in CUDA source."""

    @tilelang.jit
    def ldg32_pred_kernel(X, Y):
        N = T.const("N")
        X: T.Tensor[[N], T.float32]
        Y: T.Tensor[[N], T.float32]

        with T.Kernel(N, threads=32) as pid:
            # Only load for the first half of elements
            Y[pid] = T.reinterpret(T.ldg32(X[pid], pred=pid < N // 2), T.float32)

    X = torch.randn(128, dtype=torch.float32, device="cuda")
    Y = torch.zeros(128, dtype=torch.float32, device="cuda")

    ldg32_pred_kernel(X, Y)
    src = ldg32_pred_kernel.get_kernel_source(N=128)
    print("=== ldg32 predicated codegen ===")
    print(src)
    # Verify codegen - should have load_global_32_conditional with two arguments and non-trivial predicate
    assert "load_global_32_conditional" in src, "Expected load_global_32_conditional call in generated CUDA source"

    # Verify correctness
    Y_ref = torch.zeros(128, dtype=torch.float32, device="cuda")
    for i in range(128):
        if i < 64:
            Y_ref[i] = X[i]
        else:
            Y_ref[i] = 0

    torch.testing.assert_close(Y, Y_ref, atol=1e-5, rtol=1e-5)


@tilelang.testing.requires_cuda
def test_ldg64_predicated_codegen():
    """Test that ldg64 with predicate generates tl::load_global_64_conditional(ptr, pred) in CUDA source."""

    @tilelang.jit
    def ldg64_pred_kernel(X, Y):
        N = T.const("N")
        X: T.Tensor[[N], T.float32]
        Y: T.Tensor[[N], T.float32]

        with T.Kernel(N // 2, threads=32) as pid:
            # Only load for the first half of elements
            Y[pid * 2 : pid * 2 + 2] = T.reinterpret(T.ldg64(X[pid * 2 : pid * 2 + 2], pred=pid < N // 4), T.float32x2)

    X = torch.randn(128, dtype=torch.float32, device="cuda")
    Y = torch.zeros(128, dtype=torch.float32, device="cuda")

    ldg64_pred_kernel(X, Y)

    # Verify codegen
    src = ldg64_pred_kernel.get_kernel_source(N=128)
    print("=== ldg64 predicated codegen ===")
    print(src)
    assert "load_global_64_conditional" in src, "Expected load_global_64_conditional call in generated CUDA source"

    # Verify correctness
    Y_ref = torch.zeros(128, dtype=torch.float32, device="cuda")
    for i in range(128):
        if i < 64:
            Y_ref[i] = X[i]
        else:
            Y_ref[i] = 0

    torch.testing.assert_close(Y, Y_ref, atol=1e-5, rtol=1e-5)


@tilelang.testing.requires_cuda
def test_ldg128_predicated_codegen():
    """Test that ldg128 with predicate generates tl::load_global_128_conditional(ptr, pred) in CUDA source."""

    @tilelang.jit
    def ldg128_pred_kernel(X, Y):
        N = T.const("N")
        X: T.Tensor[[N], T.float32]
        Y: T.Tensor[[N], T.float32]

        with T.Kernel(N // 4, threads=32) as pid:
            # Only load for the first half of elements
            Y[pid * 4 : pid * 4 + 4] = T.reinterpret(T.ldg128(X[pid * 4 : pid * 4 + 4], pred=pid < N // 8), T.float32x4)

    X = torch.randn(128, dtype=torch.float32, device="cuda")
    Y = torch.zeros(128, dtype=torch.float32, device="cuda")

    ldg128_pred_kernel(X, Y)

    # Verify codegen
    src = ldg128_pred_kernel.get_kernel_source(N=128)
    print("=== ldg128 predicated codegen ===")
    print(src)
    assert "load_global_128_conditional" in src, "Expected load_global_128_conditional call in generated CUDA source"

    # Verify correctness
    Y_ref = torch.zeros(128, dtype=torch.float32, device="cuda")
    for i in range(128):
        if i < 64:
            Y_ref[i] = X[i]
        else:
            Y_ref[i] = 0

    torch.testing.assert_close(Y, Y_ref, atol=1e-5, rtol=1e-5)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
def test_ldg256_predicated_codegen():
    """Test that ldg256 with predicate generates tl::load_global_256_conditional(ptr, pred) in CUDA source."""

    @tilelang.jit
    def ldg256_pred_kernel(X, Y):
        N = T.const("N")
        X: T.Tensor[[N], T.float32]
        Y: T.Tensor[[N], T.float32]

        with T.Kernel(N // 8, threads=32) as pid:
            # Only load for the first half of elements
            Y[pid * 8 : pid * 8 + 8] = T.reinterpret(T.ldg256(X[pid * 8 : pid * 8 + 8], pred=pid < N // 16), T.float32x8)

    X = torch.randn(256, dtype=torch.float32, device="cuda")
    Y = torch.zeros(256, dtype=torch.float32, device="cuda")

    ldg256_pred_kernel(X, Y)

    # Verify codegen
    src = ldg256_pred_kernel.get_kernel_source(N=256)
    print("=== ldg256 predicated codegen ===")
    print(src)
    assert "load_global_256_conditional" in src, "Expected load_global_256_conditional call in generated CUDA source"
    # Verify correctness
    Y_ref = torch.zeros(256, dtype=torch.float32, device="cuda")
    for i in range(256):
        if i < 128:
            Y_ref[i] = X[i]
        else:
            Y_ref[i] = 0

    torch.testing.assert_close(Y, Y_ref, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    tilelang.testing.main()
