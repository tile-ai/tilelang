"""Tests for store_global_32, store_global_64, store_global_128, store_global_256 intrinsics codegen using eager jit style."""

import tilelang
import tilelang.language as T
import tilelang.testing
import torch


@tilelang.testing.requires_cuda
def test_stg32_codegen():
    """Test that stg32 generates tl::store_global_32 in CUDA source."""

    @tilelang.jit
    def stg32_kernel(X, Y):
        N = T.const("N")
        X: T.Tensor[[N], T.float32]
        Y: T.Tensor[[N], T.float32]

        with T.Kernel(N, threads=32) as pid:
            val = T.reinterpret(X[pid], T.uint32)
            T.stg32(Y[pid], val)

    X = torch.randn(128, dtype=torch.float32, device="cuda")
    Y = torch.empty(128, dtype=torch.float32, device="cuda")

    stg32_kernel(X, Y)
    src = stg32_kernel.get_kernel_source(N=128)
    print("=== stg32 codegen ===")
    print(src)
    # Verify codegen
    assert "store_global_32" in src, "Expected store_global_32 call in generated CUDA source"

    # Verify correctness
    torch.testing.assert_close(Y, X, atol=1e-5, rtol=1e-5)


@tilelang.testing.requires_cuda
def test_stg64_codegen():
    """Test that stg64 generates tl::store_global_64 in CUDA source."""

    @tilelang.jit
    def stg64_kernel(X, Y):
        N = T.const("N")
        X: T.Tensor[[N], T.float32]
        Y: T.Tensor[[N], T.float32]

        with T.Kernel(N // 2, threads=32) as pid:
            val = T.ldg64(X[pid * 2 : pid * 2 + 2])
            T.stg64(Y[pid * 2 : pid * 2 + 2], val)

    X = torch.randn(128, dtype=torch.float32, device="cuda")
    Y = torch.empty(128, dtype=torch.float32, device="cuda")

    stg64_kernel(X, Y)

    # Verify codegen
    src = stg64_kernel.get_kernel_source(N=128)
    print("=== stg64 codegen ===")
    print(src)
    assert "store_global_64" in src, "Expected store_global_64 call in generated CUDA source"

    # Verify correctness
    torch.testing.assert_close(Y, X, atol=1e-5, rtol=1e-5)


@tilelang.testing.requires_cuda
def test_stg128_codegen():
    """Test that stg128 generates tl::store_global_128 in CUDA source."""

    @tilelang.jit
    def stg128_kernel(X, Y):
        N = T.const("N")
        X: T.Tensor[[N], T.float32]
        Y: T.Tensor[[N], T.float32]

        with T.Kernel(N // 4, threads=32) as pid:
            val = T.ldg128(X[pid * 4 : pid * 4 + 4])
            T.stg128(Y[pid * 4 : pid * 4 + 4], val)

    X = torch.randn(128, dtype=torch.float32, device="cuda")
    Y = torch.empty(128, dtype=torch.float32, device="cuda")

    stg128_kernel(X, Y)

    # Verify codegen
    src = stg128_kernel.get_kernel_source(N=128)
    print("=== stg128 codegen ===")
    print(src)
    assert "store_global_128" in src, "Expected store_global_128 call in generated CUDA source"

    # Verify correctness
    torch.testing.assert_close(Y, X, atol=1e-5, rtol=1e-5)


@tilelang.testing.requires_cuda
def test_lds_sts32_codegen():
    """Test that lds32/sts32 generate shared memory helper calls."""

    @tilelang.jit
    def lds_sts32_kernel(X, Y):
        N = T.const("N")
        X: T.Tensor[[N], T.float32]
        Y: T.Tensor[[N], T.float32]

        with T.Kernel(N, threads=32) as pid:
            S = T.alloc_shared((N,), T.float32)
            T.sts32(S[pid], T.reinterpret(X[pid], T.uint32))
            T.sync_threads()
            val = T.lds32(S[pid])
            T.stg32(Y[pid], val)

    X = torch.randn(128, dtype=torch.float32, device="cuda")
    Y = torch.empty(128, dtype=torch.float32, device="cuda")

    lds_sts32_kernel(X, Y)
    src = lds_sts32_kernel.get_kernel_source(N=128)
    print("=== lds/sts32 codegen ===")
    print(src)
    assert "store_shared_32" in src
    assert "load_shared_32" in src
    torch.testing.assert_close(Y, X, atol=1e-5, rtol=1e-5)


@tilelang.testing.requires_cuda
def test_lds_sts64_codegen():
    """Test that lds64/sts64 generate shared memory helper calls."""

    @tilelang.jit
    def lds_sts64_kernel(X, Y):
        N = T.const("N")
        X: T.Tensor[[N], T.float32]
        Y: T.Tensor[[N], T.float32]

        with T.Kernel(N // 2, threads=32) as pid:
            S = T.alloc_shared((N,), T.float32)
            val = T.ldg64(X[pid * 2 : pid * 2 + 2])
            T.sts64(S[pid * 2 : pid * 2 + 2], val)
            T.sync_threads()
            val = T.lds64(S[pid * 2 : pid * 2 + 2])
            T.stg64(Y[pid * 2 : pid * 2 + 2], val)

    X = torch.randn(128, dtype=torch.float32, device="cuda")
    Y = torch.empty(128, dtype=torch.float32, device="cuda")

    lds_sts64_kernel(X, Y)
    src = lds_sts64_kernel.get_kernel_source(N=128)
    print("=== lds/sts64 codegen ===")
    print(src)
    assert "store_shared_64" in src
    assert "load_shared_64" in src
    torch.testing.assert_close(Y, X, atol=1e-5, rtol=1e-5)


@tilelang.testing.requires_cuda
def test_lds_sts128_codegen():
    """Test that lds128/sts128 generate shared memory helper calls."""

    @tilelang.jit
    def lds_sts128_kernel(X, Y):
        N = T.const("N")
        X: T.Tensor[[N], T.float32]
        Y: T.Tensor[[N], T.float32]

        with T.Kernel(N // 4, threads=32) as pid:
            S = T.alloc_shared((N,), T.float32)
            val = T.ldg128(X[pid * 4 : pid * 4 + 4])
            T.sts128(S[pid * 4 : pid * 4 + 4], val)
            T.sync_threads()
            val = T.lds128(S[pid * 4 : pid * 4 + 4])
            T.stg128(Y[pid * 4 : pid * 4 + 4], val)

    X = torch.randn(128, dtype=torch.float32, device="cuda")
    Y = torch.empty(128, dtype=torch.float32, device="cuda")

    lds_sts128_kernel(X, Y)
    src = lds_sts128_kernel.get_kernel_source(N=128)
    print("=== lds/sts128 codegen ===")
    print(src)
    assert "store_shared_128" in src
    assert "load_shared_128" in src
    torch.testing.assert_close(Y, X, atol=1e-5, rtol=1e-5)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
def test_stg256_codegen():
    """Test that stg256 generates tl::store_global_256 in CUDA source."""

    @tilelang.jit
    def stg256_kernel(X, Y):
        N = T.const("N")
        X: T.Tensor[[N], T.float32]
        Y: T.Tensor[[N], T.float32]

        with T.Kernel(N // 8, threads=32) as pid:
            val = T.ldg256(X[pid * 8 : pid * 8 + 8])
            T.stg256(Y[pid * 8 : pid * 8 + 8], val)

    X = torch.randn(256, dtype=torch.float32, device="cuda")
    Y = torch.empty(256, dtype=torch.float32, device="cuda")

    stg256_kernel(X, Y)

    # Verify codegen
    src = stg256_kernel.get_kernel_source(N=256)
    print("=== stg256 codegen ===")
    print(src)
    assert "store_global_256" in src, "Expected store_global_256 call in generated CUDA source"

    # Verify correctness
    torch.testing.assert_close(Y, X, atol=1e-5, rtol=1e-5)


@tilelang.testing.requires_cuda
def test_stg32_predicated_codegen():
    """Test that stg32 with predicate generates tl::store_global_32_conditional(ptr, val, pred) in CUDA source."""

    @tilelang.jit
    def stg32_pred_kernel(X, Y):
        N = T.const("N")
        X: T.Tensor[[N], T.float32]
        Y: T.Tensor[[N], T.float32]

        with T.Kernel(N, threads=32) as pid:
            val = T.reinterpret(X[pid], T.uint32)
            # Only store for the first half of elements
            T.stg32(Y[pid], val, pred=pid < N // 2)

    X = torch.randn(128, dtype=torch.float32, device="cuda")
    Y = torch.zeros(128, dtype=torch.float32, device="cuda")

    stg32_pred_kernel(X, Y)
    src = stg32_pred_kernel.get_kernel_source(N=128)
    print("=== stg32 predicated codegen ===")
    print(src)
    # Verify codegen - should have store_global_32 with predicate
    assert "store_global_32" in src, "Expected store_global_32 call in generated CUDA source"


@tilelang.testing.requires_cuda
def test_stg64_predicated_codegen():
    """Test that stg64 with predicate generates tl::store_global_64_conditional(ptr, val, pred) in CUDA source."""

    @tilelang.jit
    def stg64_pred_kernel(X, Y):
        N = T.const("N")
        X: T.Tensor[[N], T.float32]
        Y: T.Tensor[[N], T.float32]

        with T.Kernel(N // 2, threads=32) as pid:
            val = T.ldg64(X[pid * 2 : pid * 2 + 2])
            # Only store for the first half of elements
            T.stg64(Y[pid * 2 : pid * 2 + 2], val, pred=pid < N // 4)

    X = torch.randn(128, dtype=torch.float32, device="cuda")
    Y = torch.zeros(128, dtype=torch.float32, device="cuda")

    stg64_pred_kernel(X, Y)

    # Verify codegen
    src = stg64_pred_kernel.get_kernel_source(N=128)
    print("=== stg64 predicated codegen ===")
    print(src)
    assert "store_global_64" in src, "Expected store_global_64 call in generated CUDA source"


@tilelang.testing.requires_cuda
def test_stg128_predicated_codegen():
    """Test that stg128 with predicate generates tl::store_global_128_conditional(ptr, val, pred) in CUDA source."""

    @tilelang.jit
    def stg128_pred_kernel(X, Y):
        N = T.const("N")
        X: T.Tensor[[N], T.float32]
        Y: T.Tensor[[N], T.float32]

        with T.Kernel(N // 4, threads=32) as pid:
            val = T.ldg128(X[pid * 4 : pid * 4 + 4])
            # Only store for the first half of elements
            T.stg128(Y[pid * 4 : pid * 4 + 4], val, pred=pid < N // 8)

    X = torch.randn(128, dtype=torch.float32, device="cuda")
    Y = torch.zeros(128, dtype=torch.float32, device="cuda")

    stg128_pred_kernel(X, Y)

    # Verify codegen
    src = stg128_pred_kernel.get_kernel_source(N=128)
    print("=== stg128 predicated codegen ===")
    print(src)
    assert "store_global_128" in src, "Expected store_global_128 call in generated CUDA source"


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
def test_stg256_predicated_codegen():
    """Test that stg256 with predicate generates tl::store_global_256_conditional(ptr, val, pred) in CUDA source."""

    @tilelang.jit
    def stg256_pred_kernel(X, Y):
        N = T.const("N")
        X: T.Tensor[[N], T.float32]
        Y: T.Tensor[[N], T.float32]

        with T.Kernel(N // 8, threads=32) as pid:
            val = T.ldg256(X[pid * 8 : pid * 8 + 8])
            # Only store for the first half of elements
            T.stg256(Y[pid * 8 : pid * 8 + 8], val, pred=pid < N // 16)

    X = torch.randn(256, dtype=torch.float32, device="cuda")
    Y = torch.zeros(256, dtype=torch.float32, device="cuda")

    stg256_pred_kernel(X, Y)

    # Verify codegen
    src = stg256_pred_kernel.get_kernel_source(N=256)
    print("=== stg256 predicated codegen ===")
    print(src)
    assert "store_global_256" in src, "Expected store_global_256 call in generated CUDA source"


if __name__ == "__main__":
    tilelang.testing.main()
