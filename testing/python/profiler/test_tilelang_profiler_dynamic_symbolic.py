"""Test for profiler with dynamic symbolic constraints."""

import tilelang
import tilelang.testing
import tilelang.language as T
import torch
import pytest

SINGLE_SYMBOLIC_M_VALUES = [128, 256, 512]
MULTIPLE_SYMBOLIC_CONSTRAINTS = [(128, 128), (256, 512), (512, 256)]


@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


@pytest.mark.parametrize("m_val", SINGLE_SYMBOLIC_M_VALUES, ids=[f"m={m}" for m in SINGLE_SYMBOLIC_M_VALUES])
def test_profiler_dynamic_symbolic_single(m_val):
    """Test profiler with a single dynamic symbolic variable."""
    M = T.dynamic("m")
    N = 256
    K = 256
    block_M = 64
    block_N = 64
    block_K = 32

    kernel = matmul(M, N, K, block_M, block_N, block_K)
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)

    latency = profiler.do_bench(dynamic_symbolic_constraints={"m": m_val})
    assert latency > 0, f"Expected positive latency for m={m_val}, got {latency}"
    print(f"Latency (m={m_val}): {latency:.3f} ms")


@pytest.mark.parametrize(
    ("m_val", "n_val"),
    MULTIPLE_SYMBOLIC_CONSTRAINTS,
    ids=[f"m={m}-n={n}" for m, n in MULTIPLE_SYMBOLIC_CONSTRAINTS],
)
def test_profiler_dynamic_symbolic_multiple(m_val, n_val):
    """Test profiler with multiple dynamic symbolic variables."""
    M = T.dynamic("m")
    N = T.dynamic("n")
    K = 256
    block_M = 64
    block_N = 64
    block_K = 32

    kernel = matmul(M, N, K, block_M, block_N, block_K)
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)

    latency = profiler.do_bench(dynamic_symbolic_constraints={"m": m_val, "n": n_val})
    assert latency > 0, f"Expected positive latency for m={m_val}, n={n_val}, got {latency}"
    print(f"Latency (m={m_val}, n={n_val}): {latency:.3f} ms")


@pytest.mark.parametrize("m_val", SINGLE_SYMBOLIC_M_VALUES, ids=[f"m={m}" for m in SINGLE_SYMBOLIC_M_VALUES])
def test_profiler_dynamic_symbolic_correctness(m_val):
    """Test that kernel with dynamic symbolic produces correct results."""
    M = T.dynamic("m")
    N = 256
    K = 256
    block_M = 64
    block_N = 64
    block_K = 32

    kernel = matmul(M, N, K, block_M, block_N, block_K)

    a = torch.randn(m_val, K, dtype=torch.float16, device="cuda")
    b = torch.randn(K, N, dtype=torch.float16, device="cuda")

    c = kernel(a, b)
    ref_c = a @ b

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print(f"Correctness test passed for m={m_val}")


def test_profiler_dynamic_symbolic_missing_constraint():
    """Test that missing constraint raises appropriate error."""
    M = T.dynamic("m")
    N = 256
    K = 256
    block_M = 64
    block_N = 64
    block_K = 32

    kernel = matmul(M, N, K, block_M, block_N, block_K)
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)

    # Test that missing constraint raises ValueError
    try:
        profiler.do_bench(dynamic_symbolic_constraints={"wrong_name": 256})
        raise ValueError("Expected ValueError for missing constraint")
    except ValueError as e:
        assert "m" in str(e), f"Error message should mention missing variable 'm', got: {e}"
        print(f"Correctly raised error for missing constraint: {e}")


def test_profiler_dynamic_symbolic_with_input_tensors():
    """Test that input_tensors takes precedence over dynamic_symbolic_constraints."""
    M = T.dynamic("m")
    N = 256
    K = 256
    block_M = 64
    block_N = 64
    block_K = 32

    kernel = matmul(M, N, K, block_M, block_N, block_K)
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)

    # Manually create input tensors
    # Note: out_idx=[-1] means C is output, so we only need A and B as inputs
    concrete_M = 512
    a = torch.randn(concrete_M, K, dtype=torch.float16, device="cuda")
    b = torch.randn(K, N, dtype=torch.float16, device="cuda")

    # input_tensors should take precedence
    latency = profiler.do_bench(input_tensors=[a, b])
    assert latency > 0, f"Expected positive latency, got {latency}"
    print(f"Latency with manual input_tensors (M={concrete_M}): {latency:.3f} ms")


if __name__ == "__main__":
    tilelang.testing.main()
