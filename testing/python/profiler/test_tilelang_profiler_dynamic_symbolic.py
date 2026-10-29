"""Test for profiler with dynamic symbolic constraints."""

import tilelang
import tilelang.testing
import tilelang.language as T
import torch
import pytest
from tilelang.engine.param import KernelParam
from tilelang.profiler import Profiler

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


@pytest.mark.parametrize("dtype", ["int32", "int64"])
@pytest.mark.parametrize("value", [7, 16])
def test_profiler_substitutes_symbolic_shape_expressions(dtype, value):
    m = tilelang.tvm.tirx.Var("m", dtype)
    n = tilelang.tvm.tirx.Var("n", dtype)
    param = KernelParam("float32", [m, m + 1, 2 * n, (m + 7) // 8, m % 4, 3])
    profiler = Profiler([param], [], tilelang.TensorSupplyType.Zero)

    concrete = profiler._substitute_dynamic_symbols(param, {"m": value, "n": 5})

    assert concrete.shape == [value, value + 1, 10, (value + 7) // 8, value % 4, 3]
    assert concrete.dtype == param.dtype
    assert param.shape[0].same_as(m)
    assert isinstance(param.shape[1], tilelang.tvm.tirx.Add)


def test_profiler_substitution_reports_missing_variable_in_expression():
    m = T.dynamic("m")
    param = KernelParam("float32", [m + 1])
    profiler = Profiler([param], [], tilelang.TensorSupplyType.Zero)

    with pytest.raises(ValueError, match="Dynamic symbolic variable 'm' not found in constraints"):
        profiler._substitute_dynamic_symbols(param, {"n": 8})


def test_profiler_substitution_preserves_large_int64_dimensions():
    m = tilelang.tvm.tirx.Var("m", "int64")
    param = KernelParam("float32", [m + 1])
    profiler = Profiler([param], [], tilelang.TensorSupplyType.Zero)

    concrete = profiler._substitute_dynamic_symbols(param, {"m": 2**32})

    assert concrete.shape == [2**32 + 1]


def test_profiler_substitution_keeps_static_dimensions():
    param = KernelParam("float32", [2, 3])
    profiler = Profiler([param], [], tilelang.TensorSupplyType.Zero)

    concrete = profiler._substitute_dynamic_symbols(param, {"unused": 4})

    assert concrete.shape == [2, 3]


def test_profiler_benchmarks_symbolic_shape_expressions_on_cpu(monkeypatch):
    monkeypatch.setattr(tilelang.utils.device, "get_current_device", lambda: torch.device("cpu"))
    m = T.dynamic("m")
    param = KernelParam("float32", [m, m + 1])
    profiler = Profiler([param], [], tilelang.TensorSupplyType.Zero)
    observed_shapes = []

    def operation(value):
        observed_shapes.append(tuple(value.shape))
        return value + 1

    latency = profiler.do_bench(
        func=operation,
        backend="wall",
        device="cpu",
        n_warmup=1,
        n_repeat=2,
        dynamic_symbolic_constraints={"m": 3},
    )

    assert latency >= 0
    assert observed_shapes
    assert all(shape == (3, 4) for shape in observed_shapes)


if __name__ == "__main__":
    tilelang.testing.main()
