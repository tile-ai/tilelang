import pytest

import tilelang.testing
from tilelang import language as T


def clamp_within_bounds(
    N,
    block_N,
    dtype,
    min_val=None,
    max_val=None,
):
    import tilelang.language as T

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=block_N) as bx:
            A_shared = T.alloc_shared([block_N], dtype)
            T.copy(A[bx * block_N], A_shared)
            for i in T.Parallel(block_N):
                A_shared[i] = T.clamp(A_shared[i], min_val=min_val, max_val=max_val)
            T.copy(A_shared, B[bx * block_N])

    return main


def run_clamp(
    N,
    block_N,
    dtype,
    min=None,
    max=None,
):
    program = clamp_within_bounds(N, block_N, dtype, min, max)

    kernel = tilelang.compile(program, out_idx=[1])
    profiler = kernel.get_profiler()

    def ref_program(A):
        import torch

        output = torch.clamp(A, min, max)
        return output

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def clamp_value_range(
    N,
    block_N,
    dtype,
):
    import tilelang.language as T

    @T.prim_func
    def main(
        A: T.Tensor((1, N), dtype),
        B: T.Tensor((1, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=block_N) as bx:
            # A_shared = T.alloc_shared([1, block_N], dtype=dtype)
            A_frag = T.alloc_fragment([1, block_N], dtype=dtype)
            min_frag = T.alloc_fragment([1], dtype=dtype)
            max_frag = T.alloc_fragment([1], dtype=dtype)
            T.copy(A[0, bx * block_N], A_frag)
            T.reduce_min(A_frag, min_frag, dim=1)
            T.reduce_max(A_frag, max_frag, dim=1)
            for i in T.Parallel(block_N):
                # A_frag[0, i] = T.max(A_frag[0, i], min_frag[0] * 0.5)
                # A_frag[0, i] = T.min(A_frag[0, i], max_frag[0] * 0.5)
                A_frag[0, i] = T.clamp(A_frag[0, i], min_frag[0] * 0.5, max_frag[0] * 0.5)
            T.copy(A_frag, B[0, bx * block_N])

    return main


def run_clamp_value_range(
    N,
    block_N,
    dtype,
):
    program = clamp_value_range(
        N,
        block_N,
        dtype,
    )
    kernel = tilelang.compile(program, out_idx=[1])

    import torch

    # Convert string dtype to torch.dtype
    torch_dtype = dtype.as_torch()

    def ref_program(A):
        min_val = torch.min(A) * 0.5
        max_val = torch.max(A) * 0.5
        output = torch.clamp(A, min_val, max_val)
        return output

    A = torch.randint(-5, 5, (1, N)).cuda().to(dtype=torch_dtype)
    B = kernel(A)
    ref_b = ref_program(A)
    torch.testing.assert_close(B, ref_b)


def test_clamp():
    # clamp tests for float16 and float32
    run_clamp(1024, 128, T.float16, -0.05, 0.05)
    run_clamp(1024, 128, T.float32, -0.06, 0.05)
    run_clamp_value_range(1024, 128, T.float16)
    run_clamp_value_range(1024, 128, T.float32)


CLAMP_NAN_DTYPES = [
    T.float16,
    T.bfloat16,
    T.float32,
    T.float64,
    T.float8_e4m3,
    T.float8_e5m2,
]


def clamp_nan_kernel(N, block_N, dtype):
    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=block_N) as bx:
            for i in T.Parallel(block_N):
                C[bx * block_N + i] = T.clamp(A[bx * block_N + i], T.cast(-1.0, dtype), T.cast(1.0, dtype))

    return main


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", CLAMP_NAN_DTYPES)
def test_clamp_propagates_nan(dtype):
    """Clamp must propagate NaN, unlike a plain CUDA fmin/fmax composition."""
    import torch

    N = 32
    kernel = tilelang.compile(clamp_nan_kernel(N, N, dtype))
    torch_dtype = dtype.as_torch()

    a = torch.linspace(-2.0, 2.0, N, device="cuda", dtype=torch.float32)
    a[7] = float("nan")
    a = a.to(torch_dtype)

    # Caller-allocated output: rebuilding a torch tensor from an fp8 DLPack
    # tensor is not supported on every torch version.
    C = torch.empty(N, device="cuda", dtype=torch_dtype)
    kernel(a, C)
    got = C.float()

    assert torch.isnan(got[7]).item(), f"T.clamp dropped NaN for {dtype}: got {got[7].item()} instead of NaN"
    ref = torch.clamp(a.float(), -1.0, 1.0)
    finite = ~torch.isnan(ref)
    torch.testing.assert_close(got[finite], ref[finite], atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [T.float16, T.bfloat16, T.float32, T.float64, T.float8_e4m3])
def test_clamp_scalar_and_vector_type_promotion(dtype):
    from tvm import tirx

    for lanes in [1, 4]:
        x = T.Var("x", dtype.with_lanes(lanes))
        result = T.clamp(x, -1.0, T.float64(1.0))
        assert result.dtype == T.float64.with_lanes(lanes)
        assert isinstance(result, tirx.Call)
        assert result.op.same_as(tirx.op.Op.get("tl.clamp"))
        assert all(arg.dtype == result.dtype for arg in result.args)


@tilelang.testing.requires_cuda
def test_clamp_python_literals():
    import torch

    @T.prim_func
    def main(C: T.Tensor((4,), "float32")):
        with T.Kernel(1, threads=1):
            C[0] = T.clamp(0.5, -1.0, 1.0)
            C[1] = T.clamp(2, -1, 1)
            C[2] = T.clamp(-5, -1.0, 1.0)
            C[3] = T.clamp(float("nan"), -1.0, 1.0)

    c = torch.empty(4, device="cuda")
    tilelang.compile(main)(c)
    torch.testing.assert_close(c, torch.tensor([0.5, 1.0, -1.0, float("nan")], device="cuda"), equal_nan=True)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", CLAMP_NAN_DTYPES)
def test_clamp_vectorized_operands(dtype):
    import torch

    N = 128

    @T.prim_func
    def main(A: T.Tensor((N,), dtype), Lo: T.Tensor((N,), dtype), Hi: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(N):
                C[i] = T.clamp(A[i], Lo[i], Hi[i])

    kernel = tilelang.compile(main, target="cuda")
    source = kernel.get_kernel_source()
    if dtype in (T.float16, T.bfloat16):
        assert "tl::clamp2(" in source
    else:
        assert "tl::clamp(" in source
    if dtype == T.float32:
        assert "*(float4*)" in source

    a = torch.linspace(-2.0, 2.0, N, dtype=torch.float64, device="cuda")
    lo = torch.full_like(a, -1.0)
    hi = torch.full_like(a, 1.0)
    a[1], lo[2], hi[3] = float("nan"), float("nan"), float("nan")
    lo[4:8] = 5.0  # Reversed bounds must choose the upper bound.
    lo[8:12], hi[8:12] = 0.5, 0.5
    a[12], a[13] = float("inf"), -float("inf")
    a, lo, hi = (value.to(dtype.as_torch()) for value in (a, lo, hi))
    c = torch.empty_like(a)
    kernel(a, lo, hi, c)
    ref = torch.clamp(a.double(), lo.double(), hi.double())
    torch.testing.assert_close(c.double(), ref, equal_nan=True, atol=0, rtol=0)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", [T.float16, T.bfloat16, T.float32, T.float64, T.float8_e4m3])
def test_clamp_vector_input_scalar_bounds(dtype):
    import torch

    @T.prim_func
    def main(A: T.Tensor((128,), dtype), C: T.Tensor((128,), dtype)):
        with T.Kernel(1, threads=32):
            idx = T.Ramp(T.get_thread_binding() * 4, 1, 4)
            C[idx] = T.clamp(A[idx], T.cast(-1.0, dtype), T.cast(1.0, dtype))

    a = torch.tensor([float("nan"), -2.0, 0.5, 2.0], device="cuda").repeat(32).to(dtype.as_torch())
    c = torch.empty_like(a)
    tilelang.compile(main)(a, c)
    torch.testing.assert_close(c.float(), a.float().clamp(-1, 1), equal_nan=True, atol=0, rtol=0)


@tilelang.testing.requires_cuda
def test_clamp_evaluates_each_operand_once():
    import torch

    @T.prim_func
    def main(A: T.Tensor((3,), "float32"), C: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=1):
            C[0] = T.clamp(
                T.atomic_add(A[0], 0.25, return_prev=True),
                T.atomic_add(A[1], 0.25, return_prev=True),
                T.atomic_add(A[2], 0.25, return_prev=True),
            )

    a = torch.tensor([0.0, -1.0, 1.0], device="cuda")
    c = torch.empty(1, device="cuda")
    tilelang.compile(main)(a, c)
    torch.testing.assert_close(a, torch.tensor([0.25, -0.75, 1.25], device="cuda"))
    torch.testing.assert_close(c, torch.zeros_like(c))


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_clamp_identical_vector_atomics_evaluate_independently():
    import torch

    @T.prim_func
    def main(A: T.Tensor((4,), "float32"), V: T.Tensor((4,), "float32"), C: T.Tensor((4,), "float32")):
        with T.Kernel(1, threads=1):
            C[T.Ramp(0, 1, 4)] = T.clamp(
                T.atomic_addx4(A, V, return_prev=True),
                T.atomic_addx4(A, V, return_prev=True),
                T.atomic_addx4(A, V, return_prev=True),
            )

    a = torch.zeros(4, device="cuda")
    v = torch.full_like(a, 0.25)
    c = torch.empty_like(a)
    tilelang.compile(main)(a, v, c)
    torch.testing.assert_close(a, torch.full_like(a, 0.75))
    torch.testing.assert_close(c, torch.full_like(c, 0.25))


FP8_OPS = {
    "max": lambda a, b, dtype: T.max(a, b),
    "min": lambda a, b, dtype: T.min(a, b),
    "clamp": lambda a, b, dtype: T.clamp(a, T.cast(-1.0, dtype), T.cast(1.0, dtype)),
}


def fp8_operand_kernel(N, block_N, dtype, op):
    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=block_N) as bx:
            for i in T.Parallel(block_N):
                C[bx * block_N + i] = FP8_OPS[op](A[bx * block_N + i], B[bx * block_N + i], dtype)

    return main


def run_fp8_min_max_clamp(N, block_N, dtype, op):
    import torch

    kernel = tilelang.compile(fp8_operand_kernel(N, block_N, dtype, op))
    torch_dtype = dtype.as_torch()

    # Integers are exactly representable in both float8 formats, so the
    # reference comparison can be exact.
    a = torch.randint(-4, 5, (N,), device="cuda").to(torch.float32)
    b = torch.randint(-4, 5, (N,), device="cuda").to(torch.float32)
    if op == "max":
        ref = torch.maximum(a, b)
    elif op == "min":
        ref = torch.minimum(a, b)
    else:
        ref = a.clamp(-1.0, 1.0)

    # Allocate the output here instead of using out_idx: rebuilding a torch
    # tensor from an fp8 DLPack tensor is not supported on every torch version.
    C = torch.empty(N, device="cuda", dtype=torch_dtype)
    kernel(a.to(torch_dtype), b.to(torch_dtype), C)
    torch.testing.assert_close(C.float(), ref.to(torch_dtype).float(), atol=0, rtol=0)


def fp8_operand_vectorized_kernel(N, block_N, dtype, op, vec=4):
    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=block_N // vec) as bx:
            for i in T.Parallel(block_N // vec):
                for v in T.vectorized(vec):
                    idx = bx * block_N + i * vec + v
                    C[idx] = FP8_OPS[op](A[idx], B[idx], dtype)

    return main


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", [T.float8_e4m3, T.float8_e5m2])
@pytest.mark.parametrize("op", list(FP8_OPS))
def test_fp8_min_max_clamp(dtype, op):
    run_fp8_min_max_clamp(1024, 128, dtype, op)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("op", list(FP8_OPS))
def test_fp8_vectorized_min_max(op):
    # Vectorizing reaches the per-lane fallback, a separate overload-resolution
    # context from the scalar expression.
    tilelang.compile(fp8_operand_vectorized_kernel(1024, 128, T.float8_e4m3, op), out_idx=[2], target="cuda")


@tilelang.testing.requires_cuda
def test_fp8_e4m3fn_min_max():
    # float8_e4m3fn shares the emitted CUDA type with float8_e4m3; clamp covers
    # both min and max.
    tilelang.compile(fp8_operand_kernel(1024, 128, T.float8_e4m3fn, "clamp"), out_idx=[2], target="cuda")


if __name__ == "__main__":
    tilelang.testing.main()
