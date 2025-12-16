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


if __name__ == "__main__":
    tilelang.testing.main()
