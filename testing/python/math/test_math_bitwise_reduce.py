import tilelang
import tilelang.language as T
import torch
import tilelang.testing


@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
    },
)
def bitwise_reduce(
    M,
    N,
    block_M,
    block_N,
    name,
    func,
    clear=True,
):
    @T.prim_func
    def reduce_func(
        A: T.Tensor((M, N), T.int32),
        B: T.Tensor((M), T.int32),
        Output: T.Tensor((M), T.int32),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), T.int32)
            A_fragment = T.alloc_fragment((block_M, block_N), T.int32)
            B_shared = T.alloc_shared((block_M,), T.int32)
            B_fragment = T.alloc_fragment((block_M), T.int32)
            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(A_shared, A_fragment)
            T.copy(B[by * block_M], B_shared)
            T.copy(B_shared, B_fragment)
            func(A_fragment, B_fragment, clear=clear)
            T.copy(B_fragment, Output[by * block_M])

    return reduce_func


def run_single_bitwise_reduce(
    name,
    func,
    clear=True,
):
    M, N = 32, 32
    block_M, block_N = 32, 32
    kernel = bitwise_reduce(M, N, block_M, block_N, name, func, clear)

    # Generate test data that exercises all bit patterns for robust bitwise reduce testing
    a = torch.zeros((M, N), device="cuda", dtype=torch.int32)

    # Fill with patterns that will produce meaningful results for bitwise operations:
    # - Different bit patterns across rows/columns
    # - Mix of 0s and 1s in various positions
    # - Some all-1s and all-0s patterns for edge cases
    for i in range(M):
        for j in range(N):
            # Create varied bit patterns:
            # Row-based pattern: alternating bits based on row index
            row_pattern = (i & 0xF) << (i % 4)  # 4-bit patterns shifted by row

            # Column-based pattern: different bit positions set based on column
            col_pattern = 1 << (j % 31)  # Single bit set at different positions

            # Combine patterns with XOR to create diverse bit distributions
            # Add some deterministic "noise" based on position
            position_factor = (i * N + j) % 256

            # Final value combines all patterns
            a[i, j] = (row_pattern ^ col_pattern ^ position_factor) & 0xFFFFFFFF

            if i % 4 == 0:
                a[i, j] &= ~(0x1 << (i // 4))
            elif i % 2 == 0:
                a[i, j] |= 0x1 << (i // 2)

    if name == "reduce_bitand":
        expected = torch.full((M,), -1, device="cuda", dtype=torch.int32)
    elif name == "reduce_bitor" or name == "reduce_bitxor":
        expected = torch.full((M,), 0, device="cuda", dtype=torch.int32)
    else:
        raise ValueError("Invalid name: {}".format(name))

    output = kernel(a, expected)

    for i in range(M):
        for j in range(N):
            if name == "reduce_bitand":
                expected[i] = expected[i] & a[i, j]
            elif name == "reduce_bitor":
                expected[i] = expected[i] | a[i, j]
            elif name == "reduce_bitxor":
                expected[i] = expected[i] ^ a[i, j]
            else:
                raise ValueError("Invalid name: {}".format(name))
    assert torch.all(output == expected)
    print("✓ {} with clear={} test passed".format(name, clear))


@tilelang.testing.requires_cuda
def test_bitwise_reduce_ops():
    run_single_bitwise_reduce("reduce_bitand", T.reduce_bitand, clear=True)
    run_single_bitwise_reduce("reduce_bitor", T.reduce_bitor, clear=True)
    run_single_bitwise_reduce("reduce_bitxor", T.reduce_bitxor, clear=True)
    run_single_bitwise_reduce("reduce_bitand", T.reduce_bitand, clear=False)
    run_single_bitwise_reduce("reduce_bitor", T.reduce_bitor, clear=False)
    run_single_bitwise_reduce("reduce_bitxor", T.reduce_bitxor, clear=False)


if __name__ == "__main__":
    tilelang.testing.main()
