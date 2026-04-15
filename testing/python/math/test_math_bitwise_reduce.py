import pytest
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
    a,
    clear=True,
):
    M, N = 32, 32
    block_M, block_N = 32, 32
    kernel = bitwise_reduce(M, N, block_M, block_N, name, func, clear)

    if name == "reduce_bitand":
        expected = torch.full((M,), -1, device="cuda", dtype=torch.int32)
    elif name == "reduce_bitor" or name == "reduce_bitxor":
        expected = torch.full((M,), 0, device="cuda", dtype=torch.int32)
    else:
        raise ValueError("Invalid name: {}".format(name))

    output = kernel(a, expected)

    expected = reference_bitwise_reduce(name, a)
    assert torch.all(output == expected)
    print("✓ {} with clear={} test passed".format(name, clear))


def reference_bitwise_reduce(name, a):
    if name == "reduce_bitand":
        op = torch.bitwise_and
        identity = -1
    elif name == "reduce_bitor":
        op = torch.bitwise_or
        identity = 0
    elif name == "reduce_bitxor":
        op = torch.bitwise_xor
        identity = 0
    else:
        raise ValueError("Invalid name: {}".format(name))

    reduced = a
    while reduced.shape[1] > 1:
        if reduced.shape[1] % 2:
            padding = torch.full(
                (reduced.shape[0], 1),
                identity,
                device=reduced.device,
                dtype=reduced.dtype,
            )
            reduced = torch.cat([reduced, padding], dim=1)
        reduced = op(reduced[:, 0::2], reduced[:, 1::2])
    return reduced[:, 0]


@pytest.fixture(scope="module")
def bitwise_reduce_input():
    M, N = 32, 32
    rows = torch.arange(M, dtype=torch.int32)[:, None]
    cols = torch.arange(N, dtype=torch.int32)[None, :]

    row_pattern = (rows & 0xF) << (rows % 4)
    col_pattern = torch.bitwise_left_shift(torch.ones_like(cols), cols % 31)
    position_factor = (rows * N + cols) % 256

    a = row_pattern ^ col_pattern ^ position_factor

    clear_rows = (rows % 4) == 0
    clear_bits = torch.bitwise_left_shift(torch.ones_like(rows), rows // 4)
    a = torch.where(clear_rows, a & torch.bitwise_not(clear_bits), a)

    set_rows = ((rows % 4) != 0) & ((rows % 2) == 0)
    set_bits = torch.bitwise_left_shift(torch.ones_like(rows), rows // 2)
    a = torch.where(set_rows, a | set_bits, a)

    return a.to(device="cuda")


BITWISE_REDUCE_OPS = [
    ("reduce_bitand", T.reduce_bitand),
    ("reduce_bitor", T.reduce_bitor),
    ("reduce_bitxor", T.reduce_bitxor),
]


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(("name", "func"), BITWISE_REDUCE_OPS, ids=[name for name, _ in BITWISE_REDUCE_OPS])
@pytest.mark.parametrize("clear", [True, False], ids=["clear", "no-clear"])
def test_bitwise_reduce_ops(bitwise_reduce_input, name, func, clear):
    run_single_bitwise_reduce(name, func, bitwise_reduce_input, clear=clear)


if __name__ == "__main__":
    tilelang.testing.main()
