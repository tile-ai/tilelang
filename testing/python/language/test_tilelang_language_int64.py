import tilelang
import tilelang.language as T


@tilelang.jit
def fill_symbolic(value: float, dtype=T.bfloat16):
    n = T.symbolic("n", "int64")
    block_n = 512

    @T.prim_func
    def main(x: T.Tensor[n, dtype]):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(n, block_n), threads=128) as bx:
            # Doesn't yet work with int64-shaped global tensor
            # T.fill(x[bx * block_n : (bx + 1) * block_n], value)
            for i in T.Parallel(block_n):
                x[bx * block_n + i] = value

    return main


def run_fill_symbolic(n: int):
    import torch

    x = torch.zeros(n, dtype=torch.bfloat16, device="cuda")
    fill_symbolic(1.0)(x)
    assert x.min() == 1.0 and x.max() == 1.0


def test_fill_symbolic():
    # Requires 8GB VRAM
    run_fill_symbolic(2**32)


@tilelang.jit
def fill_static(n: int, value: float, dtype=T.bfloat16):
    block_n = 512

    @T.prim_func
    def main(x: T.Tensor[n, dtype]):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(n, block_n), threads=128) as bx:
            # Doesn't yet work with int64-shaped global tensor
            # T.fill(x[bx * block_n : (bx + 1) * block_n], value)
            for i in T.Parallel(block_n):
                x[bx * block_n + i] = value

    return main


def run_fill_static(n: int):
    import torch

    x = torch.zeros(n, dtype=torch.bfloat16, device="cuda")
    fill_static(n, 1.0)(x)
    assert x.min() == 1.0 and x.max() == 1.0


def test_fill_static():
    # Requires 8GB VRAM
    run_fill_static(2**32)


if __name__ == "__main__":
    test_fill_symbolic()
    test_fill_static()
