import warnings

import pytest

import tilelang
import tilelang.language as T


# 2^32 bf16 elements = 8 GiB. This is the upper-bound stress target; the
# helper below scales down only when the GPU cannot fit it.
_DEFAULT_TARGET_N = 2**32
# 2^28 bf16 elements = 512 MiB. Below this the test is no longer a useful
# fill-stress — skip instead of running a token-sized fill.
_FLOOR_N = 2**28


def _select_int64_stress_n(
    bytes_per_elem: int = 2,
    target_n: int = _DEFAULT_TARGET_N,
    floor_n: int = _FLOOR_N,
) -> int:
    """Pick the largest power-of-two element count that fits in free VRAM.

    Returns ``target_n`` (8 GiB at bf16) when the GPU has enough free
    memory; otherwise halves until the allocation fits with a 512 MiB / 10%
    headroom margin. Emits a UserWarning whenever the size is scaled below
    ``target_n`` so reduced coverage is visible in the test log. Skips the
    test if even ``floor_n`` (~512 MiB) cannot be allocated.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("int64 stress test requires a CUDA device")

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    # Reserve headroom so the kernel launch / Torch caching / cuBLAS
    # workspaces don't push us into OOM at the boundary. On an 8 GiB part
    # only ~7 GiB is typically free even at idle.
    headroom = max(512 * 1024 * 1024, free_bytes // 10)
    usable = max(0, free_bytes - headroom)

    n = target_n
    while n > floor_n and n * bytes_per_elem > usable:
        n //= 2

    if n * bytes_per_elem > usable:
        pytest.skip(
            f"int64 stress test needs at least "
            f"{(floor_n * bytes_per_elem) / (1024**3):.2f} GiB of free GPU "
            f"memory (got {free_bytes / (1024**3):.2f} GiB free, "
            f"{total_bytes / (1024**3):.2f} GiB total)"
        )

    if n < target_n:
        warnings.warn(
            f"int64 stress test scaled down: requested n={target_n} "
            f"({target_n * bytes_per_elem / (1024**3):.2f} GiB) but only "
            f"{free_bytes / (1024**3):.2f} GiB free / "
            f"{total_bytes / (1024**3):.2f} GiB total VRAM available; "
            f"using n={n} ({n * bytes_per_elem / (1024**3):.2f} GiB) instead.",
            stacklevel=2,
        )

    return n


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
    # Targets 8 GiB; falls back in powers of two on smaller GPUs (warns).
    run_fill_symbolic(_select_int64_stress_n())


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
    # Targets 8 GiB; falls back in powers of two on smaller GPUs (warns).
    run_fill_static(_select_int64_stress_n())


if __name__ == "__main__":
    test_fill_symbolic()
    test_fill_static()
