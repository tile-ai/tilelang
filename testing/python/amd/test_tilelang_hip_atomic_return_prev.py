import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


def _atomic_add_vector_return_prev_program(lanes):
    atomic_add = T.atomic_addx2 if lanes == 2 else T.atomic_addx4

    @T.prim_func
    def main(
        Dst: T.Tensor((lanes,), T.float32),
        Val: T.Tensor((lanes,), T.float32),
        Prev: T.Tensor((lanes,), T.float32),
    ):
        with T.Kernel(1, threads=1):
            Prev[0:lanes] = atomic_add(Dst[0], Val[0], return_prev=True)

    return main


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("lanes", [2, 4])
def test_atomic_add_vector_return_prev(lanes):
    kernel = tilelang.compile(_atomic_add_vector_return_prev_program(lanes), target="hip")
    assert f"AtomicAddx{lanes}Ret" in kernel.get_kernel_source()

    dst = torch.arange(1, lanes + 1, dtype=torch.float32, device="cuda")
    val = torch.arange(1, lanes + 1, dtype=torch.float32, device="cuda") * 10
    prev = torch.zeros(lanes, dtype=torch.float32, device="cuda")

    kernel(dst, val, prev)

    torch.testing.assert_close(prev, torch.arange(1, lanes + 1, dtype=torch.float32, device="cuda"))
    torch.testing.assert_close(
        dst,
        torch.arange(1, lanes + 1, dtype=torch.float32, device="cuda") * 11,
    )
