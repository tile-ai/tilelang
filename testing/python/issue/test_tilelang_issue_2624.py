import tilelang
import tilelang.language as T
import tilelang.testing
import torch
import pytest

N = 4


def make_kernel(op, dt):
    @T.prim_func
    def main(A: T.Tensor((N,), dt), B: T.Tensor((1,), dt)):
        with T.Kernel(1, threads=N):
            As = T.alloc_fragment((N,), dt)
            Bs = T.alloc_fragment((1,), dt)
            T.copy(A, As)
            (T.reduce_absmax if op == "absmax" else T.reduce_max)(As, Bs, dim=0)
            T.copy(Bs, B)

    return main


@pytest.mark.parametrize(
    "op,dtype",
    [
        ("absmax", "uint32"),
        ("absmax", "uint64"),
        ("abssum", "uint32"),
        ("abssum", "uint64"),
    ],
)
def test_reduce_absmax_abssum_with_positive_input(op: str, dtype: str):
    A = torch.tensor([100, 10, 20, 30], dtype=getattr(torch, dtype), device="cuda")
    raw_op = tilelang.compile(make_kernel(op.replace("abs", ""), dtype), out_idx=[1])(A)
    abs_op = tilelang.compile(make_kernel(op, dtype), out_idx=[1])(A)
    torch.testing.assert_close(raw_op, abs_op, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
