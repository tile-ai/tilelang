"""Stores to a T.alloc_var whose value hoists statements during codegen
(if_then_else -> `T condval; if (...) {...}`, floordiv by a dynamic divisor
-> `int rmod = ...`): the hoisted statements must be printed before the
assignment, not in the middle of it."""

import torch

import tilelang
import tilelang.language as T
import tilelang.testing


def var_assign_kernel(N, threads=128):
    @T.prim_func
    def kernel(A: T.Tensor((N,), T.int32), B: T.Tensor((N,), T.int32), P: T.Tensor((2,), T.int32)):
        with T.Kernel(T.ceildiv(N, threads), threads=threads) as bx:
            cap = P[0]
            div = P[1]
            for i in T.Parallel(threads):
                idx = bx * threads + i
                if idx < N:
                    v = T.alloc_var(T.int32)
                    w = T.alloc_var(T.int32)
                    # if_then_else is a Call that CodeGenC prints as statements
                    v = T.if_then_else(A[idx] >= cap, A[idx] - cap, A[idx])
                    # floor division by a dynamic divisor goes through a Let
                    w = (v + div - 1) // div
                    B[idx] = v + w * 1000

    return kernel


@tilelang.testing.requires_metal
def test_var_assign_hoisted_statements():
    N = 1024
    kernel = tilelang.compile(var_assign_kernel(N), target="metal")
    src = kernel.get_kernel_source()
    # every hoisted temporary is declared on its own line, never after `v = `
    for line in src.splitlines():
        stripped = line.strip()
        assert not (stripped.startswith("v = ") and ("int " in stripped or "{" in stripped)), line
        assert not (stripped.startswith("w = ") and ("int " in stripped or "{" in stripped)), line
    a = torch.randint(0, 100, (N,), dtype=torch.int32, device="mps")
    b = torch.empty(N, dtype=torch.int32, device="mps")
    cap, div = 60, 7
    kernel(a, b, torch.tensor([cap, div], dtype=torch.int32, device="mps"))
    torch.mps.synchronize()
    v = torch.where(a >= cap, a - cap, a)
    w = (v + div - 1) // div
    torch.testing.assert_close(b, v + w * 1000)


if __name__ == "__main__":
    tilelang.testing.main()
