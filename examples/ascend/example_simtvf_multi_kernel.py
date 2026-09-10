"""Example: two distinct SimtVF prim funcs in one file on Ascend NPU.

This exercises the multi-prim-func case: several VF-bearing kernel factories
living in a single source file. Each factory compiles independently, so the
generated VF helpers are namespaced by the kernel's global_symbol
(``<global_symbol>_simt_vf_<idx>``) — two kernels never collide on a bare
``simt_vf_0`` when they share one generated .asc.
"""

import tilelang
import tilelang.ascend.language as T


def vector_add(N):
    """C = A + B using SimtVF."""

    @T.prim_func
    def main(
        A: T.Buffer((N,), "float32"),
        B: T.Buffer((N,), "float32"),
        C: T.Buffer((N,), "float32"),
    ):
        with T.Kernel(1) as _, T.SimtVF(threads=128):
            for i in T.Parallel(N):
                C[i] = A[i] + B[i]

    return main


def vector_scale(N, scale):
    """C = A * scale using SimtVF — a second, distinct kernel factory."""

    @T.prim_func
    def main(
        A: T.Buffer((N,), "float32"),
        C: T.Buffer((N,), "float32"),
    ):
        with T.Kernel(1) as _, T.SimtVF(threads=64):
            for i in T.Parallel(N):
                C[i] = A[i] * T.float32(scale)

    return main


def ref_add(a, b):
    return a + b


def ref_scale(a, scale):
    return a * scale


if __name__ == "__main__":
    import torch

    N = 1024
    SCALE = 3.0
    device = torch.device("npu")

    print(f"Compiling vector_add (N={N}) ...")
    add_kernel = tilelang.compile(vector_add(N), target="ascend", out_idx=-1)
    print(f"Compiling vector_scale (N={N}, scale={SCALE}) ...")
    scale_kernel = tilelang.compile(vector_scale(N, SCALE), target="ascend", out_idx=-1)
    print("Compilation succeeded!")

    a = torch.randn(N, dtype=torch.float32, device=device)
    b = torch.randn(N, dtype=torch.float32, device=device)

    c_add = add_kernel(a, b)
    c_scale = scale_kernel(a)
    torch.npu.synchronize()

    if not torch.equal(c_add, ref_add(a, b)):
        raise AssertionError("vector_add results mismatch!")
    if not torch.equal(c_scale, ref_scale(a, SCALE)):
        raise AssertionError("vector_scale results mismatch!")
    print("Verification passed!")
