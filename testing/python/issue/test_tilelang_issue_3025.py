import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.utils.language import retrieve_stride


M = 4
N = 3
PITCH = 8


def _make_store_kernel():
    @T.prim_func
    def main(dst: T.StridedTensor((M, N), (PITCH, 1), "int32")):
        with T.Kernel(1, threads=1):
            for row in T.serial(M):
                T.stg32(dst[row, 0], T.Cast("uint32", row + 1))

    return main


def _make_load_kernel():
    @T.prim_func
    def main(
        src: T.StridedTensor((M, N), (PITCH, 1), "int32"),
        out: T.Tensor((M,), "int32"),
    ):
        with T.Kernel(1, threads=1):
            for row in T.serial(M):
                out[row] = T.reinterpret(T.ldg32(src[row, 0]), "int32")

    return main


def test_retrieve_stride_preserves_scalar_rank():
    scalar = T.Tensor((), "float32")
    assert retrieve_stride(scalar) == []


@tilelang.testing.requires_cuda
def test_strided_stg32_uses_declared_stride():
    kernel = tilelang.compile(_make_store_kernel(), target="cuda")
    physical = torch.zeros(M * PITCH, dtype=torch.int32, device="cuda")
    view = physical.as_strided((M, N), (PITCH, 1))

    kernel(view)

    expected = torch.arange(1, M + 1, dtype=torch.int32, device="cuda")
    torch.testing.assert_close(view[:, 0], expected, rtol=0, atol=0)
    source = kernel.get_kernel_source()
    assert "dst[(row * 8)]" in source
    assert "dst[(row * 3)]" not in source


@tilelang.testing.requires_cuda
def test_strided_ldg32_uses_declared_stride():
    physical = torch.arange(M * PITCH, dtype=torch.int32, device="cuda")
    view = physical.as_strided((M, N), (PITCH, 1))
    out = torch.empty(M, dtype=torch.int32, device="cuda")
    kernel = tilelang.compile(_make_load_kernel(), target="cuda")

    kernel(view, out)

    torch.testing.assert_close(out, view[:, 0], rtol=0, atol=0)
    source = kernel.get_kernel_source()
    assert "src[(row * 8)]" in source
    assert "src[(row * 3)]" not in source


if __name__ == "__main__":
    tilelang.testing.main()
