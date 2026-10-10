"""On-device T.fill inside SimdVF, including multi-buffered pipeline slots."""

import pytest
import tilelang
import tilelang.ascend.language as T
import tilelang.testing
import torch


def test_simdvf_fill_full_unversioned_ub_npu():
    @T.prim_func
    def kernel(O: T.Buffer((64,), "float32")):
        with T.Kernel(1):
            temp = T.alloc_shared((64,), "float32")
            with T.SimdVF():
                T.fill(temp, 1.0)
            T.copy(temp, O)

    compiled = tilelang.compile(kernel, target="ascend", out_idx=-1)
    out = compiled()
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), torch.full((64,), 1.0, dtype=torch.float32), rtol=0, atol=0)


def test_simdvf_fill_2d_ub_npu():
    @T.prim_func
    def kernel(O: T.Buffer((32, 64), "float32")):
        with T.Kernel(1):
            temp = T.alloc_shared((32, 64), "float32")
            with T.SimdVF():
                T.fill(temp, 0.0)
            T.copy(temp, O)

    compiled = tilelang.compile(kernel, target="ascend", out_idx=-1)
    out = compiled()
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), torch.zeros((32, 64), dtype=torch.float32), rtol=0, atol=0)


def test_simdvf_fill_multibuffered_ub_inside_pipeline_npu():
    tile = 64

    @T.prim_func
    def kernel(A: T.Buffer((8 * tile,), "float32"), C: T.Buffer((8 * tile,), "float32")):
        with T.Kernel(1):
            acc = T.alloc_shared((tile,), "float32")
            tmp = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({acc: 2, tmp: 2})
            for w in T.Pipelined(8, num_stages=2):
                T.copy(A[w * tile : (w + 1) * tile], tmp)
                with T.SimdVF():
                    T.fill(acc, 1.0)
                    for i in T.Parallel(tile):
                        tmp[i] = tmp[i] + acc[i]
                T.copy(tmp, C[w * tile : (w + 1) * tile])

    compiled = tilelang.compile(kernel, target="ascend", out_idx=-1)
    device = torch.device("npu")
    torch.manual_seed(0)
    a = torch.randn(8 * tile, dtype=torch.float32, device=device)
    c = compiled(a)
    torch.npu.synchronize()
    torch.testing.assert_close(c, a + 1.0, rtol=0, atol=1e-5)


def test_simdvf_fill_aligned_subregion_ub_npu():
    @T.prim_func
    def kernel(O: T.Buffer((64,), "float32")):
        with T.Kernel(1):
            temp = T.alloc_shared((128,), "float32")
            with T.SimdVF():
                T.fill(temp[8:72], 1.0)
            T.copy(temp[8:72], O)

    compiled = tilelang.compile(kernel, target="ascend", out_idx=-1)
    out = compiled()
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), torch.full((64,), 1.0, dtype=torch.float32), rtol=0, atol=0)


@pytest.mark.parametrize("start", [1, 4])
def test_simdvf_fill_misaligned_subregion_rejected(start):
    @T.prim_func
    def kernel(O: T.Buffer((64,), "float32")):
        with T.Kernel(1):
            temp = T.alloc_shared((128,), "float32")
            with T.SimdVF():
                T.fill(temp[start : start + 64], 1.0)
            T.copy(temp[start : start + 64], O)

    with pytest.raises(ValueError, match="32-byte-aligned"):
        tilelang.compile(kernel, target="ascend", out_idx=-1)


if __name__ == "__main__":
    tilelang.testing.main()
