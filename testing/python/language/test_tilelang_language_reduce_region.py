import pytest
import torch

import tilelang as tl
import tilelang.language as T
import tilelang.testing


@pytest.mark.parametrize("dim", [0, -1])
def test_reduce_fragment_region_has_actionable_error(dim):
    with pytest.raises(ValueError, match=r"T\.reduce_\* does not support slicing a fragment buffer.*T\.alloc_reducer"):

        @T.prim_func
        def main(A: T.Tensor((128,), T.float32), B: T.Tensor((1,), T.float32)):
            with T.Kernel(1, threads=128):
                src = T.alloc_fragment((128,), T.float32)
                dst = T.alloc_fragment((1,), T.float32)
                T.copy(A, src)
                T.reduce_sum(src[0:64], dst, dim=dim)
                T.copy(dst, B)


def test_reduce_fragment_output_region_has_actionable_error():
    with pytest.raises(ValueError, match=r"slicing a fragment buffer.*sliced output.*T\.alloc_reducer"):

        @T.prim_func
        def main(A: T.Tensor((128,), T.float32), B: T.Tensor((1,), T.float32)):
            with T.Kernel(1, threads=128):
                src = T.alloc_fragment((128,), T.float32)
                dst = T.alloc_fragment((1,), T.float32)
                T.copy(A, src)
                T.reduce_sum(src, dst[0:1], dim=0)
                T.copy(dst, B)


def _make_shared_region_sum(start):
    @T.prim_func
    def main(A: T.Tensor((128,), T.float32), B: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=128):
            src = T.alloc_shared((128,), T.float32)
            dst = T.alloc_fragment((1,), T.float32)
            T.copy(A, src)
            T.reduce_sum(src[start : start + 64], dst, dim=-1)
            T.copy(dst, B)

    return main


def _make_shared_input_and_output_region_sum(start, clear):
    @T.prim_func
    def main(A: T.Tensor((128,), T.float32), B: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=128):
            src = T.alloc_shared((128,), T.float32)
            dst = T.alloc_shared((2,), T.float32)
            T.copy(A, src)
            T.fill(dst, 1.0)
            T.reduce_sum(src[start : start + 64], dst[1:2], dim=-1, clear=clear)
            T.copy(dst[1:2], B)

    return main


@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
@pytest.mark.parametrize("start", [0, 64])
def test_reduce_shared_region(start):
    kernel = tl.compile(_make_shared_region_sum(start), out_idx=-1)
    a = torch.arange(128, dtype=torch.float32, device="cuda")
    b = kernel(a)
    torch.testing.assert_close(b, a[start : start + 64].sum().reshape(1), atol=0, rtol=0)


@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
@pytest.mark.parametrize("start", [0, 64])
@pytest.mark.parametrize("clear", [True, False])
def test_reduce_shared_input_and_output_regions(start, clear):
    kernel = tl.compile(_make_shared_input_and_output_region_sum(start, clear), out_idx=-1)
    a = torch.arange(128, dtype=torch.float32, device="cuda")
    b = kernel(a)
    expected = a[start : start + 64].sum().reshape(1) + int(not clear)
    torch.testing.assert_close(b, expected, atol=0, rtol=0)


def _make_partial_sum(start):
    @T.prim_func
    def main(A: T.Tensor((128,), T.float32), B: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=128):
            src = T.alloc_fragment((128,), T.float32)
            T.copy(A, src)

            acc = T.alloc_reducer((1,), T.float32, op="sum")
            T.reducer_init(acc)
            for i in T.Parallel(128):
                if (i >= start) & (i < start + 64):
                    T.reducer_update(acc[0], src[i])

            result = T.alloc_fragment((1,), T.float32)
            T.finalize_reducer(acc, result)
            T.copy(result, B)

    return main


@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
@pytest.mark.parametrize("start", [0, 64])
def test_partial_fragment_reduce_with_alloc_reducer(start):
    """Preserve the fragment's full layout domain and predicate contributions."""
    kernel = tl.compile(_make_partial_sum(start), out_idx=-1)
    a = torch.arange(128, dtype=torch.float32, device="cuda")
    b = kernel(a)
    torch.testing.assert_close(b, a[start : start + 64].sum().reshape(1), atol=0, rtol=0)


if __name__ == "__main__":
    tilelang.testing.main()
