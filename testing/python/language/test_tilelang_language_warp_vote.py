"""Tests for warp-vote / warp-ballot / block-sync-with-predicate intrinsics.

Covered intrinsics
------------------
T.any_sync          – __any_sync  / __any  (HIP)
T.all_sync          – __all_sync  / __all  (HIP)
T.ballot_sync       – __ballot_sync→uint64 (CUDA, zero-ext) / __ballot uint64 (HIP, all lanes)
T.ballot            – full-warp ballot_sync / __ballot uint64 (HIP, all lanes)
T.activemask        – __activemask→uint64 (CUDA, zero-ext) / __ballot(1) uint64 (HIP, all lanes)
T.__ffs             – __ffs / __ffsll (CUDA)
T.__fns             – __fns (CUDA)
T.syncthreads_count – __syncthreads_count
T.syncthreads_and   – __syncthreads_and
T.syncthreads_or    – __syncthreads_or
"""

import tilelang
import tilelang.language as T
import pytest
import torch
import tilelang.testing


PREDICATE_INTRINSICS = [
    T.any_sync,
    T.all_sync,
    T.ballot_sync,
    T.ballot,
    T.syncthreads_count,
    T.syncthreads_and,
    T.syncthreads_or,
]


@pytest.mark.parametrize("intrinsic", PREDICATE_INTRINSICS)
def test_predicate_intrinsics_reject_float(intrinsic):
    predicate = tilelang.tvm.tirx.FloatImm("float32", 0.5)

    with pytest.raises(TypeError, match="requires an integer or boolean predicate"):
        intrinsic(predicate)


@pytest.mark.parametrize("predicate", [1, True], ids=["integer", "boolean"])
@pytest.mark.parametrize("intrinsic", PREDICATE_INTRINSICS)
def test_predicate_intrinsics_accept_integer_or_boolean(intrinsic, predicate):
    result = intrinsic(predicate)

    assert isinstance(result, tilelang.tvm.tirx.Call)


def _all_warp_vote_kernel():
    @T.prim_func
    def main(
        B0: T.Tensor((32,), T.int32),
        B1: T.Tensor((32,), T.int32),
        B2: T.Tensor((32,), T.int64),
        B3: T.Tensor((32,), T.int64),
        B4: T.Tensor((32,), T.int32),
        B5: T.Tensor((32,), T.int32),
        B6: T.Tensor((32,), T.int64),
        B7: T.Tensor((32,), T.int32),
        B8: T.Tensor((32,), T.int32),
        B9: T.Tensor((32,), T.int32),
        B10: T.Tensor((32,), T.int32),
        B11: T.Tensor((32,), T.int32),
        B12: T.Tensor((32,), T.int32),
        B13: T.Tensor((32,), T.int32),
        B14: T.Tensor((32,), T.int32),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            B0[tx] = T.any_sync(tx == 0)
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            B1[tx] = T.all_sync(1)
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            B2[tx] = T.cast(T.ballot_sync(tx == 0), T.int64)
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            B3[tx] = T.cast(T.ballot(1), T.int64)
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            mask = T.ballot_sync(tx >= 7)
            B4[tx] = T.__ffs(mask) - 1
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            mask = T.cast(T.ballot_sync(tx >= 7), T.uint32)
            B5[tx] = T.cast(T.__fns(mask, 0, 3), T.int32)
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            B6[tx] = T.cast(T.activemask(), T.int64)
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            B7[tx] = T.syncthreads_count(tx < 16)
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            B8[tx] = T.syncthreads_and(1)
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            B9[tx] = T.syncthreads_and(tx != 0)
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            B10[tx] = T.syncthreads_or(tx == 0)
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            B11[tx] = T.syncthreads_or(0)
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            value = T.if_then_else(tx < 16, 1, 2)
            B12[tx] = T.cast(T.match_any_sync(value), T.int32)
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            B13[tx] = T.cast(T.match_all_sync(7), T.int32)
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            B14[tx] = T.cast(T.match_all_sync(tx), T.int32)

    return main


@tilelang.testing.requires_cuda
def test_warp_vote_runtime():
    kernel = tilelang.compile(_all_warp_vote_kernel(), out_idx=list(range(15)), target="cuda")
    results = kernel()
    src = kernel.get_kernel_source()
    for intrinsic in (
        "__any_sync",
        "__all_sync",
        "__ballot_sync",
        "__ffsll",
        "__fns(",
        "__activemask",
        "__syncthreads_count",
        "__syncthreads_and",
        "__syncthreads_or",
        "__match_any_sync",
        "__match_all_sync",
    ):
        assert intrinsic in src, f"Expected {intrinsic} in source:\n{src}"

    assert torch.all(results[0] != 0)
    assert torch.all(results[1] != 0)
    assert int(results[2][0]) & 1
    assert (int(results[2][0]) >> 32) == 0
    assert (int(results[3][0]) & 0xFFFFFFFF) == 0xFFFFFFFF
    assert (int(results[3][0]) >> 32) == 0
    assert torch.all(results[4] == 7)
    assert torch.all(results[5] == 9)
    assert (int(results[6][0]) & 0xFFFFFFFF) == 0xFFFFFFFF
    assert (int(results[6][0]) >> 32) == 0
    assert torch.all(results[7] == 16)
    assert torch.all(results[8] != 0)
    assert torch.all(results[9] == 0)
    assert torch.all(results[10] != 0)
    assert torch.all(results[11] == 0)

    observed_u32 = results[12].to(torch.int64) & 0xFFFFFFFF
    expected = torch.tensor([0x0000FFFF] * 16 + [0xFFFF0000] * 16, dtype=torch.int64, device="cuda")
    assert torch.equal(observed_u32, expected)
    assert torch.all(results[13] == -1)
    assert torch.all(results[14] == 0)


@tilelang.testing.requires_cuda
def test_any_sync_preserves_nonzero_wide_integer_predicate():
    """A nonzero int64 predicate whose low 32 bits are zero must test true.

    The CUDA vote builtin takes a 32-bit int; passing the raw int64 value
    narrows 1 << 32 to 0 and silently flips the predicate to false.
    """

    @tilelang.jit
    def kernel():
        @T.prim_func
        def main(A: T.Tensor((32,), "int64"), Out: T.Tensor((1,), "int32")):
            with T.Kernel(1, threads=32):
                tx = T.get_thread_binding()
                result = T.any_sync(A[tx])
                if tx == 0:
                    Out[0] = result

        return main

    a = torch.zeros((32,), dtype=torch.int64, device="cuda")
    a[5] = 1 << 32
    out = torch.zeros((1,), dtype=torch.int32, device="cuda")
    kernel()(a, out)
    assert out.item() != 0, f"any_sync narrowed 1<<32 to zero: got {out.item()}"

    # A normalization that treats every wide predicate as true must not pass.
    a.zero_()
    out.zero_()
    kernel()(a, out)
    assert out.item() == 0, f"any_sync returned nonzero for an all-zero int64 predicate: got {out.item()}"


if __name__ == "__main__":
    tilelang.testing.main()
