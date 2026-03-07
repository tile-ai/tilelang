"""Tests for warp-vote / warp-ballot / block-sync-with-predicate intrinsics.

Covered intrinsics
------------------
T.any_sync          – __any_sync  / __any  (HIP)
T.all_sync          – __all_sync  / __all  (HIP)
T.ballot_sync       – __ballot_sync→uint64 (CUDA, zero-ext) / __ballot uint64 (HIP, all lanes)
T.ballot            – full-warp ballot_sync / __ballot uint64 (HIP, all lanes)
T.activemask        – __activemask→uint64 (CUDA, zero-ext) / __ballot(1) uint64 (HIP, all lanes)
T.syncthreads_count – __syncthreads_count
T.syncthreads_and   – __syncthreads_and
T.syncthreads_or    – __syncthreads_or
"""

import tilelang
import tilelang.language as T
import torch
import tilelang.testing


# ---------------------------------------------------------------------------
# any_sync
# ---------------------------------------------------------------------------


@tilelang.jit
def kernel_any_sync():
    """Lane 0 has a non-zero predicate; any_sync should return non-zero for all lanes."""

    @T.prim_func
    def main(
        B: T.Tensor((32,), "int32"),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            val = T.any_sync(0xFFFFFFFF, tx == 0)
            B[tx] = val

    return main


@tilelang.testing.requires_cuda
def test_any_sync():
    b = torch.zeros((32,), device="cuda", dtype=torch.int32)
    kernel = kernel_any_sync()
    src = kernel.get_kernel_source()
    assert "__any_sync" in src or "__any" in src, f"Expected __any_sync/__any in source:\n{src}"
    kernel(b)
    # any lane (lane 0) has predicate==1 → result must be non-zero for all lanes
    assert torch.all(b != 0), f"Expected all non-zero, got {b}"


# ---------------------------------------------------------------------------
# all_sync
# ---------------------------------------------------------------------------


@tilelang.jit
def kernel_all_sync():
    """All lanes always pass predicate 1 → all_sync should return non-zero."""

    @T.prim_func
    def main(
        B: T.Tensor((32,), "int32"),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            val = T.all_sync(0xFFFFFFFF, 1)
            B[tx] = val

    return main


@tilelang.testing.requires_cuda
def test_all_sync():
    b = torch.zeros((32,), device="cuda", dtype=torch.int32)
    kernel = kernel_all_sync()
    src = kernel.get_kernel_source()
    assert "__all_sync" in src or "__all" in src, f"Expected __all_sync/__all in source:\n{src}"
    kernel(b)
    assert torch.all(b != 0), f"Expected all non-zero, got {b}"


# ---------------------------------------------------------------------------
# ballot_sync
# ---------------------------------------------------------------------------


@tilelang.jit
def kernel_ballot_sync():
    """Only lane 0 has a non-zero predicate → bit 0 of ballot must be set."""

    @T.prim_func
    def main(
        B: T.Tensor((32,), "int64"),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            mask = T.ballot_sync(0xFFFFFFFF, tx == 0)
            B[tx] = T.cast(mask, "int64")

    return main


@tilelang.testing.requires_cuda
def test_ballot_sync():
    b = torch.zeros((32,), device="cuda", dtype=torch.int64)
    kernel = kernel_ballot_sync()
    src = kernel.get_kernel_source()
    assert "__ballot_sync" in src or "__ballot" in src, f"Expected __ballot_sync/__ballot in source:\n{src}"
    kernel(b)
    # All lanes read the same ballot value; bit 0 must be set (lane 0 had pred=1)
    assert int(b[0]) & 1, f"Expected bit 0 set in ballot result, got {int(b[0]):#018x}"
    # upper 32 bits must be zero on CUDA (32-wide warp)
    assert (int(b[0]) >> 32) == 0, f"Expected upper 32 bits zero on CUDA, got {int(b[0]):#018x}"


# ---------------------------------------------------------------------------
# ballot  (full-warp convenience wrapper)
# ---------------------------------------------------------------------------


@tilelang.jit
def kernel_ballot():
    """All lanes pass predicate 1 → lower 32 bits of ballot must all be set."""

    @T.prim_func
    def main(
        B: T.Tensor((32,), "int64"),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            mask = T.ballot(1)
            B[tx] = T.cast(mask, "int64")

    return main


@tilelang.testing.requires_cuda
def test_ballot():
    b = torch.zeros((32,), device="cuda", dtype=torch.int64)
    kernel = kernel_ballot()
    src = kernel.get_kernel_source()
    assert "__ballot_sync" in src or "__ballot" in src, f"Expected __ballot_sync/__ballot in source:\n{src}"
    kernel(b)
    # With predicate=1 for all 32 lanes the lower 32 bits should be 0xFFFFFFFF;
    # upper 32 bits are 0 on CUDA (32-wide warp).
    assert (int(b[0]) & 0xFFFFFFFF) == 0xFFFFFFFF, f"Expected lower 32 bits all set, got {int(b[0]):#018x}"
    assert (int(b[0]) >> 32) == 0, f"Expected upper 32 bits zero on CUDA, got {int(b[0]):#018x}"


# ---------------------------------------------------------------------------
# activemask
# ---------------------------------------------------------------------------


@tilelang.jit
def kernel_activemask():
    """All 32 threads are active → lower 32 bits of activemask must all be set."""

    @T.prim_func
    def main(
        B: T.Tensor((32,), "int64"),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            mask = T.activemask()
            B[tx] = T.cast(mask, "int64")

    return main


@tilelang.testing.requires_cuda
def test_activemask():
    b = torch.zeros((32,), device="cuda", dtype=torch.int64)
    kernel = kernel_activemask()
    src = kernel.get_kernel_source()
    assert "__activemask" in src or "__ballot" in src, f"Expected __activemask/__ballot in source:\n{src}"
    kernel(b)
    # All 32 lanes active → lower 32 bits = 0xFFFFFFFF; upper 32 bits = 0 on CUDA.
    assert (int(b[0]) & 0xFFFFFFFF) == 0xFFFFFFFF, f"Expected lower 32 bits all set, got {int(b[0]):#018x}"
    assert (int(b[0]) >> 32) == 0, f"Expected upper 32 bits zero on CUDA, got {int(b[0]):#018x}"


# ---------------------------------------------------------------------------
# syncthreads_count
# ---------------------------------------------------------------------------


@tilelang.jit
def kernel_syncthreads_count():
    """Exactly half the threads (lanes 0–15) pass predicate 1."""

    @T.prim_func
    def main(
        B: T.Tensor((32,), "int32"),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            cnt = T.syncthreads_count(tx < 16)
            B[tx] = cnt

    return main


@tilelang.testing.requires_cuda
def test_syncthreads_count():
    b = torch.zeros((32,), device="cuda", dtype=torch.int32)
    kernel = kernel_syncthreads_count()
    src = kernel.get_kernel_source()
    assert "__syncthreads_count" in src, f"Expected __syncthreads_count in source:\n{src}"
    kernel(b)
    assert torch.all(b == 16), f"Expected all 16, got {b}"


# ---------------------------------------------------------------------------
# syncthreads_and
# ---------------------------------------------------------------------------


@tilelang.jit
def kernel_syncthreads_and_true():
    """All threads pass predicate 1 → syncthreads_and returns non-zero."""

    @T.prim_func
    def main(
        B: T.Tensor((32,), "int32"),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            result = T.syncthreads_and(1)
            B[tx] = result

    return main


@tilelang.jit
def kernel_syncthreads_and_false():
    """Thread 0 passes predicate 0 → syncthreads_and returns 0."""

    @T.prim_func
    def main(
        B: T.Tensor((32,), "int32"),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            result = T.syncthreads_and(tx != 0)
            B[tx] = result

    return main


@tilelang.testing.requires_cuda
def test_syncthreads_and():
    b = torch.zeros((32,), device="cuda", dtype=torch.int32)
    kernel = kernel_syncthreads_and_true()
    src = kernel.get_kernel_source()
    assert "__syncthreads_and" in src, f"Expected __syncthreads_and in source:\n{src}"
    kernel(b)
    assert torch.all(b != 0), f"Expected all non-zero, got {b}"

    b2 = torch.zeros((32,), device="cuda", dtype=torch.int32)
    kernel2 = kernel_syncthreads_and_false()
    kernel2(b2)
    assert torch.all(b2 == 0), f"Expected all 0, got {b2}"


# ---------------------------------------------------------------------------
# syncthreads_or
# ---------------------------------------------------------------------------


@tilelang.jit
def kernel_syncthreads_or_true():
    """At least one thread (lane 0) passes predicate 1 → syncthreads_or != 0."""

    @T.prim_func
    def main(
        B: T.Tensor((32,), "int32"),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            result = T.syncthreads_or(tx == 0)
            B[tx] = result

    return main


@tilelang.jit
def kernel_syncthreads_or_false():
    """No thread passes predicate 1 → syncthreads_or returns 0."""

    @T.prim_func
    def main(
        B: T.Tensor((32,), "int32"),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            result = T.syncthreads_or(0)
            B[tx] = result

    return main


@tilelang.testing.requires_cuda
def test_syncthreads_or():
    b = torch.zeros((32,), device="cuda", dtype=torch.int32)
    kernel = kernel_syncthreads_or_true()
    src = kernel.get_kernel_source()
    assert "__syncthreads_or" in src, f"Expected __syncthreads_or in source:\n{src}"
    kernel(b)
    assert torch.all(b != 0), f"Expected all non-zero, got {b}"

    b2 = torch.zeros((32,), device="cuda", dtype=torch.int32)
    kernel2 = kernel_syncthreads_or_false()
    kernel2(b2)
    assert torch.all(b2 == 0), f"Expected all 0, got {b2}"


if __name__ == "__main__":
    tilelang.testing.main()
