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
            val = T.any_sync(tx == 0)
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
            val = T.all_sync(1)
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
            mask = T.ballot_sync(tx == 0)
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
# __ffs
# ---------------------------------------------------------------------------


@tilelang.jit
def kernel_ffs_ballot_sync():
    """Find the first lane whose ballot predicate is true."""

    @T.prim_func
    def main(
        B: T.Tensor((32,), "int32"),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            mask = T.ballot_sync(tx >= 7)
            B[tx] = T.__ffs(mask) - 1

    return main


@tilelang.testing.requires_cuda
def test_ffs_ballot_sync():
    b = torch.zeros((32,), device="cuda", dtype=torch.int32)
    kernel = kernel_ffs_ballot_sync()
    src = kernel.get_kernel_source()
    assert "__ballot_sync" in src, f"Expected __ballot_sync in source:\n{src}"
    assert "__ffsll" in src, f"Expected __ffsll for uint64 ballot mask in source:\n{src}"
    kernel(b)
    assert torch.all(b == 7), f"Expected all lanes to find lane 7, got {b}"


# ---------------------------------------------------------------------------
# __fns
# ---------------------------------------------------------------------------


@tilelang.jit
def kernel_fns_ballot_sync():
    """Find the third active lane in a ballot mask."""

    @T.prim_func
    def main(
        B: T.Tensor((32,), "int32"),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            mask = T.cast(T.ballot_sync(tx >= 7), "uint32")
            B[tx] = T.cast(T.__fns(mask, 0, 3), "int32")

    return main


@tilelang.testing.requires_cuda
def test_fns_ballot_sync():
    b = torch.zeros((32,), device="cuda", dtype=torch.int32)
    kernel = kernel_fns_ballot_sync()
    src = kernel.get_kernel_source()
    assert "__ballot_sync" in src, f"Expected __ballot_sync in source:\n{src}"
    assert "__fns(" in src, f"Expected __fns in source:\n{src}"
    kernel(b)
    assert torch.all(b == 9), f"Expected all lanes to find lane 9, got {b}"


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


# ---------------------------------------------------------------------------
# match_any_sync
# ---------------------------------------------------------------------------


@tilelang.jit
def kernel_match_any_sync():
    """Lanes 0-15 share value 1; lanes 16-31 share value 2. match_any_sync
    should return 0x0000FFFF for the first half and 0xFFFF0000 for the
    second half."""

    @T.prim_func
    def main(
        B: T.Tensor((32,), "int32"),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            value = T.if_then_else(tx < 16, 1, 2)
            peers = T.match_any_sync(value)
            B[tx] = T.cast(peers, "int32")

    return main


@tilelang.testing.requires_cuda
def test_match_any_sync():
    b = torch.zeros((32,), device="cuda", dtype=torch.int32)
    kernel = kernel_match_any_sync()
    src = kernel.get_kernel_source()
    assert "__match_any_sync" in src, f"Expected __match_any_sync in source:\n{src}"
    kernel(b)
    # Reinterpret the int32 buffer as uint32 to compare against bitmasks
    # whose high bit is set (0xFFFF0000 overflows int32).
    observed_u32 = b.to(torch.int64) & 0xFFFFFFFF
    expected = torch.tensor([0x0000FFFF] * 16 + [0xFFFF0000] * 16, dtype=torch.int64, device="cuda")
    assert torch.equal(observed_u32, expected), f"Expected {expected}, got {observed_u32}"


# ---------------------------------------------------------------------------
# match_all_sync
# ---------------------------------------------------------------------------


@tilelang.jit
def kernel_match_all_sync_true():
    """All lanes share value 7 → match_all_sync returns the full mask."""

    @T.prim_func
    def main(
        B: T.Tensor((32,), "int32"),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            result = T.match_all_sync(7)
            B[tx] = T.cast(result, "int32")

    return main


@tilelang.jit
def kernel_match_all_sync_false():
    """Lanes disagree → match_all_sync returns 0."""

    @T.prim_func
    def main(
        B: T.Tensor((32,), "int32"),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            result = T.match_all_sync(tx)
            B[tx] = T.cast(result, "int32")

    return main


@tilelang.testing.requires_cuda
def test_match_all_sync():
    b = torch.zeros((32,), device="cuda", dtype=torch.int32)
    kernel = kernel_match_all_sync_true()
    src = kernel.get_kernel_source()
    assert "__match_all_sync" in src, f"Expected __match_all_sync in source:\n{src}"
    kernel(b)
    assert torch.all(b == -1), f"Expected all 0xFFFFFFFF (sign-extended -1), got {b}"

    b2 = torch.zeros((32,), device="cuda", dtype=torch.int32)
    kernel_match_all_sync_false()(b2)
    assert torch.all(b2 == 0), f"Expected all 0, got {b2}"


# ---------------------------------------------------------------------------
# Predicate width
#
# The predicate parameter follows the native CUDA/HIP 32-bit `int` semantics:
# a wider integer is narrowed before being tested for non-zero. These tests pin
# that contract (see docs/programming_guides/instructions.md) so that callers
# who pass a wide integer keep seeing the documented behaviour, and they cover
# the recommended `value != 0` spelling for testing every bit of the value.
#
# Every intrinsic named in the documentation is exercised. Each one is
# normalised to a 0/1 verdict so that a single family covers "any"-style
# intrinsics (non-zero as soon as one lane votes) and "all"-style intrinsics
# (non-zero only when every lane votes).
# ---------------------------------------------------------------------------

# 2**32 has all-zero low 32 bits, so it narrows to 0 and must vote false.
_WIDE_ONLY_VALUE = 1 << 32

#: Intrinsics that report non-zero only when every lane's predicate is set.
_ALL_INTRINSICS = ["all_sync", "syncthreads_and"]
#: Intrinsics that report non-zero as soon as a single lane's predicate is set.
_ANY_INTRINSICS = ["any_sync", "syncthreads_count", "syncthreads_or"]
#: Intrinsics that return the whole lane bitmask instead of a single verdict.
_BALLOT_INTRINSICS = ["ballot_sync", "ballot"]
_WIDE_PREDICATE_INTRINSICS = _ANY_INTRINSICS + _ALL_INTRINSICS

#: The "all" intrinsics need every lane to vote; the "any" ones need only lane 0.
_ALL_LANES_VOTE = {name: (name in _ALL_INTRINSICS) for name in _WIDE_PREDICATE_INTRINSICS}


def _wide_predicate_verdict(intrinsic: str, predicate):
    """Evaluate ``intrinsic(predicate)`` as a 0/1 verdict."""
    if intrinsic == "any_sync":
        return T.cast(T.any_sync(predicate) != 0, "int32")
    if intrinsic == "all_sync":
        return T.cast(T.all_sync(predicate) != 0, "int32")
    if intrinsic == "syncthreads_count":
        return T.syncthreads_count(predicate)
    if intrinsic == "syncthreads_and":
        return T.cast(T.syncthreads_and(predicate) != 0, "int32")
    if intrinsic == "syncthreads_or":
        return T.cast(T.syncthreads_or(predicate) != 0, "int32")
    raise ValueError(f"{intrinsic!r} returns a lane bitmask, not a 0/1 verdict")


def _ballot_mask(intrinsic: str, predicate):
    """Raw ``uint64`` lane bitmask of a ballot intrinsic."""
    if intrinsic == "ballot_sync":
        return T.ballot_sync(predicate)
    if intrinsic == "ballot":
        return T.ballot(predicate)
    raise ValueError(f"{intrinsic!r} does not return a mask")


def _wide_predicate_mask_kernel(intrinsic: str, compare_explicitly: bool):
    """Evaluate a ballot intrinsic over an int64 predicate of ``2**32``.

    Only lane 0 carries ``2**32``, so the correct mask keeps exactly bit 0.
    The mask is written to an ``int64`` buffer: for a 32-lane warp the mask has
    no bit above 31, so the sign bit stays clear and the value round-trips
    exactly without relying on an unsigned torch dtype.
    """
    assert intrinsic in _BALLOT_INTRINSICS

    @tilelang.jit
    def build():
        @T.prim_func
        def main(B: T.Tensor((1,), "int64")):
            with T.Kernel(1, threads=32):
                tx = T.get_thread_binding()
                wide = T.int64(_WIDE_ONLY_VALUE) if tx == 0 else T.int64(0)
                predicate = (wide != T.int64(0)) if compare_explicitly else wide
                mask = _ballot_mask(intrinsic, predicate)
                if tx == 0:
                    B[0] = T.cast(mask, "int64")

        return main

    return build


def _run_wide_predicate_mask(intrinsic: str, compare_explicitly: bool) -> int:
    b = torch.zeros((1,), device="cuda", dtype=torch.int64)
    _wide_predicate_mask_kernel(intrinsic, compare_explicitly)()(b)
    return int(b[0].item())


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("intrinsic", _BALLOT_INTRINSICS)
def test_wide_predicate_ballot_mask_is_exact(intrinsic):
    """Ballot must return exactly lane 0's bit, not merely "some" bit."""
    # 2**32 narrows to 0, so no lane votes and the mask must be empty.
    narrowed = _run_wide_predicate_mask(intrinsic, compare_explicitly=False)
    assert narrowed == 0, f"{intrinsic}: expected mask 0x0 (2**32 narrows to 0), got 0x{narrowed:x}"

    # Comparing with 0 first restores the only voting lane, which is lane 0.
    explicit = _run_wide_predicate_mask(intrinsic, compare_explicitly=True)
    assert explicit == 0x1, f"{intrinsic}: expected mask 0x1, got 0x{explicit:x}"


def _wide_predicate_kernel(intrinsic: str, compare_explicitly: bool):
    """Evaluate one intrinsic over an int64 predicate of ``2**32``.

    Lanes that should vote carry ``2**32``; the remaining lanes carry 0.
    ``compare_explicitly`` selects the documented ``value != 0`` spelling.
    """
    vote_every_lane = _ALL_LANES_VOTE[intrinsic]

    @tilelang.jit
    def build():
        @T.prim_func
        def main(B: T.Tensor((32,), "int32")):
            with T.Kernel(1, threads=32):
                tx = T.get_thread_binding()
                if vote_every_lane:
                    wide = T.int64(_WIDE_ONLY_VALUE)
                else:
                    wide = T.int64(_WIDE_ONLY_VALUE) if tx == 0 else T.int64(0)
                predicate = (wide != T.int64(0)) if compare_explicitly else wide
                B[tx] = _wide_predicate_verdict(intrinsic, predicate)

        return main

    return build


def _run_wide_predicate(intrinsic: str, compare_explicitly: bool) -> torch.Tensor:
    b = torch.full((32,), -1, device="cuda", dtype=torch.int32)
    _wide_predicate_kernel(intrinsic, compare_explicitly)()(b)
    return b


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("intrinsic", _WIDE_PREDICATE_INTRINSICS)
def test_wide_predicate_narrows_to_low_32_bits(intrinsic):
    """A 2**32 int64 predicate narrows to 0, so no lane votes."""
    b = _run_wide_predicate(intrinsic, compare_explicitly=False)
    assert torch.all(b == 0), f"{intrinsic}: expected every lane to report 0 (2**32 narrows to 0), got {b}"


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("intrinsic", _WIDE_PREDICATE_INTRINSICS)
def test_wide_predicate_with_explicit_zero_compare(intrinsic):
    """Comparing with 0 first makes every bit of the value participate."""
    b = _run_wide_predicate(intrinsic, compare_explicitly=True)
    assert torch.all(b == 1), f"{intrinsic}: expected every lane to report 1, got {b}"


if __name__ == "__main__":
    tilelang.testing.main()
