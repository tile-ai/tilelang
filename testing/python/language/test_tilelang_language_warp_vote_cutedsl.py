"""Warp-vote / warp-match / bit-search / predicate-barrier intrinsics on CuTeDSL.

Mirrors test_tilelang_language_warp_vote.py with ``target="cutedsl"`` and checks
the exact values the CUDA intrinsics return:

T.any_sync / T.all_sync           – __any_sync / __all_sync (0 or 1)
T.ballot_sync / T.ballot          – __ballot_sync zero-extended to uint64
T.activemask                      – __activemask zero-extended to uint64
T.match_any_sync / match_all_sync – __match_{any,all}_sync, 32- and 64-bit
T.__ffs                           – __ffs / __ffsll
T.__fns                           – __fns
T.syncthreads_count / _and / _or  – __syncthreads_count / _and / _or
T.named_barrier_arrive            – bar.arrive
T.tvm_thread_allreduce            – the shared-memory tree uses activemask/ballot_sync
"""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


FULL_MASK = 0xFFFFFFFF


def _require_cutedsl():
    try:
        from tilelang.jit.adapter.cutedsl.checks import check_cutedsl_available

        check_cutedsl_available()
    except (ImportError, ModuleNotFoundError, RuntimeError, AssertionError) as err:
        pytest.skip(f"CuTeDSL is not available: {err}")


def _compile(func, **kwargs):
    _require_cutedsl()
    return tilelang.compile(func, target="cutedsl", **kwargs)


def _as_u32(t):
    return t.to(torch.int64) & FULL_MASK


def _expect(values, dtype=torch.int64):
    return torch.tensor(values, dtype=dtype, device="cuda")


# ---------------------------------------------------------------------------
# any_sync / all_sync
# ---------------------------------------------------------------------------


def kernel_any_all_sync():
    @T.prim_func
    def main(B: T.Tensor((4, 32), "int32")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            B[0, tx] = T.any_sync(tx == 0)
            B[1, tx] = T.any_sync(tx == 99)
            # Non-boolean predicates count as true when non-zero, not by bit 0.
            B[2, tx] = T.all_sync(tx + 2)
            B[3, tx] = T.all_sync(tx != 5)

    return main


@tilelang.testing.requires_cuda
def test_any_all_sync():
    kernel = _compile(kernel_any_all_sync())
    src = kernel.get_kernel_source()
    assert "tl.__any_sync(" in src, src
    assert "tl.__all_sync(" in src, src
    b = torch.full((4, 32), -7, device="cuda", dtype=torch.int32)
    kernel(b)
    expected = torch.tensor([1, 0, 1, 0], device="cuda", dtype=torch.int32)[:, None].expand(4, 32)
    torch.testing.assert_close(b, expected)


# ---------------------------------------------------------------------------
# ballot_sync / ballot
# ---------------------------------------------------------------------------


def kernel_ballot():
    @T.prim_func
    def main(B: T.Tensor((4, 32), "int64")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            B[0, tx] = T.cast(T.ballot_sync(tx == 0), "int64")
            # Bit 31 set: the 32-bit ballot must be zero-extended, not sign-extended.
            B[1, tx] = T.cast(T.ballot_sync(tx == 31), "int64")
            B[2, tx] = T.cast(T.ballot(1), "int64")
            B[3, tx] = 0
            if tx < 16:
                B[3, tx] = T.cast(T.ballot_sync(tx % 2 == 0, mask=0x0000FFFF), "int64")

    return main


@tilelang.testing.requires_cuda
def test_ballot():
    kernel = _compile(kernel_ballot())
    src = kernel.get_kernel_source()
    assert "tl.ballot_sync(" in src, src
    assert "tl.ballot(" in src, src
    b = torch.zeros((4, 32), device="cuda", dtype=torch.int64)
    kernel(b)
    torch.testing.assert_close(b[0], _expect([0x1] * 32))
    torch.testing.assert_close(b[1], _expect([0x80000000] * 32))
    torch.testing.assert_close(b[2], _expect([FULL_MASK] * 32))
    torch.testing.assert_close(b[3], _expect([0x5555] * 16 + [0] * 16))


# ---------------------------------------------------------------------------
# activemask
# ---------------------------------------------------------------------------


def kernel_activemask():
    @T.prim_func
    def main(B: T.Tensor((32,), "int64")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            B[tx] = T.cast(T.activemask(), "int64")

    return main


@tilelang.testing.requires_cuda
def test_activemask():
    kernel = _compile(kernel_activemask())
    src = kernel.get_kernel_source()
    assert "tl.activemask()" in src, src
    b = torch.zeros((32,), device="cuda", dtype=torch.int64)
    kernel(b)
    # All 32 lanes are active; the upper 32 bits must be zero.
    torch.testing.assert_close(b, _expect([FULL_MASK] * 32))


# ---------------------------------------------------------------------------
# __ffs / __ffsll
# ---------------------------------------------------------------------------


def kernel_ffs():
    @T.prim_func
    def main(B: T.Tensor((3, 32), "int32")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            B[0, tx] = T.__ffs(T.ballot_sync(tx >= 7)) - 1
            v32 = T.if_then_else(tx == 0, T.int32(0), T.int32(1) << tx)
            B[1, tx] = T.__ffs(v32)
            v64 = T.if_then_else(tx == 0, T.int64(0), T.int64(1) << T.cast(tx + 32, "int64"))
            B[2, tx] = T.__ffs(v64)

    return main


@tilelang.testing.requires_cuda
def test_ffs():
    kernel = _compile(kernel_ffs())
    src = kernel.get_kernel_source()
    assert "tl.__ffs(" in src, src
    assert "tl.__ffsll(" in src, src
    b = torch.full((3, 32), -7, device="cuda", dtype=torch.int32)
    kernel(b)
    i32 = torch.int32
    torch.testing.assert_close(b[0], _expect([7] * 32, i32))
    # Lane 31 sets the sign bit; zero inputs return 0.
    torch.testing.assert_close(b[1], _expect([0] + [lane + 1 for lane in range(1, 32)], i32))
    torch.testing.assert_close(b[2], _expect([0] + [lane + 33 for lane in range(1, 32)], i32))


# ---------------------------------------------------------------------------
# __fns
# ---------------------------------------------------------------------------


FNS_MASK = 0xA5A5F00F


def _fns_reference(mask, base, offset):
    """PTX fns.b32 semantics."""
    if offset == 0:
        return base if (mask >> base) & 1 else FULL_MASK
    pos, count, inc = base, abs(offset) - 1, 1 if offset > 0 else -1
    while 0 <= pos < 32:
        if (mask >> pos) & 1:
            if count == 0:
                return pos
            count -= 1
        pos += inc
    return FULL_MASK


def kernel_fns():
    @T.prim_func
    def main(B: T.Tensor((2, 32), "int32")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            mask = T.cast(T.ballot_sync(tx >= 7), "uint32")
            B[0, tx] = T.cast(T.__fns(mask, 0, 3), "int32")
            B[1, tx] = T.cast(T.__fns(T.uint32(FNS_MASK), tx, tx % 7 - 3), "int32")

    return main


@tilelang.testing.requires_cuda
def test_fns():
    kernel = _compile(kernel_fns())
    src = kernel.get_kernel_source()
    assert "tl.__fns(" in src, src
    b = torch.zeros((2, 32), device="cuda", dtype=torch.int32)
    kernel(b)
    torch.testing.assert_close(b[0], _expect([9] * 32, torch.int32))
    expected = [_fns_reference(FNS_MASK, lane, lane % 7 - 3) for lane in range(32)]
    torch.testing.assert_close(_as_u32(b[1]), _expect(expected))


# ---------------------------------------------------------------------------
# syncthreads_count / syncthreads_and / syncthreads_or
# ---------------------------------------------------------------------------


def kernel_syncthreads_reduce():
    @T.prim_func
    def main(B: T.Tensor((7, 64), "int32")):
        with T.Kernel(1, threads=64):
            tx = T.get_thread_binding()
            B[0, tx] = T.syncthreads_count(tx < 16)
            B[1, tx] = T.syncthreads_count(tx & 4)
            B[2, tx] = T.syncthreads_and(tx + 1)
            B[3, tx] = T.syncthreads_and(tx != 0)
            B[4, tx] = T.syncthreads_or(tx == 63)
            B[5, tx] = T.syncthreads_or(tx & 64)
            B[6, tx] = T.syncthreads_or(0)

    return main


@tilelang.testing.requires_cuda
def test_syncthreads_reduce():
    kernel = _compile(kernel_syncthreads_reduce())
    src = kernel.get_kernel_source()
    for name in ("tl.__syncthreads_count(", "tl.__syncthreads_and(", "tl.__syncthreads_or("):
        assert name in src, src
    b = torch.full((7, 64), -7, device="cuda", dtype=torch.int32)
    kernel(b)
    expected = torch.tensor([16, 32, 1, 0, 1, 0, 0], device="cuda", dtype=torch.int32)[:, None].expand(7, 64)
    torch.testing.assert_close(b, expected)


# ---------------------------------------------------------------------------
# match_any_sync / match_all_sync
# ---------------------------------------------------------------------------


def kernel_match():
    @T.prim_func
    def main(B: T.Tensor((7, 32), "int32")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            B[0, tx] = T.cast(T.match_any_sync(T.if_then_else(tx < 16, 1, 2)), "int32")
            # The halves differ only above bit 31, so a 32-bit match would merge them.
            high = T.cast(tx // 16, "int64") << T.int64(40)
            B[1, tx] = T.cast(T.match_any_sync(high), "int32")
            B[2, tx] = T.cast(T.match_any_sync(T.if_then_else(tx < 16, T.float32(1.0), T.float32(2.0))), "int32")
            B[3, tx] = T.cast(T.match_all_sync(7), "int32")
            B[4, tx] = T.cast(T.match_all_sync(tx), "int32")
            B[5, tx] = T.cast(T.match_all_sync(T.int64(1) << T.int64(40)), "int32")
            B[6, tx] = T.cast(T.match_all_sync(high), "int32")

    return main


@tilelang.testing.requires_cuda
def test_match():
    kernel = _compile(kernel_match())
    src = kernel.get_kernel_source()
    assert "tl.__match_any_sync(" in src, src
    assert "tl.__match_all_sync(" in src, src
    b = torch.full((7, 32), -7, device="cuda", dtype=torch.int32)
    kernel(b)
    halves = [0x0000FFFF] * 16 + [0xFFFF0000] * 16
    torch.testing.assert_close(_as_u32(b[0]), _expect(halves))
    torch.testing.assert_close(_as_u32(b[1]), _expect(halves))
    torch.testing.assert_close(_as_u32(b[2]), _expect(halves))
    torch.testing.assert_close(_as_u32(b[3]), _expect([FULL_MASK] * 32))
    torch.testing.assert_close(_as_u32(b[4]), _expect([0] * 32))
    torch.testing.assert_close(_as_u32(b[5]), _expect([FULL_MASK] * 32))
    torch.testing.assert_close(_as_u32(b[6]), _expect([0] * 32))


# ---------------------------------------------------------------------------
# named_barrier_arrive (mirrors testing/python/issue/test_tilelang_issue_2188.py)
# ---------------------------------------------------------------------------


def kernel_named_barrier_arrive():
    @T.prim_func
    def kernel(inp: T.Tensor((1,), T.int32), out: T.Tensor((1,), T.int32)):
        with T.Kernel(1, threads=64) as _:
            tx = T.get_thread_binding()
            slot = T.alloc_shared((1,), T.int32)

            ready_barrier = 10
            empty_barrier = 11
            participant_threads = 64

            if tx < 32:
                if tx == 0:
                    slot[0] = inp[0] + 1
                T.named_barrier_arrive(ready_barrier, participant_threads)
                T.sync_threads(empty_barrier, participant_threads)
            else:
                T.sync_threads(ready_barrier, participant_threads)
                if tx == 32:
                    out[0] = slot[0]
                T.named_barrier_arrive(empty_barrier, participant_threads)

    return kernel


@tilelang.testing.requires_cuda
def test_named_barrier_arrive():
    kernel = _compile(kernel_named_barrier_arrive(), out_idx=[1])
    src = kernel.get_kernel_source()
    assert "tl.named_barrier_arrive(10, 64)" in src, src
    assert "tl.named_barrier_arrive(11, 64)" in src, src
    inp = torch.tensor([41], device="cuda", dtype=torch.int32)
    torch.testing.assert_close(kernel(inp).cpu(), torch.tensor([42], dtype=torch.int32))


# ---------------------------------------------------------------------------
# bitwise_not on bool
# ---------------------------------------------------------------------------


def kernel_bool_bitwise_not():
    @T.prim_func
    def main(A: T.Tensor((2,), "bool"), B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=1):
            for i in T.serial(2):
                B[i] = T.Cast("int32", ~A[i])
            B[2] = T.Cast("int32", ~T.bool(True))
            B[3] = T.Cast("int32", ~T.bool(False))

    return main


@tilelang.testing.requires_cuda
def test_bool_bitwise_not():
    kernel = _compile(kernel_bool_bitwise_not(), out_idx=[1])
    src = kernel.get_kernel_source()
    assert "~True" not in src and "~False" not in src, src
    a = torch.tensor([False, True], device="cuda")
    torch.testing.assert_close(kernel(a), _expect([1, 0, 0, 1], torch.int32))


# ---------------------------------------------------------------------------
# tvm_thread_allreduce shared-memory tree (emits tl::activemask / tl::ballot_sync)
# ---------------------------------------------------------------------------


def kernel_thread_allreduce_sum(threads, dtype):
    @T.prim_func
    def main(A: T.Tensor((threads,), dtype), C: T.Tensor((1,), dtype)):
        with T.Kernel(1, threads=threads):
            tk = T.get_thread_binding(0)
            v = T.alloc_local((1,), dtype)
            v[0] = A[tk]
            Cr = T.alloc_local((1,), dtype)
            with T.attr(
                T.comm_reducer(lambda x, y: x + y, [T.cast(0, dtype)]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(T.uint32(1), v[0], True, Cr[0], tk, dtype="handle"))
            C[0] = Cr[0]

    return main


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("threads", [8, 32, 64])
@pytest.mark.parametrize("dtype", ["int8", "int16"])
def test_thread_allreduce_buffer_tree(threads, dtype):
    # Sub-32-bit types skip the warp-shuffle reduction and take the shared
    # memory tree, whose in-warp stage is masked by activemask/ballot_sync.
    kernel = _compile(
        kernel_thread_allreduce_sum(threads, dtype),
        out_idx=[1],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    src = kernel.get_kernel_source()
    assert "tl.activemask()" in src, src
    assert "tl.ballot_sync(" in src, src
    torch_dtype = getattr(torch, dtype)
    a = (torch.arange(threads, device="cuda") % 5 + 1).to(torch_dtype)
    torch.testing.assert_close(kernel(a)[0], a.sum().to(torch_dtype))


if __name__ == "__main__":
    tilelang.testing.main()
