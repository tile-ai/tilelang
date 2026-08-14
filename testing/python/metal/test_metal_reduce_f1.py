"""Metal fragment-reduce F1 single-simdgroup plan enforcement (D14/D16/D17/D18).

D14 made the F1 discipline plan-level: every thread_step of a fragment
reduce's XOR-butterfly must stay inside one 32-lane simdgroup (Metal has
no named barriers; cross-simdgroup in-place combine relies on SIMD
lockstep ordering that does not hold across simdgroups, and the raw
thread index addresses the scratch buffer without a tid < nt guard).

D16 (Codex review 54fd8741 HIGH-1, cross-validated by internal audit
AS-24) tightened the criterion to per-step nt = extent*scale > 32 ->
reject: CheckAllReduceWidth (src/backend/common/op/reduce.h) only
requires logical_width == extent to be a positive power of two and
scale > 0; scale itself need NOT be a power of two, so the public
constructor _make_allreduce_dim0_scale_kernel(T.reduce_sum, 16, 3)
generated a 48-thread plan (offsets 24/12/6/3) whose partner 32^24 = 56
is OOB on the scratch and outside the threadgroup.

D17 (Codex review 6dc03835 HIGH-1 mathematical deepening) closes the
remaining closure window: nt <= 32 is NOT sufficient.  The raw XOR
butterfly is closed on the participating range [0, nt) iff nt is a power
of two — each offset must flip a single lane bit below log2(nt); for a
non-power-of-two nt the partner of a participating lane escapes the
range even though every offset is < 32 (e.g. extent=8, scale=3, nt=24:
partner 16^12 = 28 >= 24).

D18 (Codex review 2c5ae8b7 HIGH-1, this round) closes the ACTUAL
EXECUTION domain: the raw butterfly runs on ALL N threadgroup threads
(no tid < nt guard), so the closure domain is the whole prefix [0, N),
not just the first block [0, nt).  The largest mask nt/2 partitions
lanes into aligned nt-blocks; [0, N) is closed under every mask iff
N % nt == 0 (N = threadgroup extent, compile-time constant).  A
misaligned extent (e.g. (8,2) with N=24: nt=16, N%16=8) lets the last
incomplete block read partners >= N (OOB: scratch sized N) or
never-written codegen-padding slots.  The criterion is now per-step

    reject iff nt = extent*scale is not a power of two
                OR nt > 32
                OR N % nt != 0
    (allow iff nt is a power of two AND nt <= 32 AND N % nt == 0)

(v2's 32*1 = 32 with N=128, power-of-two replication plans like
16*2 = 32, and complete multi-block closures like (8,2) with N=32 =
2*16 stay inside XOR-closed simdgroups and are allowed).

These tests are compile/lower-only (no GPU launch): the Metal backend
must reject unsafe plans at lowering time with the single-simdgroup
diagnostic (naming the step's scale, the offending nt, and the
threadgroup extent N), and must accept every plan whose nt is a power
of two <= 32 with N % nt == 0, with all XOR offsets < 32 AND the raw
[0, N) execution prefix closed under every mask found in the MSL.  The
runtime-correctness half lives in the workspace verification suite
(reduce_f1_planlevel_regression.py, GPU-gated).
"""

import re

import pytest

import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang import tvm as tvm


def _make_allreduce_dim0_scale_kernel(reduce_fn, logical_width, scale, threads=None):
    """Copy of the upstream public constructor
    (testing/python/language/test_tilelang_language_reduce.py), used as
    the HIGH-1 repro path for the Metal backend.

    threads defaults to logical_width * scale (N == nt); D18 decouples
    the threadgroup extent from the plan's nt so misaligned N % nt
    threadgroups are reachable through the same public constructor.
    """
    if threads is None:
        threads = logical_width * scale

    @T.prim_func
    def kernel(
        A: T.Tensor((logical_width, scale), T.float32),
        B: T.Tensor((scale,), T.float32),
    ):
        with T.Kernel(1, threads=threads):
            src = T.alloc_fragment((logical_width, scale), T.float32)
            dst = T.alloc_fragment((scale,), T.float32)
            T.copy(A, src)
            reduce_fn(src, dst, dim=0)
            T.copy(dst, B)

    return kernel


def _xor_closed(offsets, n):
    """The raw execution prefix [0, n) must be closed under every XOR
    mask used in the MSL.

    XOR with a mask is a permutation of the integers, so the image of
    [0, n) is [0, n) iff the maximum partner stays inside the range.
    D17: for the participating range [0, nt) this holds iff nt is a
    power of two (a non-power-of-two nt e.g. 24 with masks 12/6/3 lets
    some partner escape even though every mask is < 32).  D18: the raw
    butterfly runs on ALL N threads without a tid < nt guard, so the
    domain that must actually be closed is the whole prefix [0, N);
    with nt a power of two this holds iff N % nt == 0 (every nt-block
    complete).
    """
    assert n > 0
    return all(max(tid ^ mask for tid in range(n)) < n for mask in offsets)


def _lower_metal(prim_func):
    target = tvm.target.Target("metal", tvm.target.Target("llvm"))
    with target:
        artifact = tilelang.lower(
            prim_func,
            target=target,
            target_host="llvm",
            enable_host_codegen=False,
            enable_device_compile=False,
        )
    return artifact.kernel_source or ""


@pytest.mark.parametrize(
    ("logical_width", "scale", "nt"),
    [
        (16, 3, 48),  # HIGH-1 (D16) repro: 32^24 = 56 -> OOB on 48-thread
        # group; nt is neither a power of two nor <= 32
        (8, 5, 40),  # same window: offset 20, partner 32^20 = 52 -> OOB
    ],
)
def test_f1_rejects_nt_gt_32_non_pow2_scale(logical_width, scale, nt):
    with pytest.raises(
        Exception,
        match=rf"single-simdgroup.*scale={scale}.*"
        rf"nt = extent\*scale = {nt}.*\b32\b",
    ):
        _lower_metal(_make_allreduce_dim0_scale_kernel(T.reduce_sum, logical_width, scale))


@pytest.mark.parametrize(
    ("logical_width", "scale", "nt"),
    [
        # D17 HIGH-1 closure window: nt <= 32 yet NOT a power of two.
        # Masks 12/6/3: partner 16^12 = 28 >= 24 escapes [0, 24) (OOB
        # scratch / never-written slot) even though every mask < 32.
        (8, 3, 24),
        # Masks 6/3: partner 8^6 = 14 >= 12 escapes [0, 12); slots
        # 12..15 exist only in codegen padding and are never written.
        (4, 3, 12),
    ],
)
def test_f1_rejects_nt_le_32_non_pow2_scale(logical_width, scale, nt):
    with pytest.raises(
        Exception,
        match=rf"single-simdgroup.*scale={scale}.*"
        rf"nt = extent\*scale = {nt}.*\b32\b",
    ):
        _lower_metal(_make_allreduce_dim0_scale_kernel(T.reduce_sum, logical_width, scale))


@pytest.mark.parametrize(
    ("logical_width", "scale", "nt"),
    [
        (16, 4, 64),  # offsets 32..4: first offset crosses the boundary
        (32, 2, 64),  # offsets 32..2: first offset crosses the boundary
        (64, 1, 64),  # cross-group extent (D14 family)
        (128, 1, 128),  # cross-group extent (D14 family)
    ],
)
def test_f1_rejects_nt_gt_32_pow2_scale(logical_width, scale, nt):
    with pytest.raises(
        Exception,
        match=rf"single-simdgroup.*scale={scale}.*"
        rf"nt = extent\*scale = {nt}.*\b32\b",
    ):
        _lower_metal(_make_allreduce_dim0_scale_kernel(T.reduce_sum, logical_width, scale))


@pytest.mark.parametrize(
    ("logical_width", "scale", "threads", "nt"),
    [
        # D18 HIGH-1 (Codex review 2c5ae8b7) execution-domain window:
        # nt is a power of two <= 32 (passes the D17 gate) but the
        # threadgroup extent is NOT an integer multiple of nt, so the
        # last incomplete nt-block reads partners >= N (OOB: scratch is
        # sized N) or never-written codegen-padding slots.  All these
        # shapes reach the F1 gate (probe-verified) and were silently
        # ALLOWED before D18.
        # (8,2,24): nt=16, N%16=8; partners 24..31 escape the 24-slot
        #           scratch (Codex compile-only repro: max partners
        #           tid 16..23 ^ 8 = 24..31).
        (8, 2, 24, 16),
        # (2,4,12): nt=8, N%8=4; tid 8..11 ^ 4 = 12..15 read slots that
        #           exist only in codegen padding and are never written.
        (2, 4, 12, 8),
        # (16,2,40): nt=32, N%32=8; task-specified (nt=32, N=40) case.
        (16, 2, 40, 32),
        (8, 4, 40, 32),  # same N=40 misalignment, nt=32
        (4, 4, 20, 16),  # nt=16, N%16=4
        (2, 2, 6, 4),  # nt=4, N%4=2
        (8, 2, 56, 16),  # N=3*nt+8 boundary: tail block incomplete
    ],
)
def test_f1_rejects_misaligned_threadgroup(logical_width, scale, threads, nt):
    """D18: N % nt != 0 -> compile-time rejection with the tail-block
    alignment diagnostic (this is the HIGH-1 closure of the ACTUAL
    [0, N) execution prefix)."""
    with pytest.raises(
        Exception,
        match=rf"single-simdgroup.*scale={scale}.*"
        rf"nt = extent\*scale = {nt}.*N = {threads}.*"
        rf"not an integer multiple of nt = {nt}",
    ):
        _lower_metal(_make_allreduce_dim0_scale_kernel(T.reduce_sum, logical_width, scale, threads))


@pytest.mark.parametrize(
    ("logical_width", "scale", "threads", "nt"),
    [
        # N == nt single-block closures (D17 family, threads unchanged)
        (32, 1, 32, 32),  # v2 boundary: full 32-lane closure, offsets 16..1
        (16, 2, 32, 32),  # power-of-two replication: offsets 16..2
        (8, 4, 32, 32),  # power-of-two replication: offsets 16..4
        (16, 1, 16, 16),  # sub-simdgroup closure: offsets 8..1
        (8, 2, 16, 16),  # boundary (D17): power-of-two nt, offsets 8..2
        (2, 4, 8, 8),  # boundary (D17): power-of-two nt, offset 4
        (4, 8, 32, 32),  # replicated small fragment: offsets 16..8
        # D18 multi-block closures (N = k*nt complete blocks; every
        # block computes its own copy of the same group total, owners
        # store): the [0, N) execution prefix must be closed under every
        # mask — this is the property N % nt == 0 guarantees.
        (8, 2, 32, 16),  # N = 2*nt: two complete 16-lane closures
        (2, 4, 16, 8),  # N = 2*nt
        (2, 4, 32, 8),  # N = 4*nt
        (8, 2, 64, 16),  # N = 4*nt
        (4, 4, 32, 16),  # N = 2*nt
        (16, 2, 96, 32),  # N = 3*nt
        (8, 2, 128, 16),  # N = 8*nt (v2-scale threadgroup extent)
    ],
)
def test_f1_allows_pow2_nt_le_32_closure(logical_width, scale, threads, nt):
    src = _lower_metal(_make_allreduce_dim0_scale_kernel(T.reduce_sum, logical_width, scale, threads))
    # every XOR butterfly offset must be < 32 (closure inside one
    # simdgroup; offsets are nt/2 halving down to scale), the max offset
    # must be nt/2, the threadgroup extent must be an integer multiple
    # of nt (D18: complete nt-blocks in the [0, N) execution prefix),
    # and the raw [0, N) prefix must be closed under every mask in the
    # MSL (D17 closure on the participating range, D18 on the actual
    # execution domain)
    offsets = [int(v) for v in re.findall(r"\^ ?(\d+)", src)]
    assert offsets, "no XOR butterfly offsets found in MSL"
    assert all(o < 32 for o in offsets)
    assert max(offsets) == nt // 2
    assert threads % nt == 0, (
        f"threadgroup extent {threads} is not a multiple of nt={nt}: incomplete tail block would escape the execution prefix"
    )
    assert _xor_closed(offsets, threads), f"XOR masks {sorted(set(offsets))} not closed on [0, {threads})"


if __name__ == "__main__":
    tilelang.testing.main()
