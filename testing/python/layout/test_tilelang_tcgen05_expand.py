# Pin ExpandTcgen05Layout against the CUTLASS Copy_Traits ground truth.
#
# The C++ side stores each tcgen05.ld/st atom as the per-warp, single-issue
# CUTLASS ``Copy_Traits<...1x>`` TV layout and derives every replication --
# warps, the high-datapath duplicate issue of the 16-datapath shapes, .xN
# column repetitions, warpgroups -- by one blocked product over a row-major
# replication grid.  These tests evaluate the resulting fragment at a
# SYMBOLIC (datapath, column) coordinate built from the hand-evaluated
# DstLayout formulas of cute/atom/copy_traits_sm100.hpp -- combined with the
# register order of TileLang's tl::tcgen05_{ld,st}_32dpNNbNx wrappers (the
# duplicate issue's registers land after all repetitions) -- and let the
# arithmetic analyzer prove the (thread, value) result equals the wrapper's
# slot for ALL lanes/registers/repetitions at once.

import pytest

import tilelang.testing
from tilelang import _ffi_api
from tilelang import tvm
from tilelang.layout import cute


def _dp_col(meta, t, j, n, d, w):
    """CUTLASS DstLayout: lane t, per-repetition register j, repetition n,
    duplicate issue d, warp w -> (datapath, b32 column)."""
    if meta == "32dp32b":
        return t + 32 * w, n
    if meta == "16dp64b":
        return 8 * (t % 2) + t // 4 + 16 * d + 32 * w, (t // 2) % 2 + 2 * n
    if meta == "16dp128b":
        return t // 4 + 8 * j + 16 * d + 32 * w, t % 4 + 4 * n
    if meta == "16dp256b":
        return t // 4 + 8 * (j // 2) + 16 * d + 32 * w, 2 * (t % 4) + j % 2 + 8 * n
    raise KeyError(meta)


# (meta suffix, columns of one repetition, registers per repetition,
#  duplicate issues per warp)
ATOM_SPECS = [
    ("32dp32b", 1, 1, 1),
    ("16dp64b", 2, 1, 2),
    ("16dp128b", 4, 2, 2),
    ("16dp256b", 8, 4, 2),
]


def _get_meta(name):
    return getattr(_ffi_api, "get_tcgen05_meta_" + name)()


def _column_major_tile(rows, cols):
    return cute.make_layout((rows, cols), stride=(cute.E(0), cute.E(1)))


@pytest.mark.parametrize(("meta", "width", "regs", "ndup"), ATOM_SPECS, ids=[s[0] for s in ATOM_SPECS])
@pytest.mark.parametrize("num_wgs", [1, 2], ids=["1wg", "2wg"])
@pytest.mark.parametrize("chunks", [1, 2, 4], ids=["x1", "x2", "x4"])
def test_expand_matches_cutlass_dst_layout(meta, width, regs, ndup, num_wgs, chunks):
    num_threads = 128 * num_wgs
    cols = width * chunks * num_wgs
    plan = _ffi_api.ExpandTcgen05Layout(_get_meta("ld_" + meta), _column_major_tile(128, cols), num_threads)
    assert plan is not None, f"{meta} must expand a (128,{cols}) tile"
    assert plan.num_chunks_each_wg == chunks
    assert plan.num_issues == 1
    assert plan.vals_per_issue == regs * chunks * ndup

    # Bind one symbolic variable per replication axis and prove the fragment
    # inverts the DstLayout formulas for every assignment simultaneously.
    ana = tvm.arith.Analyzer()

    def var(name, extent):
        v = tvm.tirx.Var(name, "int32")
        ana.bind(v, tvm.ir.Range(0, extent))
        return v

    t = var("t", 32)  # lane within the warp
    j = var("j", regs)  # register within one repetition
    n = var("n", chunks)  # .xN repetition
    d = var("d", ndup)  # duplicate high-datapath issue
    w = var("w", 4)  # warp within the warpgroup
    g = var("g", num_wgs)  # warpgroup

    dp, col = _dp_col(meta, t, j, n, d, w)
    col = col + g * chunks * width  # each warpgroup owns a column slice

    # Wrapper slot: threads are linear in (lane, warp, warpgroup); each
    # thread's registers run per-repetition regs fastest, then repetitions,
    # then the duplicate issue's registers after all repetitions.
    thread_expect = t + 32 * (w + 4 * g)
    value_expect = j + regs * (n + chunks * d)

    # Normalize the (thread@0, value@1) coordinate to rank 2 by adding the
    # zero ArithmeticTuple, exactly as FragmentToTileLang does.
    thread_got, value_got = cute.to_python(cute.from_python(plan.fragment((dp, col))) + (0, 0))
    assert ana.can_prove_equal(thread_got, thread_expect), (
        f"{meta} wgs={num_wgs} x{chunks}: fragment thread {thread_got} != DstLayout thread {thread_expect}"
    )
    assert ana.can_prove_equal(value_got, value_expect), (
        f"{meta} wgs={num_wgs} x{chunks}: fragment value {value_got} != DstLayout value {value_expect}"
    )


def test_expand_16dp_requires_pow2_single_issue():
    # The duplicate issue's registers append after ALL repetitions, so a
    # multi-issue (non-power-of-two) chunk count would interleave wrongly;
    # Expand must refuse rather than emit a wrong layout.
    tile = _column_major_tile(128, 6)
    assert _ffi_api.ExpandTcgen05Layout(_get_meta("ld_16dp64b"), tile, 128) is None
    # 32dp32b chains exactly, so the same shape is fine there.
    assert _ffi_api.ExpandTcgen05Layout(_get_meta("ld_32dp32b"), tile, 128) is not None


def test_expand_16dp_respects_max_chunks():
    # One .xN issue caps at 128 b32 columns of registers: 16dp256b (width 8)
    # allows at most 32 repetitions.
    meta = _get_meta("ld_16dp256b")
    assert _ffi_api.ExpandTcgen05Layout(meta, _column_major_tile(128, 256), 128) is not None
    assert _ffi_api.ExpandTcgen05Layout(meta, _column_major_tile(128, 512), 128) is None


def test_expand_gapped_tile_issues_per_batch():
    # A column slice of a batched accumulator leaves per-batch serialized
    # gaps: one tcgen05 issue per batch entry (rest iteration), later issues
    # appending registers.
    tile = cute.Layout.parse("(3,128,64):(128@1,1@0,1@1)")
    plan = _ffi_api.ExpandTcgen05Layout(_get_meta("st_32dp32b"), tile, 128)
    assert plan is not None
    assert str(plan.fragment) == "(3,128,64):(64@1,1@0,1@1)"
    assert (plan.num_chunks_each_wg, plan.num_issues, plan.vals_per_issue) == (64, 3, 64)
    assert [plan.rest_domain(i) for i in range(3)] == [0, 1, 2]


if __name__ == "__main__":
    tilelang.testing.main()
