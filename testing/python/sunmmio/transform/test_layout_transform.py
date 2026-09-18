"""Tests for the Sunmmio DRAM<->RSRAM layout-transform lowering.

A copy between a row-major DRAM buffer and a ZZ RSRAM buffer is split, in
CopyNode::LowerSunmmioDmaCopy, into a plain tl.dma_copy through a staging
RSRAM buffer plus a tl.sunmmio_layout_transform that re-blocks the data.  A
copy whose src/dst layouts already match stays a plain tl.dma_copy.

The kernels declare their DRAM tensors the way examples/gemm/
sunmmio_example_gemm.py does — T.MeshTensor on the Sunmmio device mesh,
carrying an explicit layout.
"""

import pytest
import tilelang
import tilelang.testing
import tilelang as tl
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.utils.target import determine_target
from tilelang.layout import (
    is_same_layout,
    make_aligned_row_major,
    make_mx_row_major_layout,
    make_mxznz_layout,
    make_mxzz_layout,
    make_row_major,
    make_zz_layout,
)
from tvm import tir
from tvm.tir import Block
from tvm.tir.stmt_functor import post_order_visit

tilelang.env.disable_cache()

M, N = 128, 128
DTYPE = T.float16
MX_DTYPE = T.mxfp8


def apply_passes_through_lower_tile_op(mod, target):
    """Apply the Sunmmio pass pipeline up to and including LowerTileOp."""
    mod = tvm.tir.transform.BindTarget(target)(mod)
    mod = tl.transform.ResolveSunmmioMeshSymbols()(mod)
    mod = tl.transform.AddWrapperForSingleBufStore()(mod)
    mod = tl.transform.LegalizeNegativeIndex()(mod)
    mod = tl.transform.InjectAssumes()(mod)
    mod = tl.transform.Simplify()(mod)
    mod = tl.transform.InferSramScope()(mod)
    mod = tl.transform.LegalizeSunmmioDataPath()(mod)
    mod = tl.transform.LayoutReducer()(mod)
    mod = tl.transform.SunmmioLayoutInference()(mod)
    mod = tl.transform.LowerTileOp()(mod)
    return mod


def collect_call_names(func):
    """Return the list of intrinsic call op-names in the function body."""
    names = []

    def visit(node):
        if isinstance(node, tir.Call) and hasattr(node.op, "name"):
            names.append(node.op.name)

    post_order_visit(func.body, visit)
    return names


def collect_alloc_buffers(func):
    """Return all buffers declared in Block alloc_buffers."""
    bufs = []

    def visit(node):
        if isinstance(node, Block):
            bufs.extend(node.alloc_buffers)

    post_order_visit(func.body, visit)
    return bufs


def staging_buffers(func):
    return [b for b in collect_alloc_buffers(func) if b.name.endswith("_layout_stage")]


def transform_leg_region_ranks(func):
    """Return [(rankA, rankB), ...], the rank of each region argument of every
    sunmmio_layout_transform call.  A region is T.region(buf[...], mask, *exts),
    so its rank is len(args) - 2."""
    legs = []

    def visit(node):
        if isinstance(node, tir.Call) and hasattr(node.op, "name") and node.op.name == "tl.sunmmio_layout_transform":
            legs.append((len(node.args[0].args) - 2, len(node.args[1].args) - 2))

    post_order_visit(func.body, visit)
    return legs


def extract_layout_map(func):
    """Return the layout_map from the function's block annotations."""
    layout_map = {}

    def visit(node):
        nonlocal layout_map
        if isinstance(node, Block) and "layout_map" in node.annotations:
            layout_map = node.annotations["layout_map"]

    post_order_visit(func.body, visit)
    return layout_map


# ---------------------------------------------------------------------------
# Kernels — declared like examples/gemm/sunmmio_example_gemm.py
# ---------------------------------------------------------------------------


def _dram(shape, layout):
    """A DRAM tensor on the Sunmmio device mesh, carrying an explicit layout —
    the T.MeshTensor declaration style of sunmmio_example_gemm.py.

    A fully-replicated policy keeps each core's buffer at the declared shape:
    these are single-core layout tests (unlike the sharded GEMM example), so
    sharding must not perturb the shapes the assertions rely on.
    """
    policy = T.placement.replicated()
    return T.MeshTensor(shape, policy, DTYPE, layout=layout)


def _dram_typed(shape, dtype, layout):
    policy = T.placement.replicated()
    return T.MeshTensor(shape, policy, dtype, layout=layout)


def mismatched_copy_kernel():
    """Row-major DRAM <-> ZZ RSRAM (default), both directions — combos 1 & 2."""

    @T.prim_func
    def main(
        A: _dram((M, N), make_row_major((M, N))),
        B: _dram((M, N), make_row_major((M, N))),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((M, N), DTYPE, scope="shared.rsram")
            T.copy(A, A_shared)  # DRAM(row-major) -> RSRAM(ZZ default)
            T.copy(A_shared, B)  # RSRAM(ZZ) -> DRAM(row-major)

    return tvm.IRModule({"main": main})


def mxzz_dram_to_mx_row_major_rsram_kernel():
    @T.prim_func
    def main(
        A: _dram_typed(
            (64, 64),
            MX_DTYPE,
            make_mxzz_layout((64, 64), dtype=MX_DTYPE),
        ),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((64, 64), MX_DTYPE, scope="shared.rsram")
            T.annotate_layout({A_shared: make_mx_row_major_layout((64, 64), dtype=MX_DTYPE)})
            T.copy(A, A_shared)

    return tvm.IRModule({"main": main})


def mxznz_dram_to_mx_row_major_rsram_kernel():
    @T.prim_func
    def main(
        A: _dram_typed(
            (64, 64),
            MX_DTYPE,
            make_mxznz_layout((64, 64), dtype=MX_DTYPE),
        ),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((64, 64), MX_DTYPE, scope="shared.rsram")
            T.annotate_layout({A_shared: make_mx_row_major_layout((64, 64), dtype=MX_DTYPE)})
            T.copy(A, A_shared)

    return tvm.IRModule({"main": main})


def zz_dram_kernel():
    """ZZ DRAM <-> row-major RSRAM, both directions — combos 3 & 4."""

    @T.prim_func
    def main(
        A: _dram((M, N), make_zz_layout((M, N), [0, 1], (32, 32))),
        B: _dram((M, N), make_zz_layout((M, N), [0, 1], (32, 32))),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((M, N), DTYPE, scope="shared.rsram")
            T.annotate_layout({A_shared: make_row_major((M, N))})
            T.copy(A, A_shared)  # DRAM(ZZ) -> RSRAM(row-major)
            T.copy(A_shared, B)  # RSRAM(row-major) -> DRAM(ZZ)

    return tvm.IRModule({"main": main})


def matched_copy_kernel():
    """Row-major DRAM and row-major RSRAM (annotated) — layouts match, so the
    copy stays a plain dma_copy."""

    @T.prim_func
    def main(A: _dram((M, N), make_row_major((M, N)))):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((M, N), DTYPE, scope="shared.rsram")
            T.annotate_layout({A_shared: make_row_major((M, N))})
            T.copy(A, A_shared)  # DRAM(row-major) -> RSRAM(row-major), matched

    return tvm.IRModule({"main": main})


def zz_dram_reduce_kernel():
    """End-to-end: ZZ DRAM (3x256x256) -> RSRAM, axis-2 reduction -> row-major
    RSRAM (3x256), copied back to ZZ DRAM (3x256).

    Out_sh needs no annotation: the reduction is over axis 2 — a blocked dim
    of the ZZ layout — so ReduceOp::InferLayout assigns Out_sh a row-major
    layout, making the final store a ZZ <- row-major transform.  The matched
    ZZ->ZZ load stays a plain dma_copy.
    """
    b, h, w = 3, 256, 256

    @T.prim_func
    def main(
        A: _dram((b, h, w), make_zz_layout((b, h, w), [1, 2], (32, 32))),
        Out: _dram((b, h), make_zz_layout((b, h), [0, 1], (32, 32))),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_sh = T.alloc_shared((b, h, w), DTYPE, scope="shared.rsram")
            Out_sh = T.alloc_shared((b, h), DTYPE, scope="shared.rsram")
            T.copy(A, A_sh)  # ZZ DRAM -> ZZ RSRAM (matched)
            T.reduce_sum(A_sh, Out_sh, dim=2)  # axis-2 reduce -> row-major Out_sh
            T.copy(Out_sh, Out)  # row-major RSRAM -> ZZ DRAM (transform)

    return tvm.IRModule({"main": main})


def padded_rsram_copy_kernel(rows, cols):
    """Unpadded row-major DRAM <-> alignment-padded row-major RSRAM, both
    directions.  The leading-dimension pitch mismatch (e.g. cols 40 -> 64)
    forces a pad/unpad sunmmio_layout_transform rather than a plain dma_copy."""

    @T.prim_func
    def main(
        A: _dram((rows, cols), make_row_major((rows, cols))),
        B: _dram((rows, cols), make_row_major((rows, cols))),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            X = T.alloc_shared((rows, cols), DTYPE, scope="shared.rsram")
            T.annotate_layout({X: make_aligned_row_major((rows, cols), "float16", 64)})
            T.copy(A, X)  # DRAM(unpadded) -> RSRAM(padded): PAD
            T.copy(X, B)  # RSRAM(padded) -> DRAM(unpadded): UNPAD

    return tvm.IRModule({"main": main})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_mismatched_copy_splits_into_dma_and_transform():
    """A row-major DRAM <-> ZZ RSRAM copy is split via an RSRAM staging buffer."""
    target = determine_target("Sunmmio", return_object=True)
    mod = mismatched_copy_kernel()
    with tvm.target.Target(target):
        mod = apply_passes_through_lower_tile_op(mod, target)
        func = list(mod.functions.values())[0]

        names = collect_call_names(func)
        # One dma_copy + one transform per copy, both directions.
        assert names.count("tl.dma_copy") == 2, names
        assert names.count("tl.sunmmio_layout_transform") == 2, names

        # One RSRAM staging buffer per split copy.
        stages = staging_buffers(func)
        assert len(stages) == 2, [b.name for b in stages]
        for b in stages:
            assert b.scope() == "shared.rsram", b.scope()


def test_zz_dram_splits_with_zz_staging():
    """ZZ DRAM <-> row-major RSRAM (combos 3 & 4): splits, and the staging
    buffer is registered with the ZZ layout so its dma_copy leg is matched."""
    target = determine_target("Sunmmio", return_object=True)
    mod = zz_dram_kernel()
    with tvm.target.Target(target):
        mod = apply_passes_through_lower_tile_op(mod, target)
        func = list(mod.functions.values())[0]

        names = collect_call_names(func)
        assert names.count("tl.dma_copy") == 2, names
        assert names.count("tl.sunmmio_layout_transform") == 2, names

        stages = staging_buffers(func)
        assert len(stages) == 2, [b.name for b in stages]

        # The staging buffers must be registered in layout_map (the
        # RegisterLayout callback) — that is what makes the dma_copy leg a
        # layout-matched ZZ->ZZ transfer.
        layout_map = extract_layout_map(func)
        registered = {b.name for b in layout_map if b.name.endswith("_layout_stage")}
        assert len(registered) == 2, registered


def test_mxzz_dram_splits_with_mxzz_staging():
    """MX layout transform is only valid for MXZZ <-> MX row-major.  The staging
    buffer must preserve the DRAM MXZZ layout kind instead of becoming row-major."""
    target = determine_target("Sunmmio", return_object=True)
    mod = mxzz_dram_to_mx_row_major_rsram_kernel()
    with tvm.target.Target(target):
        mod = apply_passes_through_lower_tile_op(mod, target)
        func = list(mod.functions.values())[0]

        names = collect_call_names(func)
        assert names.count("tl.dma_copy") == 1, names
        assert names.count("tl.sunmmio_layout_transform") == 1, names

        stages = staging_buffers(func)
        assert len(stages) == 1, [b.name for b in stages]

        layout_map = extract_layout_map(func)
        stage_layout = layout_map[stages[0]]
        assert is_same_layout(stage_layout, make_mxzz_layout((64, 64), dtype=MX_DTYPE))


def test_mxznz_dram_to_mx_row_major_rsram_is_rejected():
    """SUVM transform_layout currently supports only MXZZ <-> MX row-major."""
    target = determine_target("Sunmmio", return_object=True)
    mod = mxznz_dram_to_mx_row_major_rsram_kernel()
    with tvm.target.Target(target), pytest.raises(Exception, match="MX row-major <-> MXZZ"):
        apply_passes_through_lower_tile_op(mod, target)


def test_matched_layout_stays_plain_dma_copy():
    """A copy whose src/dst layouts match keeps a bare dma_copy — no transform."""
    target = determine_target("Sunmmio", return_object=True)
    mod = matched_copy_kernel()
    with tvm.target.Target(target):
        mod = apply_passes_through_lower_tile_op(mod, target)
        func = list(mod.functions.values())[0]

        names = collect_call_names(func)
        assert names.count("tl.dma_copy") == 1, names
        assert "tl.sunmmio_layout_transform" not in names, names
        assert staging_buffers(func) == [], [b.name for b in staging_buffers(func)]


def test_padded_rsram_copy_splits_into_pad_unpad():
    """A padded RSRAM <-> unpadded DRAM copy splits into dma + transform in BOTH
    directions: the store leg must not fast-path despite IsLayoutMatch ignoring
    the leading-dimension padding."""
    target = determine_target("Sunmmio", return_object=True)
    mod = padded_rsram_copy_kernel(2, 40)  # 40 -> padded 64; leading 2 <= 32
    with tvm.target.Target(target):
        mod = apply_passes_through_lower_tile_op(mod, target)
        func = list(mod.functions.values())[0]

        names = collect_call_names(func)
        # One dma_copy + one transform per copy (PAD on load, UNPAD on store).
        assert names.count("tl.dma_copy") == 2, names
        assert names.count("tl.sunmmio_layout_transform") == 2, names
        stages = staging_buffers(func)
        assert len(stages) == 2, [b.name for b in stages]


def test_padded_rsram_copy_rejects_leading_dim_over_32():
    """pad/unpad supports only a leading dimension (product of all dims except
    the last) <= 32; a larger one must fail loudly."""
    target = determine_target("Sunmmio", return_object=True)
    mod = padded_rsram_copy_kernel(40, 40)  # leading 40 > 32
    with tvm.target.Target(target), pytest.raises(Exception, match="leading dimension"):
        apply_passes_through_lower_tile_op(mod, target)


def test_padded_rsram_copy_leading_dim_32_is_allowed():
    """Leading dim exactly 32 is at the limit and allowed (boundary)."""
    target = determine_target("Sunmmio", return_object=True)
    mod = padded_rsram_copy_kernel(32, 40)  # leading 32 == limit
    with tvm.target.Target(target):
        mod = apply_passes_through_lower_tile_op(mod, target)
        func = list(mod.functions.values())[0]
        assert collect_call_names(func).count("tl.sunmmio_layout_transform") == 2


def test_padded_rsram_copy_leading_dim_33_rejected():
    """Leading dim 33 is one past the limit and rejected (boundary)."""
    target = determine_target("Sunmmio", return_object=True)
    mod = padded_rsram_copy_kernel(33, 40)  # leading 33 > limit
    with tvm.target.Target(target), pytest.raises(Exception, match="leading dimension"):
        apply_passes_through_lower_tile_op(mod, target)


def padded_rsram_copy_kernel_3d(d0, d1, cols):
    @T.prim_func
    def main(
        A: _dram((d0, d1, cols), make_row_major((d0, d1, cols))),
        B: _dram((d0, d1, cols), make_row_major((d0, d1, cols))),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            X = T.alloc_shared((d0, d1, cols), DTYPE, scope="shared.rsram")
            T.annotate_layout({X: make_aligned_row_major((d0, d1, cols), "float16", 64)})
            T.copy(A, X)
            T.copy(X, B)

    return tvm.IRModule({"main": main})


def test_padded_rsram_copy_rank3_splits():
    """Rank-3 padded copy: the 32-limit is on the product of all dims except the
    last (2*3 = 6 <= 32), and the inner 40 -> 64 padding still drives a split."""
    target = determine_target("Sunmmio", return_object=True)
    mod = padded_rsram_copy_kernel_3d(2, 3, 40)
    with tvm.target.Target(target):
        mod = apply_passes_through_lower_tile_op(mod, target)
        func = list(mod.functions.values())[0]
        assert collect_call_names(func).count("tl.sunmmio_layout_transform") == 2


def unit_dim_source_kernel():
    """4D DRAM sliced with unit dims (A[0,:,0,:], logical [2,64]) into a 2D ZZ
    RSRAM buffer.  The copy frontend's shape check sees the squeezed [2,64], but
    the lowered DRAM source region keeps its full 4D extents [1,2,1,64] (a
    region's rank equals its buffer's).  The staging must follow the RSRAM
    region [2,64] so the transform leg is rank-matched; the DMA leg carries the
    degenerate dims and folds them away."""

    @T.prim_func
    def main(A: _dram((2, 2, 2, 64), make_row_major((2, 2, 2, 64)))):
        with T.Kernel(1, threads=128) as (bx,):
            S = T.alloc_shared((2, 64), DTYPE, scope="shared.rsram")
            T.annotate_layout({S: make_zz_layout((2, 64), [0, 1], (32, 32))})
            T.copy(A[0, :, 0, :], S)  # DRAM region [1,2,1,64] -> ZZ RSRAM [2,64]

    return tvm.IRModule({"main": main})


def test_unit_dim_source_staging_follows_rsram_shape():
    """A unit-dim 4D DRAM source: the staging follows the RSRAM region [2,64],
    not the 4D DRAM region, so the transform leg is rank-matched (2D <-> 2D)."""
    target = determine_target("Sunmmio", return_object=True)
    mod = unit_dim_source_kernel()
    with tvm.target.Target(target):
        mod = apply_passes_through_lower_tile_op(mod, target)
        func = list(mod.functions.values())[0]

        names = collect_call_names(func)
        assert names.count("tl.sunmmio_layout_transform") == 1, names

        # Staging follows the RSRAM region [2,64], not the 4D DRAM region.
        stages = staging_buffers(func)
        assert len(stages) == 1, [b.name for b in stages]
        assert [int(x) for x in stages[0].shape] == [2, 64], stages[0].shape

        # The transform leg pairs the [2,64] staging with the [2,64] RSRAM
        # buffer: both rank 2.  (Before the fix the staging was [1,2,1,64], so
        # the transform leg was a broken 4D <-> 2D pairing.)
        assert transform_leg_region_ranks(func) == [(2, 2)], transform_leg_region_ranks(func)


def unit_dim_dest_kernel():
    """Store direction: ZZ RSRAM -> row-major DRAM 4D dest sliced with unit dims
    (B[0,:,0,:]).  The matched ZZ->ZZ load stays a plain dma_copy; only the store
    splits, and its staging follows the RSRAM region [2,64], not the 4D dest."""

    @T.prim_func
    def main(
        A: _dram((2, 64), make_zz_layout((2, 64), [0, 1], (32, 32))),
        B: _dram((2, 2, 2, 64), make_row_major((2, 2, 2, 64))),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            X = T.alloc_shared((2, 64), DTYPE, scope="shared.rsram")
            T.annotate_layout({X: make_zz_layout((2, 64), [0, 1], (32, 32))})
            T.copy(A, X)  # ZZ DRAM -> ZZ RSRAM: matched, plain dma
            T.copy(X, B[0, :, 0, :])  # ZZ RSRAM -> rm DRAM 4D unit-dim: store split

    return tvm.IRModule({"main": main})


def test_unit_dim_dest_staging_follows_rsram_shape():
    """Store direction with a unit-dim 4D DRAM dest: the staging follows the
    RSRAM region [2,64] and the transform leg is rank-matched (2D <-> 2D)."""
    target = determine_target("Sunmmio", return_object=True)
    mod = unit_dim_dest_kernel()
    with tvm.target.Target(target):
        mod = apply_passes_through_lower_tile_op(mod, target)
        func = list(mod.functions.values())[0]

        names = collect_call_names(func)
        # Matched load (plain dma) + store split (dma + transform).
        assert names.count("tl.dma_copy") == 2, names
        assert names.count("tl.sunmmio_layout_transform") == 1, names

        stages = staging_buffers(func)
        assert len(stages) == 1, [b.name for b in stages]
        assert [int(x) for x in stages[0].shape] == [2, 64], stages[0].shape
        assert transform_leg_region_ranks(func) == [(2, 2)], transform_leg_region_ranks(func)


def unit_dim_source_pad_kernel():
    """Pad/unpad path with a unit-dim source: 4D DRAM slice [1,2,1,40] -> padded
    (aligned) RSRAM [2,40].  Exercises the 32-limit branch with unit dims (the
    leading product 1*2*1 = 2 stays <= 32)."""

    @T.prim_func
    def main(A: _dram((2, 2, 2, 40), make_row_major((2, 2, 2, 40)))):
        with T.Kernel(1, threads=128) as (bx,):
            X = T.alloc_shared((2, 40), DTYPE, scope="shared.rsram")
            T.annotate_layout({X: make_aligned_row_major((2, 40), "float16", 64)})
            T.copy(A[0, :, 0, :], X)  # DRAM [1,2,1,40] -> padded RSRAM [2,40]

    return tvm.IRModule({"main": main})


def test_unit_dim_source_pad_unpad_path():
    """Unit-dim source on the pad/unpad path: staging follows the RSRAM region
    [2,40] and the pad transform leg is rank-matched."""
    target = determine_target("Sunmmio", return_object=True)
    mod = unit_dim_source_pad_kernel()
    with tvm.target.Target(target):
        mod = apply_passes_through_lower_tile_op(mod, target)
        func = list(mod.functions.values())[0]

        assert collect_call_names(func).count("tl.sunmmio_layout_transform") == 1
        stages = staging_buffers(func)
        assert len(stages) == 1, [b.name for b in stages]
        assert [int(x) for x in stages[0].shape] == [2, 40], stages[0].shape
        assert transform_leg_region_ranks(func) == [(2, 2)], transform_leg_region_ranks(func)


def leading_unit_dim_3d_kernel():
    """Unit dim in the leading position of a 3D source: A[0,:,:] -> [1,2,64]
    region into a ZZ RSRAM [2,64].  Varies the squeezed-dim position/rank."""

    @T.prim_func
    def main(A: _dram((2, 2, 64), make_row_major((2, 2, 64)))):
        with T.Kernel(1, threads=128) as (bx,):
            S = T.alloc_shared((2, 64), DTYPE, scope="shared.rsram")
            T.annotate_layout({S: make_zz_layout((2, 64), [0, 1], (32, 32))})
            T.copy(A[0, :, :], S)  # DRAM [1,2,64] -> ZZ RSRAM [2,64]

    return tvm.IRModule({"main": main})


def test_leading_unit_dim_3d_source():
    """A 3D source with a leading unit dim: staging follows the RSRAM region
    [2,64] and the transform leg is rank-matched."""
    target = determine_target("Sunmmio", return_object=True)
    mod = leading_unit_dim_3d_kernel()
    with tvm.target.Target(target):
        mod = apply_passes_through_lower_tile_op(mod, target)
        func = list(mod.functions.values())[0]

        assert collect_call_names(func).count("tl.sunmmio_layout_transform") == 1
        stages = staging_buffers(func)
        assert len(stages) == 1, [b.name for b in stages]
        assert [int(x) for x in stages[0].shape] == [2, 64], stages[0].shape
        assert transform_leg_region_ranks(func) == [(2, 2)], transform_leg_region_ranks(func)


def test_matched_rank_staging_equals_rsram_region():
    """Backward-compat: with no unit dims the RSRAM region equals the DRAM
    region, so anchoring the staging on the RSRAM side leaves the staging shape
    (and every transform leg) unchanged."""
    target = determine_target("Sunmmio", return_object=True)
    mod = mismatched_copy_kernel()  # (M,N) row-major DRAM <-> ZZ RSRAM, both dirs
    with tvm.target.Target(target):
        mod = apply_passes_through_lower_tile_op(mod, target)
        func = list(mod.functions.values())[0]

        stages = staging_buffers(func)
        assert len(stages) == 2, [b.name for b in stages]
        for b in stages:
            assert [int(x) for x in b.shape] == [M, N], b.shape
        assert transform_leg_region_ranks(func) == [(2, 2), (2, 2)], transform_leg_region_ranks(func)


def test_aligned_rsram_copy_already_32_multiple_stays_plain():
    """When the RSRAM 'aligned' layout is a no-op (inner already a 32-multiple)
    it matches the unpadded DRAM, so the copy stays a plain dma_copy."""
    target = determine_target("Sunmmio", return_object=True)

    @T.prim_func
    def main(
        A: _dram((2, 64), make_row_major((2, 64))),
        B: _dram((2, 64), make_row_major((2, 64))),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            X = T.alloc_shared((2, 64), DTYPE, scope="shared.rsram")
            T.annotate_layout({X: make_aligned_row_major((2, 64), "float16", 64)})
            T.copy(A, X)
            T.copy(X, B)

    with tvm.target.Target(target):
        mod = apply_passes_through_lower_tile_op(tvm.IRModule({"main": main}), target)
        func = list(mod.functions.values())[0]
        names = collect_call_names(func)
        assert "tl.sunmmio_layout_transform" not in names, names
        assert names.count("tl.dma_copy") == 2, names


def test_dma_transform_dependency_inserts_unit_sync():
    target = determine_target("Sunmmio", return_object=True)
    mod = mismatched_copy_kernel()
    with tvm.target.Target(target):
        mod = apply_passes_through_lower_tile_op(mod, target)
        # InjectSunmmioSync only processes global device functions; that attr is
        # set by the host/device split later in the real pipeline, so set it
        # here to exercise the pass directly on the layout-transform op.
        func = mod["main"].with_attr("tir.is_global_func", True)
        mod = tvm.IRModule({"main": func})
        mod = tl.transform.InjectSunmmioSync()(mod)
        func = list(mod.functions.values())[0]

        names = collect_call_names(func)
        assert "tl.dma_copy" in names, names
        assert "tl.sunmmio_layout_transform" in names, names
        assert "tl.sunmmio_sync" in names, names
        assert not any("token" in name for name in names), names

        script = func.script()
        assert script.index("T.dma_copy") < script.index("T.sunmmio_sync") < script.index("T.sunmmio_layout_transform")


def test_end_to_end_zz_dram_reduce_roundtrip():
    """ZZ DRAM -> RSRAM -> axis-2 reduce -> row-major RSRAM -> ZZ DRAM.

    The matched ZZ->ZZ load stays a plain dma_copy; only the row-major -> ZZ
    store is split into a sunmmio_layout_transform + dma_copy.
    """
    target = determine_target("Sunmmio", return_object=True)
    mod = zz_dram_reduce_kernel()
    with tvm.target.Target(target):
        mod = apply_passes_through_lower_tile_op(mod, target)
        func = list(mod.functions.values())[0]
        print(func)

        names = collect_call_names(func)
        # ZZ->ZZ load (plain) + the transform's dma leg = 2 dma_copy.
        assert names.count("tl.dma_copy") == 2, names
        # Only the row-major -> ZZ store needs a transform.
        assert names.count("tl.sunmmio_layout_transform") == 1, names
        # The axis-2 reduction lowered to the vector-core in-tile reduce.
        assert names.count("tl.vector_core_in_tile_reduce") == 1, names

        # One ZZ staging buffer (mirrors the ZZ DRAM output), registered in
        # layout_map by the RegisterLayout callback.
        stages = staging_buffers(func)
        assert len(stages) == 1, [b.name for b in stages]
        assert stages[0].scope() == "shared.rsram", stages[0].scope()
        layout_map = extract_layout_map(func)
        registered = {b.name for b in layout_map if b.name.endswith("_layout_stage")}
        assert registered == {stages[0].name}, registered


if __name__ == "__main__":
    tilelang.testing.main()
