"""Regression tests for GitHub issue #2537.

The warp-partition search (``ComputeDefaultWarpPartition`` /
``ComputeWgmmaWarpPartition``, ``src/cuda/op/gemm.cc``) accepted candidates
without checking that the warps *cover* the M/N extents.  Depending on the
shape this either silently left trailing row/column bands unwritten
(``warp_col_tiles = N // n_warp`` floors) or tripped internal
ICHECK/AssertionError where a clean diagnostic belongs.  The fix restricts
the search to covering partitions and raises a clear front-end error when
none exists.
"""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.language import GemmWarpPolicy

# Front-end error emitted by the fixed partition search.
_REJECT = "No valid warp partition"

# Tests whose docstring reasons about the *default MMA* search must pin the
# backend: on sm_90 `SelectInst` silently reroutes any M >= 64 / warps % 4 == 0
# shape to the WGMMA search, which is a different code path with different
# constraints (and, for N = 104, a pre-existing descriptor-layer limitation
# unrelated to the partition search).  Shapes with M < 64 or an odd warp count
# can never take the WGMMA route and need no pin.
_FORCE_MMA = {"tl.disable_wgmma": True}


def _matmul(M, N, K, threads, policy):
    """Single-block GEMM with an M x N accumulator fragment."""

    @T.prim_func
    def main(A: T.Tensor((M, K), "float16"), B: T.Tensor((K, N), "float16"), C: T.Tensor((M, N), "float32")):
        with T.Kernel(1, 1, threads=threads):
            A_shared = T.alloc_shared((M, K), "float16")
            B_shared = T.alloc_shared((K, N), "float16")
            C_frag = T.alloc_fragment((M, N), "float32")
            T.clear(C_frag)
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.gemm(A_shared, B_shared, C_frag, policy=policy)
            T.copy(C_frag, C)

    return main


def _check(M, N, K, threads, policy, pass_configs=None):
    kernel = tilelang.compile(_matmul(M, N, K, threads, policy), out_idx=[2], pass_configs=pass_configs)
    torch.manual_seed(0)
    A = torch.randint(0, 3, (M, K), dtype=torch.float16, device="cuda")
    B = torch.randint(0, 3, (K, N), dtype=torch.float16, device="cuda")
    ref = A.float() @ B.float()
    out = kernel(A, B).float()
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Silent wrong-code faces: uncovered partitions must now be rejected.
# ---------------------------------------------------------------------------


@tilelang.testing.requires_cuda
def test_square_uncovered_n_rejected():
    """Issue #2537 repro: Square, N=104, 12 warps picked n_warp=12 and left
    columns 96..103 unwritten (12 * (104 // 12) = 96).  Must reject now."""
    with pytest.raises(Exception, match=_REJECT):
        tilelang.compile(_matmul(16, 104, 32, 384, GemmWarpPolicy.Square), out_idx=[2])


@tilelang.testing.requires_cuda
def test_square_uncovered_m_rejected():
    """M-axis sibling: Square, M=288, 17 warps picked m_warp=17 and left rows
    272..287 unwritten (17 * (288 // 17) = 272).  Must reject now."""
    with pytest.raises(Exception, match=_REJECT):
        tilelang.compile(_matmul(288, 8, 32, 544, GemmWarpPolicy.Square), out_idx=[2])


@tilelang.testing.requires_cuda
def test_fullrow_uncovered_n_rejected():
    """FullRow, M=16 forces m_warp=1, n_warp=12 does not cover N=104."""
    with pytest.raises(Exception, match=_REJECT):
        tilelang.compile(_matmul(16, 104, 32, 384, GemmWarpPolicy.FullRow), out_idx=[2])


@tilelang.testing.requires_cuda
def test_fullcol_uncovered_m_rejected():
    """FullCol, N=8 falls back to n_warp=1, m_warp=20 does not cover M=336.

    Pinned to the MMA search: on sm_90 this shape would route to the WGMMA
    search instead (covered by test_wgmma_fullcol_uncovered_m_rejected)."""
    with pytest.raises(Exception, match=_REJECT):
        tilelang.compile(_matmul(336, 8, 32, 640, GemmWarpPolicy.FullCol), out_idx=[2], pass_configs=_FORCE_MMA)


# ---------------------------------------------------------------------------
# Covering shapes that still compute: ground-truth checks.
# ---------------------------------------------------------------------------


@tilelang.testing.requires_cuda
def test_square_covering_warps_numeric():
    """13 warps cover N=104 (13 * 8 = 104): the shape itself is computable."""
    _check(16, 104, 32, 416, GemmWarpPolicy.Square)


@tilelang.testing.requires_cuda
def test_square_divisible_control_numeric():
    """Control: N=96 divides evenly across 12 warps; unchanged by the fix."""
    _check(16, 96, 32, 384, GemmWarpPolicy.Square)


@tilelang.testing.requires_cuda
def test_fullcol_fallback_covering_numeric():
    """FullCol, N=104, 12 warps used to die on an internal ICHECK
    (n_warp = 104 / 8 = 13 does not divide 12).  The fixed search falls back
    to (m_warp=12, n_warp=1), which covers M=192 x N=104: must compute.

    Pinned to the MMA path: N=104 (208 bytes/row of fp16) fits no canonical
    GMMA swizzle atom, so the WGMMA route rejects this shape in the descriptor
    layer on any warp partition, before and after the partition fix alike."""
    _check(192, 104, 32, 384, GemmWarpPolicy.FullCol, pass_configs=_FORCE_MMA)


@tilelang.testing.requires_cuda
def test_fullrow_fallback_covering_numeric():
    """FullRow, M=48, 8 warps used to die on an internal ICHECK
    (m_warp = 48 / 16 = 3 does not divide 8).  The fixed search falls back
    to (m_warp=1, n_warp=8), which covers M=48 x N=64: must compute."""
    _check(48, 64, 32, 256, GemmWarpPolicy.FullRow)


@tilelang.testing.requires_cuda
def test_square_no_partition_clear_error():
    """Square, 16x16 with 7 warps has no covering partition at all.  Used to
    die on an internal ICHECK; must now raise the clear front-end error."""
    with pytest.raises(Exception, match=_REJECT):
        tilelang.compile(_matmul(16, 16, 32, 224, GemmWarpPolicy.Square), out_idx=[2])


# ---------------------------------------------------------------------------
# WGMMA path (sm_90).  N tiles are multiples of 16 in every case that
# executes, steering clear of the unrelated n_dim=8 descriptor defect
# (issue #2593).
# ---------------------------------------------------------------------------


@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_wgmma_uncovered_n_rejected():
    """WGMMA Square, N=104, 12 warps picked (4, 3): 3 * (104 // 3) = 102 < 104
    and warp_col_tiles=34 tripped an AssertionError.  Must reject cleanly."""
    with pytest.raises(Exception, match=_REJECT):
        tilelang.compile(_matmul(64, 104, 32, 384, GemmWarpPolicy.Square), out_idx=[2])


@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_wgmma_uncovered_m_rejected():
    """WGMMA Square, M=336, 20 warps picked m_warp=20 (covers only 320 rows)
    and passed every emitter assert - the silent WGMMA face.  Must reject."""
    with pytest.raises(Exception, match=_REJECT):
        tilelang.compile(_matmul(336, 16, 32, 640, GemmWarpPolicy.Square), out_idx=[2])


@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_wgmma_fullcol_uncovered_m_rejected():
    """WGMMA FullCol retry loop picked (20, 1) without covering M=336."""
    with pytest.raises(Exception, match=_REJECT):
        tilelang.compile(_matmul(336, 16, 32, 640, GemmWarpPolicy.FullCol), out_idx=[2])


@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_wgmma_square_conserved_control_numeric():
    """Control: WGMMA Square, M=64, N=96, 4 warps picks (4, 1) both before and
    after the fix (warp_col_tiles=96); the partition is conserved and the
    result must match ground truth."""
    _check(64, 96, 32, 128, GemmWarpPolicy.Square)


@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_wgmma_fullrow_factoring_numeric():
    """WGMMA FullRow, M=128, 12 warps used to die on an internal ICHECK
    (cand=8 does not divide 12).  The fixed search picks (4, 3) with
    warp_col_tiles=64, covering 128 x 192: must compute."""
    _check(128, 192, 32, 384, GemmWarpPolicy.FullRow)


if __name__ == "__main__":
    tilelang.testing.main()
