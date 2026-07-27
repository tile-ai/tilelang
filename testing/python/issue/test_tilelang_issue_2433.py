"""Regression for tile-ai/tilelang issue #2433."""

import torch
import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.jit
def _persistent_visit_kernel(MB, NB, sm_num):

    @T.prim_func
    def kernel(Visited: T.Tensor((MB, NB), T.int32)):
        with T.Kernel(sm_num, threads=1) as block_id:
            for bx, by in T.Persistent([MB, NB], sm_num, block_id):
                Visited[bx, by] += 1

    return kernel


@tilelang.jit
def _persistent_visit_kernel_3d(D0, D1, D2, sm_num):

    @T.prim_func
    def kernel(Visited: T.Tensor((D0, D1, D2), T.int32)):
        with T.Kernel(sm_num, threads=1) as block_id:
            for i, j, k in T.Persistent([D0, D1, D2], sm_num, block_id):
                Visited[i, j, k] += 1

    return kernel


def run_persistent_visit(m_blocks: int, n_blocks: int, sm_num: int):
    kernel = _persistent_visit_kernel(m_blocks, n_blocks, sm_num)
    visited = torch.zeros(m_blocks, n_blocks, device="cuda", dtype=torch.int32)
    kernel(visited)
    expected = torch.ones_like(visited)
    assert torch.equal(visited, expected), (
        f"T.Persistent visited grid mismatch for "
        f"m_blocks={m_blocks}, n_blocks={n_blocks}, sm_num={sm_num}: "
        f"missing={(visited == 0).nonzero(as_tuple=False).tolist()}, "
        f"overflow={(visited > 1).nonzero(as_tuple=False).tolist()}"
    )


def run_persistent_visit_3d(d0: int, d1: int, d2: int, sm_num: int):
    kernel = _persistent_visit_kernel_3d(d0, d1, d2, sm_num)
    visited = torch.zeros(d0, d1, d2, device="cuda", dtype=torch.int32)
    kernel(visited)
    expected = torch.ones_like(visited)
    assert torch.equal(visited, expected), (
        f"T.Persistent 3D visited grid mismatch for "
        f"({d0},{d1},{d2}), sm_num={sm_num}: "
        f"missing={(visited == 0).nonzero(as_tuple=False).tolist()}, "
        f"overflow={(visited > 1).nonzero(as_tuple=False).tolist()}"
    )


@tilelang.testing.requires_cuda
def test_issue_2433():
    """Reproduce the failure table from the issue and the divisor controls.

    With group_size defaulting to min(8, n_blocks), only the rows where
    n_blocks % group_size != 0 trip the bug; the divisor rows act as
    controls to make sure the fix doesn't regress them.
    """
    # Divisor controls (must always pass).
    for n_blocks in (1, 2, 4, 8):
        run_persistent_visit(m_blocks=4, n_blocks=n_blocks, sm_num=3)

    # Non-divisor cases -- exactly the failing rows from the issue.
    for n_blocks in (9, 10, 11, 12):
        run_persistent_visit(m_blocks=4, n_blocks=n_blocks, sm_num=3)

    # Single-wave control from the issue (sm_num >= total tiles): proves
    # the bug is divisibility, not wave-count related.
    run_persistent_visit(m_blocks=4, n_blocks=9, sm_num=36)

    # Boundary: leading dim not a multiple of group_size either -- only the
    # last dim is grouped, so this must still pass.
    run_persistent_visit(m_blocks=7, n_blocks=11, sm_num=5)

    # Boundary: 3D domain with non-divisor last dim, exercising the
    # mixed-radix decode + range guard at higher rank.
    run_persistent_visit_3d(d0=3, d1=5, d2=10, sm_num=4)

    # Boundary: more SMs than tiles, so several waves end up entirely in
    # the padded tail and must be skipped by the range guard.
    run_persistent_visit(m_blocks=2, n_blocks=3, sm_num=16)


if __name__ == "__main__":
    tilelang.testing.main()
