"""Regression test for issue #2524.

T.reduce_sum / T.reduce_max with batch > 1 silently returned wrong results
when reducing_threads < blockDim. The workspace was allocated with
reducing_threads as the stride, but the device template indexed with
threadIdx.x (the full block index), causing cross-batch overwrites.
"""

import torch

import tilelang
import tilelang.language as T
from tilelang.testing import requires_cuda


def _make_reduce_kernel(batch: int, op: str):
    """Build a kernel that reduces a (2, 8, 16) fragment along dim=1.

    Each block has 128 threads. The reduce dimension (dim=1, length 8) only
    involves 64 threads, so reducing_threads < blockDim — this is the
    condition that triggered issue #2524.
    """

    @T.prim_func
    def main(
        A: T.Tensor((16, 8, 16), "float32"),
        C: T.Tensor((16, 16), "float32"),
    ):
        with T.Kernel(8, threads=128) as bx:
            a_s = T.alloc_shared((2, 8, 16), "float32")
            a_f = T.alloc_fragment((2, 8, 16), "float32")
            c_f = T.alloc_fragment((2, 16), "float32")
            T.copy(A[bx * 2, 0, 0], a_s)
            T.copy(a_s, a_f)
            if op == "sum":
                T.reduce_sum(a_f, c_f, dim=1, batch=batch)
            elif op == "max":
                T.reduce_max(a_f, c_f, dim=1, batch=batch)
            elif op == "min":
                T.reduce_min(a_f, c_f, dim=1, batch=batch)
            T.copy(c_f, C[bx * 2, 0])

    return main


def _torch_ref(A: torch.Tensor, op: str):
    if op == "sum":
        return A.sum(dim=1)
    elif op == "max":
        return A.max(dim=1).values
    elif op == "min":
        return A.min(dim=1).values
    raise ValueError(f"Unknown op: {op}")


@requires_cuda
def test_issue_2524_reduce_sum_batch_gt_1():
    """reduce_sum with batch=2 — reducing_threads(64) < blockDim(128)."""
    tilelang.disable_cache()
    torch.manual_seed(0)
    A = torch.randint(0, 8, (16, 8, 16)).float().cuda()
    ref = _torch_ref(A, "sum").cpu()

    kernel = tilelang.compile(
        _make_reduce_kernel(batch=2, op="sum"),
        target="cuda",
    )
    C = torch.empty((16, 16), dtype=torch.float32, device="cuda")
    kernel(A, C)
    torch.cuda.synchronize()

    diff = (C.cpu() - ref).abs()
    ndiff = int((diff > 1e-4).sum())
    assert ndiff == 0, f"Batch=2 sum had {ndiff} mismatches (max diff {diff.max().item():g})"


@requires_cuda
def test_issue_2524_reduce_sum_batch_eq_1():
    """reduce_sum with batch=1 — sanity check (should always pass)."""
    tilelang.disable_cache()
    torch.manual_seed(0)
    A = torch.randint(0, 8, (16, 8, 16)).float().cuda()
    ref = _torch_ref(A, "sum").cpu()

    kernel = tilelang.compile(
        _make_reduce_kernel(batch=1, op="sum"),
        target="cuda",
    )
    C = torch.empty((16, 16), dtype=torch.float32, device="cuda")
    kernel(A, C)
    torch.cuda.synchronize()

    diff = (C.cpu() - ref).abs()
    ndiff = int((diff > 1e-4).sum())
    assert ndiff == 0, f"Batch=1 sum had {ndiff} mismatches (max diff {diff.max().item():g})"


@requires_cuda
def test_issue_2524_reduce_max_batch_gt_1():
    """reduce_max with batch=2 — reducing_threads(64) < blockDim(128)."""
    tilelang.disable_cache()
    torch.manual_seed(0)
    A = torch.randint(0, 8, (16, 8, 16)).float().cuda()
    ref = _torch_ref(A, "max").cpu()

    kernel = tilelang.compile(
        _make_reduce_kernel(batch=2, op="max"),
        target="cuda",
    )
    C = torch.empty((16, 16), dtype=torch.float32, device="cuda")
    kernel(A, C)
    torch.cuda.synchronize()

    diff = (C.cpu() - ref).abs()
    ndiff = int((diff > 1e-4).sum())
    assert ndiff == 0, f"Batch=2 max had {ndiff} mismatches (max diff {diff.max().item():g})"


@requires_cuda
def test_issue_2524_reduce_min_batch_gt_1():
    """reduce_min with batch=2 — reducing_threads(64) < blockDim(128)."""
    tilelang.disable_cache()
    torch.manual_seed(0)
    A = torch.randint(0, 8, (16, 8, 16)).float().cuda()
    ref = _torch_ref(A, "min").cpu()

    kernel = tilelang.compile(
        _make_reduce_kernel(batch=2, op="min"),
        target="cuda",
    )
    C = torch.empty((16, 16), dtype=torch.float32, device="cuda")
    kernel(A, C)
    torch.cuda.synchronize()

    diff = (C.cpu() - ref).abs()
    ndiff = int((diff > 1e-4).sum())
    assert ndiff == 0, f"Batch=2 min had {ndiff} mismatches (max diff {diff.max().item():g})"


if __name__ == "__main__":
    tilelang.testing.main()
