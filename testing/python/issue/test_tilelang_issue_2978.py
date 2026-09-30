"""`T.ws(-1)` is documented to skip the threadIdx binding, so it runs everywhere.

A negative id was merged as an ordinary warp-group index instead, which built the
range `[-warp_group_size, 0)` -- true for no thread -- so the region ran on no
thread and every store inside it was dropped without a diagnostic.
"""

import pytest
import torch

import tilelang
import tilelang.testing
import tilelang.language as T

N = 256


def _writer_kernel(ws):
    """Zero every element, then write 5 from inside the `ws` region (if any)."""

    @T.prim_func
    def main(Out: T.Tensor((N,), "int32")):
        with T.Kernel(1, threads=N):
            tx = T.get_thread_binding()
            Out[tx] = 0
            if ws is None:
                Out[tx] = 5
            else:
                with T.ws(ws):
                    Out[tx] = 5

    return main


@tilelang.testing.requires_cuda
def test_ws_negative_one_runs_on_every_thread():
    out = tilelang.compile(_writer_kernel(-1), out_idx=[0], target="cuda")().cpu()
    assert int((out == 5).sum().item()) == N


@tilelang.testing.requires_cuda
def test_ws_negative_one_matches_an_unwrapped_store():
    """Control: the same store outside any `ws` region runs on every thread."""

    plain = tilelang.compile(_writer_kernel(None), out_idx=[0], target="cuda")().cpu()
    neg_one = tilelang.compile(_writer_kernel(-1), out_idx=[0], target="cuda")().cpu()
    torch.testing.assert_close(neg_one, plain, rtol=0, atol=0)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("group", [0, 1])
def test_ws_group_index_still_selects_one_group(group):
    """Control: a real group index still restricts the region to its own threads."""

    out = tilelang.compile(_writer_kernel(group), out_idx=[0], target="cuda")().cpu()
    assert int((out == 5).sum().item()) == N // 2


if __name__ == "__main__":
    tilelang.testing.main()
