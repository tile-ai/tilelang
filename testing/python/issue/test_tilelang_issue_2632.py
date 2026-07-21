"""Regression test for GitHub issue #2632.

A contracting (many-to-one) forward map passed to ``T.annotate_layout`` on a
shared buffer used to be misread by ``makeBufferWithLayout`` as a replication
factor: the buffer was rewritten to a higher rank and compilation aborted on an
unrelated ``BufferStore`` rank ICHECK (``2 vs. 1``). It must instead be
rejected with a ``ValueError`` naming the buffer and the non-injectivity.

Note: the guard is a pigeonhole check (codomain extent < domain extent). It
covers every layout that can reach the broken replication reinterpretation;
non-injective maps whose output bounding box is not smaller than the domain
(e.g. ``lambda i: (i % 64) * 4``) do not crash there and are out of scope
for this fix.
"""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.layout import Layout

N = 128

REJECT_MATCH = r"shared buffer `\w+`.*must be injective"


def _roundtrip_kernel(fwd):

    @T.prim_func
    def main(A: T.Tensor((N,), "int32"), Out: T.Tensor((N,), "int32")):
        with T.Kernel(1, threads=N):
            s = T.alloc_shared((N,), "int32")
            T.annotate_layout({s: Layout((N,), fwd)})
            for i in T.Parallel(N):
                s[i] = A[i]
            for i in T.Parallel(N):
                Out[i] = s[i]

    return main


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "fwd",
    [
        pytest.param(lambda i: i % 64, id="mod64-2-to-1"),
        pytest.param(lambda i: i * 0, id="zero-N-to-1"),
    ],
)
def test_contracting_shared_layout_rejected(fwd):
    """A contracting many-to-one map must be a clean ValueError, not an ICE."""
    with pytest.raises(ValueError, match=REJECT_MATCH):
        tilelang.compile(_roundtrip_kernel(fwd), out_idx=[1], target="cuda")


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "fwd",
    [
        pytest.param(lambda i: i, id="identity"),
        pytest.param(lambda i: i ^ 1, id="xor-pair-swap"),
        pytest.param(lambda i: i * 2, id="grow-i-times-2"),
        pytest.param(lambda i: i + 10, id="grow-i-plus-10"),
    ],
)
def test_injective_shared_layout_still_compiles_and_runs(fwd):
    """Injective maps (including codomain-growing ones) must keep working."""
    kernel = tilelang.compile(_roundtrip_kernel(fwd), out_idx=[1], target="cuda")
    A = torch.arange(N, dtype=torch.int32, device="cuda")
    torch.testing.assert_close(kernel(A), A)


if __name__ == "__main__":
    tilelang.testing.main()
