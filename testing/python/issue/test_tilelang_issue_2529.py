import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.engine.lower import lower
from tilelang.layout import make_linear_layout


def _make_tma_store_kernel(n, block, annotate_linear=False):
    @T.prim_func
    def main(b: T.Tensor((n,), T.float32)):
        with T.Kernel(T.ceildiv(n, block), threads=128) as bx:
            shared = T.alloc_shared((block,), T.float32)
            if annotate_linear:
                T.annotate_layout({shared: make_linear_layout(shared)})
            for i in T.Parallel(block):
                shared[i] = T.cast(bx * block + i, T.float32)
            T.copy(shared, b[bx * block])

    return main


def _lower_source(n, block, annotate_linear=False):
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90"})
    with target:
        artifact = lower(
            _make_tma_store_kernel(n, block, annotate_linear),
            target=target,
            enable_device_compile=False,
        )
    return artifact.kernel_source


def test_tma_store_selection_uses_copy_bounds():
    full_tile_source = _lower_source(1024, 256)
    # The unannotated rank-1 tail exercises descriptor layout inference.
    partial_tile_source = _lower_source(1000, 256)
    # A linear layout cannot itself force descriptor lowering, so this also
    # verifies that Lower re-derives the tail bounds from its analyzer.
    partial_tile_linear_layout_source = _lower_source(1000, 256, annotate_linear=True)

    assert "tma_store" in full_tile_source
    assert "CUtensorMap" not in full_tile_source
    assert "tma_store" in partial_tile_source
    assert "CUtensorMap" in partial_tile_source
    assert "tma_store" in partial_tile_linear_layout_source
    assert "CUtensorMap" in partial_tile_linear_layout_source


@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_tma_store_partial_tile_runtime():
    n, block = 1000, 256
    kernel = tilelang.compile(_make_tma_store_kernel(n, block, annotate_linear=True), target="cuda")

    sentinel = -1.0
    padded_n = (n + block - 1) // block * block
    backing = torch.full((padded_n,), sentinel, device="cuda")
    kernel(backing[:n])
    torch.cuda.synchronize()

    expected = torch.arange(n, dtype=torch.float32, device="cuda")
    torch.testing.assert_close(backing[:n], expected)
    assert torch.all(backing[n:] == sentinel)


if __name__ == "__main__":
    test_tma_store_selection_uses_copy_bounds()
    test_tma_store_partial_tile_runtime()
