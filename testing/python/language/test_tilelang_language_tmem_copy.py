"""Register<->TMEM copy roundtrips over general annotated TMEM layouts.

Each kernel allocates one fragment and one TMEM buffer of the same logical
shape, annotates the TMEM buffer with an explicit ``lambda *coords ->
[datapath, column]`` layout, stores registers into TMEM (tcgen05.st), loads
them back (tcgen05.ld), and writes the result out; the test verifies the
roundtrip is the identity.  This exercises the generic CuTe-algebra TMEM
copy path (InferTMemLayout / ExpandTcgen05Layout / LowerTmem) over
multi-dimensional, permuted, and interleaved physical layouts, with and
without slicing.
"""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing

PASS_CONFIGS = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}

# (name, logical shape, forward map to [datapath, column], threads)
# The layout in CuTe spelling is given alongside each case.
LAYOUT_CASES = [
    # (128,128):(1@0,1@1) -- the standard accumulator fragment.
    ("std_2d", (128, 128), lambda i, j: [i, j], 128),
    # (128,128):(1@1,1@0) -- transposed: rows map to columns.
    ("transposed_2d", (128, 128), lambda i, j: [j, i], 128),
    # (3,128,128):(128@1,1@0,1@1) -- leading batch repeated along columns.
    ("batched_3d", (3, 128, 128), lambda a, i, j: [i, a * 128 + j], 128),
    # (3,32,64,4):(64@1,1@0,1@1,32@0) -- datapaths split across two modes.
    ("interleaved_4d", (3, 32, 64, 4), lambda a, i, j, k: [i + 32 * k, a * 64 + j], 128),
    # (128,128):(1@0,1@1) spread across two warpgroups.
    ("std_2d_2wg", (128, 128), lambda i, j: [i, j], 256),
]


def _make_roundtrip_kernel(shape, forward, threads):
    @T.prim_func
    def main(
        A: T.Tensor(shape, T.float32),
        D: T.Tensor(shape, T.float32),
    ):
        with T.Kernel(1, threads=threads):
            A_shared = T.alloc_shared(shape, T.float32)
            A_frag = T.alloc_fragment(shape, T.float32)
            tmem = T.alloc_tmem(shape, T.float32)
            B_frag = T.alloc_fragment(shape, T.float32)
            B_shared = T.alloc_shared(shape, T.float32)

            T.annotate_layout({tmem: T.Layout(shape, forward)})
            T.copy(A, A_shared)
            T.copy(A_shared, A_frag)
            T.copy(A_frag, tmem)
            T.copy(tmem, B_frag)
            T.copy(B_frag, B_shared)
            T.copy(B_shared, D)

    return main


def _last_dim_slice(buf, rank, lo, hi):
    """buf[:, ..., lo:hi] spelled with explicit full slices per rank."""
    index = tuple([slice(None)] * (rank - 1) + [slice(lo, hi)])
    return buf[index]


def _make_sliced_roundtrip_kernel(shape, forward, threads, nsplit):
    """Split the store and the load along the last dim into nsplit slices."""
    last = shape[-1]
    assert last % nsplit == 0
    step = last // nsplit
    rank = len(shape)

    @T.prim_func
    def main(
        A: T.Tensor(shape, T.float32),
        D: T.Tensor(shape, T.float32),
    ):
        with T.Kernel(1, threads=threads):
            A_shared = T.alloc_shared(shape, T.float32)
            A_frag = T.alloc_fragment(shape, T.float32)
            tmem = T.alloc_tmem(shape, T.float32)
            B_frag = T.alloc_fragment(shape, T.float32)
            B_shared = T.alloc_shared(shape, T.float32)

            T.annotate_layout({tmem: T.Layout(shape, forward)})
            T.copy(A, A_shared)
            T.copy(A_shared, A_frag)
            for s in range(nsplit):
                T.copy(
                    _last_dim_slice(A_frag, rank, s * step, (s + 1) * step),
                    _last_dim_slice(tmem, rank, s * step, (s + 1) * step),
                )
            for s in range(nsplit):
                T.copy(
                    _last_dim_slice(tmem, rank, s * step, (s + 1) * step),
                    _last_dim_slice(B_frag, rank, s * step, (s + 1) * step),
                )
            T.copy(B_frag, B_shared)
            T.copy(B_shared, D)

    return main


def _make_asymmetric_roundtrip_kernel(shape, forward, threads, nsplit):
    """Sliced stores, one whole-buffer load.

    Asymmetric on purpose: a bug that mislocates a slice's registers on the
    store is NOT cancelled by the same bug on the load, unlike the symmetric
    roundtrips above.
    """
    last = shape[-1]
    assert last % nsplit == 0
    step = last // nsplit
    rank = len(shape)

    @T.prim_func
    def main(
        A: T.Tensor(shape, T.float32),
        D: T.Tensor(shape, T.float32),
    ):
        with T.Kernel(1, threads=threads):
            A_shared = T.alloc_shared(shape, T.float32)
            A_frag = T.alloc_fragment(shape, T.float32)
            tmem = T.alloc_tmem(shape, T.float32)
            B_frag = T.alloc_fragment(shape, T.float32)
            B_shared = T.alloc_shared(shape, T.float32)

            T.annotate_layout({tmem: T.Layout(shape, forward)})
            T.copy(A, A_shared)
            T.copy(A_shared, A_frag)
            for s in range(nsplit):
                T.copy(
                    _last_dim_slice(A_frag, rank, s * step, (s + 1) * step),
                    _last_dim_slice(tmem, rank, s * step, (s + 1) * step),
                )
            T.copy(tmem, B_frag)
            T.copy(B_frag, B_shared)
            T.copy(B_shared, D)

    return main


def _make_batch_sliced_roundtrip_kernel(shape, forward, threads):
    """Store/load one leading-dim batch entry at a time (dynamic origins)."""
    tail = tuple([slice(None)] * (len(shape) - 1))

    @T.prim_func
    def main(
        A: T.Tensor(shape, T.float32),
        D: T.Tensor(shape, T.float32),
    ):
        with T.Kernel(1, threads=threads):
            A_shared = T.alloc_shared(shape, T.float32)
            A_frag = T.alloc_fragment(shape, T.float32)
            tmem = T.alloc_tmem(shape, T.float32)
            B_frag = T.alloc_fragment(shape, T.float32)
            B_shared = T.alloc_shared(shape, T.float32)

            T.annotate_layout({tmem: T.Layout(shape, forward)})
            T.copy(A, A_shared)
            T.copy(A_shared, A_frag)
            for a in T.serial(shape[0]):
                T.copy(A_frag[(a, *tail)], tmem[(a, *tail)])
            for a in T.serial(shape[0]):
                T.copy(tmem[(a, *tail)], B_frag[(a, *tail)])
            T.copy(B_frag, B_shared)
            T.copy(B_shared, D)

    return main


def _run_roundtrip(kernel_func, shape):
    kernel = tilelang.compile(kernel_func, target="cuda", pass_configs=PASS_CONFIGS)
    source = kernel.get_kernel_source()
    assert "tcgen05_st_" in source and "tcgen05_ld_" in source
    a = torch.randn(*shape, device="cuda", dtype=torch.float32)
    d = torch.empty(*shape, device="cuda", dtype=torch.float32)
    kernel(a, d)
    torch.testing.assert_close(d, a, rtol=0.0, atol=0.0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
@pytest.mark.parametrize(
    ("name", "shape", "forward", "threads"),
    LAYOUT_CASES,
    ids=[case[0] for case in LAYOUT_CASES],
)
def test_tmem_copy_roundtrip(name, shape, forward, threads):
    _run_roundtrip(_make_roundtrip_kernel(shape, forward, threads), shape)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
@pytest.mark.parametrize(
    ("name", "shape", "forward", "threads"),
    # A batched layout sliced along its column mode leaves per-batch column
    # gaps in TMEM; the copy iterates the contiguous chunks, one tcgen05
    # issue per batch entry (rest iteration).
    [case for case in LAYOUT_CASES if case[0] in ("std_2d", "std_2d_2wg", "batched_3d")],
    ids=["std_2d", "std_2d_2wg", "batched_3d"],
)
def test_tmem_copy_roundtrip_sliced_last_dim(name, shape, forward, threads):
    _run_roundtrip(_make_sliced_roundtrip_kernel(shape, forward, threads, nsplit=2), shape)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
@pytest.mark.parametrize(
    ("name", "shape", "forward", "threads"),
    [case for case in LAYOUT_CASES if case[0] in ("std_2d", "batched_3d")],
    ids=["std_2d", "batched_3d"],
)
def test_tmem_copy_sliced_store_whole_load(name, shape, forward, threads):
    _run_roundtrip(_make_asymmetric_roundtrip_kernel(shape, forward, threads, nsplit=2), shape)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
@pytest.mark.parametrize(
    ("name", "shape", "forward", "threads"),
    [case for case in LAYOUT_CASES if case[0] in ("batched_3d", "interleaved_4d")],
    ids=["batched_3d", "interleaved_4d"],
)
def test_tmem_copy_roundtrip_sliced_batch(name, shape, forward, threads):
    _run_roundtrip(_make_batch_sliced_roundtrip_kernel(shape, forward, threads), shape)


if __name__ == "__main__":
    tilelang.testing.main()
