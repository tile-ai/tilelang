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


def _make_16bit_roundtrip_kernel(shape, forward, dtype):
    @T.prim_func
    def main(
        A: T.Tensor(shape, dtype),
        D: T.Tensor(shape, dtype),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared(shape, dtype)
            A_frag = T.alloc_fragment(shape, dtype)
            tmem = T.alloc_tmem(shape, dtype)
            B_frag = T.alloc_fragment(shape, dtype)
            B_shared = T.alloc_shared(shape, dtype)

            T.annotate_layout({tmem: T.Layout(shape, forward)})
            T.copy(A, A_shared)
            T.copy(A_shared, A_frag)
            T.copy(A_frag, tmem)
            # Clobber both fragments so a partial load cannot be masked by
            # stale register values that happen to alias the store's.
            T.fill(A_frag, -1.0)
            T.fill(B_frag, -2.0)
            T.copy(tmem, B_frag)
            T.copy(B_frag, B_shared)
            T.copy(B_shared, D)

    return main


def _make_shared_allocation_roundtrip_kernel(wide_cols, narrow_cols):
    """Roundtrip two TMEM buffers that share one physical tcgen05.alloc.

    LowerSharedTmem packs the narrow buffer at column ``wide_cols`` of the
    allocation both use, because rounding their columns up to a power of two
    together costs nothing.  The *wide* roundtrip is what catches a dropped
    column offset: the narrow buffer would still round-trip if its store and its
    load dropped the offset together, but its store would land on the wide
    buffer's first columns.
    """
    wide_shape = (128, wide_cols)
    narrow_shape = (128, narrow_cols)

    @T.prim_func
    def main(
        A: T.Tensor(wide_shape, T.float32),
        B: T.Tensor(narrow_shape, T.float32),
        DA: T.Tensor(wide_shape, T.float32),
        DB: T.Tensor(narrow_shape, T.float32),
    ):
        with T.Kernel(1, threads=128):
            a_shared = T.alloc_shared(wide_shape, T.float32)
            a_in_frag = T.alloc_fragment(wide_shape, T.float32)
            a_out_frag = T.alloc_fragment(wide_shape, T.float32)
            b_shared = T.alloc_shared(narrow_shape, T.float32)
            b_in_frag = T.alloc_fragment(narrow_shape, T.float32)
            b_out_frag = T.alloc_fragment(narrow_shape, T.float32)
            wide_tmem = T.alloc_tmem(wide_shape, T.float32)
            narrow_tmem = T.alloc_tmem(narrow_shape, T.float32)

            T.annotate_layout(
                {
                    wide_tmem: T.Layout(wide_shape, lambda i, j: [i, j]),
                    narrow_tmem: T.Layout(narrow_shape, lambda i, j: [i, j]),
                }
            )
            T.copy(A, a_shared)
            T.copy(a_shared, a_in_frag)
            T.copy(a_in_frag, wide_tmem)
            T.copy(B, b_shared)
            T.copy(b_shared, b_in_frag)
            T.copy(b_in_frag, narrow_tmem)

            T.copy(wide_tmem, a_out_frag)
            T.copy(a_out_frag, a_shared)
            T.copy(a_shared, DA)
            T.copy(narrow_tmem, b_out_frag)
            T.copy(b_out_frag, b_shared)
            T.copy(b_shared, DB)

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
    ("name", "shape", "forward", "modifier"),
    [
        # CUTLASS FrgTypeC keeps each 16-bit accumulator in the low half of
        # its own 32-bit storage slot (column stride 2), so the copy plans
        # in b32 columns and gathers with the pack::16b/unpack::16b
        # modifiers.
        ("pack16b", (128, 64), lambda i, j: [i, 2 * j], True),
        # Two 16-bit values packed per b32 column move as whole columns with
        # the plain instruction.
        ("packed_pair", (128, 128), lambda i, j: [i, j], False),
    ],
    ids=["pack16b", "packed_pair"],
)
def test_tmem_copy_roundtrip_16bit(name, shape, forward, modifier):
    kernel = tilelang.compile(
        _make_16bit_roundtrip_kernel(shape, forward, T.bfloat16),
        target="cuda",
        pass_configs=PASS_CONFIGS,
    )
    source = kernel.get_kernel_source()
    ld_line = next(line for line in source.splitlines() if "tcgen05_ld_" in line)
    assert (", true>" in ld_line) == modifier
    a = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    d = torch.empty(*shape, device="cuda", dtype=torch.bfloat16)
    kernel(a, d)
    torch.testing.assert_close(d, a, rtol=0.0, atol=0.0)


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


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_tmem_copy_roundtrip_across_a_shared_allocation():
    wide_cols, narrow_cols = 96, 32
    kernel = tilelang.compile(
        _make_shared_allocation_roundtrip_kernel(wide_cols, narrow_cols),
        target="cuda",
        pass_configs=PASS_CONFIGS,
    )
    source = kernel.get_kernel_source()
    # 96 + 32 columns round up to the same 128 the wide buffer needs alone, so
    # the two buffers share a single allocation.
    assert source.count("tl::tmem_allocate") == 1
    assert "tl::tmem_allocate((&(wide_tmem[0])), 128)" in source
    assert "tcgen05_st_" in source and "tcgen05_ld_" in source

    a = torch.randn(128, wide_cols, device="cuda", dtype=torch.float32)
    b = torch.randn(128, narrow_cols, device="cuda", dtype=torch.float32)
    da = torch.empty_like(a)
    db = torch.empty_like(b)
    kernel(a, b, da, db)
    torch.testing.assert_close(da, a, rtol=0.0, atol=0.0)
    torch.testing.assert_close(db, b, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    tilelang.testing.main()
