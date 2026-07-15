"""Numerical and MLIR-structural tests for TileIR's tile-copy views:
T.transpose (tl.tileop.transpose) and the TileIR backend's recognition of
the fixed-4-row TMA gather/scatter surface (T.tma_gather4 / T.tma_scatter4).

Note: deliberately does NOT `from __future__ import annotations`. The
jit factories below reference shape variables (m, n, k, ...) only inside
parameter type annotations (e.g. `T.Tensor((n, k), ...)`), never in the
function body. Under PEP 563 (deferred/string annotations), such names are
never captured as closure freevars, so resolving the annotations later
raises NameError before tracing even starts. Eager (non-deferred)
annotations sidestep this since they evaluate immediately in the enclosing
scope.
"""

import pytest
import tilelang
import tilelang.language as T

# ---------------------------------------------------------------------------
# Guard: cuda_tile MLIR bindings required to build any kernel
# ---------------------------------------------------------------------------

try:
    from cuda_tile._mlir import ir as _ir  # noqa: F401

    _HAS_CUDA_TILE = True
except ImportError:
    _HAS_CUDA_TILE = False

skip_no_cuda_tile = pytest.mark.skipif(
    not _HAS_CUDA_TILE,
    reason="cuda_tile MLIR bindings unavailable",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _skip_if_tileir_toolchain_unavailable():
    from tilelang.tileir.checks import TileIRDependencyError, check_tileir_available

    try:
        check_tileir_available()
    except TileIRDependencyError as exc:
        pytest.skip(f"CUDA TileIR toolchain unavailable: {exc}")


def _setup_gpu():
    """Return (torch, major, minor, target_str) or skip."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("No CUDA GPU available")
    major, minor = torch.cuda.get_device_capability()
    target_str = f"tileir -arch=sm_{major}{minor}"
    return torch, major, minor, target_str


# ---------------------------------------------------------------------------
# Test: T.tma_gather4 / T.tma_scatter4 structural probes (TileIR backend
# RECOGNIZES the existing T.tma_gather4/scatter4 surface via new
# CopyGather/CopyScatter ops).
# ---------------------------------------------------------------------------


def _gather4_jit(n: int = 64, k: int = 64, K_box: int | None = None, col: int = 0, rows=(10, 11, 12, 13)):
    """Gather 4 rows of a (n,k) GLOBAL src, at column OFFSET ``col``, into a
    (4, K_box) SHARED tile, then copy the tile out to a (4, K_box) GLOBAL dst
    so the result is host-observable."""
    K_box = k if K_box is None else K_box
    rows = list(rows)

    @tilelang.jit
    def make():
        @T.prim_func
        def main(
            Src: T.Tensor((n, k), "float32"),
            Dst: T.Tensor((4, K_box), "float32"),
        ):
            with T.Kernel(1, threads=128):
                smem = T.alloc_shared((4, K_box), "float32")
                mbar = T.alloc_barrier(1)
                T.tma_gather4(Src, smem, col, rows, barrier=mbar)
                T.copy(smem, Dst)

        return main

    return make


def _scatter4_jit(n: int = 64, k: int = 64, K_box: int | None = None, col: int = 0, rows=(10, 11, 12, 13)):
    """Scatter a (4, K_box) SHARED tile (loaded from a (4, K_box) GLOBAL src) into 4
    rows of a (n, k) GLOBAL dst, at column OFFSET ``col``."""
    K_box = k if K_box is None else K_box
    rows = list(rows)

    @tilelang.jit
    def make():
        @T.prim_func
        def main(
            Src: T.Tensor((4, K_box), "float32"),
            Dst: T.Tensor((n, k), "float32"),
        ):
            with T.Kernel(1, threads=128):
                smem = T.alloc_shared((4, K_box), "float32")
                T.copy(Src, smem)
                T.tma_scatter4(smem, Dst, col, rows)

        return main

    return make


@skip_no_cuda_tile
def test_tma_gather4_emits_gather_scatter_view():
    _skip_if_tileir_toolchain_unavailable()
    from tilelang.tileir.pipeline import build_tileir_module
    from tilelang.tileir.semantic import materialize_launch_nest
    from tilelang.tileir.launch import _split_grid_sync_primfunc

    pf = _gather4_jit().get_tir()
    pf = materialize_launch_nest(pf)
    pf = _split_grid_sync_primfunc(pf)
    mlir = str(build_tileir_module(pf, arch="sm_100"))
    assert "make_gather_scatter_view" in mlir
    # Gather-specific: the sparse (gather) index is loaded via load_view_tko
    # on the gsview, not the plain partition_view store path.
    assert "load_view_tko" in mlir


@skip_no_cuda_tile
def test_tma_scatter4_emits_gather_scatter_view():
    """Structural mirror of test_tma_gather4_emits_gather_scatter_view: the
    scatter path builds its gather_scatter_view over the DESTINATION buffer
    and stores through it (store_view_tko), the dual of the gather path's
    load_view_tko."""
    _skip_if_tileir_toolchain_unavailable()
    from tilelang.tileir.pipeline import build_tileir_module
    from tilelang.tileir.semantic import materialize_launch_nest
    from tilelang.tileir.launch import _split_grid_sync_primfunc

    pf = _scatter4_jit().get_tir()
    pf = materialize_launch_nest(pf)
    pf = _split_grid_sync_primfunc(pf)
    mlir = str(build_tileir_module(pf, arch="sm_100"))
    assert "make_gather_scatter_view" in mlir
    assert "store_view_tko" in mlir


def test_tma_gather4_traces_to_tir():
    """Frontend-only sanity: T.tma_gather4 must build a PrimFunc (TIR
    construction succeeds) without needing the cuda_tile MLIR bindings or a
    GPU. Lowering to MLIR is covered above."""
    pf = _gather4_jit().get_tir()
    assert pf is not None


def test_tma_scatter4_traces_to_tir():
    """Frontend-only sanity for T.tma_scatter4, mirroring the gather case."""
    pf = _scatter4_jit().get_tir()
    assert pf is not None


# ---------------------------------------------------------------------------
# Test: end-to-end GPU numerics for T.tma_gather4 / T.tma_scatter4
# ---------------------------------------------------------------------------


@skip_no_cuda_tile
@pytest.mark.parametrize("col,K_box", [(0, 64), (16, 32)], ids=["col0", "col16"])
def test_tma_gather4_numerical(col, K_box):
    """Gather4 of 4 distinct rows at column OFFSET ``col`` -- i.e. the
    gathered box is ``src[row, col:col+K_box]``. The col=16 variant pins the
    dim-1 (column) index path separately from col=0 and verifies that
    T.tma_gather4/scatter4's ``col`` drives the column offset. Pure data
    movement, so the comparison is bit-exact -- rtol=0, atol=0."""
    _skip_if_tileir_toolchain_unavailable()
    torch, *_ = _setup_gpu()

    n, k = 64, 64
    rows = [10, 11, 12, 13]
    kernel = tilelang.compile(_gather4_jit(n, k, K_box=K_box, col=col, rows=rows).get_tir(), execution_backend="tileir")

    src = torch.randn(n, k, dtype=torch.float32, device="cuda")
    dst = torch.empty(4, K_box, dtype=torch.float32, device="cuda")
    kernel(src, dst)

    expected = src[rows, col : col + K_box]
    torch.testing.assert_close(dst, expected, rtol=0, atol=0)


@skip_no_cuda_tile
def test_tma_gather4_numerical_duplicate_rows():
    """Gather4 with a DUPLICATE row index (10 repeated): a pure gather reads
    each row independently, so repeats are not a race (unlike scatter4).
    Bit-exact -- rtol=0, atol=0."""
    _skip_if_tileir_toolchain_unavailable()
    torch, *_ = _setup_gpu()

    n, k = 64, 64
    rows = [10, 10, 12, 13]
    kernel = tilelang.compile(_gather4_jit(n, k, rows=rows).get_tir(), execution_backend="tileir")

    src = torch.randn(n, k, dtype=torch.float32, device="cuda")
    dst = torch.empty(4, k, dtype=torch.float32, device="cuda")
    kernel(src, dst)

    expected = src[rows, :]
    torch.testing.assert_close(dst, expected, rtol=0, atol=0)


@skip_no_cuda_tile
@pytest.mark.parametrize("col,K_box", [(0, 64), (16, 32)], ids=["col0", "col16"])
def test_tma_scatter4_numerical(col, K_box):
    """Scatter4 into a sentinel-filled dst -- only the 4 scattered rows at
    column OFFSET ``col`` should change, and every other row and column must
    retain the sentinel bit-exactly (rtol=0, atol=0). The col=16 variant pins
    the dim-1 (column) index path separately from col=0, which alone could not
    distinguish a correct column-offset lowering from a hardcoded-zero bug --
    ensuring untouched rows AND untouched columns (0:16 and 48:64) are
    preserved."""
    _skip_if_tileir_toolchain_unavailable()
    torch, *_ = _setup_gpu()

    n, k = 64, 64
    rows = [10, 11, 12, 13]
    kernel = tilelang.compile(_scatter4_jit(n, k, K_box=K_box, col=col, rows=rows).get_tir(), execution_backend="tileir")

    src = torch.randn(4, K_box, dtype=torch.float32, device="cuda")
    sentinel = torch.full((n, k), -12345.0, dtype=torch.float32, device="cuda")
    dst = sentinel.clone()
    kernel(src, dst)

    expected = sentinel.clone()
    expected[rows, col : col + K_box] = src
    torch.testing.assert_close(dst, expected, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Test: T.tma_gather4 / T.tma_scatter4 rejection paths
# ---------------------------------------------------------------------------


def test_tma_gather4_rejects_five_rows():
    """Existing frontend check (copy_op.py's T.tma_gather4): exactly 4 row
    indices are required. No GPU / cuda_tile toolchain needed -- the
    rejection happens at trace time, before any TIR is lowered."""
    with pytest.raises(ValueError, match="exactly 4 row indices"):
        _gather4_jit(rows=(10, 11, 12, 13, 14)).get_tir()


def test_tma_scatter4_rejects_five_rows():
    """Scatter4 dual of test_tma_gather4_rejects_five_rows."""
    with pytest.raises(ValueError, match="exactly 4 row indices"):
        _scatter4_jit(rows=(10, 11, 12, 13, 14)).get_tir()


def _hand_crafted_gather4_sliced_shared_kernel(*, is_gather: bool):
    """Build a (kernel, program) pair with a hand-crafted 2-region
    ``tl.tileop.copy`` ``SemanticStmt`` annotated ``is_gather4`` /
    ``is_scatter4``, whose SHARED-side region is SLICED (extent 2 on a
    declared (4, 64) SHARED buffer) -- bypassing the ``T.tma_gather4`` /
    ``T.tma_scatter4`` frontend entirely.

    The real frontend can never reach this path: it only ever takes a WHOLE
    ``T.alloc_shared`` buffer for the SHARED side (never a
    ``BufferRegion``/slice), so the lowering-layer guard
    (``_require_full_buffer_region`` in ``_lower_gather4_scatter4``) can only
    be exercised this way, via a hand-crafted ``SemanticStmt``. The
    GLOBAL-side region uses the frontend's usual placeholder ``(4, 64)``
    shape; only the SHARED-side extent is validated by this lowering path.
    """
    from tilelang.tileir.semantic import SemanticBuffer, SemanticKernel, SemanticStmt, SemanticProgram, SemanticRegion

    src_buf = SemanticBuffer(name="src", shape=(64, 64), dtype="float32", scope="global" if is_gather else "shared")
    dst_buf = SemanticBuffer(name="dst", shape=(4, 64), dtype="float32", scope="shared" if is_gather else "global")

    ann_key = "annotation.is_gather4" if is_gather else "annotation.is_scatter4"
    shared_region_shape = (2, 64)  # sliced: declared SHARED dim is 4
    global_region_shape = (4, 64)  # the frontend's usual dummy shape
    src_region_shape = global_region_shape if is_gather else shared_region_shape
    dst_region_shape = shared_region_shape if is_gather else global_region_shape

    copy_stmt = SemanticStmt(
        kind="tile_op",
        attrs=(("op", "tl.tileop.copy"), (ann_key, "1")),
        regions=(
            SemanticRegion(buffer="src", access="read", indices=("0", "0"), shape=src_region_shape),
            SemanticRegion(buffer="dst", access="write", indices=("0", "0"), shape=dst_region_shape),
        ),
    )
    body = SemanticStmt(kind="seq", children=(copy_stmt,))
    kernel = SemanticKernel(
        name="gather4_sliced_shared_guard_test",
        grid=("1", "1", "1"),
        threads=("128", "1", "1"),
        alloc_buffers=(src_buf, dst_buf),
        body=body,
    )
    program = SemanticProgram(
        name="gather4_sliced_shared_guard_test_prog",
        params=(src_buf, dst_buf),
        global_alloc_buffers=(),
        kernels=(kernel,),
    )
    return kernel, program


@pytest.mark.parametrize("is_gather", [True, False], ids=["gather4", "scatter4"])
def test_tma_gather4_scatter4_lowering_rejects_sliced_shared_region(is_gather):
    """A sliced SHARED-side (4, K_box) tile region must raise at lowering.
    The frontend can never produce one (T.tma_gather4/T.tma_scatter4 always
    take a whole T.alloc_shared buffer), so this is a defensive guard
    exercised only via a hand-crafted SemanticStmt."""
    from tilelang.tileir.errors import TileIRLoweringNotImplementedError
    from tilelang.tileir.lowering.sem_to_ir import lower_kernel
    from tilelang.tileir.ir.builder import IRBuilder

    kernel, program = _hand_crafted_gather4_sliced_shared_kernel(is_gather=is_gather)
    builder = IRBuilder()
    with pytest.raises(TileIRLoweringNotImplementedError, match="(?i)partial|whole buffer"):
        lower_kernel(kernel, builder, program=program)


def _hand_crafted_copy_extra_regions_kernel():
    """Build a kernel with a hand-crafted 3-region ``tl.tileop.copy``
    ``SemanticStmt`` WITHOUT gather4/scatter4 annotation -- a general
    index-buffer gather/scatter form (src, dst, index_buffer) this backend
    does not support. The lowering guard must reject this with a clear
    message."""
    from tilelang.tileir.semantic import SemanticBuffer, SemanticKernel, SemanticStmt, SemanticProgram, SemanticRegion

    src_buf = SemanticBuffer(name="src", shape=(64, 64), dtype="float32", scope="global")
    dst_buf = SemanticBuffer(name="dst", shape=(4, 64), dtype="float32", scope="shared")
    idx_buf = SemanticBuffer(name="idx", shape=(4,), dtype="int32", scope="global")

    # 3-region copy: src (read), dst (write), idx (read for a general gather form).
    copy_stmt = SemanticStmt(
        kind="tile_op",
        attrs=(("op", "tl.tileop.copy"),),
        regions=(
            SemanticRegion(buffer="src", access="read", indices=("0", "0"), shape=(4, 64)),
            SemanticRegion(buffer="dst", access="write", indices=("0", "0"), shape=(4, 64)),
            SemanticRegion(buffer="idx", access="read", indices=("0",), shape=(4,)),
        ),
    )
    body = SemanticStmt(kind="seq", children=(copy_stmt,))
    kernel = SemanticKernel(
        name="copy_extra_regions_guard_test",
        grid=("1", "1", "1"),
        threads=("128", "1", "1"),
        alloc_buffers=(src_buf, dst_buf, idx_buf),
        body=body,
    )
    program = SemanticProgram(
        name="copy_extra_regions_guard_test_prog",
        params=(src_buf, dst_buf, idx_buf),
        global_alloc_buffers=(),
        kernels=(kernel,),
    )
    return kernel, program


def test_tma_copy_lowering_rejects_extra_regions_without_annotation():
    """A 3+-region ``tl.tileop.copy`` without gather4/scatter4 annotation
    must raise at lowering, naming the inconsistent region count. This guards
    against a general index-buffer gather/scatter form this backend does not
    support."""
    from tilelang.tileir.errors import TileIRLoweringError
    from tilelang.tileir.lowering.sem_to_ir import lower_kernel
    from tilelang.tileir.ir.builder import IRBuilder

    kernel, program = _hand_crafted_copy_extra_regions_kernel()
    builder = IRBuilder()
    with pytest.raises(TileIRLoweringError, match="3 regions"):
        lower_kernel(kernel, builder, program=program)


# ---------------------------------------------------------------------------
# Test: T.transpose (tl.tileop.transpose)
# ---------------------------------------------------------------------------


def _transpose_permute_jit(m: int, n: int):
    """SHARED -> SHARED transpose (ct.permute path): copy A GLOBAL->SHARED,
    T.transpose(A_s, At_s), copy At_s (SHARED, shape (n,m)) out to GLOBAL."""

    @tilelang.jit
    def make():
        @T.prim_func
        def main(
            A: T.Tensor((m, n), "float32"),
            Out: T.Tensor((n, m), "float32"),
        ):
            with T.Kernel(1, threads=128):
                A_s = T.alloc_shared((m, n), "float32")
                At_s = T.alloc_shared((n, m), "float32")
                T.copy(A, A_s)
                T.transpose(A_s, At_s)
                T.copy(At_s, Out)

        return main

    return make


def _transpose_strided_jit(m: int, n: int):
    """GLOBAL -> SHARED transpose (cuda_tile.make_strided_view path):
    T.transpose(A, At_s) directly on the GLOBAL source, then copy out."""

    @tilelang.jit
    def make():
        @T.prim_func
        def main(
            A: T.Tensor((m, n), "float32"),
            Out: T.Tensor((n, m), "float32"),
        ):
            with T.Kernel(1, threads=128):
                At_s = T.alloc_shared((n, m), "float32")
                T.transpose(A, At_s)
                T.copy(At_s, Out)

        return main

    return make


@skip_no_cuda_tile
def test_transpose_global_src_emits_strided_view():
    """A global-source transpose must use a strided view, not a partition view followed by a permute."""
    _skip_if_tileir_toolchain_unavailable()
    from tilelang.tileir.pipeline import build_tileir_module
    from tilelang.tileir.semantic import materialize_launch_nest
    from tilelang.tileir.launch import _split_grid_sync_primfunc

    pf = _transpose_strided_jit(32, 64).get_tir()
    pf = materialize_launch_nest(pf)
    pf = _split_grid_sync_primfunc(pf)
    mlir = str(build_tileir_module(pf, arch="sm_100"))
    assert "make_strided_view" in mlir


@skip_no_cuda_tile
@pytest.mark.parametrize("m,n", [(64, 64), (32, 64)], ids=["square", "nonsquare"])
def test_transpose_shared_to_shared_numerical(m, n):
    """ct.permute path: dst[j,i] = src[i,j] for a SHARED->SHARED transpose.
    Pure data movement (no floating-point arithmetic), so the comparison is
    bit-exact -- rtol=0, atol=0. Includes a non-square shape (32x64) since a
    square shape cannot distinguish a correct permute from a dim_map/shape
    mistake that happens to be self-consistent."""
    _skip_if_tileir_toolchain_unavailable()
    torch, *_ = _setup_gpu()

    kernel = tilelang.compile(_transpose_permute_jit(m, n).get_tir(), execution_backend="tileir")
    a = torch.randn(m, n, dtype=torch.float32, device="cuda")
    out = torch.empty(n, m, dtype=torch.float32, device="cuda")
    kernel(a, out)

    torch.testing.assert_close(out, a.t(), rtol=0, atol=0)


@skip_no_cuda_tile
@pytest.mark.parametrize("m,n", [(64, 64), (32, 64)], ids=["square", "nonsquare"])
def test_transpose_global_to_shared_numerical(m, n):
    """cuda_tile.make_strided_view path: dst[j,i] = src[i,j] for a
    GLOBAL->SHARED transpose. Same bit-exactness / non-square rationale as
    test_transpose_shared_to_shared_numerical above."""
    _skip_if_tileir_toolchain_unavailable()
    torch, *_ = _setup_gpu()

    kernel = tilelang.compile(_transpose_strided_jit(m, n).get_tir(), execution_backend="tileir")
    a = torch.randn(m, n, dtype=torch.float32, device="cuda")
    out = torch.empty(n, m, dtype=torch.float32, device="cuda")
    kernel(a, out)

    torch.testing.assert_close(out, a.t(), rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Test: T.transpose partial/dynamic-region guards -- covers a dynamic-extent
# region silently transposing the WHOLE buffer, a partial GLOBAL dst with no
# guard at all, an end-to-end test with a REAL partial region, and
# rank-2-buffers-only (rejected clearly).
# ---------------------------------------------------------------------------


def _mlir_text_for(make_fn) -> str:
    """Build the MLIR text for a jit factory's traced PrimFunc (the same
    "MLIR-text recipe" used by test_transpose_global_src_emits_strided_view
    above), for guard tests that only need to prove ``build_tileir_module``
    raises -- no GPU execution required."""
    from tilelang.tileir.pipeline import build_tileir_module
    from tilelang.tileir.semantic import materialize_launch_nest
    from tilelang.tileir.launch import _split_grid_sync_primfunc

    pf = make_fn.get_tir()
    pf = materialize_launch_nest(pf)
    pf = _split_grid_sync_primfunc(pf)
    return str(build_tileir_module(pf, arch="sm_100"))


@skip_no_cuda_tile
def test_transpose_static_partial_region_raises():
    """A REAL static partial GLOBAL-src region -- ``A[0:16, :]`` on a
    declared ``(32, 64)`` buffer, i.e. exactly half the rows -- must raise
    loudly at lowering time. A naive guard that compares the declared
    buffer shape against a region shape already squeezed/derived from
    ``_region_tile_shape`` would, for a region smaller than the buffer, NOT
    match -- but the general contract this test guards is: any region whose
    extent along a dim differs from the buffer's declared dim for that dim
    must be rejected, never silently transposed from a wrong sub-window."""
    _skip_if_tileir_toolchain_unavailable()
    from tilelang.tileir.errors import TileIRLoweringError

    @tilelang.jit
    def make():
        @T.prim_func
        def main(
            A: T.Tensor((32, 64), "float32"),
            Out: T.Tensor((64, 16), "float32"),
        ):
            with T.Kernel(1, threads=128):
                At_s = T.alloc_shared((64, 16), "float32")
                T.transpose(A[0:16, :], At_s)
                T.copy(At_s, Out)

        return main

    with pytest.raises(TileIRLoweringError, match="(?i)partial|full-buffer"):
        _mlir_text_for(make)


@skip_no_cuda_tile
def test_transpose_dynamic_extent_region_raises():
    """A DYNAMIC-extent region -- ``A[0:n, :]`` where ``n`` is a scalar
    ``T.int32`` kernel argument, not a compile-time constant -- must raise
    rather than being silently inflated to "the whole buffer" by
    ``_region_tile_shape``'s dynamic-dim fallback (e.g.
    ``_region_tile_shape(('seq_len - 0', 64), (128, 64)) == (128, 64)``).
    The frontend happily traces this region (unlike the rank-3 case, no
    ``T.transpose`` assertion catches it), so the lowering-time guard is the
    only place this can be caught."""
    _skip_if_tileir_toolchain_unavailable()
    from tilelang.tileir.errors import TileIRLoweringError

    @tilelang.jit
    def make():
        @T.prim_func
        def main(
            A: T.Tensor((32, 64), "float32"),
            n: T.int32,
            Out: T.Tensor((64, 32), "float32"),
        ):
            with T.Kernel(1, threads=128):
                At_s = T.alloc_shared((64, 32), "float32")
                T.transpose(A[0:n, :], At_s)
                T.copy(At_s, Out)

        return main

    with pytest.raises(TileIRLoweringError, match="(?i)dynamic-extent|partial|full-buffer"):
        _mlir_text_for(make)


@skip_no_cuda_tile
def test_transpose_rank3_buffer_raises():
    """A rank-3 buffer (e.g. a ``(1, M, N)`` BSHD-style leading batch
    dim) must be rejected with a clear "rank-N buffer" message, not a
    confusing unsqueezed-vs-squeezed shape mismatch.  ``T.transpose`` only
    asserts rank >= 2, so a rank-3 whole-buffer "transpose" traces fine at
    the frontend; the lowering guard is what must reject it."""
    _skip_if_tileir_toolchain_unavailable()
    from tilelang.tileir.errors import TileIRLoweringNotImplementedError

    @tilelang.jit
    def make():
        @T.prim_func
        def main(
            A: T.Tensor((1, 32, 64), "float32"),
            Out: T.Tensor((1, 64, 32), "float32"),
        ):
            with T.Kernel(1, threads=128):
                At_s = T.alloc_shared((1, 64, 32), "float32")
                T.transpose(A, At_s)
                T.copy(At_s, Out)

        return main

    with pytest.raises(TileIRLoweringNotImplementedError, match="rank-3"):
        _mlir_text_for(make)


# ---------------------------------------------------------------------------
# Test: memspace-combo coverage -- transpose SHARED -> GLOBAL (the permute
# path storing straight to a GLOBAL dst; the other tests above cover
# SHARED->SHARED and GLOBAL->SHARED). Pure data movement, so the assert is
# bit-exact (rtol=0, atol=0).
# ---------------------------------------------------------------------------


def _transpose_shared_to_global_jit(m: int, n: int):
    """SHARED -> GLOBAL transpose: the ``ct.permute`` (SHARED-src) path with
    ``_store_buffer_tile`` writing the permuted tile straight to a GLOBAL
    dst (partition-view store), rather than to another SHARED tile."""

    @tilelang.jit
    def make():
        @T.prim_func
        def main(
            A: T.Tensor((m, n), "float32"),
            Out: T.Tensor((n, m), "float32"),
        ):
            with T.Kernel(1, threads=128):
                A_s = T.alloc_shared((m, n), "float32")
                T.copy(A, A_s)
                T.transpose(A_s, Out)

        return main

    return make


@skip_no_cuda_tile
@pytest.mark.parametrize("m,n", [(64, 64), (32, 64)], ids=["square", "nonsquare"])
def test_transpose_shared_to_global_numerical(m, n):
    """SHARED -> GLOBAL transpose, dst[j,i] = src[i,j]. Bit-exact;
    includes a non-square shape for the same reason as the other transpose
    numeric tests (a square shape cannot distinguish a correct permute from
    a self-consistent dim_map/shape mistake)."""
    _skip_if_tileir_toolchain_unavailable()
    torch, *_ = _setup_gpu()

    kernel = tilelang.compile(_transpose_shared_to_global_jit(m, n).get_tir(), execution_backend="tileir")
    a = torch.randn(m, n, dtype=torch.float32, device="cuda")
    out = torch.empty(n, m, dtype=torch.float32, device="cuda")
    kernel(a, out)

    torch.testing.assert_close(out, a.t(), rtol=0, atol=0)
