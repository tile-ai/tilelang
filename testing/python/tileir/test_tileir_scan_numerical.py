"""TileIR scan lowering and numerical tests."""

import pytest
import tilelang
import tilelang.language as T

from tileir_test_utils import _setup_gpu, _skip_if_tileir_toolchain_unavailable, skip_no_cuda_tile


def _cummax_1d_prim_func(n: int = 256, reverse: bool = False):
    @T.prim_func
    def kern(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32")):
        with T.Kernel(1, threads=128):
            tile = T.alloc_shared((n,), "float32")
            T.copy(A, tile)
            T.cummax(src=tile, dim=0, reverse=reverse)
            T.copy(tile, B)

    return kern


def _cummax_2d_prim_func(m: int = 32, n: int = 64, dim: int = 1, reverse: bool = False, dtype: str = "float32"):
    @T.prim_func
    def kern(A: T.Tensor((m, n), dtype), B: T.Tensor((m, n), dtype)):
        with T.Kernel(1, threads=128):
            tile = T.alloc_shared((m, n), dtype)
            T.copy(A, tile)
            T.cummax(src=tile, dim=dim, reverse=reverse)
            T.copy(tile, B)

    return kern


def test_cummax_is_recognized_by_semantic_layer():
    """Frontend-only sanity: tracing a T.cummax kernel through the TileIR
    semantic layer must NOT raise TileLangSemanticError.
    """
    from tilelang.tileir.semantic import extract_semantic_program, materialize_launch_nest
    from tilelang.tileir.launch import _split_grid_sync_primfunc
    from tileir_test_utils import _semantic_stmts

    pf = _cummax_1d_prim_func(n=64)
    pf = materialize_launch_nest(pf)
    pf = _split_grid_sync_primfunc(pf)
    program = extract_semantic_program(pf)

    cummax_stmts = [
        s for k in program.kernels for s in _semantic_stmts(k.body) if s.kind == "tile_op" and dict(s.attrs).get("op") == "tl.tileop.cummax"
    ]
    assert len(cummax_stmts) == 1


@skip_no_cuda_tile
@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
def test_cummax_numerical_1d(reverse):
    _skip_if_tileir_toolchain_unavailable()
    torch, *_ = _setup_gpu()

    n = 256
    kernel = tilelang.compile(_cummax_1d_prim_func(n, reverse=reverse), execution_backend="tileir")

    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.empty(n, device="cuda", dtype=torch.float32)
    kernel(a, b)

    scan_input = torch.flip(a, dims=[0]) if reverse else a
    ref = torch.cummax(scan_input, dim=0).values
    if reverse:
        ref = torch.flip(ref, dims=[0])
    torch.testing.assert_close(b, ref, rtol=0, atol=0)


@skip_no_cuda_tile
def test_cummax_numerical_2d_axis1():
    _skip_if_tileir_toolchain_unavailable()
    torch, *_ = _setup_gpu()

    m, n = 32, 64
    kernel = tilelang.compile(_cummax_2d_prim_func(m, n, dim=1), execution_backend="tileir")

    a = torch.randn(m, n, device="cuda", dtype=torch.float32)
    b = torch.empty(m, n, device="cuda", dtype=torch.float32)
    kernel(a, b)

    ref = torch.cummax(a, dim=1).values
    torch.testing.assert_close(b, ref, rtol=0, atol=0)


@skip_no_cuda_tile
def test_cummax_numerical_2d_uint32_src():
    """uint32 src with values >= 2^31 exercises the unsigned cmp/identity path.

    This exercises the same unsigned-comparison invariant needed by plain
    `Reduce`'s "max" kind (see compute.py's `Reduce` docstring): Cumsum's
    "max" kind selects its identity and its ct.max comparison signedness from
    `self.dst.type.dtype.name`/`self.src.type.dtype.name`, which the TileIR
    type registry alias-collapses (uint32 -> int32). With SIGNED cmp (or a
    SIGNED-derived INT_MIN identity of -2^31 reinterpreted as the huge
    unsigned value 2^31), a top-bit-set (>= 2^31) value would lose to a
    top-bit-clear (< 2^31) value in the running max — this test mixes both
    classes per row so the correct (UNSIGNED) running max provably differs
    from what a hardcoded-SIGNED comparison would produce.

    Since torch.uint32 kernel-arg mapping may not be fully supported, we
    pass the bit patterns as int32 and compute the reference on the uint
    bit patterns.
    """
    _skip_if_tileir_toolchain_unavailable()
    torch, *_ = _setup_gpu()

    m, n = 16, 32
    # Construct: each row has n/2 top-bit-clear, n/2 top-bit-set, INTERLEAVED
    # (not grouped) so the running max must cross classes mid-scan.
    # Even positions: small clear values unique per row. Odd positions: set
    # values (>= 2^31) unique per row. The row's true (unsigned) cummax is
    # eventually dominated by the largest set value; a signed cummax would
    # instead be dominated by the largest clear value throughout.
    torch.manual_seed(0)
    vals_2d = torch.empty(m, n, device="cuda", dtype=torch.int64)
    half = n // 2
    for row in range(m):
        clear_vals = torch.arange(row * n, row * n + half, device="cuda", dtype=torch.int64)
        set_vals = torch.arange(row * half, row * half + half, device="cuda", dtype=torch.int64) + 2**31
        vals_2d[row, 0::2] = clear_vals
        vals_2d[row, 1::2] = set_vals

    has_clear = (vals_2d < 2**31).any(dim=1)
    has_set = (vals_2d >= 2**31).any(dim=1)
    assert (has_clear & has_set).all(), f"Row(s) missing a value class: clear={has_clear.tolist()}, set={has_set.tolist()}"

    a_uint_bits = vals_2d.to(torch.int32)

    kernel = tilelang.compile(_cummax_2d_prim_func(m, n, dim=1, dtype="uint32"), execution_backend="tileir")

    b_uint_bits = torch.empty(m, n, device="cuda", dtype=torch.int32)
    kernel(a_uint_bits, b_uint_bits)

    # Reference: compute on the actual uint bit patterns (torch.cummax on
    # int32 with bit patterns >= 2^31 would mis-order via signed comparison).
    a_uint64 = a_uint_bits.to(torch.int64) & 0xFFFFFFFF
    ref_uint64 = torch.cummax(a_uint64, dim=1).values
    ref = (ref_uint64 & 0xFFFFFFFF).to(torch.int32)
    torch.testing.assert_close(b_uint_bits, ref, rtol=0, atol=0)


@skip_no_cuda_tile
def test_cummax_lowers_to_scan_mlir():
    _skip_if_tileir_toolchain_unavailable()
    from tilelang.tileir.pipeline import build_tileir_module
    from tilelang.tileir.semantic import materialize_launch_nest
    from tilelang.tileir.launch import _split_grid_sync_primfunc

    pf = _cummax_1d_prim_func(n=64)
    pf = materialize_launch_nest(pf)
    pf = _split_grid_sync_primfunc(pf)
    mlir = str(build_tileir_module(pf, arch="sm_100"))

    assert "scan" in mlir


def test_lower_cummax_rejects_malformed_stmt():
    """A hand-built `tl.tileop.cummax` SemanticStmt with only 1 region (missing
    `dst`) must raise `TileIRLoweringError`, not warn-and-silently-skip."""
    import pytest as _pytest
    from tilelang.tileir.errors import TileIRLoweringError
    from tilelang.tileir.semantic import SemanticBuffer, SemanticKernel, SemanticStmt, SemanticProgram, SemanticRegion
    from tilelang.tileir.lowering.sem_to_ir import lower_kernel
    from tilelang.tileir.ir.builder import IRBuilder

    src_buf = SemanticBuffer(name="src", shape=(32,), dtype="float32", scope="shared")

    cummax_stmt = SemanticStmt(
        kind="tile_op",
        attrs=(("op", "tl.tileop.cummax"), ("dim", "0"), ("reverse", "0")),
        regions=(SemanticRegion(buffer="src", access="read", indices=(), shape=(32,)),),
    )
    body = SemanticStmt(kind="seq", children=(cummax_stmt,))
    kernel = SemanticKernel(
        name="cummax_guard_test",
        grid=("1", "1", "1"),
        threads=("128", "1", "1"),
        alloc_buffers=(src_buf,),
        body=body,
    )
    program = SemanticProgram(
        name="cummax_guard_test_prog",
        params=(),
        global_alloc_buffers=(),
        kernels=(kernel,),
    )

    builder = IRBuilder()
    with _pytest.raises(TileIRLoweringError, match="2 regions"):
        lower_kernel(kernel, builder, program=program)
