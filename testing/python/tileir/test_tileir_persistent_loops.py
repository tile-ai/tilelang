"""TileIR persistent-loop lowering and numerical tests."""

import pytest
import tilelang
import tilelang.language as T

from tileir_test_utils import _setup_gpu, _skip_if_tileir_toolchain_unavailable, skip_no_cuda_tile


def _persistent_gemm_prim_func(M=128, N=128, K=128, block_M=64, block_N=64, block_K=64, sm_num=2, threads=128):
    """T.Persistent GEMM: examples/gemm/example_gemm_persistent.py:36-66
    (use_persistent_primitive=True path), shrunk and simplified to float32.

    Default shapes give a 2x2 (4-tile) grid with sm_num=2 CTAs, so each CTA's
    persistent loop iterates over 2 waves -- the loop body's break-guard
    fires (`counter >= stop`) after the 2nd iteration, and the loop's
    generated `T.loop_break()` mid-body guard (for wave-count padding) never
    fires for these particular shapes; see
    `test_persistent_primitive_gemm_numerical_uneven_grid` below for a shape
    where it does.
    """

    @T.prim_func
    def kern(A: T.Tensor((M, K), "float32"), B: T.Tensor((K, N), "float32"), C: T.Tensor((M, N), "float32")):
        m_blocks = T.ceildiv(M, block_M)
        n_blocks = T.ceildiv(N, block_N)

        with T.Kernel(sm_num, threads=threads) as block_id:
            A_shared = T.alloc_shared((block_M, block_K), "float32")
            B_shared = T.alloc_shared((block_K, block_N), "float32")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            C_shared = T.alloc_shared((block_M, block_N), "float32")

            # "pm"/"pn" (NOT "bx"/"by" as in the upstream example): see the
            # NAMING HAZARD note above.
            for pm, pn in T.Persistent([m_blocks, n_blocks], sm_num, block_id):
                T.clear(C_local)
                for k in T.serial(T.ceildiv(K, block_K)):
                    T.copy(A[pm * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, pn * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)
                T.copy(C_local, C_shared)
                T.copy(C_shared, C[pm * block_M, pn * block_N])

    return kern


def test_persistent_loop_break_is_recognized_by_semantic_layer():
    """Frontend-only sanity: tracing a T.Persistent kernel through the
    TileIR semantic layer must NOT raise TileLangSemanticError. Confirms
    `tl.loop_break` -- not just `tir.break_loop` -- is recognized as a
    `tile_op` semantic statement.
    """
    from tilelang.tileir.semantic import extract_semantic_program, materialize_launch_nest
    from tilelang.tileir.launch import _split_grid_sync_primfunc
    from tileir_test_utils import _semantic_stmts

    pf = _persistent_gemm_prim_func()
    pf = materialize_launch_nest(pf)
    pf = _split_grid_sync_primfunc(pf)
    program = extract_semantic_program(pf)

    break_stmts = [
        s for k in program.kernels for s in _semantic_stmts(k.body) if s.kind == "tile_op" and dict(s.attrs).get("op") == "tl.loop_break"
    ]
    assert len(break_stmts) == 1


@skip_no_cuda_tile
def test_persistent_primitive_loop_lowers_to_break_capable_loop_mlir():
    """Structural: a `for` loop whose body contains `T.loop_break()` (the
    outer wave loop `T.Persistent` generates) lowers to `cuda_tile.loop`
    (LoopOp), not `cuda_tile.for` (ForOp), because only LoopOp supports
    early termination.
    """
    _skip_if_tileir_toolchain_unavailable()
    from tilelang.tileir.pipeline import build_tileir_module
    from tilelang.tileir.semantic import materialize_launch_nest
    from tilelang.tileir.launch import _split_grid_sync_primfunc

    pf = _persistent_gemm_prim_func()
    pf = materialize_launch_nest(pf)
    pf = _split_grid_sync_primfunc(pf)
    module = build_tileir_module(pf, arch="sm_100")
    assert module.operation.verify()
    mlir = str(module)

    assert "loop iter_values" in mlir
    assert "break" in mlir


@skip_no_cuda_tile
@pytest.mark.parametrize(
    ("M", "N", "K", "sm_num"),
    [(128, 128, 128, 2), (192, 128, 128, 4)],
    ids=["even-grid", "uneven-grid"],
)
def test_persistent_primitive_gemm_numerical(M, N, K, sm_num):
    """Exercise both regular and wave-padded persistent GEMM grids."""
    _skip_if_tileir_toolchain_unavailable()
    torch, *_ = _setup_gpu()

    kernel = tilelang.compile(_persistent_gemm_prim_func(M=M, N=N, K=K, sm_num=sm_num), execution_backend="tileir")

    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    kernel(a, b, c)

    ref = a @ b
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=5e-2)


@skip_no_cuda_tile
def test_manual_grid_stride_persistent_gemm_numerical():
    """Manual grid-stride persistent GEMM: examples/gemm/example_gemm_persistent.py
    :67-81 (use_persistent_primitive=False path), shrunk to float32 / small
    shapes. This is a plain serial `for` + scalar index math + an `if` guard
    with no `break`, so it uses the ForOp path rather than the break-capable
    LoopOp path used by `T.Persistent`.

    Uses `group_size=1` (NOT the upstream example's hardcoded 8): with
    group_size=8 and this shrunk grid's n_blocks=2, the group-swizzle math
    degenerates (index `by` never exceeds `group_size`, so the `// group_size`
    term that should select a different super-group of `bx` never advances)
    and `bx` is stuck at 0 for every tile -- verified by hand this leaves
    half the C matrix unwritten. `group_size=8` is only valid for grids where
    it evenly organizes a LARGER `n_blocks` (as in the upstream example's
    4096x4096 default shapes, n_blocks=32); it is a user swizzle CHOICE
    unrelated to TileIR backend support.

    Compared with `rtol=1e-2, atol=5e-2` (TF32 MMA path) -- see the
    tolerance note on `test_persistent_primitive_gemm_numerical` above; not
    an exact/bit-identical comparison. This test's `bx`/`by` are an EAGER
    (non-deferred) let-bind -- unlike `T.Persistent`'s deferred/replayed
    `pm`/`pn` above -- so they are NOT subject to the reserved-launch-axis
    collision guard
    (`test_persistent_loop_reserved_axis_name_collision_raises` below): an
    eager Bind correctly shadows the reserved binding in the same lowering
    scope before any later reference, which is exactly why this test's
    `bx`/`by` naming has always produced correct results.
    """
    _skip_if_tileir_toolchain_unavailable()
    torch, *_ = _setup_gpu()

    M = N = K = 128
    block_M = block_N = block_K = 64
    sm_num = 2
    group_size = 1
    threads = 128

    m_blocks = (M + block_M - 1) // block_M
    n_blocks = (N + block_N - 1) // block_N
    waves = (m_blocks * n_blocks + sm_num - 1) // sm_num

    @T.prim_func
    def kern(A: T.Tensor((M, K), "float32"), B: T.Tensor((K, N), "float32"), C: T.Tensor((M, N), "float32")):
        with T.Kernel(sm_num, threads=threads) as block_id:
            A_shared = T.alloc_shared((block_M, block_K), "float32")
            B_shared = T.alloc_shared((block_K, block_N), "float32")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            C_shared = T.alloc_shared((block_M, block_N), "float32")

            for w in T.serial(waves):
                tile_id = sm_num * w + block_id
                bx = (tile_id // group_size) % m_blocks
                by = (tile_id % group_size) + (tile_id // group_size) // m_blocks * group_size

                if bx * block_M < M and by * block_N < N:
                    T.clear(C_local)
                    for k in T.serial(T.ceildiv(K, block_K)):
                        T.copy(A[bx * block_M, k * block_K], A_shared)
                        T.copy(B[k * block_K, by * block_N], B_shared)
                        T.gemm(A_shared, B_shared, C_local)
                    T.copy(C_local, C_shared)
                    T.copy(C_shared, C[bx * block_M, by * block_N])

    kernel = tilelang.compile(kern, execution_backend="tileir")

    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    kernel(a, b, c)

    ref = a @ b
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=5e-2)


def _persistent_gemm_prim_func_colliding_names(M=128, N=128, K=128, block_M=64, block_N=64, block_K=64, sm_num=2, threads=128):
    """Identical to `_persistent_gemm_prim_func` above except the
    `T.Persistent`-unpacked coordinates are named `bx`/`by` -- exactly the
    reserved launch-axis name collision this test exercises -- instead of
    the non-colliding `pm`/`pn`.
    """

    @T.prim_func
    def kern(A: T.Tensor((M, K), "float32"), B: T.Tensor((K, N), "float32"), C: T.Tensor((M, N), "float32")):
        m_blocks = T.ceildiv(M, block_M)
        n_blocks = T.ceildiv(N, block_N)

        with T.Kernel(sm_num, threads=threads) as block_id:
            A_shared = T.alloc_shared((block_M, block_K), "float32")
            B_shared = T.alloc_shared((block_K, block_N), "float32")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            C_shared = T.alloc_shared((block_M, block_N), "float32")

            # Deliberately colliding names: the 1-D grid launch reserves the
            # TIR variable names "bx"/"by"/"bz" independently of the Python
            # identifier `block_id` used above.
            for bx, by in T.Persistent([m_blocks, n_blocks], sm_num, block_id):
                T.clear(C_local)
                for k in T.serial(T.ceildiv(K, block_K)):
                    T.copy(A[bx * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, by * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)
                T.copy(C_local, C_shared)
                T.copy(C_shared, C[bx * block_M, by * block_N])

    return kern


@skip_no_cuda_tile
def test_persistent_loop_reserved_axis_name_collision_numerical():
    """`for bx, by in T.Persistent(...)` — the persistent tile coordinates are
    named `bx`/`by`, colliding with the canonical name TileLang gives the
    kernel's blockIdx.x/y. The lowering scope keys scalar bindings by TIR
    ``Var`` OBJECT IDENTITY, so the coordinate Vars and the block-axis Vars stay
    distinct even though they share a name."""
    _skip_if_tileir_toolchain_unavailable()
    torch, *_ = _setup_gpu()

    M = N = K = 128
    kernel = tilelang.compile(
        _persistent_gemm_prim_func_colliding_names(M=M, N=N, K=K),
        execution_backend="tileir",
    )
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    kernel(a, b, c)
    torch.testing.assert_close(c, a @ b, rtol=1e-2, atol=5e-2)


def _persistent_gemm_prim_func_non_reserved_name_collision(M=128, N=128, K=128, block_M=64, block_N=64, block_K=64, sm_num=2, threads=128):
    """Build a persistent GEMM whose eager scalar and coordinate share a name."""

    @T.prim_func
    def kern(A: T.Tensor((M, K), "float32"), B: T.Tensor((K, N), "float32"), C: T.Tensor((M, N), "float32")):
        m_blocks = T.ceildiv(M, block_M)
        n_blocks = T.ceildiv(N, block_N)

        with T.Kernel(sm_num, threads=threads) as block_id:
            A_shared = T.alloc_shared((block_M, block_K), "float32")
            B_shared = T.alloc_shared((block_K, block_N), "float32")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            C_shared = T.alloc_shared((block_M, block_N), "float32")

            foo = block_id * 2

            # The persistent coordinate shadows the eager scalar by name, but
            # the two bindings remain distinct TIR Vars.
            for foo, pn in T.Persistent([m_blocks, n_blocks], sm_num, block_id):
                T.clear(C_local)
                for k in T.serial(T.ceildiv(K, block_K)):
                    T.copy(A[foo * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, pn * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)
                T.copy(C_local, C_shared)
                T.copy(C_shared, C[foo * block_M, pn * block_N])

    return kern


@skip_no_cuda_tile
def test_persistent_loop_non_reserved_name_collision_numerical():
    """A T.Persistent coordinate named `foo` shadowing an earlier eager
    `foo = block_id * 2` let-bind resolves body references by TIR Var identity."""
    _skip_if_tileir_toolchain_unavailable()
    torch, *_ = _setup_gpu()

    M = N = K = 128
    kernel = tilelang.compile(
        _persistent_gemm_prim_func_non_reserved_name_collision(M=M, N=N, K=K),
        execution_backend="tileir",
    )
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    kernel(a, b, c)
    torch.testing.assert_close(c, a @ b, rtol=1e-2, atol=5e-2)


def _break_loop_carried_accumulator_prim_func(num_waves: int = 4, break_at: int = 3):
    """`acc` is read-before-write INSIDE a `for w in T.serial(...)`
    loop whose body also directly contains `T.loop_break()`. `Out[0] =
    acc[0]` is written EVERY iteration from INSIDE the loop body (not after
    it), so this specific shape is a pure "silent wrong answer", not the
    (also real, but LOUD -- an MLIR dominance error) live-out variant the
    tile-map snapshot/restore guard separately covers.
    """

    @T.prim_func
    def kern(Out: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=32):
            acc = T.alloc_fragment((1,), "float32")
            T.clear(acc)
            for w in T.serial(num_waves):
                if w >= break_at:
                    T.loop_break()
                acc[0] += 1.0
                Out[0] = acc[0]

    return kern


@skip_no_cuda_tile
def test_break_loop_carried_accumulator_raises():
    """A `for` loop with a body `break` that also loop-carries a genuine
    (read-before-write) SHARED/REGISTER accumulator must raise
    `_UnsupportedTileIRNode` at build time, not silently compile a
    wrong-answer kernel (see the section docstring above)."""
    _skip_if_tileir_toolchain_unavailable()
    from tilelang.tileir.pipeline import build_tileir_module
    from tilelang.tileir.semantic import materialize_launch_nest
    from tilelang.tileir.launch import _split_grid_sync_primfunc
    from tilelang.tileir.errors import _UnsupportedTileIRNode

    pf = _break_loop_carried_accumulator_prim_func()
    pf = materialize_launch_nest(pf)
    pf = _split_grid_sync_primfunc(pf)

    with pytest.raises(_UnsupportedTileIRNode, match="loop-carried"):
        build_tileir_module(pf, arch="sm_100")


def _no_break_loop_carried_accumulator_prim_func(num_waves: int = 3):
    """Accumulator case without `T.loop_break()`.

    The accumulation uses the plain ForOp path, which threads loop-carried
    tiles as iter-args via `loop_carry_pass`. It must compute `num_waves`
    correctly and keep the loop-carried-tile guard scoped to the break-capable
    LoopOp path.
    """

    @T.prim_func
    def kern(Out: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=32):
            acc = T.alloc_fragment((1,), "float32")
            T.clear(acc)
            for _w in T.serial(num_waves):
                acc[0] += 1.0
                Out[0] = acc[0]

    return kern


@skip_no_cuda_tile
def test_no_break_loop_carried_accumulator_numerical():
    _skip_if_tileir_toolchain_unavailable()
    torch, *_ = _setup_gpu()

    kernel = tilelang.compile(_no_break_loop_carried_accumulator_prim_func(num_waves=3), execution_backend="tileir")
    out = torch.zeros(1, device="cuda", dtype=torch.float32)
    kernel(out)
    torch.testing.assert_close(out, torch.full((1,), 3.0, device="cuda", dtype=torch.float32))


def _break_loop_global_write_prim_func(num_waves: int = 4, break_at: int = 2):
    """`for w in T.serial(num_waves): if w >= break_at: T.loop_break();
    T.copy(X, x_frag); T.copy(x_frag, C)` -- writing GLOBAL `C` inside a
    break-capable loop -- followed by `T.copy(C, D)` AFTER the loop, reading
    the buffer the loop wrote. `x_frag` is write-then-read SCRATCH freshly
    repopulated from `X` every iteration (the persistent-GEMM shape this
    loop form targets: stage-then-flush), so it needs no iter-arg threading
    of its own (per the class docstring) -- the only cross-loop-boundary
    dependency under test is `C`'s ordering token. If post-loop ordering on
    `C` is lost, `D` can read `C`'s pre-loop (uninitialized/garbage) contents
    instead of the last in-loop write.
    """

    @T.prim_func
    def kern(X: T.Tensor((1,), "float32"), C: T.Tensor((1,), "float32"), D: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=32):
            x_frag = T.alloc_fragment((1,), "float32")
            for w in T.serial(num_waves):
                if w >= break_at:
                    T.loop_break()
                T.copy(X, x_frag)
                T.copy(x_frag, C)
            T.copy(C, D)

    return kern


@skip_no_cuda_tile
def test_break_loop_global_write_lowers_with_token_iterargs_mlir():
    """Structural: the break-capable LoopOp for `_break_loop_global_write_prim_func`
    carries per-buffer ordering tokens (LAST_OP, LAST_STORE for the single
    written GLOBAL buffer `C`) as ADDITIONAL iter-args alongside the counter
    -- i.e. 3 iter-args total, not just the bare counter.
    """
    _skip_if_tileir_toolchain_unavailable()
    from tilelang.tileir.pipeline import build_tileir_module
    from tilelang.tileir.semantic import materialize_launch_nest
    from tilelang.tileir.launch import _split_grid_sync_primfunc

    pf = _break_loop_global_write_prim_func()
    pf = materialize_launch_nest(pf)
    pf = _split_grid_sync_primfunc(pf)
    mlir = str(build_tileir_module(pf, arch="sm_100"))

    assert "loop iter_values" in mlir
    assert "break" in mlir
    # The break-capable loop's iter_values signature carries 2 extra
    # `token` typed iter-args (LAST_OP/LAST_STORE for `C`) alongside the
    # `i32` counter -- i.e. 3 iter-args, not just the bare counter.
    assert "loop iter_values(" in mlir
    loop_sig = mlir.split("loop iter_values(", 1)[1].split(")", 1)[0]
    assert loop_sig.count(",") == 2, f"expected 3 iter-args (counter + 2 tokens), got signature: {loop_sig!r}"
    # Every `break` inside the loop must forward all 3 iter-args too.
    for line in mlir.splitlines():
        if "break %" in line:
            operands = line.split("break", 1)[1].split(":", 1)[0]
            assert operands.count(",") == 2, f"break did not forward all iter-args: {line!r}"
    # The post-loop uses of `C` must reference the loop's OWN result token
    # (e.g. `%3#1`), not a raw in-region SSA value -- confirms the
    # `ctx._op_token` redirect (not just `ctx._token_map`) took effect.
    assert "#1" in mlir


@skip_no_cuda_tile
def test_break_loop_global_write_numerical():
    """`D` must read the LAST value the break-capable loop
    actually wrote into `C` (from `X`), bit-exact -- pure copies, no
    arithmetic, so `torch.equal` (not just `assert_close`) is the right
    check."""
    _skip_if_tileir_toolchain_unavailable()
    torch, *_ = _setup_gpu()

    kernel = tilelang.compile(_break_loop_global_write_prim_func(num_waves=4, break_at=2), execution_backend="tileir")
    x = torch.full((1,), 7.0, device="cuda", dtype=torch.float32)
    c = torch.full((1,), -1.0, device="cuda", dtype=torch.float32)
    d = torch.zeros(1, device="cuda", dtype=torch.float32)
    kernel(x, c, d)

    assert torch.equal(d, x), f"expected D == X == 7.0 (bit-exact), got D={d}, C={c}"
