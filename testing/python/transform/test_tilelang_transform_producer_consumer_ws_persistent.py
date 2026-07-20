"""Repro/regression tests for ProducerConsumerWS on persistent pipelines.

The ``ProducerConsumerWarpSpecialized`` pass historically failed on any
persistent kernel whose top-level structure interposes a ``ForNode`` between
the block body and the inner ``T.Pipelined`` K-loop.

Two idiomatic TileLang patterns trigger this:

1. **``T.Persistent`` primitive** (see ``examples/gemm/example_gemm_persistent.py``)::

       for by, bx in T.Persistent([...], sm_num, block_id):
           for ko in T.Pipelined(num_k, num_stages=...):
               ...

   ``T.Persistent`` expands to ``SeqStmt(Bind, ..., For(kSerial, ...))``
   with the inner ``T.Pipelined`` sitting under an ``IfThenElse`` inside
   the ``For``'s body.

2. **Hand-written ``T.serial`` tile scheduler** (the ``use_persistent_primitive=False``
   fallback in the same example)::

       for tile_id in T.serial(total_tiles):
           for ko in T.Pipelined(num_k, num_stages=...):
               ...
"""

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm as tvm
from tilelang.backend.target import determine_target


def persistent_matmul_primitive(
    M=512,
    N=512,
    K=256,
    block_M=64,
    block_N=64,
    block_K=32,
    num_stages=2,
    dtype="float16",
    accum_dtype="float32",
    threads=128,
    sm_num=4,
):

    @T.prim_func
    def main(
        A: T.Buffer((M, K), dtype),
        B: T.Buffer((K, N), dtype),
        C: T.Buffer((M, N), dtype),
    ):
        with T.Kernel(sm_num, threads=threads) as (block_id,):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            for by, bx in T.Persistent(
                [T.ceildiv(M, block_M), T.ceildiv(N, block_N)],
                sm_num,
                block_id,
            ):
                T.clear(C_local)
                for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[by * block_M, ko * block_K], A_shared)
                    T.copy(B[ko * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)
                T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def persistent_matmul_serial(
    M=256,
    N=256,
    K=256,
    block_M=64,
    block_N=64,
    block_K=32,
    num_stages=2,
    dtype="float16",
    threads=128,
):
    num_tiles_m = (M + block_M - 1) // block_M
    num_tiles_n = (N + block_N - 1) // block_N
    total_tiles = num_tiles_m * num_tiles_n
    num_k = (K + block_K - 1) // block_K

    @T.prim_func
    def main(
        A: T.Buffer((M, K), dtype),
        B: T.Buffer((K, N), dtype),
        C: T.Buffer((M, N), dtype),
    ):
        with T.Kernel(1, threads=threads) as _:
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), "float32")

            for tile_id in T.serial(total_tiles):
                by = tile_id // num_tiles_n
                bx = tile_id % num_tiles_n
                T.clear(C_local)
                for ko in T.Pipelined(num_k, num_stages=num_stages):
                    T.copy(A[by * block_M, ko * block_K], A_shared)
                    T.copy(B[ko * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)
                T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def persistent_matmul_serial_dynamic_bound(
    M=256,
    N=256,
    K=256,
    block_M=64,
    block_N=64,
    block_K=32,
    num_stages=2,
    dtype="float16",
    threads=128,
):
    """``T.serial`` persistent GEMM whose outer bound is symbolic.

    Guards against the fix accidentally depending on ``IntImm`` extents.
    """
    num_k = (K + block_K - 1) // block_K
    num_tiles_m_expr = T.ceildiv(M, block_M)
    num_tiles_n_expr = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Buffer((M, K), dtype),
        B: T.Buffer((K, N), dtype),
        C: T.Buffer((M, N), dtype),
    ):
        with T.Kernel(1, threads=threads) as _:
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), "float32")

            for tile_id in T.serial(num_tiles_m_expr * num_tiles_n_expr):
                by = tile_id // num_tiles_n_expr
                bx = tile_id % num_tiles_n_expr
                T.clear(C_local)
                for ko in T.Pipelined(num_k, num_stages=num_stages):
                    T.copy(A[by * block_M, ko * block_K], A_shared)
                    T.copy(B[ko * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)
                T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def persistent_matmul_misaligned_k(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    num_stages,
    sm_num,
    dtype="float16",
    threads=128,
):
    """``T.Persistent`` GEMM whose K-trip is *not* a multiple of ``2 * num_stages``.

    Regression for reoLantern's report on PR #2674: the block-scoped
    mbarrier phase persists across outer persistent iterations, but the
    original fix drove barrier parity purely from the inner ``k`` index,
    so any wave after the first produced wrong results (or launch
    failures) whenever ``trip % (2 * num_stages) != 0``.
    """

    @T.prim_func
    def main(
        A: T.Buffer((M, K), dtype),
        B: T.Buffer((K, N), dtype),
        C: T.Buffer((M, N), dtype),
    ):
        with T.Kernel(sm_num, threads=threads) as (block_id,):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), "float32")

            for bx, by in T.Persistent(
                [T.ceildiv(M, block_M), T.ceildiv(N, block_N)],
                sm_num,
                block_id,
            ):
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[bx * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, by * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)
                T.copy(C_local, C[bx * block_M, by * block_N])

    return main


def persistent_matmul_guarded_body(
    guard_kind: str,
    M=64,
    N=64,
    K=128,
    block_K=32,
    num_stages=2,
    dtype="float16",
    threads=128,
    num_tiles=2,
):
    """Two-tile persistent matmul whose inner K-loop is conditionally guarded.

    ``guard_kind == "lt"`` uses ``if k < 2`` (Lyscoria repro #1: the
    ``PhaseCounter`` must survive across outer iterations so tile 1
    does not desync from the block-scoped mbarrier phase).

    ``guard_kind == "eq"`` uses ``if k == 0 or k == 2`` (Lyscoria repro
    #2: shared-buffer versioning inside the guarded body uses a
    MultiVersionBuffer-supplied compound stage expression, and it must
    be rewritten to the same counter that drives barrier parity).
    """
    assert guard_kind in ("lt", "eq")

    @T.prim_func
    def main(
        A: T.Buffer((num_tiles, M, K), dtype),
        B: T.Buffer((num_tiles, K, N), dtype),
        C: T.Buffer((num_tiles, M, N), dtype),
    ):
        with T.Kernel(1, threads=threads) as (block_id,):
            A_shared = T.alloc_shared((M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, N), dtype)
            C_local = T.alloc_fragment((M, N), "float32")

            for tile, _ in T.Persistent([num_tiles, 1], 1, block_id):
                T.clear(C_local)
                for k in T.Pipelined(K // block_K, num_stages=num_stages):
                    if guard_kind == "lt":
                        if k < 2:
                            T.copy(A[tile, 0, k * block_K], A_shared)
                            T.copy(B[tile, k * block_K, 0], B_shared)
                            T.gemm(A_shared, B_shared, C_local)
                    else:
                        if k == 0 or k == 2:
                            T.copy(A[tile, 0, k * block_K], A_shared)
                            T.copy(B[tile, k * block_K, 0], B_shared)
                            T.gemm(A_shared, B_shared, C_local)
                T.copy(C_local, C[tile, 0, 0])

    return main


def persistent_matmul_variable_stages(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    num_stages,
    sm_num,
    dtype="float16",
    threads=128,
):
    """``T.Persistent`` GEMM parameterised by ``num_stages``.

    Used to stress the phase-counter machinery at ``num_stages == 1``
    (where ``PhaseCounter::StageExpr`` short-circuits to zero and only
    ``ParityExpr`` drives barrier phases) and at ``num_stages >= 3``
    (where the counter must survive multiple outer waves without
    aliasing the block-scoped mbarrier phase).
    """

    @T.prim_func
    def main(
        A: T.Buffer((M, K), dtype),
        B: T.Buffer((K, N), dtype),
        C: T.Buffer((M, N), dtype),
    ):
        with T.Kernel(sm_num, threads=threads) as (block_id,):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), "float32")

            for by, bx in T.Persistent(
                [T.ceildiv(M, block_M), T.ceildiv(N, block_N)],
                sm_num,
                block_id,
            ):
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)
                T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def persistent_matmul_guarded_deep_pipeline(
    M=64,
    N=64,
    K=192,
    block_K=32,
    num_stages=3,
    dtype="float16",
    threads=128,
    num_tiles=3,
):
    """Persistent + guarded body + deep (num_stages=3) pipeline combined.

    Stresses that the counter-based stage clock stays synchronised with
    the block-scoped mbarrier when all three features overlap:
      * multiple outer waves (Lyscoria repro #1 territory),
      * MVB-emitted compound stage expressions (Lyscoria repro #2),
      * num_stages >= 3 with parity flipping every ``num_stages`` steps.
    """

    @T.prim_func
    def main(
        A: T.Buffer((num_tiles, M, K), dtype),
        B: T.Buffer((num_tiles, K, N), dtype),
        C: T.Buffer((num_tiles, M, N), dtype),
    ):
        with T.Kernel(1, threads=threads) as (block_id,):
            A_shared = T.alloc_shared((M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, N), dtype)
            C_local = T.alloc_fragment((M, N), "float32")

            for tile, _ in T.Persistent([num_tiles, 1], 1, block_id):
                T.clear(C_local)
                for k in T.Pipelined(K // block_K, num_stages=num_stages):
                    if k < 5:
                        T.copy(A[tile, 0, k * block_K], A_shared)
                        T.copy(B[tile, k * block_K, 0], B_shared)
                        T.gemm(A_shared, B_shared, C_local)
                T.copy(C_local, C[tile, 0, 0])

    return main


def _apply_ws_pass(func):
    func = func.with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(func)
    target = determine_target({"kind": "cuda", "arch": "sm_90"}, return_object=True)
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    return tilelang.cuda.transform.ProducerConsumerWarpSpecialized()(mod)


def test_ws_pass_applies_under_t_persistent_primitive():
    """WS must descend through the ``For`` node synthesized by ``T.Persistent``."""
    func = persistent_matmul_primitive()
    mod = _apply_ws_pass(func)
    script = mod["main"].script()
    assert "tl_tiled_ws_applied" in script


def test_ws_pass_applies_under_persistent_serial():
    """Same coverage gap exercised via a hand-written ``T.serial`` scheduler."""
    func = persistent_matmul_serial()
    mod = _apply_ws_pass(func)
    script = mod["main"].script()
    assert "tl_tiled_ws_applied" in script


def test_ws_pass_applies_under_persistent_dynamic_bound():
    """The outer ``T.serial`` bound may be symbolic; the fix must not depend on IntImm."""
    func = persistent_matmul_serial_dynamic_bound()
    mod = _apply_ws_pass(func)
    script = mod["main"].script()
    assert "tl_tiled_ws_applied" in script


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_ws_end_to_end_t_persistent_matmul_correctness():
    import torch

    M, N, K = 256, 256, 256
    func = persistent_matmul_primitive(M, N, K, 64, 64, 32, num_stages=2, sm_num=4)
    kernel = tilelang.compile(func, target=determine_target(), out_idx=[2])
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(K, N, dtype=torch.float16, device="cuda")
    C = kernel(A, B)
    ref = A.float() @ B.float()
    torch.testing.assert_close(C.float(), ref, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_ws_end_to_end_persistent_serial_matmul_correctness():
    import torch

    M, N, K = 256, 256, 256
    func = persistent_matmul_serial(M, N, K, 64, 64, 32, num_stages=2)
    kernel = tilelang.compile(func, target=determine_target(), out_idx=[2])
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(K, N, dtype=torch.float16, device="cuda")
    C = kernel(A, B)
    ref = A.float() @ B.float()
    torch.testing.assert_close(C.float(), ref, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_ws_persistent_misaligned_k_matches_wsoff_reference():
    """Every wave must be correct even when ``trip % (2 * num_stages) != 0``.

    Comparing against the non-WS reference (rather than a math oracle)
    keeps the tolerance tight and pinpoints WS-specific regressions.
    """
    import torch

    # sm_num=2 with 4 tiles => 2 waves, so mbarrier phase must survive the
    # wave boundary.  trip=3 with num_stages=2 gives trip % 4 == 3, i.e.
    # a phase misalignment that would silently corrupt wave 2 under the
    # original fix.
    sm_num = 2
    M = N = 128
    block_M = block_N = 64
    block_K = 64
    K = 3 * block_K  # trip = 3, 2*num_stages = 4, trip % 4 == 3
    num_stages = 2

    func = persistent_matmul_misaligned_k(M, N, K, block_M, block_N, block_K, num_stages, sm_num=sm_num)

    # Sanity: WS must actually be applied to this func, otherwise the
    # comparison below degenerates into wsoff-vs-wsoff and would happily
    # pass even if the pass silently declined the candidate.
    ws_mod = _apply_ws_pass(func)
    assert "tl_tiled_ws_applied" in ws_mod["main"].script()

    ref_kernel = tilelang.compile(
        func,
        target=determine_target(),
        out_idx=[2],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    ws_kernel = tilelang.compile(
        func,
        target=determine_target(),
        out_idx=[2],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False},
    )
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.125
    B = torch.randn(K, N, dtype=torch.float16, device="cuda") * 0.125
    ref = ref_kernel(A, B).float()
    got = ws_kernel(A, B).float()
    torch.testing.assert_close(got, ref, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_ws_persistent_guarded_lt_matches_wsoff_reference():
    """``if k < 2`` guard: ``PhaseCounter`` must persist across outer tiles."""
    import torch

    func = persistent_matmul_guarded_body("lt")
    # Reference: non-WS compilation of the same source func.
    ref_kernel = tilelang.compile(
        func,
        target=determine_target(),
        out_idx=[2],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    ws_kernel = tilelang.compile(
        func,
        target=determine_target(),
        out_idx=[2],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False},
    )
    torch.manual_seed(0)
    A = torch.randn(2, 64, 128, dtype=torch.float16, device="cuda") * 0.125
    B = torch.randn(2, 128, 64, dtype=torch.float16, device="cuda") * 0.125
    ref = ref_kernel(A, B).float()
    got = ws_kernel(A, B).float()
    torch.testing.assert_close(got, ref, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_ws_persistent_guarded_eq_matches_wsoff_reference():
    """``if k == 0 or k == 2`` guard: MVB compound stage expression must
    be rewritten to the same counter as the barrier parity.
    """
    import torch

    func = persistent_matmul_guarded_body("eq")
    ref_kernel = tilelang.compile(
        func,
        target=determine_target(),
        out_idx=[2],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    ws_kernel = tilelang.compile(
        func,
        target=determine_target(),
        out_idx=[2],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False},
    )
    torch.manual_seed(0)
    A = torch.randn(2, 64, 128, dtype=torch.float16, device="cuda") * 0.125
    B = torch.randn(2, 128, 64, dtype=torch.float16, device="cuda") * 0.125
    ref = ref_kernel(A, B).float()
    got = ws_kernel(A, B).float()
    torch.testing.assert_close(got, ref, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_ws_persistent_num_stages_one():
    """``num_stages == 1`` exercises the ``StageExpr`` short-circuit branch.

    With a single pipeline stage the counter's ``StageExpr`` is a
    compile-time zero, so barrier parity is driven purely by
    ``ParityExpr`` (``floormod(Load, 2)``).  Verify that outer-wave
    correctness still holds under this degenerate pipeline depth.
    """
    import torch

    sm_num = 2
    M = N = 128
    block_M = block_N = 64
    block_K = 64
    K = 4 * block_K
    func = persistent_matmul_variable_stages(M, N, K, block_M, block_N, block_K, num_stages=1, sm_num=sm_num)
    kernel = tilelang.compile(func, target=determine_target(), out_idx=[2])
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.125
    B = torch.randn(K, N, dtype=torch.float16, device="cuda") * 0.125
    C = kernel(A, B)
    ref = A.float() @ B.float()
    torch.testing.assert_close(C.float(), ref, rtol=5e-2, atol=5e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_ws_persistent_deep_pipeline_num_stages_three():
    """``num_stages == 3`` deep pipeline across multiple persistent waves.

    Deep pipelines flip parity every ``num_stages`` iterations rather
    than every iteration, so an off-by-one in ``ParityExpr`` or a
    counter reset across outer tiles would immediately corrupt wave 2.
    """
    import torch

    sm_num = 2
    M = N = 128
    block_M = block_N = 64
    block_K = 64
    # trip = 5, 2*num_stages = 6 => phase-misaligned across waves.
    K = 5 * block_K
    func = persistent_matmul_variable_stages(M, N, K, block_M, block_N, block_K, num_stages=3, sm_num=sm_num)
    kernel = tilelang.compile(func, target=determine_target(), out_idx=[2])
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.125
    B = torch.randn(K, N, dtype=torch.float16, device="cuda") * 0.125
    C = kernel(A, B)
    ref = A.float() @ B.float()
    torch.testing.assert_close(C.float(), ref, rtol=5e-2, atol=5e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_ws_persistent_guarded_deep_pipeline_matches_wsoff_reference():
    """Triple-stress: persistent outer + guarded inner + deep pipeline.

    Compares the WS build against the non-WS fallback of the *same*
    source function so any counter/parity/version desynchronisation
    surfaces as a bit-level mismatch rather than a math-oracle
    tolerance breach.
    """
    import torch

    func = persistent_matmul_guarded_deep_pipeline()
    ref_kernel = tilelang.compile(
        func,
        target=determine_target(),
        out_idx=[2],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    ws_kernel = tilelang.compile(
        func,
        target=determine_target(),
        out_idx=[2],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False},
    )
    torch.manual_seed(0)
    A = torch.randn(3, 64, 192, dtype=torch.float16, device="cuda") * 0.125
    B = torch.randn(3, 192, 64, dtype=torch.float16, device="cuda") * 0.125
    ref = ref_kernel(A, B).float()
    got = ws_kernel(A, B).float()
    torch.testing.assert_close(got, ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    tilelang.testing.main()
