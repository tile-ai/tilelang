"""Runtime tests for automatic OOB tail clamping in Ascend T.copy."""

import torch
import tilelang
import tilelang.testing
from tilelang import tvm
from tilelang.ascend import language as T
from tvm import tirx
from tvm.tirx.stmt_functor import post_order_visit


HIDDEN = 576
BLOCK_M = 128
BLOCK_K = 128
GEMM_TILE_M = 128
GEMM_TILE_N = 128
GEMM_TILE_K = 128
OPAQUE_OFFSET_TILE_ROWS = 8
OPAQUE_OFFSET_COLS = 128
OPAQUE_OFFSET_OUT_ROWS = 12
OPAQUE_OFFSET_ROW = OPAQUE_OFFSET_TILE_ROWS


@tilelang.jit()
def make_1d_dynamic_copy(hidden: int, block_k: int = 128):
    @T.prim_func
    def copy_1d(
        x: T.Tensor[(hidden,), T.float32],
        out: T.Tensor[(hidden,), T.float32],
    ) -> None:
        with T.Kernel(T.ceildiv(hidden, block_k)) as pid:
            x_ub = T.alloc_shared((block_k,), "float32")
            offset = pid * block_k

            T.copy(x[offset : offset + block_k], x_ub)
            T.copy(x_ub, out[offset : offset + block_k])

    return copy_1d


@tilelang.jit()
def make_2d_dynamic_inner_copy(hidden: int, block_m: int = 128, block_k: int = 128):
    @T.prim_func
    def copy_2d(
        x: T.Tensor[(block_m, hidden), T.float32],
        out: T.Tensor[(block_m, hidden), T.float32],
    ) -> None:
        with T.Kernel(T.ceildiv(hidden, block_k)) as pid:
            x_ub = T.alloc_shared((block_m, block_k), "float32")
            offset = pid * block_k

            T.copy(x[0:block_m, offset : offset + block_k], x_ub)
            T.copy(x_ub, out[0:block_m, offset : offset + block_k])

    return copy_2d


@tilelang.jit()
def make_opaque_offset_tail_store(use_alloc_var: bool):
    @T.prim_func
    def kernel(
        tile: T.Tensor[(OPAQUE_OFFSET_TILE_ROWS, OPAQUE_OFFSET_COLS), T.float32],
        out: T.Tensor[(OPAQUE_OFFSET_OUT_ROWS, OPAQUE_OFFSET_COLS), T.float32],
        row_offset: T.int32,
    ) -> None:
        with T.Kernel(1):
            tile_ub = T.alloc_shared((OPAQUE_OFFSET_TILE_ROWS, OPAQUE_OFFSET_COLS), T.float32)
            T.copy(tile, tile_ub)
            if use_alloc_var:
                offset = T.alloc_var(T.int32, init=row_offset)
                T.copy(tile_ub, out[offset : offset + OPAQUE_OFFSET_TILE_ROWS, :])
            else:
                T.copy(tile_ub, out[row_offset : row_offset + OPAQUE_OFFSET_TILE_ROWS, :])

    return kernel


def make_opaque_tile_aligned_gm_to_l1_kernel():
    @T.prim_func
    def kernel(
        A: T.Buffer((8192, 128), "bfloat16"),
        B: T.Buffer((256, 128), "bfloat16"),
        C: T.Buffer((256, 256), "float32"),
        tile_idx: T.int32,
    ):
        with T.Kernel(1):
            m_tile = T.alloc_var(T.int32, init=tile_idx)
            a_l1 = T.alloc_l1((256, 128), "bfloat16")
            b_l1 = T.alloc_l1((256, 128), "bfloat16")
            acc = T.alloc_l0c((256, 256), "float32")

            T.copy(A[m_tile * 256 : (m_tile + 1) * 256, :], a_l1)
            T.copy(B, b_l1)
            T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, C)

    return kernel


def make_transpose_gm_to_l1_oob_kernel(m: int, n: int, k: int, tile_m: int, tile_n: int, tile_k: int):
    """GEMM whose A is loaded transposed GM->L1 with a non-square L1 tile.

    A is stored as (K, M) and loaded with transpose=True into an (tile_m, tile_k)
    L1 tile. When tile_m != tile_k the transposed axes must be paired correctly
    during OOB tail clamping; otherwise the clamp compares/clamps the wrong axis.
    """
    m_tiles = (m + tile_m - 1) // tile_m
    n_tiles = (n + tile_n - 1) // tile_n
    k_tiles = (k + tile_k - 1) // tile_k

    @T.prim_func
    def kernel(
        A: T.Buffer((k, m), "bfloat16"),
        B: T.Buffer((n, k), "bfloat16"),
        C: T.Buffer((m, n), "float32"),
    ):
        with T.Kernel(m_tiles * n_tiles) as bx:
            m_tile = bx // n_tiles
            n_tile = bx % n_tiles
            a_l1 = T.alloc_l1((tile_m, tile_k), "bfloat16")
            b_l1 = T.alloc_l1((tile_n, tile_k), "bfloat16")
            a_l0 = T.alloc_l0a((tile_m, tile_k), "bfloat16")
            b_l0 = T.alloc_l0b((tile_n, tile_k), "bfloat16")
            acc = T.alloc_l0c((tile_m, tile_n), "float32")

            for kt in T.serial(k_tiles):
                T.copy(
                    A[kt * tile_k : (kt + 1) * tile_k, m_tile * tile_m : (m_tile + 1) * tile_m],
                    a_l1,
                    transpose=True,
                )
                T.copy(
                    B[n_tile * tile_n : (n_tile + 1) * tile_n, kt * tile_k : (kt + 1) * tile_k],
                    b_l1,
                )
                T.copy(a_l1, a_l0)
                T.copy(b_l1, b_l0)
                T.gemm(a_l0, b_l0, acc, transpose_B=True, clear_accum=(kt == 0))

            T.copy(
                acc,
                C[m_tile * tile_m : (m_tile + 1) * tile_m, n_tile * tile_n : (n_tile + 1) * tile_n],
            )

    return kernel


def make_bf16_gemm_oob_kernel(m: int, n: int, k: int):
    m_tiles = (m + GEMM_TILE_M - 1) // GEMM_TILE_M
    n_tiles = (n + GEMM_TILE_N - 1) // GEMM_TILE_N
    k_tiles = (k + GEMM_TILE_K - 1) // GEMM_TILE_K

    @T.prim_func
    def kernel(
        A: T.Buffer((m, k), "bfloat16"),
        B: T.Buffer((n, k), "bfloat16"),
        C: T.Buffer((m, n), "float32"),
    ):
        with T.Kernel(m_tiles * n_tiles) as bx:
            m_tile = bx // n_tiles
            n_tile = bx % n_tiles
            a_l1 = T.alloc_l1((GEMM_TILE_M, GEMM_TILE_K), "bfloat16")
            b_l1 = T.alloc_l1((GEMM_TILE_N, GEMM_TILE_K), "bfloat16")
            a_l0 = T.alloc_l0a((GEMM_TILE_M, GEMM_TILE_K), "bfloat16")
            b_l0 = T.alloc_l0b((GEMM_TILE_N, GEMM_TILE_K), "bfloat16")
            acc = T.alloc_l0c((GEMM_TILE_M, GEMM_TILE_N), "float32")

            for kt in T.serial(k_tiles):
                # Edge requests use full tiles; lowering clamps each dimension
                # to the remaining logical GM region.
                T.copy(
                    A[
                        m_tile * GEMM_TILE_M : (m_tile + 1) * GEMM_TILE_M,
                        kt * GEMM_TILE_K : (kt + 1) * GEMM_TILE_K,
                    ],
                    a_l1,
                )
                T.copy(
                    B[
                        n_tile * GEMM_TILE_N : (n_tile + 1) * GEMM_TILE_N,
                        kt * GEMM_TILE_K : (kt + 1) * GEMM_TILE_K,
                    ],
                    b_l1,
                )
                T.copy(a_l1, a_l0)
                T.copy(b_l1, b_l0)
                T.gemm(a_l0, b_l0, acc, transpose_B=True, clear_accum=(kt == 0))

            T.copy(
                acc,
                C[
                    m_tile * GEMM_TILE_M : (m_tile + 1) * GEMM_TILE_M,
                    n_tile * GEMM_TILE_N : (n_tile + 1) * GEMM_TILE_N,
                ],
            )

    return kernel


def make_dynamic_guard_multibuffer_kernel():
    @T.prim_func
    def kernel(
        src: T.Buffer((4, 16, 64), "float32"),
        out: T.Buffer((4, 16, 64), "float32"),
        meta: T.Buffer((2,), "int32"),
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((16, 64), "float32")
            group = T.int32(meta[0])
            rows = T.max(T.min(T.int32(meta[1]), 16), 0)
            for _ in T.serial(2):
                T.copy(src[group, 0:rows, :], ub[0:rows, :])
                T.copy(ub[0:rows, :], out[group, 0:rows, :])

    return kernel


def make_dual_copy_oob_kernel():
    @T.prim_func
    def kernel(out: T.Buffer((192, 128), "float32")):
        with T.Kernel(1):
            tile = T.alloc_shared((128, 128), "float32")
            a_l1 = T.alloc_l1((16, 16), "bfloat16")
            b_l1 = T.alloc_l1((16, 16), "bfloat16")
            accum = T.alloc_l0c((16, 16), "float32")
            T.dual_copy(tile, out[64:320, :])
            T.gemm(a_l1, b_l1, accum, transpose_B=True, clear_accum=True)

    return kernel


def test_dual_copy_is_rewritten_before_oob_clamping() -> None:
    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class CaptureCopyRegion:
        def run_after_pass(self, mod, info):
            if info.name not in {"tl.RewriteDualCopy", "tl.AscendInsertOOBPadding"}:
                return

            regions = []

            def visit(node):
                if not isinstance(node, tirx.Call) or getattr(node.op, "name", "") not in ("tl.tileop.ascend_copy", "tl.tileop.copy"):
                    return
                destination = node.args[1]
                load = destination.args[0]
                regions.append(
                    {
                        "annotations": set(node.annotations),
                        "row_min": str(load.indices[0]),
                        "row_extent": str(destination.args[2]),
                    }
                )

            func = next(func for func in mod.functions.values() if isinstance(func, tirx.PrimFunc))
            post_order_visit(func.body, visit)
            snapshots[info.name] = regions

    with tvm.transform.PassContext(instruments=[CaptureCopyRegion()]):
        source = tilelang.lower(make_dual_copy_oob_kernel(), target="ascend").kernel_source

    assert "__global__ __mix__(1, 2)" in source
    rewritten = snapshots["tl.RewriteDualCopy"]
    clamped = snapshots["tl.AscendInsertOOBPadding"]
    assert len(rewritten) == len(clamped) == 1
    assert not ({"dual_dst_ctl", "double"} & rewritten[0]["annotations"])
    assert not ({"dual_dst_ctl", "double"} & clamped[0]["annotations"])
    assert "sid" in rewritten[0]["row_min"]
    assert rewritten[0]["row_extent"] == "128"
    assert "sid" in clamped[0]["row_extent"]
    assert clamped[0]["row_extent"] != rewritten[0]["row_extent"]


def test_copy_oob_guards_are_lowered_after_auto_schedule() -> None:
    snapshots = {}
    wanted = {
        "tl.AscendInsertOOBPadding",
        "tl.AnnotateMultiBufferEligible",
        "tl.AutoSchedule",
        "tl.InsertSync",
        "tl.AscendLowerTileOp",
    }

    @tvm.ir.instrument.pass_instrument
    class CapturePassIR:
        def run_after_pass(self, mod, info):
            if info.name in wanted:
                snapshots[info.name] = mod.script()

    with tvm.transform.PassContext(opt_level=3, instruments=[CapturePassIR()]):
        tilelang.lower(make_dynamic_guard_multibuffer_kernel(), target="ascend")

    assert snapshots.keys() >= wanted

    def generated_guard_lines(pass_name):
        return [
            line.strip()
            for line in snapshots[pass_name].splitlines()
            if line.lstrip().startswith("if ") and ("group" in line or "rows" in line)
        ]

    # Neither the empty `rows` tile nor the fixed-axis `group` bounds check may
    # turn the first UB write into a conditional write before eligibility and
    # scheduling have run.
    assert not generated_guard_lines("tl.AscendInsertOOBPadding")
    assert not generated_guard_lines("tl.AnnotateMultiBufferEligible")
    assert not generated_guard_lines("tl.AutoSchedule")
    scheduled = snapshots["tl.AutoSchedule"]
    assert "T.ascend_set_flag" not in scheduled
    assert "T.ascend_wait_flag" not in scheduled
    synchronized = snapshots["tl.InsertSync"]
    assert "T.ascend_set_flag" in synchronized
    assert "T.ascend_wait_flag" in synchronized
    annotation_lines = [line for line in snapshots["tl.AnnotateMultiBufferEligible"].splitlines() if "multi_buffer_eligible" in line]
    assert any("ub" in line for line in annotation_lines), annotation_lines

    # LowerTileOp reconstructs one runtime predicate from the clamped semantic
    # extents. It must cover both reasons for an empty DMA.
    lowered_guards = generated_guard_lines("tl.AscendLowerTileOp")
    assert any("group" in line and "rows" in line for line in lowered_guards), lowered_guards


def test_gm_to_l1_tile_aligned_opaque_offset_elides_row_fill() -> None:
    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class CapturePassIR:
        def run_after_pass(self, mod, info):
            if info.name == "tl.AscendInsertOOBPadding":
                snapshots[info.name] = mod.script()

    with tvm.transform.PassContext(opt_level=3, instruments=[CapturePassIR()]):
        tilelang.lower(make_opaque_tile_aligned_gm_to_l1_kernel(), target="ascend")

    padded = snapshots["tl.AscendInsertOOBPadding"]
    assert "ascend_fill_l1" not in padded


def test_gm_to_l1_oob_fill_keeps_precise_region_until_lowering() -> None:
    snapshots = {}
    wanted = {
        "tl.AscendInsertOOBPadding",
        "tl.InsertSync",
        "tl.AscendLowerTileOp",
    }

    @tvm.ir.instrument.pass_instrument
    class CapturePassIR:
        def run_after_pass(self, mod, info):
            if info.name in wanted:
                snapshots[info.name] = mod.script()

    with tvm.transform.PassContext(opt_level=3, instruments=[CapturePassIR()]):
        tilelang.lower(make_bf16_gemm_oob_kernel(250, 250, 250), target="ascend")

    padded = snapshots["tl.AscendInsertOOBPadding"]
    assert padded.count("T.fill(T.region(") == 2
    assert "ascend_fill_l1" not in padded
    assert "a_l1[T.min(128, 250 - bx // 2 * 128), 0]" in padded
    assert "b_l1[T.min(128, 250 - bx % 2 * 128), 0]" in padded

    synchronized = snapshots["tl.InsertSync"]
    assert synchronized.count("T.fill(T.region(") == 2
    assert 'T.ascend_pipe_barrier("PIPE_MTE2")' not in synchronized

    lowered = snapshots["tl.AscendLowerTileOp"]
    assert "T.fill(" not in lowered
    assert lowered.count("T.ascend_fill_l1(") == 2


def test_copy_oob_1d_tail() -> None:
    device = "npu"

    x_1d = torch.arange(HIDDEN, dtype=torch.float32, device=device)
    out_1d = torch.empty_like(x_1d)
    make_1d_dynamic_copy(HIDDEN, BLOCK_K)(x_1d, out_1d)
    torch.npu.synchronize()
    torch.testing.assert_close(out_1d, x_1d)


def test_copy_oob_2d_inner_tail() -> None:
    device = "npu"

    x_2d = torch.arange(BLOCK_M * HIDDEN, dtype=torch.float32, device=device).reshape(BLOCK_M, HIDDEN)
    out_2d = torch.empty_like(x_2d)
    make_2d_dynamic_inner_copy(HIDDEN, BLOCK_M, BLOCK_K)(x_2d, out_2d)
    torch.npu.synchronize()
    torch.testing.assert_close(out_2d, x_2d)


def test_copy_oob_tail_with_opaque_nonnegative_min() -> None:
    tile = torch.arange(
        1,
        OPAQUE_OFFSET_TILE_ROWS * OPAQUE_OFFSET_COLS + 1,
        dtype=torch.float32,
        device="npu",
    ).reshape(OPAQUE_OFFSET_TILE_ROWS, OPAQUE_OFFSET_COLS)
    out_elements = OPAQUE_OFFSET_OUT_ROWS * OPAQUE_OFFSET_COLS

    # Both forms are opaque to the arithmetic analyzer, but the Ascend copy
    # contract guarantees their runtime values are non-negative.
    for use_alloc_var in (False, True):
        pool = torch.zeros(2 * out_elements, dtype=torch.float32, device="npu")
        out = pool[:out_elements].reshape(OPAQUE_OFFSET_OUT_ROWS, OPAQUE_OFFSET_COLS)
        neighbor = pool[out_elements:]

        make_opaque_offset_tail_store(use_alloc_var)(tile, out, OPAQUE_OFFSET_ROW)
        torch.npu.synchronize()

        valid_rows = OPAQUE_OFFSET_OUT_ROWS - OPAQUE_OFFSET_ROW
        torch.testing.assert_close(out[OPAQUE_OFFSET_ROW:], tile[:valid_rows])
        assert torch.count_nonzero(out[:OPAQUE_OFFSET_ROW]).item() == 0
        assert torch.count_nonzero(neighbor).item() == 0


def test_bf16_gemm_oob_multi_axis_tail() -> None:
    m, n, k = 250, 250, 250
    torch.manual_seed(42)
    a = torch.randn((m, k), dtype=torch.bfloat16, device="npu")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="npu")

    kernel = tilelang.compile(
        make_bf16_gemm_oob_kernel(m, n, k),
        out_idx=-1,
        target="ascend",
    )

    actual = kernel(a, b)
    torch.npu.synchronize()

    expected = a.float() @ b.float().T
    assert actual.shape == (m, n)

    for m_tile in range((m + GEMM_TILE_M - 1) // GEMM_TILE_M):
        for n_tile in range((n + GEMM_TILE_N - 1) // GEMM_TILE_N):
            m_slice = slice(m_tile * GEMM_TILE_M, min((m_tile + 1) * GEMM_TILE_M, m))
            n_slice = slice(n_tile * GEMM_TILE_N, min((n_tile + 1) * GEMM_TILE_N, n))
            torch.testing.assert_close(
                actual[m_slice, n_slice],
                expected[m_slice, n_slice],
                rtol=1e-2,
                atol=1e-2,
            )


def test_transpose_gm_to_l1_oob_nonsquare_tile() -> None:
    # Transposed GM->L1 load with a non-square L1 tile (tile_m != tile_k) and M/K
    # both OOB. Exercises transposed-axis pairing during OOB tail clamping.
    m, n, k = 200, 128, 100
    tile_m, tile_n, tile_k = 128, 128, 32
    torch.manual_seed(7)
    a_t = torch.randn((k, m), dtype=torch.bfloat16, device="npu")  # A stored as (K, M)
    b = torch.randn((n, k), dtype=torch.bfloat16, device="npu")

    kernel = tilelang.compile(
        make_transpose_gm_to_l1_oob_kernel(m, n, k, tile_m, tile_n, tile_k),
        out_idx=-1,
        target="ascend",
    )
    actual = kernel(a_t, b)
    torch.npu.synchronize()

    expected = a_t.float().T @ b.float().T  # A logical = (M, K); C = A @ B.T
    assert actual.shape == (m, n)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    tilelang.testing.main()
