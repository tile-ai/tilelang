"""Split-K GEMM with inter-core sync and atomic reduction on Ascend.

The kernel uses AutoSchedule for intra-core pipe synchronization and keeps the
global split-K rendezvous explicit:

* ``bx % split_k`` selects a contiguous K partition.
* ``bx // split_k`` selects the output-tile worker group.
* Split 0 initializes the output with an ordinary L0C-to-GM copy.
* With ``deterministic=False``, all AICs rendezvous once before the remaining
  splits atomically add their L0C partial results through FixPipe.
* With ``deterministic=True``, split ``s`` writes in phase ``s`` and all AICs
  rendezvous after every phase, fixing the FP32 reduction order.

The global inter-core barrier is intentionally shared by all physical cores.
Consequently, every core must execute the same number of output-tile iterations.
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.profiler import do_bench


def gemm_splitk(M_DIM=8192, K_DIM=8192, N_DIM=8192, split_k=4, deterministic=False):
    """Build a BF16 x BF16 -> FP32 manual split-K GEMM.

    When ``deterministic`` is enabled, partial results are committed in
    increasing split-ID order with an inter-core barrier after every split.
    Otherwise, split 0 initializes the output and all other splits issue their
    atomic stores concurrently after one barrier.
    """

    NUM_BLOCKS = 32
    TILE_M = 256
    TILE_N = 256
    TILE_K = 256
    NUM_STAGES = 2
    INTER_CORE_FLAG = 0

    if split_k <= 1:
        raise ValueError(f"split_k must be greater than 1, got {split_k}")
    if NUM_BLOCKS % split_k != 0:
        raise ValueError(f"split_k must divide {NUM_BLOCKS}, got {split_k}")
    if M_DIM % TILE_M != 0 or N_DIM % TILE_N != 0:
        raise ValueError(f"M and N must be multiples of ({TILE_M}, {TILE_N}), got ({M_DIM}, {N_DIM})")
    if K_DIM % TILE_K != 0:
        raise ValueError(f"K must be a multiple of {TILE_K}, got {K_DIM}")

    M_TILES = M_DIM // TILE_M
    N_TILES = N_DIM // TILE_N
    K_TILES = K_DIM // TILE_K
    NUM_GROUPS = NUM_BLOCKS // split_k
    OUT_TILES = M_TILES * N_TILES
    if K_TILES % split_k != 0:
        raise ValueError(f"K tile count must be divisible by split_k, got K_TILES={K_TILES}, split_k={split_k}")
    if OUT_TILES % NUM_GROUPS != 0:
        raise ValueError(
            f"output tile count must be divisible by the number of worker groups, got OUT_TILES={OUT_TILES}, NUM_GROUPS={NUM_GROUPS}"
        )

    K_TILES_PER_SPLIT = K_TILES // split_k
    TILES_PER_GROUP = OUT_TILES // NUM_GROUPS
    WINDOW = min(4, M_TILES)
    MAIN_ROW = M_TILES // WINDOW - 1
    TAIL_WIN = M_TILES - MAIN_ROW * WINDOW

    @T.macro
    def aswt_swizzle(tile_idx):
        m_tile = T.alloc_var("int32")
        n_tile = T.alloc_var("int32")
        row_idx = tile_idx // N_TILES // WINDOW
        if row_idx < MAIN_ROW:
            m_tile = row_idx * WINDOW + tile_idx % WINDOW
            n_tile = (tile_idx // WINDOW) % N_TILES
        else:
            tail_idx = tile_idx - MAIN_ROW * WINDOW * N_TILES
            m_tile = MAIN_ROW * WINDOW + tail_idx % TAIL_WIN
            n_tile = (tail_idx // TAIL_WIN) % N_TILES
        if row_idx % 2 != 0:
            n_tile = N_TILES - 1 - n_tile
        return m_tile, n_tile

    @T.prim_func
    def main(
        X: T.Buffer((M_DIM, K_DIM), "bfloat16"),
        W: T.Buffer((N_DIM, K_DIM), "bfloat16"),
        C: T.Buffer((M_DIM, N_DIM), "float32"),
    ):
        with T.Kernel(NUM_BLOCKS) as bx:
            split_id = bx % split_k
            group_id = bx // split_k

            x_l1 = T.alloc_l1((TILE_M, TILE_K), "bfloat16")
            w_l1 = T.alloc_l1((TILE_N, TILE_K), "bfloat16")
            res = T.alloc_l0c((TILE_M, TILE_N), "float32")

            if split_id != 0:
                T.set_atomic("add", "float32")

            for out_tile in T.Serial(TILES_PER_GROUP):
                tile_idx = out_tile * NUM_GROUPS + group_id
                m_tile, n_tile = aswt_swizzle(tile_idx)

                for local_kt in T.Pipelined(K_TILES_PER_SPLIT, num_stages=NUM_STAGES):
                    kt = split_id * K_TILES_PER_SPLIT + local_kt
                    T.copy(
                        X[
                            m_tile * TILE_M : (m_tile + 1) * TILE_M,
                            kt * TILE_K : (kt + 1) * TILE_K,
                        ],
                        x_l1,
                    )
                    T.copy(
                        W[
                            n_tile * TILE_N : (n_tile + 1) * TILE_N,
                            kt * TILE_K : (kt + 1) * TILE_K,
                        ],
                        w_l1,
                    )
                    T.gemm(
                        x_l1,
                        w_l1,
                        res,
                        transpose_B=True,
                        clear_accum=(local_kt == 0),
                    )

                if deterministic:
                    with T.PerCoreTask():
                        # Commit one split per phase. Every AIC executes every
                        # barrier, so phase s+1 cannot start until split s's
                        # FixPipe store is globally visible.
                        for store_split in T.Serial(split_k):
                            if split_id == store_split:
                                T.copy(res, C[m_tile * TILE_M, n_tile * TILE_N])
                            T.ascend_sync_inter_arrive("PIPE_FIX", INTER_CORE_FLAG)
                            T.ascend_sync_inter_wait("PIPE_FIX", INTER_CORE_FLAG)
                else:
                    with T.PerCoreTask():
                        # Split 0 initializes C with a normal FixPipe store. Since
                        # the arrive is also queued on FixPipe, the global barrier
                        # cannot release until every group's initializing store is
                        # visible.
                        if split_id == 0:
                            T.copy(res, C[m_tile * TILE_M, n_tile * TILE_N])

                        T.ascend_sync_inter_arrive("PIPE_FIX", INTER_CORE_FLAG)
                        T.ascend_sync_inter_wait("PIPE_FIX", INTER_CORE_FLAG)

                        # The other splits only issue their atomic stores after
                        # split 0 has initialized every output tile in this wave.
                        if split_id != 0:
                            T.copy(res, C[m_tile * TILE_M, n_tile * TILE_N])

            if split_id != 0:
                T.set_atomic_none()

    return main


def ref_program(x, w):
    return x.float() @ w.float().T


# (M, N, K, split_k). These aligned shapes have too few 256 x 256 output
# tiles to occupy all 32 AICs without splitting K.
BENCHMARK_CASES = (
    (256, 256, 8192, 32),
    (256, 512, 8192, 16),
    (256, 1024, 8192, 8),
    (1024, 256, 8192, 8),
    (512, 512, 8192, 8),
    (512, 1024, 8192, 4),
)


def run_regression_perf(
    M_DIM=512,
    K_DIM=8192,
    N_DIM=512,
    split_k=8,
    deterministic=False,
):
    """Compile and benchmark one Split-K configuration, returning milliseconds."""
    import torch

    device = torch.device("npu")
    kernel = tilelang.compile(
        gemm_splitk(M_DIM, K_DIM, N_DIM, split_k, deterministic=deterministic),
        out_idx=-1,
    )
    x = torch.randn(M_DIM, K_DIM, dtype=torch.bfloat16, device=device)
    w = torch.randn(N_DIM, K_DIM, dtype=torch.bfloat16, device=device)
    kernel(x, w)
    torch.npu.synchronize()

    prof = do_bench(
        lambda: kernel(x, w),
        backend="msprof_detail",
        _n_warmup=30,
        _n_repeat=50,
    )
    flops = 2.0 * M_DIM * N_DIM * K_DIM
    print(f"    [deterministic={deterministic}] {prof.dur_us:.3f} us/iter  |  {prof.tflops(flops):.3f} TFLOPS")
    return prof.dur_ns / 1e6


if __name__ == "__main__":
    import torch

    DETERMINISTIC_MODES = (False, True)
    NUM_WARMUPS = 30
    NUM_REPEATS = 50
    PRINT_KERNEL_SOURCE = False

    device = torch.device("npu")
    summary = []

    for m_dim, n_dim, k_dim, split_k in BENCHMARK_CASES:
        print(f"\n=== M={m_dim}, N={n_dim}, K={k_dim}, split_k={split_k} ===")
        x = torch.randn(m_dim, k_dim, dtype=torch.bfloat16, device=device)
        w = torch.randn(n_dim, k_dim, dtype=torch.bfloat16, device=device)
        expected = ref_program(x, w)

        for deterministic in DETERMINISTIC_MODES:
            print(f"Compiling deterministic={deterministic} ...")
            kernel = tilelang.compile(
                gemm_splitk(m_dim, k_dim, n_dim, split_k, deterministic=deterministic),
                out_idx=-1,
            )
            if PRINT_KERNEL_SOURCE:
                print("\n--- Generated Ascend Source ---")
                print(kernel.get_kernel_source())

            c = kernel(x, w)
            torch.npu.synchronize()
            torch.testing.assert_close(c, expected, rtol=1e-2, atol=1e-2)
            print("Correctness check passed!")

            if deterministic:
                c_repeat = kernel(x, w)
                torch.npu.synchronize()
                assert torch.equal(c, c_repeat), "Deterministic split-K produced different results across runs"
                print("Determinism check passed!")

            prof = do_bench(
                lambda kernel=kernel, x=x, w=w: kernel(x, w),
                backend="msprof_detail",
                _n_warmup=NUM_WARMUPS,
                _n_repeat=NUM_REPEATS,
            )
            flops = 2.0 * m_dim * n_dim * k_dim
            tflops = prof.tflops(flops)
            print(
                f"{prof.dur_us:.3f} us/iter  |  {tflops:.3f} TFLOPS  |  "
                f"MAD {prof.aic_mad * 100:.3f}%  |  MTE2 {prof.aic_mte2 * 100:.3f}%  |  "
                f"MTE1 {prof.aic_mte1 * 100:.3f}%  |  FIX {prof.aic_fixpipe * 100:.3f}%  |  "
                f"Scalar {prof.aic_scalar * 100:.3f}%"
            )
            summary.append(
                (
                    m_dim,
                    n_dim,
                    k_dim,
                    split_k,
                    deterministic,
                    prof.dur_us,
                    tflops,
                    prof.aic_mad * 100,
                    prof.aic_mte2 * 100,
                    prof.aic_mte1 * 100,
                    prof.aic_fixpipe * 100,
                )
            )

        del x, w, expected
        torch.npu.synchronize()
        torch.npu.empty_cache()

    print("\n=== Summary ===")
    for m_dim, n_dim, k_dim, split_k, deterministic, dur_us, tflops, mad, mte2, mte1, fix in summary:
        print(
            f"M={m_dim:4d}, N={n_dim:4d}, K={k_dim:5d}, split_k={split_k:2d}, deterministic={str(deterministic):5s}  |  "
            f"{dur_us:7.3f} us  |  {tflops:7.3f} TFLOPS  |  MAD {mad:6.3f}%  |  "
            f"MTE2 {mte2:6.3f}%  |  MTE1 {mte1:6.3f}%  |  FIX {fix:6.3f}%"
        )
