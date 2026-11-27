"""Tests for T.transpose shared memory transpose primitive."""

import tilelang
import tilelang.testing
import tilelang.language as T
import torch


def tilelang_transpose(M, N, block_M, block_N, dtype=T.float16):
    """Kernel: read tile from A into shared, transpose in shared, write to B.

    A is (M, N), B is (M, N).
    B = A.T.T = A when block_M == M and block_N == N (single tile).
    Actually: we read A tile (block_M, block_N) into shared,
    transpose to (block_N, block_M) in shared, then write to B
    so B[bx*block_N + j, by*block_M + i] = A[by*block_M + i, bx*block_N + j]
    i.e., B = A.T
    """

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((N, M), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            tile = T.alloc_shared((block_M, block_N), dtype)
            tile_T = T.alloc_shared((block_N, block_M), dtype)

            # Load from global to shared
            T.copy(
                A[by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N],
                tile,
            )
            # Transpose in shared memory
            T.transpose(tile, tile_T)
            # Store transposed tile back to global
            T.copy(
                tile_T,
                B[bx * block_N : (bx + 1) * block_N, by * block_M : (by + 1) * block_M],
            )

    return main


def run_tilelang_transpose(M=128, N=128, block_M=128, block_N=128, dtype=T.float16):
    program = tilelang_transpose(M, N, block_M, block_N, dtype)
    kernel = tilelang.compile(
        program,
        out_idx=[1],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    a = torch.randn(M, N, device="cuda", dtype=getattr(torch, dtype))
    b = kernel(a)
    expected = a.T
    torch.testing.assert_close(b, expected, rtol=1e-2, atol=1e-2)
    print(f"PASS: transpose M={M}, N={N}, block_M={block_M}, block_N={block_N}")


def tilelang_transpose_square(M, block_M, dtype=T.float16):
    """Simpler test: square transpose with single tile."""

    @T.prim_func
    def main(
        A: T.Tensor((M, M), dtype),
        B: T.Tensor((M, M), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(M, block_M), threads=128) as (bx, by):
            tile = T.alloc_shared((block_M, block_M), dtype)
            tile_T = T.alloc_shared((block_M, block_M), dtype)

            T.copy(
                A[by * block_M : (by + 1) * block_M, bx * block_M : (bx + 1) * block_M],
                tile,
            )
            T.transpose(tile, tile_T)
            T.copy(
                tile_T,
                B[bx * block_M : (bx + 1) * block_M, by * block_M : (by + 1) * block_M],
            )

    return main


def run_tilelang_transpose_square(M=256, block_M=128, dtype=T.float16):
    program = tilelang_transpose_square(M, block_M, dtype)
    kernel = tilelang.compile(
        program,
        out_idx=[1],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    a = torch.randn(M, M, device="cuda", dtype=getattr(torch, dtype))
    b = kernel(a)
    expected = a.T
    torch.testing.assert_close(b, expected, rtol=1e-2, atol=1e-2)
    print(f"PASS: square transpose M={M}, block_M={block_M}")


def _smem_optin_bytes() -> int:
    """Per-block opt-in dynamic shared memory limit (bytes) of the current CUDA
    device, queried via cudaDevAttrMaxSharedMemoryPerBlockOptin.

    Reference per-block opt-in caps:
        sm_70 (V100)         : 96 KB
        sm_75 (T4 / Turing)  : 64 KB
        sm_80 (A100)         : 164 KB
        sm_86 / sm_89 (Ada)  : 99 KB
        sm_90 (H100)         : 228 KB
        sm_100 (B100)        : 228 KB
        sm_120 (consumer Blackwell, RTX 50-series): 99 KB
    """
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return getattr(props, "shared_memory_per_block_optin", 0)


# Transpose test tiers. Each entry is (label, M, N, block_M, block_N).
# `T.transpose` allocates two shared tiles (`tile` + `tile_T`), each
# block_M * block_N * sizeof(dtype) bytes, so the per-block shared-memory
# requirement is 2 * block_M * block_N * elem_bytes plus minor headroom.
# Tiers are ordered smallest -> largest. The top tier matches the original
# (pre-downgrade) test footprint of 128 KB; we never auto-promote past that.
# Smaller tiers exist so consumer cards (e.g. sm_120 with a 99 KB opt-in cap)
# still exercise the codepath instead of skipping outright.
_TRANSPOSE_TIERS = [
    # label              M    N    block_M block_N    # tile pair (fp16) -> total smem
    ("small_64x64", 128, 128, 64, 64),  # 2 * 64*64*2 = 16 KB
    ("default_128x128_square", 128, 128, 128, 128),  # 2 * 128*128*2 = 64 KB
    ("default_128x128_multi_tile", 256, 256, 128, 128),  # 64 KB, 2x2 tiling
    ("wide_128x256", 128, 256, 128, 256),  # 2 * 128*256*2 = 128 KB (original top)
]


@tilelang.testing.requires_cuda
def test_tilelang_transpose():
    import warnings

    optin = _smem_optin_bytes()  # codespell:ignore
    elem_bytes = 2  # fp16
    ran: list[str] = []
    skipped: list[str] = []
    for label, M, N, block_M, block_N in _TRANSPOSE_TIERS:
        required = 2 * block_M * block_N * elem_bytes
        if required > optin:  # codespell:ignore
            skipped.append(f"{label} (needs {required} B)")
            continue
        run_tilelang_transpose(M=M, N=N, block_M=block_M, block_N=block_N)
        ran.append(label)

    # If we did not get to the top tier, emit a runtime warning so it shows
    # up in the pytest warnings summary instead of silently downgrading.
    top_label = _TRANSPOSE_TIERS[-1][0]
    if skipped and top_label not in ran:
        warnings.warn(
            f"test_tilelang_transpose downgraded on this device (shared_memory_per_block_optin={optin} B): ran {ran}; skipped {skipped}",  # codespell:ignore
            stacklevel=1,
        )

    if not ran:
        import pytest

        smallest_required = 2 * _TRANSPOSE_TIERS[0][3] * _TRANSPOSE_TIERS[0][4] * elem_bytes
        pytest.skip(f"No transpose tier fits this device (optin={optin} B, smallest tier needs {smallest_required} B)")  # codespell:ignore


@tilelang.testing.requires_cuda
def test_tilelang_transpose_square():
    run_tilelang_transpose_square(M=128, block_M=128)
    run_tilelang_transpose_square(M=256, block_M=128)
    run_tilelang_transpose_square(M=512, block_M=128)


@tilelang.testing.requires_cuda
def test_tilelang_transpose_same_rank_dynamic_slice():
    @T.prim_func
    def main(X: T.Tensor((2, 4), T.float32), H: T.Tensor((1,), T.int32), Y: T.Tensor((4, 2), T.float32)):
        with T.Kernel(1, threads=32):
            Xs = T.alloc_shared((2, 4), T.float32)
            Ts = T.alloc_shared((4, 2), T.float32)
            T.copy(X, Xs)
            T.fill(Ts, -1)
            if H[0] == 4:
                T.transpose(Xs, Ts[0 : H[0], 0:2])
            T.copy(Ts, Y)

    kernel = tilelang.compile(main, out_idx=[2])
    x = torch.arange(8, dtype=torch.float32, device="cuda").reshape(2, 4)
    h = torch.tensor([4], dtype=torch.int32, device="cuda")
    torch.testing.assert_close(kernel(x, h), x.T, rtol=0, atol=0)
    h.fill_(3)
    torch.testing.assert_close(kernel(x, h), torch.full_like(x.T, -1), rtol=0, atol=0)

    # Guarded dynamic slice must also compile when src is multi-buffered by T.Pipelined.
    @T.prim_func
    def pipelined_guarded(X: T.Tensor((4, 32, 64), T.float32), W: T.Tensor((1,), T.int32), Y: T.Tensor((64, 32), T.float32)):
        with T.Kernel(1, threads=128):
            Xs = T.alloc_shared((32, 64), T.float32)
            Ts = T.alloc_shared((64, 32), T.float32)
            Acc = T.alloc_fragment((64, 32), T.float32)
            T.clear(Acc)
            for t in T.Pipelined(4, num_stages=2):
                T.copy(X[t, 0, 0], Xs)
                T.fill(Ts, 0.0)
                if W[0] == 64:
                    T.transpose(Xs, Ts[0 : W[0], 0:32])
                for i, j in T.Parallel(64, 32):
                    Acc[i, j] += Ts[i, j]
            T.copy(Acc, Y)

    pkernel = tilelang.compile(
        pipelined_guarded,
        out_idx=[2],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    xp = torch.randint(-2, 3, (4, 32, 64), dtype=torch.int32).to(torch.float32).cuda()
    w = torch.tensor([64], dtype=torch.int32, device="cuda")
    torch.testing.assert_close(pkernel(xp, w), xp.transpose(1, 2).sum(dim=0), rtol=0, atol=0)
    w.fill_(32)
    torch.testing.assert_close(pkernel(xp, w), torch.zeros((64, 32), dtype=torch.float32, device="cuda"), rtol=0, atol=0)


def _run_pipelined_transpose(H: int, W: int, disable_ws: bool):
    steps = 4

    @T.prim_func
    def main(X: T.Tensor((steps, H, W), T.float32), Y: T.Tensor((W, H), T.float32)):
        with T.Kernel(1, threads=128):
            Xs = T.alloc_shared((H, W), T.float32)
            Ts = T.alloc_shared((W, H), T.float32)
            Acc = T.alloc_fragment((W, H), T.float32)
            T.clear(Acc)
            for t in T.Pipelined(steps, num_stages=2):
                T.copy(X[t, 0, 0], Xs)
                T.transpose(Xs, Ts)
                for i, j in T.Parallel(W, H):
                    Acc[i, j] += Ts[i, j]
            T.copy(Acc, Y)

    pass_configs = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True} if disable_ws else None
    kernel = tilelang.compile(main, out_idx=[1], pass_configs=pass_configs)
    x = torch.randint(-2, 3, (steps, H, W), dtype=torch.int32).to(torch.float32).cuda()
    y = kernel(x)
    torch.testing.assert_close(y, x.transpose(1, 2).sum(dim=0), rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_tilelang_transpose_pipelined():
    """T.transpose with leading unit stage dim on src or dst."""
    # Multi-buffered src (3D) -> unversioned dst (2D), non-square and unit-axis.
    _run_pipelined_transpose(32, 64, disable_ws=True)
    _run_pipelined_transpose(32, 64, disable_ws=False)
    _run_pipelined_transpose(1, 32, disable_ws=True)

    # Unversioned src (2D) -> sliced 3D dst with leading unit dim.
    steps, H, W = 4, 32, 64

    @T.prim_func
    def dst_leading_unit(X: T.Tensor((steps, H, W), T.float32), Y: T.Tensor((W, H), T.float32)):
        with T.Kernel(1, threads=128):
            Xs = T.alloc_shared((H, W), T.float32)
            Ts = T.alloc_shared((2, W, H), T.float32)
            Acc = T.alloc_fragment((W, H), T.float32)
            T.clear(Acc)
            for t in T.serial(steps):
                T.copy(X[t, 0, 0], Xs)
                T.transpose(Xs, Ts[t % 2, 0:W, 0:H])
                for i, j in T.Parallel(W, H):
                    Acc[i, j] += Ts[t % 2, i, j]
            T.copy(Acc, Y)

    kernel = tilelang.compile(
        dst_leading_unit,
        out_idx=[1],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    x = torch.randint(-2, 3, (steps, H, W), dtype=torch.int32).to(torch.float32).cuda()
    torch.testing.assert_close(kernel(x), x.transpose(1, 2).sum(dim=0), rtol=0, atol=0)

    # Mismatched transposed slice extents on different-rank operands must be rejected.
    import pytest

    @T.prim_func
    def dst_mismatched_slice(X: T.Tensor((2, 4), T.float32), Y: T.Tensor((2, 4, 2), T.float32)):
        with T.Kernel(1, threads=32):
            Xs = T.alloc_shared((2, 4), T.float32)
            Ts = T.alloc_shared((2, 4, 2), T.float32)
            T.copy(X, Xs)
            T.transpose(Xs, Ts[1, 0:1, 0:2])
            T.copy(Ts, Y)

    with pytest.raises(Exception, match="Transpose extent mismatch"):
        tilelang.compile(
            dst_mismatched_slice,
            out_idx=[1],
            pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
        )


if __name__ == "__main__":
    tilelang.testing.main()
