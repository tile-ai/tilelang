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


def tilelang_transpose_matrix_case(M, N, dtype, src_offset=0, swizzle=False, dst_col=0, consumer=False, address_view=False):
    """Transpose one stage of a multi-buffered source, as T.Pipelined produces:
    the source region has a leading extent-1 axis that the destination lacks.

    `dst_col` shifts the destination region right so it crosses the buffer edge,
    and `consumer` runs the transpose on threads 128..255 only."""
    from tilelang.layout import make_swizzled_layout

    stages = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, src_offset + N), dtype),
        B: T.Tensor((N, M), dtype),
    ):
        with T.Kernel(1, threads=256 if consumer else 128) as _:
            tile = T.alloc_shared((stages, M, src_offset + N), dtype)
            tile_T = T.alloc_shared((N, M), dtype)
            if swizzle:
                T.annotate_layout({tile: make_swizzled_layout(tile), tile_T: make_swizzled_layout(tile_T)})
            if address_view:
                view = T.make_tensor_from_addr(T.address_of(tile[stages - 1, 0, src_offset]), (M, N), dtype, storage_scope="shared.dyn")
                for i, j in T.Parallel(M, N):
                    view[i, j] = A[i, src_offset + j]
                T.transpose(view, tile_T)
            else:
                T.copy(A, tile[stages - 1, :, :])
                if dst_col:
                    T.fill(tile_T, 0)
                if consumer:
                    if T.get_thread_binding() >= 128:
                        T.transpose(tile[stages - 1, :, src_offset : src_offset + N], tile_T)
                else:
                    T.transpose(tile[stages - 1, :, src_offset : src_offset + N], tile_T[:, dst_col : dst_col + M])
            T.copy(tile_T, B)

    return main


def _bits(x):
    return x.view({1: torch.uint8, 2: torch.int16, 4: torch.int32}[x.element_size()])


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tilelang_transpose_matrix_instructions():
    # (dtype, M, N, keyword arguments, uses ldmatrix/stmatrix)
    cases = [
        (T.float16, 40, 72, {}, True),
        (T.bfloat16, 64, 128, {"consumer": True}, True),
        (T.float8_e4m3fn, 64, 512, {}, True),
        (T.float8_e4m3fn, 64, 128, {"swizzle": True}, True),
        (T.int8, 48, 80, {}, True),
        # rows that do not start on a 16-byte boundary keep the SIMT loop
        (T.float16, 16, 16, {"src_offset": 4}, False),
        (T.float8_e4m3fn, 16, 16, {"src_offset": 8}, False),
        # a destination region past the buffer edge keeps the predicated loop
        (T.float16, 16, 16, {"dst_col": 8}, False),
        # Contiguous rows in a view whose base address is offset by one byte.
        (T.uint8, 16, 16, {"src_offset": 1, "address_view": True}, False),
        (T.float32, 32, 64, {}, False),
    ]
    for dtype, M, N, kwargs, matrix in cases:
        kernel = tilelang.compile(
            tilelang_transpose_matrix_case(M, N, dtype, **kwargs),
            target="cuda",
            pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
        )
        assert ("ptx_stmatrix" in kernel.get_kernel_source()) == matrix, (dtype, M, N, kwargs)
        src_offset, dst_col = kwargs.get("src_offset", 0), kwargs.get("dst_col", 0)
        torch_dtype = getattr(torch, dtype)
        a = (
            torch.randint(0, 256, (M, src_offset + N), dtype=torch.uint8, device="cuda").view(torch_dtype)
            if torch_dtype.itemsize == 1
            else torch.randn(M, src_offset + N, device="cuda").to(torch_dtype)
        )
        expected = torch.zeros(N, M, dtype=torch_dtype, device="cuda")
        expected[:, dst_col:] = a[: M - dst_col, src_offset:].t()
        b = torch.empty(N, M, dtype=torch_dtype, device="cuda")
        kernel(a, b)
        torch.testing.assert_close(_bits(b), _bits(expected), rtol=0, atol=0)


def tilelang_transpose_pipelined_gemm(K, M, N, block_K, dtype):
    """C = A^T @ B, transposing each pipelined stage of A in shared memory so the
    transpose reads a multi-buffered stage and a GEMM consumes its output."""

    @T.prim_func
    def main(
        A: T.Tensor((K, M), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((block_K, M), dtype)
            A_T = T.alloc_shared((M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, N), dtype)
            C_local = T.alloc_fragment((M, N), T.float32)
            T.clear(C_local)
            for k in T.Pipelined(K // block_K, num_stages=2):
                T.copy(A[k * block_K, 0], A_shared)
                T.copy(B[k * block_K, 0], B_shared)
                T.transpose(A_shared, A_T)
                T.gemm(A_T, B_shared, C_local)
            T.copy(C_local, C)

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tilelang_transpose_pipelined_gemm():
    K, M, N, block_K = 256, 64, 64, 64
    kernel = tilelang.compile(tilelang_transpose_pipelined_gemm(K, M, N, block_K, T.float16), target="cuda")
    assert "ptx_stmatrix" in kernel.get_kernel_source()
    a = torch.randn(K, M, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float32)
    kernel(a, b, c)
    torch.testing.assert_close(c, a.float().t() @ b.float(), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    tilelang.testing.main()
