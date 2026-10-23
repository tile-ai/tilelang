"""PTO target through the shared tvm_ffi host execution path.

Covers output allocation, dynamic shapes, dynamic shared memory,
multi-kernel and mixed cube/vector programs, exported libraries, and
host-controlled launches.
"""

import pytest
import torch

import tilelang
import tilelang.testing
from tilelang.ascend import language as T
from tilelang.ascend.torch_exchange import install_torch_npu_stream_exchange


pytestmark = pytest.mark.pto

ROWS = T.dynamic("rows")
WIDTH = 128


def vector_add(n: int, tile: int):
    @T.prim_func
    def main(
        a: T.Tensor((n,), T.float32),
        b: T.Tensor((n,), T.float32),
        c: T.Tensor((n,), T.float32),
    ):
        with T.Kernel(n // tile) as bx, T.SimtVF(threads=tile):
            for i in T.Parallel(tile):
                idx = bx * tile + i
                c[idx] = a[idx] + b[idx]

    return main


@T.prim_func
def dynamic_copy(
    source: T.Tensor((ROWS, WIDTH), T.bfloat16),
    destination: T.Tensor((ROWS, WIDTH), T.bfloat16),
):
    with T.Kernel(ROWS) as row:
        tile = T.alloc_shared((WIDTH,), T.bfloat16)
        T.copy(source[row, :], tile)
        T.copy(tile, destination[row, :])


@T.prim_func
def dynamic_outputs(
    positive: T.Tensor((ROWS, WIDTH), T.float32),
    source: T.Tensor((ROWS, WIDTH), T.float32),
    negative: T.Tensor((ROWS, WIDTH), T.float32),
    bias: T.int32,
):
    with T.Kernel(ROWS) as row, T.SimtVF(threads=WIDTH):
        for i in T.Parallel(WIDTH):
            positive[row, i] = source[row, i] + T.float32(bias)
            negative[row, i] = source[row, i] - T.float32(bias)


def two_kernels(n: int, tile: int):
    @T.prim_func
    def main(
        a: T.Tensor((n,), T.float32),
        b: T.Tensor((n,), T.float32),
        tmp: T.Tensor((n,), T.float32),
        out: T.Tensor((n,), T.float32),
    ):
        with T.Kernel(n // tile) as bx, T.SimtVF(threads=tile):
            for i in T.Parallel(tile):
                idx = bx * tile + i
                tmp[idx] = a[idx] + b[idx]

        with T.Kernel(n // tile) as bx, T.SimtVF(threads=tile):
            for i in T.Parallel(tile):
                idx = bx * tile + i
                out[idx] = tmp[idx] * T.float32(2.0)

    return main


def dynamic_shared_copy():
    n = T.dynamic("n")

    @T.prim_func
    def copy(src: T.Tensor((n,), T.float32), dst: T.Tensor((n,), T.float32)):
        with T.Kernel(1):
            ub = T.alloc_shared((T.ceildiv(n, 64) * 64,), T.float32)
            T.copy(src, ub[:n])
            T.copy(ub[:n], dst)

    return copy


def accumulate(n: int, tile: int, iters: int):
    @T.prim_func
    def main(
        a: T.Tensor((n,), T.float32),
        out: T.Tensor((n,), T.float32),
    ):
        for _ in T.serial(iters):
            with T.Kernel(n // tile) as bx, T.SimtVF(threads=tile):
                for i in T.Parallel(tile):
                    idx = bx * tile + i
                    out[idx] = out[idx] + a[idx]

    return main


def mixed_gemm(m_dim: int, k_dim: int, n_dim: int):
    num_blocks = 32
    tile_m, tile_n, tile_k = 16, 16, 32
    m_tiles, n_tiles = m_dim // tile_m, n_dim // tile_n
    k_tiles = k_dim // tile_k
    tiles_per_core = (m_tiles * n_tiles) // num_blocks
    window = min(4, m_tiles)
    num_stages = 2
    main_row = m_tiles // window - 1
    tail_win = m_tiles - main_row * window

    @T.macro
    def aswt_swizzle(tile_idx):
        m_tile = T.alloc_var("int32")
        n_tile = T.alloc_var("int32")
        row_idx = tile_idx // n_tiles // window
        if row_idx < main_row:
            m_tile = row_idx * window + tile_idx % window
            n_tile = (tile_idx // window) % n_tiles
        else:
            tail_idx = tile_idx - main_row * window * n_tiles
            m_tile = main_row * window + tail_idx % tail_win
            n_tile = (tail_idx // tail_win) % n_tiles
        if row_idx % 2 != 0:
            n_tile = n_tiles - 1 - n_tile
        return m_tile, n_tile

    @T.prim_func
    def main(
        x: T.Buffer((m_dim, k_dim), "bfloat16"),
        w: T.Buffer((n_dim, k_dim), "bfloat16"),
        c: T.Buffer((m_dim, n_dim), "float32"),
    ):
        with T.Kernel(num_blocks) as bx:
            res = T.alloc_l0c((tile_m, tile_n), "float32")
            x_l1 = T.alloc_l1((num_stages, tile_m, tile_k), "bfloat16")
            w_l1 = T.alloc_l1((num_stages, tile_n, tile_k), "bfloat16")
            temp = T.alloc_shared((tile_m // 2, tile_n), "float32")

            with T.Cube():
                for f in T.Serial(num_stages):
                    T.ascend_set_flag("MTE1_MTE2", f)
                T.ascend_set_flag("FIX_M", 0)

                for out_tile in T.Serial(tiles_per_core):
                    tile_idx = out_tile * num_blocks + bx
                    m_tile, n_tile = aswt_swizzle(tile_idx)

                    for kt in T.Serial(k_tiles):
                        f = kt % num_stages
                        T.ascend_wait_flag("MTE1_MTE2", f)
                        T.copy(x[m_tile * tile_m : (m_tile + 1) * tile_m, kt * tile_k : (kt + 1) * tile_k], x_l1[f, :, :])
                        T.copy(w[n_tile * tile_n : (n_tile + 1) * tile_n, kt * tile_k : (kt + 1) * tile_k], w_l1[f, :, :])
                        T.ascend_set_flag("MTE2_MTE1", f)

                        T.ascend_wait_flag("MTE2_MTE1", f)
                        if kt == 0:
                            T.ascend_wait_flag("FIX_M", 0)
                        T.gemm(x_l1[f, :, :], w_l1[f, :, :], res, transpose_B=True, clear_accum=(kt == 0))
                        T.ascend_set_flag("MTE1_MTE2", f)

                    T.ascend_set_flag("M_FIX", 0)
                    T.ascend_wait_flag("M_FIX", 0)
                    T.ascend_cross_core_wait_flag(4, "PIPE_FIX", 4)
                    T.ascend_cross_core_wait_flag(4, "PIPE_FIX", 20)
                    T.dual_copy(res, temp)
                    T.ascend_cross_core_set_flag(4, "PIPE_FIX", 6)
                    T.ascend_cross_core_set_flag(4, "PIPE_FIX", 22)
                    T.ascend_set_flag("FIX_M", 0)

                for f in T.Serial(num_stages):
                    T.ascend_wait_flag("MTE1_MTE2", f)
                T.ascend_wait_flag("FIX_M", 0)
                T.ascend_cross_core_wait_flag(4, "PIPE_FIX", 4)
                T.ascend_cross_core_wait_flag(4, "PIPE_FIX", 20)

            with T.Vector() as sid:
                my_aic_flag = 6
                my_aiv_flag = 4
                T.ascend_cross_core_set_flag(4, "PIPE_MTE3", my_aiv_flag)

                for out_tile in T.Serial(tiles_per_core):
                    tile_idx = out_tile * num_blocks + bx
                    m_tile, n_tile = aswt_swizzle(tile_idx)
                    T.ascend_cross_core_wait_flag(4, "PIPE_MTE3", my_aic_flag)
                    T.copy(temp, c[m_tile * tile_m + sid * (tile_m // 2), n_tile * tile_n])
                    T.ascend_cross_core_set_flag(4, "PIPE_MTE3", my_aiv_flag)

    return main


def test_pto_auto_resolves_tvm_ffi():
    from tilelang.backend.module import create_backend_context

    context = create_backend_context("pto", None, "auto")
    assert context.execution_backend.name == "tvm_ffi"


@tilelang.testing.requires_ascend
def test_pto_tvm_ffi_vecadd_callee_allocated():
    n, tile = 64, 32
    kernel = tilelang.compile(vector_add(n, tile), target="pto", out_idx=-1, execution_backend="tvm_ffi")

    a = torch.randn(n, device="npu")
    b = torch.randn(n, device="npu")
    out = kernel(a, b)
    torch.npu.synchronize()

    torch.testing.assert_close(out.cpu(), a.cpu() + b.cpu(), rtol=0, atol=0)


@tilelang.testing.requires_ascend
def test_pto_tvm_ffi_dynamic_shape():
    kernel = tilelang.compile(dynamic_copy, target="pto", out_idx=-1, execution_backend="tvm_ffi")

    for rows in (1, 3, 17):
        source = torch.randn(rows, WIDTH, dtype=torch.bfloat16, device="npu")
        out = kernel(source)
        torch.npu.synchronize()
        torch.testing.assert_close(out.cpu(), source.cpu(), rtol=0, atol=0)


@tilelang.testing.requires_ascend
def test_pto_tvm_ffi_dynamic_outputs():
    kernel = tilelang.compile(dynamic_outputs, out_idx=[0, 2], target="pto", execution_backend="tvm_ffi")

    rows = 3
    reference = torch.arange(rows * WIDTH, dtype=torch.float32).reshape(rows, WIDTH)
    source = reference.npu()
    torch.npu.synchronize()

    positive, negative = kernel(source, 9)

    assert positive.device == negative.device == source.device
    torch.testing.assert_close(positive.cpu(), reference + 9, atol=0, rtol=0)
    torch.testing.assert_close(negative.cpu(), reference - 9, atol=0, rtol=0)


@tilelang.testing.requires_ascend
def test_pto_tvm_ffi_multiple_kernels():
    n, tile = 64, 32
    kernel = tilelang.compile(two_kernels(n, tile), target="pto", out_idx=[2, 3], execution_backend="tvm_ffi")

    a = torch.randn(n, device="npu")
    b = torch.randn(n, device="npu")
    tmp, out = kernel(a, b)
    torch.npu.synchronize()

    torch.testing.assert_close(tmp.cpu(), a.cpu() + b.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(out.cpu(), (a.cpu() + b.cpu()) * 2, rtol=0, atol=0)


@tilelang.testing.requires_ascend
def test_pto_tvm_ffi_dynamic_shared_memory():
    kernel = tilelang.compile(dynamic_shared_copy(), target="pto", out_idx=-1, execution_backend="tvm_ffi")

    for size in (1, 65, 1000):
        source = torch.arange(size, dtype=torch.float32, device="npu")
        out = kernel(source)
        torch.npu.synchronize()
        torch.testing.assert_close(out.cpu(), source.cpu(), rtol=0, atol=0)


@tilelang.testing.requires_ascend
def test_pto_tvm_ffi_mixed_cube_vector_gemm():
    m, k, n = 128, 64, 128
    kernel = tilelang.compile(
        mixed_gemm(m, k, n),
        target="pto",
        out_idx=-1,
        execution_backend="tvm_ffi",
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False},
    )

    x = torch.randn(m, k, dtype=torch.bfloat16, device="npu")
    w = torch.randn(n, k, dtype=torch.bfloat16, device="npu")
    c = kernel(x, w)
    torch.npu.synchronize()

    expected = x.float().cpu() @ w.float().cpu().T
    torch.testing.assert_close(c.cpu(), expected, atol=1e-2, rtol=1e-2)


def test_pto_tvm_ffi_source_accessors():
    n, tile = 64, 32
    kernel = tilelang.compile(vector_add(n, tile), target="pto", out_idx=-1, execution_backend="tvm_ffi")

    device_source = kernel.get_kernel_source(kernel_only=True)
    assert "@pto.jit" in device_source

    host_source = kernel.get_host_source()
    assert host_source and "target" in host_source


@tilelang.testing.requires_ascend
def test_pto_tvm_ffi_export_library_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path))

    n, tile = 64, 32
    kernel = tilelang.compile(vector_add(n, tile), target="pto", out_idx=-1, execution_backend="tvm_ffi")

    a = torch.randn(n, device="npu")
    b = torch.randn(n, device="npu")
    direct = kernel(a, b)
    torch.npu.synchronize()

    library = tmp_path / "kernel.so"
    kernel.export_library(str(library))

    module = tilelang.tvm.runtime.load_module(str(library))
    entry = module.get_function("main", query_imports=True)
    install_torch_npu_stream_exchange()
    # The callee-allocated-output ABI takes the inputs plus one extra Torch
    # tensor that anchors output allocation to Torch's allocator.
    loaded = entry(a, b, a)
    torch.npu.synchronize()

    torch.testing.assert_close(loaded.cpu(), direct.cpu(), rtol=0, atol=0)


@tilelang.testing.requires_ascend
def test_pto_tvm_ffi_kernel_call_in_host_loop(tmp_path, monkeypatch):
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path))

    n, tile, iters = 64, 32, 3
    kernel = tilelang.compile(accumulate(n, tile, iters), target="pto", out_idx=None, execution_backend="tvm_ffi")

    a = torch.randn(n, device="npu")
    init = torch.randn(n, device="npu")
    out = init.clone()
    kernel(a, out)
    torch.npu.synchronize()

    # Accumulating is non-idempotent, so matching the sequential reference
    # proves every launch in the host loop ran.
    expected = init.cpu()
    for _ in range(iters):
        expected = expected + a.cpu()
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
