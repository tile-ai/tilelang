"""Numerical copy tails, including transposed Cube loads and untouched neighbors."""

import pytest
import torch
import tilelang
import tilelang.ascend.language as T


@pytest.mark.parametrize("rank", [1, 2], ids=["vector-tail", "matrix-column-tail"])
def test_dma_tail_roundtrip(rank):
    shape = (80,) if rank == 1 else (4, 80)
    tile_shape = (64,) if rank == 1 else (4, 64)

    @T.prim_func
    def main(src: T.Tensor(shape, "float32"), out: T.Tensor(shape, "float32")):
        with T.Kernel(2) as block:
            ub = T.alloc_shared(tile_shape, "float32")
            if rank == 1:
                T.copy(src[block * 64 : (block + 1) * 64], ub)
                T.copy(ub, out[block * 64 : (block + 1) * 64])
            else:
                T.copy(src[:, block * 64 : (block + 1) * 64], ub)
                T.copy(ub, out[:, block * 64 : (block + 1) * 64])

    src = torch.arange(80 if rank == 1 else 320, device="npu", dtype=torch.float32).reshape(shape)
    actual = tilelang.compile(main, out_idx=-1, target="ascend")(src)
    torch.testing.assert_close(actual, src)


@pytest.mark.parametrize("mutable_offset", [False, True], ids=["scalar-parameter", "local-variable"])
def test_opaque_store_tail_preserves_neighboring_storage(mutable_offset):
    @T.prim_func
    def main(src: T.Tensor((8, 32), "float32"), out: T.Tensor((12, 32), "float32"), row_offset: T.int32):
        with T.Kernel(1):
            ub = T.alloc_shared((8, 32), "float32")
            T.copy(src, ub)
            if mutable_offset:
                offset = T.alloc_var("int32", init=row_offset)
                T.copy(ub, out[offset : offset + 8, :])
            else:
                T.copy(ub, out[row_offset : row_offset + 8, :])

    src = torch.arange(1, 257, device="npu", dtype=torch.float32).reshape(8, 32)
    pool = torch.zeros((24, 32), device="npu")
    kernel = tilelang.compile(main, target="ascend")
    kernel(src, pool[:12], 8)
    expected = torch.zeros_like(pool)
    expected[8:12] = src[:4]
    torch.testing.assert_close(pool, expected)


@pytest.mark.parametrize("transpose", [False, True], ids=["direct", "transposed-nonsquare-load"])
def test_gemm_copy_tails_across_all_matrix_axes(transpose):
    m, n, k = 31, 19, 47
    tile_m, tile_n, tile_k = 16, 16, 32
    a_shape = (k, m) if transpose else (m, k)

    @T.prim_func
    def main(A: T.Tensor(a_shape, "bfloat16"), B: T.Tensor((n, k), "bfloat16"), C: T.Tensor((m, n), "float32")):
        with T.Kernel(4) as block:
            a = T.alloc_l1((tile_m, tile_k), "bfloat16")
            b = T.alloc_l1((tile_n, tile_k), "bfloat16")
            a0 = T.alloc_l0a((tile_m, tile_k), "bfloat16")
            b0 = T.alloc_l0b((tile_n, tile_k), "bfloat16")
            acc = T.alloc_l0c((tile_m, tile_n), "float32")
            for step in T.serial(2):
                if transpose:
                    T.copy(A[step * tile_k : (step + 1) * tile_k, block // 2 * tile_m : (block // 2 + 1) * tile_m], a, transpose=True)
                else:
                    T.copy(A[block // 2 * tile_m : (block // 2 + 1) * tile_m, step * tile_k : (step + 1) * tile_k], a)
                T.copy(B[block % 2 * tile_n : (block % 2 + 1) * tile_n, step * tile_k : (step + 1) * tile_k], b)
                T.copy(a, a0)
                T.copy(b, b0)
                T.gemm(a0, b0, acc, transpose_B=True, clear_accum=step == 0)
            T.copy(acc, C[block // 2 * tile_m : (block // 2 + 1) * tile_m, block % 2 * tile_n : (block % 2 + 1) * tile_n])

    a = torch.randn(a_shape, device="npu", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="npu", dtype=torch.bfloat16)
    actual = tilelang.compile(main, out_idx=-1, target="ascend")(a, b)
    expected = (a.float().T if transpose else a.float()) @ b.float().T
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
