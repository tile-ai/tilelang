"""L1 subregions feed L0 reductions with transpose and accumulation intact."""

import pytest
import torch
import tilelang
import tilelang.ascend.language as T


@pytest.mark.parametrize(
    "dtype,transpose,k_tiles,stages",
    [
        pytest.param("bfloat16", True, 1, 1, id="transposed-tile"),
        pytest.param("bfloat16", True, 2, 2, id="transposed-pipeline"),
        pytest.param("float32", False, 2, 1, id="fp32-sub-k-accumulation"),
    ],
)
def test_l1_sub_k_regions_accumulate_in_l0(dtype, transpose, k_tiles, stages):
    m, n, tile_k = 16, 32, 32
    k = k_tiles * tile_k
    a_shape = (k, m) if transpose else (m, k)
    b_shape = (k, n) if transpose else (n, k)

    @T.prim_func
    def main(A: T.Tensor(a_shape, dtype), B: T.Tensor(b_shape, dtype), C: T.Tensor((m, n), "float32")):
        with T.Kernel(1):
            a = T.alloc_l1(a_shape, dtype)
            b = T.alloc_l1(b_shape, dtype)
            a0 = T.alloc_l0a((m, tile_k), dtype)
            b0 = T.alloc_l0b((n, tile_k), dtype)
            acc = T.alloc_l0c((m, n), "float32")
            T.copy(A, a)
            T.copy(B, b)
            if dtype == "float32":
                T.set_hf32_mode("nearest_even")
            for step in T.Pipelined(k_tiles, num_stages=stages):
                if transpose:
                    T.copy(a[step * tile_k : (step + 1) * tile_k, :], a0, transpose=True)
                    T.copy(b[step * tile_k : (step + 1) * tile_k, :], b0, transpose=True)
                else:
                    T.copy(a[:, step * tile_k : (step + 1) * tile_k], a0)
                    T.copy(b[:, step * tile_k : (step + 1) * tile_k], b0)
                T.gemm(a0, b0, acc, transpose_B=True, clear_accum=step == 0)
            T.copy(acc, C)

    a = torch.randn(a_shape, device="npu", dtype=getattr(torch, dtype))
    b = torch.randn(b_shape, device="npu", dtype=getattr(torch, dtype))
    actual = tilelang.compile(main, out_idx=-1, target="ascend")(a, b)
    expected = a.float().T @ b.float() if transpose else a.float() @ b.float().T
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
