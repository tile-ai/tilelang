"""
Test for ND→NZ scatter + 2D copy_ubuf_to_cbuf on Ascend (UB→L1, V→C direction).

Dataflow:
  AIV:  GM(D) → UB → L1 (nd2nz scatter + 2D copy_ubuf_to_cbuf)
  AIC:  GM(I) → L1 (MTE2, hardware ND→NZ) → wait AIV → gemm(I, D) → L0C → GM(C)

Verify: I @ D = D  (identity matmul), so output C should equal input D.
"""

import pytest
import torch
import tilelang
import tilelang.language as T
import tilelang.testing


def nd2nz_matmul(
    M: int,
    N: int,
    src_dtype: str,
    mul_dtype: str,
    num_aiv: int = 2,
    split_dim: int = 0,
):
    """Build a Mix kernel that tests the nd2nz path.

    M must be a multiple of 16 and (M * 4) % 32 == 0 (holds for any M % 8 == 0).
    """

    @T.prim_func
    def main(
        I_gm: T.Buffer((N, N), mul_dtype),  # Identity matrix (GM, ND)
        X_gm: T.Buffer((M, N), src_dtype),  # Test data (GM, ND)
        O_gm: T.Buffer((M, N), "float32"),  # Output (GM)
    ):
        with T.MixedKernel(1) as (pid, sid):
            i_l1 = T.alloc_l1((N, N), mul_dtype)  # Identity in L1 (NZ)
            x_l1 = T.alloc_l1((M, N), mul_dtype)  # Test data in L1 (NZ, written by AIV)
            o_l0c = T.alloc_l0c((M, N), "float32")

            for _i in T.Pipelined(1):
                x_ub_shape = [M, N]
                x_ub_shape[split_dim] //= num_aiv
                x_ub = T.alloc_shared(x_ub_shape, src_dtype)
                if num_aiv == 1:
                    if sid == 0:
                        T.copy(X_gm, x_ub)
                        T.copy(x_ub, x_l1)
                else:
                    T.dual_copy(X_gm, x_ub)
                    T.dual_copy(x_ub, x_l1)
                T.copy(I_gm, i_l1)
                T.gemm(x_l1, i_l1, o_l0c, transpose_B=True, clear_accum=True)
                T.copy(o_l0c, O_gm)

    return main


def _to_torch_dtype(dt: str):
    if dt == "float":
        return torch.float32
    if dt in ("bfloat16_t", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {dt}")


@pytest.mark.parametrize("num_aiv", [1, 2])
@pytest.mark.parametrize(
    "M,N,src_dtype,mul_dtype",
    [
        *[(m, n, "float", "bfloat16") for m in [32, 64, 128] for n in [64, 128]],
        *[(m, n, "float", "float") for m in [32, 64, 128] for n in [64, 128]],
        *[(m, n, "bfloat16", "float") for m in [32, 64, 128] for n in [128, 256]],
    ],
)
def test_nd2nz_matmul(M, N, src_dtype, mul_dtype, num_aiv):
    torch.manual_seed(42)

    src_td = _to_torch_dtype(src_dtype)
    mul_td = _to_torch_dtype(mul_dtype)

    I_gm = torch.eye(N, dtype=mul_td).npu()
    X_gm = torch.arange(M * N).view(M, N).to(dtype=src_td).npu()

    # Compile
    program = nd2nz_matmul(M, N, src_dtype, mul_dtype, num_aiv=num_aiv, split_dim=0)
    kernel = tilelang.compile(
        program,
        out_idx=-1,
    )

    # Run
    O_gm = kernel(I_gm, X_gm)
    torch.npu.synchronize()

    # Verify: I @ X = X  (identity matmul)
    expected = X_gm.to(dtype=mul_td)
    actual = O_gm.to(dtype=mul_td)
    rel_err = ((expected - actual).abs() / expected.abs().clamp(min=1.0)).max().item()
    assert rel_err < 1e-3, f"M={M} N={N} src_dtype={src_dtype} mul_dtype={mul_dtype} num_aiv={num_aiv} rel_err={rel_err}"


@pytest.mark.parametrize("split_dim", [0, 1])
def test_nd2nz_matmul_reuses_mixed_kernel_sid(split_dim):
    source = tilelang.lower(nd2nz_matmul(64, 128, "float", "float", num_aiv=2, split_dim=split_dim), target="ascend").kernel_source

    assert "__global__ __mix__(1, 2)" in source
    assert source.count("get_subblockid()") == 1
    assert source.count("copy_gm_to_ubuf_align_v2") == 1
    assert source.count("copy_ubuf_to_cbuf") == 1


if __name__ == "__main__":
    tilelang.testing.main()
