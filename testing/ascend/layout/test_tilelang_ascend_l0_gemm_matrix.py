import pytest
import torch
import tilelang
import tilelang.ascend.language as T
import tilelang.testing


def make_l0_gemm_kernel(M: int, K: int, N: int, dtype: str):
    @T.prim_func
    def gemm_kernel(
        X: T.Tensor((M, K), dtype),
        W: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1):
            res = T.alloc_l0c((M, N), "float32")
            x_l1 = T.alloc_l1((M, K), dtype)
            w_l1 = T.alloc_l1((N, K), dtype)
            x_l0a = T.alloc_l0a((M, K), dtype)
            w_l0b = T.alloc_l0b((N, K), dtype)
            T.copy(X, x_l1)
            T.copy(W, w_l1)
            T.copy(x_l1, x_l0a)
            T.copy(w_l1, w_l0b)
            T.gemm(x_l0a, w_l0b, res, transpose_B=True, clear_accum=True)
            T.copy(res, C)

    return gemm_kernel


@pytest.mark.parametrize("dtype,atol,rtol", [("bfloat16", 1e-2, 1e-2), ("float8_e4m3fn", 1e-1, 5e-2)])
@pytest.mark.parametrize("m,k,n", [(1, 64, 64), (15, 15, 15), (32, 32, 32), (100, 100, 100)])
def test_l0_gemm_matrix(dtype, atol, rtol, m, k, n):
    kernel = tilelang.compile(make_l0_gemm_kernel(m, k, n, dtype), target="ascend", out_idx=-1)
    x = (torch.randn(m, k, device="npu") * 0.25).to(getattr(torch, dtype))
    w = (torch.randn(n, k, device="npu") * 0.25).to(getattr(torch, dtype))
    torch.testing.assert_close(kernel(x, w), x.float() @ w.float().T, atol=atol, rtol=rtol)


def _blockscaled_kernel(m, n, trans_a, trans_b, dtype="float8_e4m3fn", use_l0=True, byte_view=False):
    k = 256 if dtype == "float4_e2m1fn" else 128
    sf_cols = k // 64
    a_shape = (k, m) if trans_a else (m, k)
    b_shape = (n, k) if trans_b else (k, n)

    @T.prim_func
    def gemm(
        a: T.Tensor((m, k // 2) if byte_view else a_shape, "int8" if byte_view else dtype),
        b: T.Tensor((n, k // 2) if byte_view else b_shape, "int8" if byte_view else dtype),
        sa: T.Tensor((sf_cols, m), "uint16"),
        sb: T.Tensor((sf_cols, n), "uint16"),
        out: T.Tensor((m, n), "float32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1(a_shape, dtype)
            b_l1 = T.alloc_l1(b_shape, dtype)
            sa_l1 = T.alloc_l1((m, sf_cols), "uint16")
            sb_l1 = T.alloc_l1((n, sf_cols), "uint16")
            a_l0 = T.alloc_l0a(a_shape, dtype)
            b_l0 = T.alloc_l0b(b_shape, dtype)
            # Transposed data tiles need the explicit logical SF shape (rows
            # always follow M/N, not the tile's leading axis).
            a_sf = T.alloc_l0a_sf(a_l0, sf_shape=(m, sf_cols))
            b_sf = T.alloc_l0b_sf(b_l0, sf_shape=(n, sf_cols))
            acc = T.alloc_l0c((m, n), "float32")
            a_data = T.view(a, a_shape, dtype) if byte_view else a
            b_data = T.view(b, b_shape, dtype) if byte_view else b
            T.copy(a_data, a_l1)
            T.copy(b_data, b_l1)
            T.copy(sa, sa_l1, transpose=True)
            T.copy(sb, sb_l1, transpose=True)
            if use_l0:
                T.copy(a_l1, a_l0)
                T.copy(sa_l1, a_sf)
                T.copy(b_l1, b_l0)
                T.copy(sb_l1, b_sf)
                T.gemm_blockscaled(a_l0, b_l0, acc, a_sf, b_sf, transpose_A=trans_a, transpose_B=trans_b, clear_accum=True)
            else:
                T.gemm_blockscaled(a_l1, b_l1, acc, sa_l1, sb_l1, transpose_B=True, clear_accum=True)
            T.copy(acc, out)

    return gemm


@tilelang.testing.requires_ascend
@pytest.mark.parametrize(
    "m,n,trans_a,trans_b",
    [
        (1, 15, False, True),
        (15, 15, False, True),
        (32, 32, False, True),
        (100, 100, False, True),
        (64, 64, False, False),
        (64, 64, True, False),
        (64, 64, True, True),
    ],
)
def test_blockscaled_l0_gemm(m, n, trans_a, trans_b):
    # Small binary fractions keep FP8 input conversion exact; vary every K32 scale group.
    a = (torch.randint(-4, 5, (m, 128)).float() / 8).to(torch.float8_e4m3fn)
    b = (torch.randint(-4, 5, (n, 128)).float() / 8).to(torch.float8_e4m3fn)
    sa = torch.randint(124, 131, (m, 4), dtype=torch.uint8)
    sb = torch.randint(124, 131, (n, 4), dtype=torch.uint8)
    expected = (a.float() * torch.pow(2.0, sa.float() - 127).repeat_interleave(32, dim=1)) @ (
        b.float() * torch.pow(2.0, sb.float() - 127).repeat_interleave(32, dim=1)
    ).T
    kernel = tilelang.compile(_blockscaled_kernel(m, n, trans_a, trans_b), target="ascend", out_idx=-1)
    actual = kernel(
        (a.T.contiguous() if trans_a else a).npu(),
        (b if trans_b else b.T.contiguous()).npu(),
        sa.view(torch.uint16).T.contiguous().npu(),
        sb.view(torch.uint16).T.contiguous().npu(),
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-5)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("use_l0,byte_view", [(True, False), (False, False), (False, True)], ids=["l0", "l1", "l1-byte-view"])
def test_packed_fp4_blockscaled_gemm(use_l0, byte_view):
    m, n, k = 16, 16, 256
    gen = torch.Generator().manual_seed(0)
    # Decode the independent E2M1 nibble table for the CPU reference.
    values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
    a_codes = torch.randint(0, 16, (m, k), generator=gen, dtype=torch.uint8)
    b_codes = torch.randint(0, 16, (n, k), generator=gen, dtype=torch.uint8)
    sa = torch.randint(125, 129, (m, k // 32), generator=gen, dtype=torch.uint8)
    sb = torch.randint(125, 129, (n, k // 32), generator=gen, dtype=torch.uint8)
    a = values[a_codes.long()] * torch.pow(2.0, sa.float() - 127).repeat_interleave(32, dim=1)
    b = values[b_codes.long()] * torch.pow(2.0, sb.float() - 127).repeat_interleave(32, dim=1)
    storage_dtype = torch.int8 if byte_view else torch.float4_e2m1fn_x2
    packed_a = (a_codes[:, ::2] | (a_codes[:, 1::2] << 4)).view(storage_dtype)
    packed_b = (b_codes[:, ::2] | (b_codes[:, 1::2] << 4)).view(storage_dtype)
    kernel = tilelang.compile(_blockscaled_kernel(m, n, False, True, "float4_e2m1fn", use_l0, byte_view), target="ascend", out_idx=-1)
    actual = kernel(packed_a.npu(), packed_b.npu(), sa.view(torch.uint16).T.contiguous().npu(), sb.view(torch.uint16).T.contiguous().npu())
    torch.testing.assert_close(actual.cpu(), a @ b.T, rtol=1e-5, atol=1e-5)
