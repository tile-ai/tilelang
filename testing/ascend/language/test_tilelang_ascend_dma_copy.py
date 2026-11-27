"""Numerical checks for strided DMA and packed sub-byte storage."""

import pytest
import torch
import tilelang
import tilelang.testing
from tilelang.ascend import language as T


def _matrix_copy_kernel(dtype, route, m=32, k=128):
    n = m
    scale_pairs = k // 64
    gm_trans_a, gm_trans_b = route == "gm-transpose-a", route == "gm-transpose-b"
    l0_trans_a, l0_trans_b = route == "l0-transpose-a", route == "l0-transpose-b"
    a_shape = (k, m) if gm_trans_a or l0_trans_a else (m, k)
    b_shape = (k, n) if gm_trans_b or l0_trans_b else (n, k)
    a_tile = (k, m) if l0_trans_a else (m, k)
    b_tile = (k, n) if l0_trans_b else (n, k)

    @T.prim_func
    def main(
        A: T.Tensor(a_shape, dtype),
        B: T.Tensor(b_shape, dtype),
        SA: T.Tensor((scale_pairs, m), "uint16"),
        SB: T.Tensor((scale_pairs, n), "uint16"),
        O: T.Tensor((m, n), "float32"),
    ):
        with T.Kernel(1):
            a1 = T.alloc_l1(a_tile, dtype)
            b1 = T.alloc_l1(b_tile, dtype)
            a0 = T.alloc_l0a(a_tile, dtype)
            b0 = T.alloc_l0b(b_tile, dtype)
            sa1 = T.alloc_l1((m, scale_pairs), "uint16")
            sb1 = T.alloc_l1((n, scale_pairs), "uint16")
            sa0 = T.alloc_l0a_sf(a0, sf_shape=(m, scale_pairs))
            sb0 = T.alloc_l0b_sf(b0, sf_shape=(n, scale_pairs))
            acc = T.alloc_l0c((m, n), "float32")
            T.copy(A, a1, transpose=gm_trans_a)
            T.copy(B, b1, transpose=gm_trans_b)
            T.copy(SA, sa1, transpose=True)
            T.copy(SB, sb1, transpose=True)
            T.copy(a1, a0)
            T.copy(b1, b0)
            T.copy(sa1, sa0)
            T.copy(sb1, sb0)
            T.gemm_blockscaled(a0, b0, acc, sa0, sb0, transpose_A=l0_trans_a, transpose_B=not l0_trans_b, clear_accum=True)
            T.copy(acc, O)

    return main


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("route", ["normal", "gm-transpose-a", "gm-transpose-b", "l0-transpose-a", "l0-transpose-b"])
def test_fp8_matrix_copy_with_mx_scales(dtype, route):
    kernel = tilelang.compile(_matrix_copy_kernel(dtype, route), target="ascend", out_idx=-1)
    torch_dtype = getattr(torch, dtype)
    for seed in (0, 1):
        generator = torch.Generator().manual_seed(seed)
        a = (torch.randint(-4, 5, (32, 128), generator=generator).float() / 8).to(torch_dtype)
        b = (torch.randint(-4, 5, (32, 128), generator=generator).float() / 8).to(torch_dtype)
        sa = torch.randint(125, 130, (32, 4), dtype=torch.uint8, generator=generator)
        sb = torch.randint(125, 130, (32, 4), dtype=torch.uint8, generator=generator)
        a_input = a.T.contiguous() if route in ("gm-transpose-a", "l0-transpose-a") else a
        b_input = b.T.contiguous() if route in ("gm-transpose-b", "l0-transpose-b") else b
        actual = kernel(
            a_input.npu(), b_input.npu(), sa.view(torch.uint16).T.contiguous().npu(), sb.view(torch.uint16).T.contiguous().npu()
        )
        torch.npu.synchronize()
        a_scaled = a.float() * torch.pow(2.0, sa.float() - 127).repeat_interleave(32, dim=1)
        b_scaled = b.float() * torch.pow(2.0, sb.float() - 127).repeat_interleave(32, dim=1)
        torch.testing.assert_close(actual.cpu(), a_scaled @ b_scaled.T, rtol=1e-5, atol=1e-5)


@tilelang.testing.requires_ascend
@pytest.mark.skipif(not hasattr(torch, "float4_e2m1fn_x2"), reason="PyTorch float4_e2m1fn_x2 dtype is unavailable")
@pytest.mark.parametrize("route", ["normal", "l0-transpose-a", "l0-transpose-b"])
def test_fp4_matrix_copy_uses_l0_transpose(route):
    kernel = tilelang.compile(_matrix_copy_kernel("float4_e2m1fn", route, 64, 64), target="ascend", out_idx=-1)
    if route != "normal":
        operand = "a" if route.endswith("a") else "b"
        assert f"asc_copy_l12l0{operand}_transpose" in kernel.get_kernel_source()
    values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
    for seed in (0, 1):
        generator = torch.Generator().manual_seed(seed)
        a = torch.randint(0, 16, (64, 64), dtype=torch.uint8, generator=generator)
        b = torch.randint(0, 16, (64, 64), dtype=torch.uint8, generator=generator)
        if seed == 0:
            if route == "l0-transpose-a":
                b = torch.eye(64, dtype=torch.uint8) * 2
            else:
                a = torch.eye(64, dtype=torch.uint8) * 2
        a_input = a.T.contiguous() if route == "l0-transpose-a" else a
        b_input = b.T.contiguous() if route == "l0-transpose-b" else b
        a_packed = (a_input[:, ::2] | (a_input[:, 1::2] << 4)).view(torch.float4_e2m1fn_x2)
        b_packed = (b_input[:, ::2] | (b_input[:, 1::2] << 4)).view(torch.float4_e2m1fn_x2)
        scales = torch.full((1, 64), 0x7F7F, dtype=torch.uint16).npu()
        actual = kernel(a_packed.npu(), b_packed.npu(), scales, scales).cpu()
        torch.testing.assert_close(actual, values[a.long()] @ values[b.long()].T, rtol=0, atol=0)


@pytest.mark.parametrize("src_width,ub_width,cols", [(4, 64, 4), (8, 16, 4), (64, 1, 1)])
def test_strided_dma_roundtrip(src_width, ub_width, cols):
    @T.prim_func
    def copy(src: T.Tensor((4, src_width), "int32"), dst: T.Tensor((4, cols), "int32")):
        with T.Kernel(1):
            ub = T.alloc_shared((4, ub_width), "int32")
            T.copy(src[:, :cols], ub[:, :cols])
            T.copy(ub[:, :cols], dst)

    kernel = tilelang.compile(copy, out_idx=-1, target="ascend")
    src = torch.arange(4 * src_width, dtype=torch.int32, device="npu").reshape(4, src_width)
    torch.testing.assert_close(kernel(src), src[:, :cols])


@pytest.mark.parametrize("rows", [0, 1, 4])
def test_dynamic_rows_with_fixed_middle_axis(rows):
    @T.prim_func
    def copy(src: T.Tensor((4, 2, 32), "float32"), dst: T.Tensor((4, 32), "float32"), n: T.int32):
        with T.Kernel(1):
            ub = T.alloc_shared((4, 32), "float32")
            if n > 0:
                T.copy(src[:n, 0, :], ub[:n, :])
                T.copy(ub[:n, :], dst[:n, :])

    kernel = tilelang.compile(copy, target="ascend")
    src = torch.arange(256, dtype=torch.float32, device="npu").reshape(4, 2, 32)
    dst = torch.full((4, 32), -1.0, device="npu")
    kernel(src, dst, rows)
    expected = torch.full_like(dst, -1.0)
    expected[:rows] = src[:rows, 0, :]
    torch.testing.assert_close(dst, expected)


def l0c_to_strided_gm_copy():
    ldd = T.dynamic("ldd", dtype="int64")
    m, n, k = 16, 16, 64

    @T.prim_func
    def kernel(
        a: T.Tensor((m, k), "bfloat16"),
        b: T.Tensor((n, k), "bfloat16"),
        out: T.StridedTensor((m, n), (ldd, 1), "float32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((m, k), "bfloat16")
            b_l1 = T.alloc_l1((n, k), "bfloat16")
            acc = T.alloc_l0c((m, n), "float32")
            T.copy(a, a_l1)
            T.copy(b, b_l1)
            T.gemm(
                a_l1,
                b_l1,
                acc,
                transpose_B=True,
                clear_accum=True,
                unit_flag_ctrl=3,
            )
            T.copy(acc, out[0, 0], unit_flag_ctrl=3)

    return kernel


def test_l0c_to_strided_gm_uses_runtime_row_stride():
    kernel = tilelang.compile(l0c_to_strided_gm_copy(), target="ascend")
    device = torch.device("npu")
    a = torch.randn((16, 64), dtype=torch.bfloat16, device=device)
    b = torch.randn((16, 64), dtype=torch.bfloat16, device=device)
    storage = torch.empty((16, 32), dtype=torch.float32, device=device)
    out = storage[:, :16]
    kernel(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a.float() @ b.float().T)


FP4_N = 64


def fp4_copy():
    @T.prim_func
    def kernel(
        src: T.Tensor((FP4_N,), T.float4_e2m1fn),
        dst: T.Tensor((FP4_N,), T.float4_e2m1fn),
    ):
        with T.Kernel(1) as _:
            buf = T.alloc_shared((FP4_N,), T.float4_e2m1fn)
            T.copy(src, buf)
            T.copy(buf, dst)

    return kernel


@pytest.mark.skipif(
    not hasattr(torch, "float4_e2m1fn_x2"),
    reason="PyTorch float4_e2m1fn_x2 dtype is unavailable",
)
def test_gm2ub2gm_fp4_copy():
    kernel = tilelang.compile(
        fp4_copy(),
        out_idx=-1,
        target="ascend",
    )

    device = torch.device("npu")
    nbytes = FP4_N // 2
    src_bytes = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, device="cpu").to(device)
    src_fp4 = src_bytes.view(torch.float4_e2m1fn_x2)
    out = kernel(src_fp4)
    torch.npu.synchronize()

    assert out.shape == (nbytes,)
    assert out.dtype == torch.float4_e2m1fn_x2
    out_bytes = out.view(torch.uint8)
    torch.testing.assert_close(out_bytes.cpu(), src_bytes.cpu())


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("reuse_pad_register", [False, True], ids=["pad-value", "data-select"])
def test_gm_to_ub_pad_value(reuse_pad_register):
    @T.prim_func
    def copy(A: T.Tensor((2, 30), "float32"), O: T.Tensor((2, 32), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((2, 32), "float32")
            if reuse_pad_register:
                T.ascend_set_copy_pad_value(-1.0, dtype="float32")
                T.copy(A, ub[:, :30], data_select=True)
            else:
                T.copy(A, ub[:, :30], pad_value=-1.0)
            T.copy(ub, O)

    a = torch.arange(60, dtype=torch.float32).reshape(2, 30)
    result = tilelang.compile(copy, target="ascend", out_idx=-1)(a.npu())
    expected = torch.full((2, 32), -1.0)
    expected[:, :30] = a
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
