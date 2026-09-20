"""Numerical checks for strided DMA and packed sub-byte storage."""

import pytest
import torch
import tilelang
from tilelang.ascend import language as T


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
