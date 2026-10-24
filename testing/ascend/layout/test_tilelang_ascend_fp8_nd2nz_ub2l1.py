"""Ascend950 FP8 ND-to-NZ scatter and UB-to-L1 GEMM correctness."""

import pytest
import torch
import tilelang
import tilelang.testing
from tilelang.ascend import language as T
from tilelang import tvm

pytest.importorskip("torch_npu")


def _scatter_call(src, dst, rows, cols):
    return tvm.tirx.Call(
        "void",
        tvm.ir.Op.get("tl.ascend_nd2nz_scatter"),
        [src, dst, T.int32(rows), T.int32(cols), T.int32(rows + 1), tvm.tirx.StringImm("fp8_e4_t"), tvm.tirx.StringImm("fp8_e4_t")],
    )


def _make_gemm_kernel(dtype, cols, route, operand):
    rows = 32
    sids = 2 if route == "dual_copy" else 1
    part_rows = rows // sids

    @T.prim_func
    def kernel(a: T.Tensor((2, rows, cols), dtype), b: T.Tensor((2, rows, cols), dtype), out: T.Tensor((2, rows, rows), "float32")):
        with T.MixedKernel(1, sids=sids):
            nd = T.alloc_shared((part_rows, cols), dtype)
            a1 = T.alloc_l1((rows, cols), dtype)
            b1 = T.alloc_l1((rows, cols), dtype)
            a0 = T.alloc_l0a((rows, cols), dtype)
            b0 = T.alloc_l0b((rows, cols), dtype)
            c0 = T.alloc_l0c((rows, rows), "float32")
            for stage in T.Pipelined(2):
                if operand == "A":
                    T.copy(b[stage, :, :], b1)
                    if route == "dual_copy":
                        T.dual_copy(a[stage, :, :], nd)
                        T.dual_copy(nd, a1)
                    else:
                        T.copy(a[stage, :, :], nd)
                        T.copy(nd, a1)
                else:
                    T.copy(a[stage, :, :], a1)
                    if route == "dual_copy":
                        T.dual_copy(b[stage, :, :], nd)
                        T.dual_copy(nd, b1)
                    else:
                        T.copy(b[stage, :, :], nd)
                        T.copy(nd, b1)
                T.copy(a1, a0)
                T.copy(b1, b0)
                T.gemm(a0, b0, c0, transpose_B=True, clear_accum=True)
                T.copy(c0, out[stage, :, :])

    return kernel


def _make_scatter_kernel(rows, cols, inside_vf):
    dtype = "float8_e4m3fn"

    @T.prim_func
    def main(input_gm: T.Tensor((rows, cols), dtype), output_gm: T.Tensor((rows + 1, cols), dtype)):
        with T.Kernel(1):
            src = T.alloc_shared((rows, cols), dtype)
            nz = T.alloc_shared(((rows + 1) * cols,), dtype)
            T.copy(input_gm, src)
            if inside_vf:
                with T.SimdVF():
                    T.evaluate(
                        _scatter_call(
                            T.access_ptr(src[0, 0], "r", extent=rows * cols), T.access_ptr(nz[0], "w", extent=(rows + 1) * cols), rows, cols
                        )
                    )
            else:
                T.evaluate(
                    _scatter_call(
                        T.access_ptr(src[0, 0], "r", extent=rows * cols), T.access_ptr(nz[0], "w", extent=(rows + 1) * cols), rows, cols
                    )
                )
            T.copy(nz, output_gm)

    return main


# Verify automatic copy/dual_copy reordering supplies either GEMM operand.
@tilelang.testing.requires_ascend
@pytest.mark.parametrize("cols", [32, 96, 256, 288])
@pytest.mark.parametrize("route", ["copy", "dual_copy"])
@pytest.mark.parametrize("operand", ["A", "B"])
def test_fp8_nd2nz_ub_to_l1_gemm(cols, route, operand):
    dtype = "float8_e4m3fn"
    kernel = tilelang.compile(_make_gemm_kernel(dtype, cols, route, operand), target="ascend", execution_backend="tvm_ffi")
    torch.manual_seed(31)
    a = torch.randint(-4, 5, (2, 32, cols)).to(getattr(torch, dtype))
    b = torch.randint(-4, 5, (2, 32, cols)).to(getattr(torch, dtype))
    out = torch.full((2, 32, 32), float("nan"), device="npu")
    kernel(a.npu(), b.npu(), out)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), a.float() @ b.float().transpose(-1, -2), rtol=0, atol=0)


# Preserve all FP8 bit patterns across full vectors/tails, inside and outside SIMD VF.
@tilelang.testing.requires_ascend
@pytest.mark.parametrize("cols", [*range(32, 513, 32), 544, 608])
@pytest.mark.parametrize("inside_vf", [False, True])
def test_fp8_scatter_intrinsic_preserves_bits(cols, inside_vf):
    rows = 32
    bits = torch.randint(0, 256, (rows * cols,), dtype=torch.uint8, generator=torch.Generator().manual_seed(31))
    bits[:256] = torch.arange(256, dtype=torch.uint8)
    bits = bits.reshape(rows, cols)
    kernel = tilelang.compile(_make_scatter_kernel(rows, cols, inside_vf), target="ascend", out_idx=-1, execution_backend="tvm_ffi")
    output = kernel(bits.view(torch.float8_e4m3fn).npu()).cpu().view(torch.uint8)
    unpacked = output.reshape(cols // 32, rows + 1, 32)[:, :rows, :].permute(1, 0, 2).reshape(rows, cols)
    torch.testing.assert_close(unpacked, bits, rtol=0, atol=0)
