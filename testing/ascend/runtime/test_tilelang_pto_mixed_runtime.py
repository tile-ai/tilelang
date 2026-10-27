"""Mixed PTO entries preserve physical subcores and compose with other entries."""

import pytest
import torch

import tilelang
import tilelang.testing
from tilelang.ascend import language as T


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("target", ["ascend", "pto"])
@pytest.mark.parametrize("blocks", [1, 3])
@pytest.mark.parametrize("sids", [1, 2])
def test_vector_only_mixed_launches_requested_subcores(target, blocks, sids):
    @T.prim_func
    def main(src: T.Tensor((blocks * 128,), "float32"), dst: T.Tensor((blocks * 128,), "float32")):
        with T.MixedKernel(blocks, sids=sids) as (bx, sid):
            ub = T.alloc_shared((64,), "float32")
            begin = bx * 128 + sid * 64
            T.copy(src[begin : begin + 64], ub)
            T.copy(ub, dst[begin : begin + 64])

    src = torch.arange(blocks * 128, dtype=torch.float32, device="npu")
    dst = torch.full_like(src, -31)
    tilelang.compile(main, target=target)(src, dst)
    expected = torch.full_like(src, -31).view(blocks, 128)
    expected[:, : sids * 64] = src.view(blocks, 128)[:, : sids * 64]
    torch.testing.assert_close(dst, expected.flatten(), rtol=0, atol=0)


def _mixed_sequence(kind, blocks, sids):
    if kind == "mixed_plain":

        @T.prim_func
        def main(
            src: T.Tensor((blocks * 128,), "float32"),
            scratch: T.Tensor((blocks * 128,), "float32"),
            dst: T.Tensor((blocks * 128,), "float32"),
        ):
            with T.MixedKernel(blocks, sids=sids) as (bx, sid):
                first = T.alloc_shared((64,), "float32")
                begin = bx * 128 + sid * 64
                T.copy(src[begin : begin + 64], first)
                T.copy(first, scratch[begin : begin + 64])
            with T.Kernel(blocks) as bx:
                second = T.alloc_shared((128,), "float32")
                T.copy(scratch[bx * 128 : (bx + 1) * 128], second)
                T.copy(second, dst[bx * 128 : (bx + 1) * 128])

    elif kind == "plain_mixed":

        @T.prim_func
        def main(
            src: T.Tensor((blocks * 128,), "float32"),
            scratch: T.Tensor((blocks * 128,), "float32"),
            dst: T.Tensor((blocks * 128,), "float32"),
        ):
            with T.Kernel(blocks) as bx:
                first = T.alloc_shared((128,), "float32")
                T.copy(src[bx * 128 : (bx + 1) * 128], first)
                T.copy(first, scratch[bx * 128 : (bx + 1) * 128])
            with T.MixedKernel(blocks, sids=sids) as (bx, sid):
                second = T.alloc_shared((64,), "float32")
                begin = bx * 128 + sid * 64
                T.copy(scratch[begin : begin + 64], second)
                T.copy(second, dst[begin : begin + 64])

    else:
        assert kind == "mixed_mixed"

        @T.prim_func
        def main(
            src: T.Tensor((blocks * 128,), "float32"),
            scratch: T.Tensor((blocks * 128,), "float32"),
            dst: T.Tensor((blocks * 128,), "float32"),
        ):
            with T.MixedKernel(blocks, sids=sids) as (bx, sid):
                first = T.alloc_shared((64,), "float32")
                begin = bx * 128 + sid * 64
                T.copy(src[begin : begin + 64], first)
                T.copy(first, scratch[begin : begin + 64])
            with T.MixedKernel(blocks, sids=sids) as (bx, sid):
                second = T.alloc_shared((64,), "float32")
                begin = bx * 128 + sid * 64
                T.copy(scratch[begin : begin + 64], second)
                T.copy(second, dst[begin : begin + 64])

    return main


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("target", ["ascend", "pto"])
@pytest.mark.parametrize("kind", ["mixed_plain", "plain_mixed", "mixed_mixed"])
@pytest.mark.parametrize("sids", [1, 2])
def test_mixed_entries_compose_in_host_launch_order(target, kind, sids):
    blocks = 3
    src = torch.arange(blocks * 128, dtype=torch.float32, device="npu")
    scratch = torch.full_like(src, -17)
    dst = torch.full_like(src, -31)
    tilelang.compile(_mixed_sequence(kind, blocks, sids), target=target)(src, scratch, dst)

    expected_scratch = torch.full_like(src, -17).view(blocks, 128)
    first_width = 128 if kind == "plain_mixed" else sids * 64
    expected_scratch[:, :first_width] = src.view(blocks, 128)[:, :first_width]
    expected_dst = torch.full_like(src, -31).view(blocks, 128)
    second_width = 128 if kind == "mixed_plain" else sids * 64
    expected_dst[:, :second_width] = expected_scratch[:, :second_width]
    torch.testing.assert_close(scratch, expected_scratch.flatten(), rtol=0, atol=0)
    torch.testing.assert_close(dst, expected_dst.flatten(), rtol=0, atol=0)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("target", ["ascend", "pto"])
@pytest.mark.parametrize("sids", [1, 2])
def test_mixed_cube_to_vector_dependency(target, sids):
    blocks = 3
    tile = 16

    @T.prim_func
    def main(
        a: T.Tensor((blocks, tile, tile), "bfloat16"),
        b: T.Tensor((blocks, tile, tile), "bfloat16"),
        dst: T.Tensor((blocks, tile, tile), "float32"),
    ):
        with T.MixedKernel(blocks, sids=sids) as (bx, _):
            a_l1 = T.alloc_l1((tile, tile), "bfloat16")
            b_l1 = T.alloc_l1((tile, tile), "bfloat16")
            acc = T.alloc_l0c((tile, tile), "float32")
            ub = T.alloc_shared((tile // sids, tile), "float32")
            T.copy(a[bx, :, :], a_l1)
            T.copy(b[bx, :, :], b_l1)
            T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=True, unit_flag_ctrl=3)
            if sids == 1:
                T.copy(acc, ub, unit_flag_ctrl=3)
                T.copy(ub, dst[bx, :, :])
            else:
                T.dual_copy(acc, ub, unit_flag_ctrl=3)
                T.dual_copy(ub, dst[bx, :, :])

    a = (torch.arange(blocks * tile * tile).reshape(blocks, tile, tile) % 7 - 3).to(dtype=torch.bfloat16, device="npu")
    b = (torch.arange(blocks * tile * tile).reshape(blocks, tile, tile) % 5 - 2).to(dtype=torch.bfloat16, device="npu")
    dst = torch.full((blocks, tile, tile), -12345, dtype=torch.float32, device="npu")
    tilelang.compile(main, target=target)(a, b, dst)
    expected = a.float() @ b.float().transpose(-1, -2)
    torch.testing.assert_close(dst, expected, rtol=0, atol=0)
