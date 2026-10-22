"""Validate compaction order, byte counts, empty streams, and stream reset."""

import pytest
import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S


@pytest.mark.parametrize("dtype", ["uint8", "int8", "int16", "int32"])
@pytest.mark.parametrize("limit", [0, 1, 37, 127])
@pytest.mark.parametrize("store_count", [False, True])
def test_compact_store(dtype, limit, store_count):
    width = torch.empty((), dtype=getattr(torch, dtype)).element_size() * 8
    lanes = 2048 // width

    @T.prim_func
    def main(X: T.Tensor((512,), dtype), Y: T.Tensor((512,), dtype), C: T.Tensor((8,), "uint32")):
        with T.Kernel(1):
            x = T.alloc_shared((512,), dtype)
            y = T.alloc_shared((512,), dtype)
            count = T.alloc_shared((8,), "uint32")
            T.copy(X, x)
            T.ascend_set_flag("MTE2_V", 0)
            T.ascend_wait_flag("MTE2_V", 0)
            with T.SimdVF():
                mask = S.pset(width)
                for block in T.serial(512 // lanes):
                    S.vsts(y[block * lanes], S.vdup(-1, dtype, mask), mask, f"NORM_B{width}")
                S.mem_bar("VST_VLD")
                for _repetition in T.serial(2):
                    stream = S.compact_store_begin(T.access_ptr(y[0], "w", extent=512))
                    for block in T.serial(512 // lanes):
                        value = S.vld(x[block * lanes])
                        selected = S.vcmps(value, limit, mask, op="lt")
                        S.compact_store(stream, value, selected)
                    if store_count:
                        S.compact_store_end(stream, count[0])
                    else:
                        S.compact_store_end(stream)
                    S.mem_bar("VST_VLD")
            T.ascend_set_flag("V_MTE3", 0)
            T.ascend_wait_flag("V_MTE3", 0)
            T.copy(y, Y)
            T.copy(count, C)

    source = (torch.arange(512) % 127).to(getattr(torch, dtype)).roll(37).npu()
    kernel = tilelang.compile(
        main,
        target="ascend",
        out_idx=[1, 2],
        pass_configs={
            "tl.enable_auto_schedule": False,
            "tl.disable_thread_storage_sync": True,
        },
    )
    output, count = kernel(source)
    expected = source.cpu()[source.cpu() < limit]
    if store_count:
        assert int(count[0].cpu()) == expected.numel() * (width // 8)
    torch.testing.assert_close(output[: expected.numel()].cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["MODE_STORED", "MODE_NO_STORED"])
def test_squeeze_mode(mode):
    @T.prim_func
    def main(X: T.Tensor((256,), "uint8"), Y: T.Tensor((256,), "uint8")):
        with T.Kernel(1):
            x = T.alloc_shared((256,), "uint8")
            y = T.alloc_shared((256,), "uint8")
            T.copy(X, x)
            T.ascend_set_flag("MTE2_V", 0)
            T.ascend_wait_flag("MTE2_V", 0)
            with T.SimdVF():
                mask = S.pset(8)
                value = S.vld(x[0])
                chosen = S.vcmps(value, 37, mask, op="lt")
                packed = S.vsqz(value, chosen, mode)
                S.vsts(y[0], packed, mask, "NORM_B8")
            T.ascend_set_flag("V_MTE3", 0)
            T.ascend_wait_flag("V_MTE3", 0)
            T.copy(y, Y)

    source = torch.arange(256).to(torch.uint8).roll(43).npu()
    kernel = tilelang.compile(
        main,
        target="ascend",
        out_idx=[1],
        pass_configs={"tl.enable_auto_schedule": False, "tl.disable_thread_storage_sync": True},
    )
    expected = source.cpu()[source.cpu() < 37]
    torch.testing.assert_close(kernel(source)[:37].cpu(), expected, rtol=0, atol=0)
    assert mode in kernel.get_kernel_source()
