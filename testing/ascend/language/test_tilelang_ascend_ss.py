"""Group-shared scalar storage: allocation, publication, and slot reuse."""

import re

import pytest
import torch

import tilelang
import tilelang.ascend.language as T
from tilelang import tvm
from tvm import tirx


def _publish_to_cube(rounds=8, stage=0):
    @T.prim_func
    def main(seed: T.Tensor((16, 16), "float16"), out: T.Tensor((rounds, 1), "uint64")):
        with T.MixedKernel(1) as (_, sid):
            words = T.alloc_ss((2, 2), "uint32")
            packed = T.view(words, shape=(2,), dtype="uint64")
            cube = T.alloc_l1((16, 16), "float16")
            # Include Cube work so the uniform scalar consumer runs on AIC.
            T.copy(seed, cube)
            for i in T.serial(rounds):
                if stage is None:
                    words[i % 2, sid] = T.Cast("uint32", (i % 3) * (100 + sid))
                    out[i, 0] = packed[i % 2]
                else:
                    with T.Stage(stage):
                        words[i % 2, sid] = T.Cast("uint32", (i % 3) * (100 + sid))
                        out[i, 0] = packed[i % 2]

    return main


def test_ss_codegen_emits_volatile_views_and_uses_mode4():
    source = tilelang.lower(_publish_to_cube(), target="ascend").kernel_source
    assert "volatile __ssbuf__" in source
    assert "(volatile __ssbuf__ uint32_t *)0" in source
    assert re.search(r"volatile __ssbuf__ uint64_t\s*\*", source)
    assert "asc_sync_intra_wait(PIPE_S" in source
    assert "asc_sync_subblock" not in source


@pytest.mark.parametrize(
    "shape,dtype,message", [((1,), "float32", "uint32 or uint64"), ((769,), "uint32", "3072"), ((0,), "uint32", "positive")]
)
def test_invalid_ss_allocation(shape, dtype, message):
    with pytest.raises(ValueError, match=message):
        T.alloc_ss(shape, dtype)


def test_dynamic_ss_allocation_is_rejected():
    with pytest.raises(ValueError, match="static shape"):
        T.alloc_ss((tirx.Var("n", "int32"),), "uint32")


@pytest.mark.parametrize("size", [2, 400])
def test_ss_multiple_allocations_and_total_capacity(size):
    @T.prim_func
    def main(out: T.Tensor((2,), "uint32")):
        with T.MixedKernel(1) as (_, sid):
            a = T.alloc_ss((size,), "uint32")
            b = T.alloc_ss((size,), "uint32")
            a[sid] = T.uint32(1)
            b[sid] = T.uint32(2)
            out[sid] = a[sid] + b[sid]

    if size == 400:
        with pytest.raises(tvm.error.InternalError, match="3072"):
            tilelang.lower(main, target="ascend")
    else:
        source = tilelang.lower(main, target="ascend").kernel_source
        assert source.count("(volatile __ssbuf__ uint8_t *)0") == 1
        # Two eight-byte allocations must have separate 32-byte aligned bases.
        stores = re.findall(r"\)\[(\d+)\] = \(uint32_t\)([12]);", source)
        assert {(int(offset), int(value)) for offset, value in stores} == {(0, 1), (1, 1), (8, 2), (9, 2)}
        assert "asc_sync_intra_" not in source


@pytest.mark.parametrize("divergent", [False, True])
def test_ss_peer_read_requires_explicit_synchronization(divergent):
    @T.prim_func
    def main(out: T.Tensor((2,), "uint32")):
        with T.MixedKernel(1) as (_, sid):
            words = T.alloc_ss((2,), "uint32")
            if divergent:
                if sid == 0:
                    words[0] = T.uint32(1)
                else:
                    words[1] = T.uint32(2)
            else:
                words[sid] = T.uint32(1)
            out[sid] = words[1 - sid]

    # Subcore projection exposes the missing AIV0 -> AIV1 ordering directly.
    with pytest.raises(tvm.error.InternalError, match="does not support synchronization between AIV0 and AIV1"):
        tilelang.lower(main, target="ascend")


def test_ss_vf_access_is_rejected():
    @T.prim_func
    def main(out: T.Tensor((256,), "uint32")):
        with T.MixedKernel(1) as (_, sid):
            words = T.alloc_ss((2,), "uint32")
            words[sid] = T.uint32(1)
            with T.SimtVF(threads=128):
                tx = T.get_thread_binding()
                out[tx + sid * 128] = words[sid]

    with pytest.raises(tvm.error.InternalError, match="SS supports scalar uint32/uint64 accesses outside VF blocks only"):
        tilelang.lower(main, target="ascend")


def _three_readers(rounds):
    @T.prim_func
    def main(out: T.Tensor((rounds, 3), "uint64")):
        with T.Kernel(1):
            words = T.alloc_ss((2,), "uint32")
            packed = T.view(words, shape=(1,), dtype="uint64")
            for i in T.serial(rounds):
                with T.Cube():
                    T.ascend_cross_core_wait_flag(4, "PIPE_S", 0)
                    T.ascend_cross_core_wait_flag(4, "PIPE_S", 16)
                    out[i, 2] = packed[0]
                    T.ascend_cross_core_set_flag(4, "PIPE_S", 0)
                    T.ascend_cross_core_set_flag(4, "PIPE_S", 16)
                    T.ascend_cross_core_wait_flag(4, "PIPE_S", 1)
                    T.ascend_cross_core_wait_flag(4, "PIPE_S", 17)
                    T.ascend_cross_core_set_flag(4, "PIPE_S", 1)
                    T.ascend_cross_core_set_flag(4, "PIPE_S", 17)
                with T.Vector() as sid:
                    words[sid] = T.Cast("uint32", (i % 3) * (100 + sid))
                    T.ascend_cross_core_set_flag(4, "PIPE_S", 0)
                    T.ascend_cross_core_wait_flag(4, "PIPE_S", 0)
                    out[i, sid] = packed[0]
                    T.ascend_cross_core_set_flag(4, "PIPE_S", 1)
                    T.ascend_cross_core_wait_flag(4, "PIPE_S", 1)

    return main


@pytest.mark.parametrize("mode", ["auto", "staged", "manual"])
def test_ss_publication_and_slot_reuse(mode):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    rounds = 32
    manual = mode == "manual"
    kernel = tilelang.compile(
        _three_readers(rounds) if manual else _publish_to_cube(rounds, stage=0 if mode == "staged" else None),
        out_idx=-1,
        target="ascend",
        pass_configs={"tl.enable_auto_schedule": not manual},
    )
    readers = 3 if manual else 1
    expected = torch.tensor([[(i % 3) * 100 + (((i % 3) * 101) << 32)] * readers for i in range(rounds)], dtype=torch.uint64)
    args = [] if manual else [torch.zeros((16, 16), dtype=torch.float16, device="npu")]
    for _ in range(10):
        actual = kernel(*args)
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
