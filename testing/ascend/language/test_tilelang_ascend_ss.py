"""Group-shared scalar storage: allocation, publication, and slot reuse."""

from tvm_tirx import ir
from testing.ascend._ir import access_tensor

import re

import pytest
import torch

import tilelang
import tilelang.ascend.language as T
import tvm_tirx
from testing.ascend._ir import allocated_buffer, nodes
from testing.ascend._memory_ir import element_offset, storage_owner


def _ss_uint32_base_words(source):
    """Resolve native typed views back to uint32 offsets in the SS pool."""
    arenas = re.findall(r"volatile __ssbuf__ uint8_t\s*\*\s*(\w+)\s*=\s*\(volatile __ssbuf__ uint8_t\s*\*\)0;", source)
    assert len(arenas) == 1
    declarations = re.findall(r"volatile __ssbuf__ uint32_t\s*\*\s*(\w+)\s*=\s*(.+);", source)
    assert declarations
    bases = {}
    for name, pointer in declarations:
        offset = re.search(rf"\(volatile __ssbuf__ char\s*\*\){re.escape(arenas[0])}\)\s*\+\s*\(int64_t\)(\d+)\)", pointer)
        assert offset is not None, pointer
        byte_offset = int(offset.group(1))
        assert byte_offset % 4 == 0
        bases[name] = byte_offset // 4
    return bases


def _publish_to_cube(rounds=8, stage=0, versions=2, version_mode="auto"):
    @T.prim_func
    def main(seed: T.Tensor((16, 16), "float16"), out: T.Tensor((rounds, 1), "uint64")):
        with T.MixedKernel(1) as (_, sid):
            words = T.alloc_ss((2,), "uint32")
            packed = T.view(words, shape=(1,), dtype="uint64")
            if versions is not None:
                T.annotate_buffer_versions({words: (versions, version_mode)})
            cube = T.alloc_l1((16, 16), "float16")
            # Include Cube work so the uniform scalar consumer runs on AIC.
            T.copy(seed, cube)
            for i in T.Pipelined(rounds, num_stages=versions or 3, annotations={"enable_offset": True}):
                if version_mode != "counter" or i % 2 == 0:
                    if stage is None:
                        words[sid] = T.Cast("uint32", (i + 1) * (100 + sid))
                        out[i, 0] = packed[0]
                    else:
                        with T.Stage(0):
                            words[sid] = T.Cast("uint32", (i + 1) * (100 + sid))
                        with T.Stage(stage):
                            out[i, 0] = packed[0]
                else:
                    out[i, 0] = T.uint64(0)

    return main


@pytest.mark.parametrize("versions,stage,version_mode", [(None, 2, "auto"), (2, None, "auto"), (3, 2, "iteration"), (3, 0, "counter")])
def test_ss_codegen_materializes_versions_and_uses_mode4(versions, stage, version_mode):
    snapshots = {}

    @ir.transform.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name == "tl.MaterializeMultiBuffer":
                snapshots[info.name] = mod

    with tvm_tirx.transform.PassContext(instruments=[Capture()]):
        source = tilelang.lower(_publish_to_cube(versions=versions, stage=stage, version_mode=version_mode), target="ascend").kernel_source
    materialized = snapshots["tl.MaterializeMultiBuffer"]
    words = allocated_buffer(materialized, "words")
    actual_versions = int(words.ty.shape[0])
    assert (2 <= actual_versions <= 3) if versions is None else actual_versions == versions
    assert tuple(int(dim) for dim in words.ty.shape) == (actual_versions, 2)
    accesses = [
        access
        for kind in (ir.TensorLoad, ir.TensorStore)
        for access in nodes(materialized, kind)
        if storage_owner(materialized, access).same_as(words)
    ]
    assert {str(access_tensor(access).ty.dtype) for access in accesses} == {"uint32", "uint64"}
    assert all(len(access.indices) == 2 for access in accesses)
    analyzer = tvm_tirx.sym.Analyzer()
    for access in accesses:
        tensor = access_tensor(access)
        # Both dtype views must step over exactly the same physical slot.
        # A version slot is two uint32 elements, or one uint64 element.
        stride = element_offset(materialized, ir.TensorLoad(tensor, [1, 0])) - element_offset(materialized, ir.TensorLoad(tensor, [0, 0]))
        assert analyzer.can_prove_equal(stride, 2 if str(tensor.ty.dtype) == "uint32" else 1)
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
        T.alloc_ss((ir.Var("n", "int32"),), "uint32")


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
        with pytest.raises(RuntimeError, match="3072"):
            tilelang.lower(main, target="ascend")
    else:
        source = tilelang.lower(main, target="ascend").kernel_source
        assert source.count("(volatile __ssbuf__ uint8_t *)0") == 1
        # Two eight-byte allocations must have separate 32-byte aligned bases.
        bases = _ss_uint32_base_words(source)
        stores = re.findall(r"\b(\w+)\[(\d+)\] = \(uint32_t\)([12]);", source)
        assert {(bases[name] + int(offset), int(value)) for name, offset, value in stores} == {(0, 1), (1, 1), (8, 2), (9, 2)}
        assert "asc_sync_intra_" not in source


@pytest.mark.parametrize("size", [96, 100], ids=["fits", "exceeds-pool"])
def test_ss_fixed_versions_check_final_allocation_capacity(size):
    @T.prim_func
    def main(seed: T.Tensor((16, 16), "float16"), out: T.Tensor((4,), "uint64")):
        with T.MixedKernel(1) as (_, sid):
            a = T.alloc_ss((size,), "uint64")
            b = T.alloc_ss((size,), "uint64")
            T.annotate_buffer_versions({a: 2, b: 2})
            cube = T.alloc_l1((16, 16), "float16")
            T.copy(seed, cube)
            for i in T.Pipelined(4, num_stages=2):
                a[sid] = T.Cast("uint64", i + sid)
                b[sid] = T.Cast("uint64", i + sid + 1)
                out[i] = a[0] + a[1] + b[0] + b[1]

    # Fixed version counts are validated after materialization and packing.
    # Two allocations with two versions occupy 3072 or 3200 bytes.
    if size == 100:
        with pytest.raises(RuntimeError, match="SS allocations require a static total size of at most 3072 bytes"):
            tilelang.lower(main, target="ascend", enable_device_compile=False)
    else:
        tilelang.lower(main, target="ascend", enable_device_compile=False)


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
    with pytest.raises(RuntimeError, match="does not support synchronization between AIV0 and AIV1"):
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

    with pytest.raises(RuntimeError, match="SS supports scalar uint32/uint64 accesses outside VF blocks only"):
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
                    words[sid] = T.Cast("uint32", (i + 1) * (100 + sid))
                    T.ascend_cross_core_set_flag(4, "PIPE_S", 0)
                    T.ascend_cross_core_wait_flag(4, "PIPE_S", 0)
                    out[i, sid] = packed[0]
                    T.ascend_cross_core_set_flag(4, "PIPE_S", 1)
                    T.ascend_cross_core_wait_flag(4, "PIPE_S", 1)

    return main


@pytest.mark.parametrize("mode", ["inferred", "auto", "staged", "counter", "manual"])
def test_ss_publication_and_slot_reuse(mode):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    rounds = 32
    manual = mode == "manual"
    kernel = tilelang.compile(
        _three_readers(rounds)
        if manual
        else _publish_to_cube(
            rounds,
            stage=2 if mode in ("inferred", "staged") else (0 if mode == "counter" else None),
            versions=None if mode == "inferred" else (2 if mode == "auto" else 3),
            version_mode="counter" if mode == "counter" else "auto",
        ),
        out_idx=-1,
        target="ascend",
        pass_configs={"tl.enable_auto_schedule": not manual},
    )
    readers = 3 if manual else 1
    expected = torch.tensor(
        [[0 if mode == "counter" and i % 2 else (i + 1) * 100 + (((i + 1) * 101) << 32)] * readers for i in range(rounds)],
        dtype=torch.uint64,
    )
    args = [] if manual else [torch.zeros((16, 16), dtype=torch.float16, device="npu")]
    for _ in range(10):
        actual = kernel(*args)
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
