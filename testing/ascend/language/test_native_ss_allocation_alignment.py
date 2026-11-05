"""The SS producer's 32-byte default coexists with explicit native alignment."""

import pytest
import tilelang
import tilelang.ascend.language as T
from testing.ascend._ir import allocated_tensors, kernel, nodes, tensor
from testing.ascend._memory_ir import address_bits
from tilelang.ascend import transform

import tvm_tirx as tvm
from tvm_tirx import ir, tirx


@pytest.mark.parametrize("dtype", ["uint32", "uint64"])
@pytest.mark.parametrize("shape", [2, (2,)])
def test_alloc_ss_emits_its_historical_32_byte_alignment(dtype, shape):
    @T.prim_func
    def main(output: T.Tensor((2,), dtype)):
        with T.Kernel(1):
            words = T.alloc_ss(shape, dtype)
            words[0] = T.Cast(dtype, 1)
            output[0] = words[0]

    assert tirx.analysis.check_well_formed(main)
    (words,) = [value for value in allocated_tensors(main) if value.ty.scope.name == "shared.ss"]
    assert tuple(int(value) for value in words.ty.shape) == (2,)
    assert str(words.ty.dtype) == dtype
    assert int(words.ty.data_alignment) == 32


@pytest.mark.parametrize("alignment", [32, 64, 128])
def test_native_ss_pool_respects_explicit_allocation_alignment(alignment):
    a, b = [tensor((2,), "uint32", name=name, scope="shared.ss", align=alignment) for name in ("a", "b")]
    before = kernel(
        [
            ir.TensorStore(a, [0], ir.const(1, "uint32")),
            ir.TensorStore(b, [0], ir.const(2, "uint32")),
        ],
        buffers=[a, b],
    )
    assert tirx.analysis.check_well_formed(before)
    saved = ir.save_json(before)
    analyzed = transform.InferBufferAliases()(before)
    assert tirx.analysis.check_well_formed(analyzed)
    after = transform.MergeUBAllocations()(analyzed)
    assert ir.save_json(before) == saved
    assert tirx.analysis.check_well_formed(after)
    (arena,) = allocated_tensors(after)
    addresses = {}
    for store in nodes(after, ir.TensorStore):
        owner, offset = address_bits(after, store)
        assert owner.same_as(arena)
        addresses[int(store.value)] = int(offset) // 8
    assert addresses == {1: 0, 2: alignment}
    assert int(arena.ty.data_alignment) >= alignment


@pytest.mark.parametrize("allocator,scope", [(T.alloc_shared, "shared.dyn"), (T.alloc_l1, "shared.l1")])
def test_other_allocation_producers_keep_the_native_default(allocator, scope):
    @T.prim_func
    def main(output: T.Tensor((2,), "float32")):
        with T.Kernel(1):
            words = allocator((2,), "float32")
            words[0] = T.float32(1)
            output[0] = words[0]

    assert tirx.analysis.check_well_formed(main)
    (words,) = [value for value in allocated_tensors(main) if value.ty.scope.name == scope]
    assert int(words.ty.data_alignment) == 64


def test_original_two_allocation_kernel_has_exact_native_store_addresses(tmp_path):
    size = 2

    @T.prim_func
    def main(out: T.Tensor((2,), "uint32")):
        with T.MixedKernel(1) as (_, sid):
            a = T.alloc_ss((size,), "uint32")
            b = T.alloc_ss((size,), "uint32")
            a[sid] = T.uint32(1)
            b[sid] = T.uint32(2)
            out[sid] = a[sid] + b[sid]

    snapshots = []

    @ir.transform.pass_instrument
    class Capture:
        def run_after_pass(self, module, info):
            if info.name == "tl.MergeUBAllocations":
                snapshots.append(module)

    saved = ir.save_json(main)
    with tvm.transform.PassContext(instruments=[Capture()]):
        artifact = tilelang.lower(main, target="ascend", enable_device_compile=False)
    assert ir.save_json(main) == saved
    (merged,) = snapshots
    assert tirx.analysis.check_well_formed(merged)
    (function,) = [
        function
        for function in merged.functions.values()
        if any(var.ty.scope.name == "shared.ss.dyn" for var in allocated_tensors(function))
    ]
    (tmp_path / "merged.json").write_text(ir.save_json(function))
    (tmp_path / "kernel.cc").write_text(artifact.kernel_source)
    (arena,) = [var for var in allocated_tensors(function) if var.ty.scope.name == "shared.ss.dyn"]
    stores = [store for store in nodes(function, ir.TensorStore) if store.dest.ty.scope.name == "shared.ss"]
    observed = []
    for store in stores:
        owner, offset_bits = address_bits(function, store)
        assert owner.same_as(arena) and int(offset_bits) % 32 == 0
        observed.append((int(offset_bits) // 32, int(store.value)))
    assert len(observed) == 4
    assert set(observed) == {(0, 1), (1, 1), (8, 2), (9, 2)}
    assert artifact.kernel_source.count("(volatile __ssbuf__ uint8_t *)0") == 1
    assert "asc_sync_intra_" not in artifact.kernel_source
