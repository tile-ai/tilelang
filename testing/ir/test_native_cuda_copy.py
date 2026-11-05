"""CUDA copy producers preserve typed addresses, effects and instruction policy."""

import os
from pathlib import Path

import pytest
import tvm_tirx
from tvm_tirx import ir, tirx

from test_native_reducer_contracts import (
    allocate,
    annotation,
    integer,
    objects,
    region,
    run_pass,
    tensor,
    tileop,
)


def function(params, body, arch="sm_90", threads=128):
    thread = ir.Var("threadIdx_x", "int32")
    launch = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), integer(threads)], [thread], {}, body)
    return (
        tirx.Function(params, [launch], ir.NoneType())
        .with_attr("global_symbol", "main")
        .with_attr("target", tvm_tirx.target.Target({"kind": "cuda", "arch": arch}))
    )


def lower(value, name):
    original = ir.save_json(value)
    inferred = run_pass("LayoutInference", value)
    result = run_pass("LowerTileOp", inferred)
    assert ir.save_json(value) == original
    assert tirx.analysis.check_well_formed(result)
    assert not tirx.analysis.undefined_vars(result)
    assert not any(call.op.name.startswith("tl.tileop.") for call in calls(result))
    destination = os.environ.get("TILELANG_OPS_ARTIFACT_DIR")
    if destination:
        directory = Path(destination)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / (name + ".json")).write_text(ir.save_json(result))
    return result


def calls(value, name=None):
    return [call for call in objects(value, ir.Call) if isinstance(call.op, ir.Op) and (name is None or call.op.name == name)]


@pytest.mark.parametrize("dtype", ["float32", "float16", "uint8"])
@pytest.mark.parametrize("size", [1, 128, 257])
def test_sync_copy_runs_real_cuda_layout_and_lowering(dtype, size):
    source = tensor("A", [size], dtype, "global")
    output = tensor("B", [size], dtype, "global")
    shared = tensor("S", [size], dtype, "shared")
    value = function(
        [source, output],
        [
            allocate(shared),
            tileop("copy", region(source), region(shared), metadata={"prefer_instruction": "sync"}),
            tileop("copy", region(shared), region(output), metadata={"prefer_instruction": "sync"}),
        ],
    )
    result = lower(value, f"cuda-sync-{dtype}-{size}")
    assert objects(result, ir.TensorLoad)
    assert objects(result, ir.TensorStore)
    assert not any("tma_" in call.op.name or "cp_async" in call.op.name for call in calls(result))


@pytest.mark.parametrize("explicit,managed", [(False, False), (True, False), (False, True)])
def test_async_copy_preserves_commit_wait_ownership(explicit, managed):
    source = tensor("A", [128], scope="global")
    shared = tensor("S", [128], scope="shared")
    metadata = {"prefer_instruction": "cp_async"}
    if explicit:
        metadata["is_async_copy"] = True
    if managed:
        metadata["no_implicit_async_commit_wait"] = True
    result = lower(
        function([source], [allocate(shared), tileop("copy", region(source), region(shared), metadata=metadata)], arch="sm_80"),
        f"cuda-async-{explicit}-{managed}",
    )
    assert calls(result, "tl.ptx_cp_async")
    assert bool(calls(result, "tl.ptx_commit_group")) == (not managed)
    assert bool(calls(result, "tl.ptx_wait_group")) == (not explicit and not managed)
    for call in calls(result, "tl.ptx_cp_async"):
        ir.assert_structural_equal(call.ty, ir.PrimType("void"))
        assert isinstance(call.args[0].ty, ir.PtrType)
        assert isinstance(call.args[1].ty, ir.PtrType)


@pytest.mark.parametrize("arch", ["sm_70", "sm_75"])
def test_explicit_async_rejects_target_without_instruction(arch):
    source = tensor("A", [128], scope="global")
    shared = tensor("S", [128], scope="shared")
    value = function(
        [source], [allocate(shared), tileop("copy", region(source), region(shared), metadata={"is_async_copy": True})], arch=arch
    )
    with pytest.raises(Exception, match="cp.async|SM80"):
        run_pass("LayoutInference", value)


@pytest.mark.parametrize("load", [False, True])
def test_tma_1d_keeps_view_origin_and_typed_pointer(load):
    memory_layout = tirx.TileLayout(tirx.layout.S[(128,)] + 16)
    global_tensor = ir.Var("G", tirx.TensorType([128], "float32", layout=memory_layout))
    shared = tensor("S", [128], scope="shared")
    barrier = tensor("mbar", [1], "uint64", "shared.barrier")
    source, destination = (global_tensor, shared) if load else (shared, global_tensor)
    metadata = {"is_tma_copy": True}
    if load:
        metadata["barrier"] = ir.TensorLoad(barrier, [0])
    value = function(
        [global_tensor], [allocate(shared), allocate(barrier), tileop("copy", region(source), region(destination), metadata=metadata)]
    )
    result = lower(value, f"cuda-tma1d-{load}")
    intrinsic = calls(result, "tl.tma_load" if load else "tl.tma_store")
    assert len(intrinsic) == 1
    assert not calls(result, "tl.create_tma_descriptor")
    global_pointer = intrinsic[0].args[1 if load else 0]
    assert global_pointer.op.name == "tl.ptr_access"
    assert isinstance(global_pointer.ty, ir.PtrType)
    assert int(tvm_tirx.sym.Analyzer().simplify(global_pointer.args[1]).value) == 16
    assert global_pointer.args[0].args[0].same_as(global_tensor)
    assert bool(calls(result, "tl.mbarrier_expect_tx")) == load
    assert not calls(result, "tl.mbarrier_wait_parity")


def test_tma_descriptor_preserves_noncontiguous_global_stride_and_offset():
    memory_layout = tirx.TileLayout(tirx.layout.S[(2, 32) : (64, 1)] + 16)
    source = ir.Var("A", tirx.TensorType([2, 32], "float32", layout=memory_layout))
    shared = tensor("S", [2, 32], scope="shared")
    barrier = tensor("mbar", [1], "uint64", "shared.barrier")
    value = function(
        [source],
        [
            allocate(shared),
            allocate(barrier),
            tileop("copy", region(source), region(shared), metadata={"is_tma_copy": True, "barrier": ir.TensorLoad(barrier, [0])}),
        ],
    )
    result = lower(value, "cuda-tma-strided-descriptor")
    descriptors = calls(result, "tl.create_tma_descriptor")
    assert descriptors
    for descriptor in descriptors:
        assert isinstance(descriptor.ty, ir.PtrType)
        assert int(descriptor.args[1].value) == 2
        analyzer = tvm_tirx.sym.Analyzer()
        assert [int(analyzer.simplify(value).value) for value in descriptor.args[5:7]] == [4, 256]
        pointer = descriptor.args[2]
        assert pointer.op.name == "tl.ptr_access"
        assert int(analyzer.simplify(pointer.args[1]).value) == 16
        assert pointer.args[0].args[0].same_as(source)


@pytest.mark.parametrize("size", [1, 15, 63])
def test_explicit_tma_rejects_non_aligned_transfer(size):
    source = tensor("A", [size], scope="global")
    shared = tensor("S", [size], scope="shared")
    barrier = tensor("mbar", [1], "uint64", "shared.barrier")
    value = function(
        [source],
        [
            allocate(shared),
            allocate(barrier),
            tileop("copy", region(source), region(shared), metadata={"is_tma_copy": True, "barrier": ir.TensorLoad(barrier, [0])}),
        ],
    )
    with pytest.raises(Exception, match="TMA|tma"):
        run_pass("LayoutInference", value)


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("origin", [(0, 0), (16, 16), (0, 1)])
def test_ldmatrix_uses_logical_slice_and_aligned_origin(transposed, origin):
    from tilelang.layout import make_gemm_fragment_8x8, make_gemm_fragment_8x8_transposed

    fragment_layout = make_gemm_fragment_8x8_transposed() if transposed else make_gemm_fragment_8x8()
    fragment_layout = fragment_layout.repeat([2, 2], repeat_on_thread=False)
    shared = tensor("S", [32, 32], "float16", "shared")
    fragment = tensor("F", [16, 16], "float16", "local.fragment")
    shared_region = ir.TensorRegion(
        shared, [ir.Range.from_min_extent(origin[0], 16), ir.Range.from_min_extent(origin[1], 16)], ir.TensorRegionType()
    )
    value = function(
        [],
        [
            allocate(shared),
            allocate(fragment),
            annotation({"layout_map": {fragment: fragment_layout}}),
            tileop("copy", shared_region, region(fragment)),
        ],
        arch="sm_80",
        threads=32,
    )
    result = lower(value, f"cuda-ldmatrix-{transposed}-{origin[0]}-{origin[1]}")
    instructions = calls(result, "tl.ptx_ldmatrix")
    assert bool(instructions) == (origin[1] % 8 == 0)
    for instruction in instructions:
        assert bool(instruction.args[0].value) == transposed
        assert isinstance(instruction.args[2].ty, ir.PtrType)
        assert isinstance(instruction.args[3].ty, ir.PtrType)


@pytest.mark.parametrize("load", [False, True])
def test_gather_scatter_descriptor_encodes_one_row_box(load):
    global_tensor = tensor("G", [8, 64], scope="global")
    shared = tensor("S", [4, 32], scope="shared")
    barrier = tensor("mbar", [1], "uint64", "shared.barrier")
    logical = ir.TensorRegion(global_tensor, [ir.Range(0, 4), ir.Range.from_min_extent(16, 32)], ir.TensorRegionType())
    source, destination = (logical, region(shared)) if load else (region(shared), logical)
    metadata = {
        "is_gather4" if load else "is_scatter4": True,
        "gather4_rows": [integer(x) for x in (1, 3, 5, 7)],
        "gather4_col": integer(16),
    }
    if load:
        metadata["barrier"] = ir.TensorLoad(barrier, [0])
    value = function(
        [global_tensor], [allocate(shared), allocate(barrier), tileop("copy", source, destination, metadata=metadata)], arch="sm_100a"
    )
    result = lower(value, f"cuda-gather-scatter-{load}")
    intrinsic_name = "tl.tma_load_gather4" if load else "tl.tma_store_scatter4"
    assert calls(result, intrinsic_name)
    for descriptor in calls(result, "tl.create_tma_descriptor"):
        rank = int(descriptor.args[1].value)
        assert rank == 2
        assert int(descriptor.args[3 + 2 * rank + 1].value) == 1
    assert not calls(result, "tl.mbarrier_expect_tx")
    assert not calls(result, "tl.mbarrier_wait_parity")


@pytest.mark.parametrize("load", [False, True])
def test_tmem_copy_uses_native_tensor_and_register_pointer(load):
    from tilelang.layout import Layout

    tmem = tensor("tmem", [128, 32], "float32", "shared.tmem")
    fragment = tensor("fragment", [128, 32], "float32", "local.fragment")
    layout = Layout([128, 32], lambda i, j: [i, j])
    source, destination = (tmem, fragment) if load else (fragment, tmem)
    value = function(
        [],
        [
            allocate(tmem),
            allocate(fragment),
            annotation({"layout_map": {tmem: layout}}),
            tileop("copy", region(source), region(destination)),
        ],
        arch="sm_100a",
    )
    result = lower(value, f"cuda-tmem-{load}")
    instructions = calls(result, "tl.tcgen05_ld" if load else "tl.tcgen05_st")
    assert instructions
    for instruction in instructions:
        assert isinstance(instruction.args[3], ir.TensorLoad)
        assert isinstance(instruction.args[6].ty, ir.PtrType)
