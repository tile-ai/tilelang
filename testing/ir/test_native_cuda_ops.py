"""CUDA tile operators use native types, dispatch and memory contracts."""

import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx

from tilelang.layout import Fragment

from test_native_cuda_copy import calls, function, lower
from test_native_reducer_contracts import allocate, annotation, integer, objects, region, tensor, tileop


def cuda_function(params, body, arch="sm_100a", threads=128, blocks=1):
    value = function(params, body, arch=arch, threads=threads)
    block = ir.Var("blockIdx_x", "int32")
    return tirx.Function(
        value.params,
        [ir.RegionStmt("tirx.launch_thread", [ir.StringImm("blockIdx.x"), integer(blocks)], [block], {}, value.body)],
        ir.NoneType(),
        attrs=value.attrs,
    )


def gemm(a, b, c, m, n, k, transpose_b=True, clear=True, metadata=None, name="gemm"):
    return tileop(
        name,
        region(a),
        region(b),
        region(c),
        integer(0, "bool"),
        integer(int(transpose_b), "bool"),
        integer(m),
        integer(n),
        integer(k),
        integer(0),
        integer(int(clear), "bool"),
        integer(0),
        integer(0),
        integer(0),
        metadata=metadata,
    )


@pytest.mark.parametrize(
    "arch,value,bulk", [("sm_100a", 0.0, True), ("sm_90", 0.0, False), ("sm_100a", -0.0, False), ("sm_100a", 2.0, False)]
)
def test_fill_bulk_keeps_zero_bit_pattern_and_architecture(arch, value, bulk):
    output = tensor("B", [128], scope="global")
    shared = tensor("S", [128], scope="shared")
    result = lower(
        function([output], [allocate(shared), tileop("fill", region(shared), ir.FloatImm("float32", value))], arch=arch),
        f"cuda-fill-{arch}-{value}",
    )
    instructions = calls(result, "tl.ptx_st_bulk_shared")
    assert bool(instructions) == bulk
    if bulk:
        assert len(instructions) == 1
        call = instructions[0]
        assert isinstance(call.args[0].ty, ir.PtrType)
        assert call.args[0].ty.scope.name in ("shared", "shared.dyn")
        assert int(call.args[1].value) == 512
        ir.assert_structural_equal(call.ty, ir.PrimType("void"))


@pytest.mark.parametrize("dtype", ["float32", "int32"])
@pytest.mark.parametrize("name", ["atomicadd", "atomicmax", "atomicmin"])
def test_atomic_tile_lowering_uses_native_pointer_and_void_result(dtype, name):
    source, output = tensor("A", [257], dtype, "global"), tensor("B", [257], dtype, "global")
    result = lower(function([source, output], [tileop(name, region(source), region(output))]), "cuda-" + name + "-" + dtype)
    expected_op = {"atomicadd": "tl.atomic_add_elem_op", "atomicmax": "tl.atomic_max_elem_op", "atomicmin": "tl.atomic_min_elem_op"}[name]
    atomics = calls(result, expected_op)
    assert atomics
    for call in atomics:
        assert isinstance(call.args[0].ty, ir.PtrType)
        ir.assert_structural_equal(call.args[0].ty.element_type, ir.PrimType(dtype))
        ir.assert_structural_equal(call.ty, ir.PrimType("void"))


@pytest.mark.parametrize(
    "arch,dtype,m,n,k,scope,expected",
    [
        ("sm_70", "float32", 32, 32, 32, "local.fragment", "cuda.fma"),
        ("sm_75", "float16", 32, 32, 32, "local.fragment", "cuda.mma"),
        ("sm_80", "float16", 32, 32, 32, "local.fragment", "cuda.mma"),
        ("sm_80", "bfloat16", 32, 32, 32, "local.fragment", "cuda.mma"),
        ("sm_90", "float16", 64, 64, 32, "local.fragment", "cuda.wgmma"),
        ("sm_100a", "float16", 64, 64, 32, "shared.tmem", "cuda.tcgen05"),
    ],
)
def test_gemm_instruction_selection_reads_native_tensor_types(arch, dtype, m, n, k, scope, expected):
    a, b = tensor("A", [m, k], dtype, "shared"), tensor("B", [n, k], dtype, "shared")
    c = tensor("C", [m, n], "float32", scope)
    parsed = tvm_ffi.get_global_func("tl.testing.ParseNativeOperator")(gemm(a, b, c, m, n, k).value)
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": arch})
    assert tvm_ffi.get_global_func("tl.GemmGetGemmInstructionKey")(parsed, 128, target) == expected


def test_explicit_two_cta_has_no_ordinary_mma_fallback():
    a, b = tensor("A", [64, 32], "float16", "shared"), tensor("B", [32, 32], "float16", "shared")
    c = tensor("C", [64, 64], "float32", "local.fragment")
    parsed = tvm_ffi.get_global_func("tl.testing.ParseNativeOperator")(gemm(a, b, c, 64, 64, 32, metadata={"use_2cta": integer(1)}).value)
    with pytest.raises(Exception, match="use_2cta.*Blackwell|TCGEN"):
        tvm_ffi.get_global_func("tl.GemmGetGemmInstructionKey")(parsed, 128, tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_80"}))


def test_tcgen_metadata_uses_native_ffi_type_arguments():
    values = tvm_ffi.get_global_func("tl.get_tcgen5_mma_meta")(64, 64, 32, ir.PrimType("float16"), ir.PrimType("float32"), True, False)
    assert len(values) == 5
    assert all(isinstance(value, ir.IntImm) for value in values)
    assert [int(value.value) for value in values[:3]] == [64, 64, 16]


def test_reduction_workspace_is_defined_inside_native_thread_region():
    source, output = tensor("A", [8, 256], scope="global"), tensor("B", [8], scope="global")
    fragment = tensor("F", [8, 256], scope="local.fragment")
    reduced = tensor("R", [8], scope="local.fragment")
    # This test exercises cross-warp storage, which a warp-local reduction
    # does not need. Use the same 128-thread ownership as the execution test.
    layout = Fragment([8, 256], forward_fn=lambda i, j: (j % 128, i * 2 + j // 128))
    result = lower(
        cuda_function(
            [source, output],
            [
                allocate(fragment),
                allocate(reduced),
                annotation({"layout_map": {fragment: layout}}),
                tileop("copy", region(source), region(fragment), metadata={"prefer_instruction": "sync"}),
                tileop("reduce", region(fragment), region(reduced), ir.StringImm("sum"), integer(1), integer(1, "bool")),
                tileop("copy", region(reduced), region(output), metadata={"prefer_instruction": "sync"}),
            ],
        ),
        "cuda-reduction-workspace-scope",
    )
    workspaces = [binding for binding in objects(result, ir.Bind) if binding.var.name.startswith("workspace")]
    assert len(workspaces) == 1
    thread = next(
        node for node in objects(result, ir.RegionStmt) if node.op.name == "tirx.launch_thread" and node.args[0].value == "threadIdx.x"
    )
    assert any(binding.same_as(workspaces[0]) for binding in objects(thread.body, ir.Bind))
    assert workspaces[0].var.ty.scope.name == "shared.dyn"


def test_tma_atomic_add_preserves_strided_destination_and_read_source():
    memory = tirx.TileLayout(tirx.layout.S[(8, 64) : (80, 1)] + 16)
    destination = ir.Var("B", tirx.TensorType([8, 64], "float32", layout=memory))
    shared = tensor("S", [8, 32], scope="shared")
    selected = ir.TensorRegion(destination, [ir.Range.from_min_extent(0, 8), ir.Range.from_min_extent(16, 32)], ir.TensorRegionType())
    value = function([destination], [allocate(shared), tileop("atomicadd", region(shared), selected, metadata={"use_tma": True})])
    result = lower(value, "cuda-atomic-tma-strided")
    descriptors = calls(result, "tl.create_tma_descriptor")
    stores = calls(result, "tl.tma_store")
    assert len(descriptors) == len(stores) == 1
    descriptor, store = descriptors[0], stores[0]
    assert isinstance(descriptor.ty, ir.PtrType)
    assert int(descriptor.args[1].value) == 2
    analyzer = tvm_tirx.sym.Analyzer()
    assert [int(analyzer.simplify(x).value) for x in descriptor.args[5:7]] == [4, 320]
    assert int(analyzer.simplify(descriptor.args[2].args[1]).value) == 16
    assert isinstance(store.args[1].ty, ir.PtrType)
    assert int(store.args[1].args[-1].value) == 1
    assert isinstance(store.args[-1], ir.Tuple)
    assert [int(x.value) for x in store.args[-3:-1]] == [1, 0]
    assert calls(result, "tl.tma_store_arrive") and calls(result, "tl.tma_store_wait")


@pytest.mark.parametrize(
    "arch,dtype,scalar,message",
    [
        ("sm_80", "float32", False, "SM90|TMA support"),
        ("sm_90", "float64", False, "does not support dtype"),
        ("sm_90", "float32", True, "PrimExpr as value"),
    ],
)
def test_explicit_tma_atomic_add_rejects_unsupported_contract(arch, dtype, scalar, message):
    destination = tensor("B", [128], dtype, "global")
    shared = tensor("S", [128], dtype, "shared")
    source = ir.FloatImm(dtype, 1) if scalar else region(shared)
    value = function(
        [destination], [allocate(shared), tileop("atomicadd", source, region(destination), metadata={"use_tma": True})], arch=arch
    )
    with pytest.raises(Exception, match=message):
        lower(value, f"cuda-atomic-tma-negative-{arch}-{dtype}-{scalar}")


@pytest.mark.parametrize(
    "arch,output_scope,expected",
    [
        ("sm_100a", "shared.tmem", "cuda.tcgen05.blockscaled"),
        ("sm_120a", "local.fragment", "cuda.mma.blockscaled"),
    ],
)
def test_blockscaled_selection_preserves_native_scale_operands(arch, output_scope, expected):
    a, b = tensor("A", [64, 32], "float8_e4m3fn", "shared"), tensor("B", [64, 32], "float8_e4m3fn", "shared")
    c = tensor("C", [64, 64], "float32", output_scope)
    sfa, sfb = tensor("SFA", [64, 1], "uint8", "shared"), tensor("SFB", [64, 1], "uint8", "shared")
    args = list(gemm(a, b, c, 64, 64, 32).value.args[:13]) + [region(sfa), region(sfb), integer(0)]
    parsed = tvm_ffi.get_global_func("tl.testing.ParseNativeOperator")(tileop("gemm_blockscaled", *args).value)
    key = tvm_ffi.get_global_func("tl.GemmGetGemmInstructionKey")(parsed, 128, tvm_tirx.target.Target({"kind": "cuda", "arch": arch}))
    assert key == expected
    assert parsed.sfaRegion.source.same_as(sfa) and parsed.sfbRegion.source.same_as(sfb)


def test_blockscaled_explicit_tcgen05_has_no_sm120_mma_fallback():
    a, b = tensor("A", [64, 32], "float8_e4m3fn", "shared"), tensor("B", [64, 32], "float8_e4m3fn", "shared")
    c = tensor("C", [64, 64], "float32", "local.fragment")
    sf = tensor("SF", [64, 1], "uint8", "shared")
    args = list(gemm(a, b, c, 64, 64, 32).value.args[:13]) + [region(sf), region(sf), integer(0)]
    parsed = tvm_ffi.get_global_func("tl.testing.ParseNativeOperator")(tileop("tcgen05_gemm_blockscaled", *args).value)
    with pytest.raises(Exception, match="SM100 TCGEN5MMA"):
        tvm_ffi.get_global_func("tl.GemmGetGemmInstructionKey")(parsed, 128, tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_120a"}))


@pytest.mark.parametrize(
    "arch,opname,expected",
    [("sm_80", "gemm_sp", "cuda.mma.sp"), ("sm_90", "gemm_sp", "cuda.wgmma.sp"), ("sm_100a", "tcgen05_gemm_sp", None)],
)
def test_sparse_selection_preserves_metadata_and_explicit_unavailable_instruction(arch, opname, expected):
    a, b = tensor("A", [64, 32], "float16", "shared"), tensor("B", [64, 64], "float16", "shared")
    c, metadata = tensor("C", [64, 64], "float32", "local.fragment"), tensor("E", [64, 4], "uint16", "shared")
    args = [
        region(a),
        region(metadata),
        region(b),
        region(c),
        integer(0),
        integer(0),
        integer(1),
        integer(64),
        integer(64),
        integer(64),
        integer(0),
        integer(1, "bool"),
    ]
    parsed = tvm_ffi.get_global_func("tl.testing.ParseNativeOperator")(tileop(opname, *args).value)
    select = tvm_ffi.get_global_func("tl.GemmSPGetGemmInstructionKey")
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": arch})
    if expected is None:
        with pytest.raises(Exception, match="sparse TCGEN5MMA lowering is not yet available"):
            select(parsed, 128, target)
    else:
        assert select(parsed, 128, target) == expected
