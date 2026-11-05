"""Opaque matrix capacity and native Metal host context preserve real IR scope."""

import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx

import tilelang
import tilelang.language as T
from tilelang.analysis._visitor import tilelang_metadata


def fragment_gemm(rows=16, columns=16, threads=128):
    @T.prim_func
    def main(C: T.Tensor((rows, columns), "float32")):
        with T.Kernel(1, threads=threads):
            a = T.alloc_shared((rows, 16), "float16")
            b = T.alloc_shared((16, columns), "float16")
            accum = T.alloc_fragment((rows, columns), "float32")
            T.clear(accum)
            T.gemm(a, b, accum)
            T.copy(accum, C)

    return main


def nodes(value, kind):
    result = []
    tvm_ffi.structural_walk(value, (kind, result.append))
    return result


def prepare(function):
    target = tvm_tirx.target.Target({"kind": "metal", "keys": ["metal", "gpu"]})
    module = ir.IRModule({"main": function})
    module = tilelang.transform.NormalizeAliasAnnotations()(module)
    module = tilelang.transform.LowerTensorSpec()(module)
    module = tirx.transform.BindTarget(target)(module)
    module = tilelang.transform.MaterializeKernelLaunch(lower_thread_binding=True, lower_grid_binding=True, default_threads=128)(module)
    return module, target


@pytest.mark.parametrize("rows,columns,threads", [(16, 16, 128), (32, 64, 128), (16, 16, 32)])
def test_promoted_matrix_keeps_logical_regions_and_per_warp_capacity(rows, columns, threads):
    module, target = prepare(fragment_gemm(rows, columns, threads))
    before = ir.save_json(module)
    promote = tvm_ffi.get_global_func("tl.transform.MetalFragmentToSimdgroup")()
    promoted = promote(module)
    assert ir.save_json(module) == before
    assert tirx.analysis.check_well_formed(promoted)
    ir.assert_structural_equal(promote(promoted), promoted)
    ir.assert_structural_equal(ir.load_json(ir.save_json(promoted)), promoted)
    definitions = [
        binding
        for binding in nodes(promoted, ir.Bind)
        if isinstance(binding.var.ty, tirx.TensorType) and str(binding.var.ty.scope) == "metal.simdgroup"
    ]
    assert len(definitions) == 1
    definition = definitions[0]
    assert [int(value) for value in definition.var.ty.shape] == [rows, columns]
    assert definition.value.attrs["tl.metal_simdgroup_physical_elements"] == rows * columns // (threads // 32)
    uses = [region for region in nodes(promoted, ir.TensorRegion) if region.source.same_as(definition.var)]
    assert uses
    assert all([int(axis.extent) for axis in region.region] == [rows, columns] for region in uses)
    import tilelang.metal

    tilelang.metal.register_backend()
    with target:
        lowered = tilelang.transform.LowerTileOp()(promoted)
    assert tirx.analysis.check_well_formed(lowered)
    definitions = [
        binding for binding in nodes(lowered, ir.Bind) if binding.var.name == "accum" and isinstance(binding.var.ty, tirx.TensorType)
    ]
    assert len(definitions) == 1
    physical = definitions[0].var
    matrices = rows * columns // (threads // 32) // 64
    assert [int(value) for value in physical.ty.shape] == [matrices * 64]
    assert "tl.metal_simdgroup_physical_elements" not in definitions[0].value.attrs
    fills = [
        call
        for call in nodes(lowered, ir.Call)
        if getattr(call.op, "name", "") == "tirx.metal.make_filled_simdgroup_matrix" and call.args[0].same_as(physical)
    ]
    assert {int(call.args[1]) for call in fills} == set(range(matrices))


def packed_call(name="device"):
    pointer = ir.Var("stack", ir.PtrType("uint8"))
    call = ir.Call("tirx.call_packed_lowered", [ir.StringImm(name), pointer, ir.const(0), ir.const(0)])
    return pointer, ir.Evaluate(call)


def test_host_context_tracks_native_scope_and_is_idempotent():
    from tilelang.metal.transform import MarkHostMetalContext

    pointer, launch = packed_call()
    unrelated = ir.RegionStmt("tl.lexical_scope", [], [], {}, [ir.Evaluate(1)])
    compute = ir.RegionStmt("tirx.compute_scope", [ir.StringImm("compute")], [], {}, [unrelated, launch])
    function = tirx.Function([pointer], [launch, compute])
    module = ir.IRModule({"main": function})
    original = ir.save_json(module)
    marked = MarkHostMetalContext()(module)
    assert ir.save_json(module) == original
    assert tirx.analysis.check_well_formed(marked)
    contexts = [region for region in nodes(marked, ir.RegionStmt) if (tilelang_metadata(region) or {}).get("key") == "metal_context"]
    assert len(contexts) == 1
    assert sum(region.op.name == "tl.lexical_scope" for region in nodes(marked, ir.RegionStmt)) == 1
    ir.assert_structural_equal(MarkHostMetalContext()(marked), marked)
