"""Keep the existing WebGPU sync_threads fallback at the final TL boundary.

WGSL source and native IR are checked here; these tests do not execute a browser
or a WebGPU device. An external call retained in the source is an evaluation
order probe, not a complete standalone shader.
"""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

import tilelang
import tilelang.language as T
from tilelang.webgpu.codegen import _lower_tilelang_sync, build_webgpu


def _objects(value, cls):
    result = []
    tvm_ffi.structural_walk(value, (cls, result.append), order="pre")
    return result


def _module(body, params=()):
    function = tirx.Function(list(params), body, ir.PrimType("void")).with_attr("global_symbol", "main")
    return ir.IRModule({"main": function})


def make_sync_kernel(mode):
    @T.prim_func
    def kernel(
        A: T.Tensor((32,), "int32"),
        B: T.Tensor((32,), "int32"),
        barrier_id: T.int32,
        thread_count: T.int32,
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            B[tx] = A[tx]
            if mode == 0:
                T.sync_threads()
            elif mode == 1:
                T.sync_threads(1)
            elif mode == 2:
                T.sync_threads(1, 32)
            elif mode == 3:
                T.sync_threads(barrier_id)
            elif mode == 4:
                T.sync_threads(barrier_id, thread_count)
            elif mode == 5:
                T.sync_threads(A[0])
            elif mode == 6:
                T.sync_threads(T.call_extern("int32", "barrier_side_effect"))
            elif mode == 7:
                T.sync_threads(1, T.call_extern("int32", "barrier_side_effect"))
            elif mode == 8:
                value = T.call_extern("int32", "barrier_side_effect")
                T.sync_threads(value)
            else:
                T.evaluate(T.call_extern("int32", "barrier_side_effect"))
                T.sync_threads(1)

    return kernel


@pytest.mark.parametrize("mode", range(10))
def test_public_sync_threads_matches_existing_webgpu_evaluation(mode, tmp_path):
    with Target("webgpu"):
        function = make_sync_kernel(mode)
        original = ir.save_json(function)
        artifact = tilelang.lower(function, target="webgpu")
    assert ir.save_json(function) == original
    source = artifact.kernel_source
    (tmp_path / "kernel.wgsl").write_text(source)
    assert source.count("workgroupBarrier();") == 1
    # Both old inline ID/count expressions were ignored by its WebGPU emitter.
    # A separate Bind or Evaluate is observable and must still execute once.
    assert source.count("barrier_side_effect") == (1 if mode in (8, 9) else 0)
    if mode in (8, 9):
        assert source.index("barrier_side_effect") < source.index("workgroupBarrier();")
    if mode in (3, 4):
        assert "barrier_id: i32" in source
    if mode == 4:
        assert "thread_count: i32" in source
    # Codegen is late: the producer IR and outlined module still retain the TL
    # operation, including its access and effect operands for earlier passes.
    calls = [call for call in _objects(artifact.device_mod, ir.Call) if isinstance(call.op, ir.Op)]
    assert sum(call.op.name == "tl.storage_sync" for call in calls) == (0 if mode == 0 else 1)


@pytest.mark.parametrize("scope", ["shared", "shared.dyn"])
def test_terminal_rewrite_preserves_locations_attributes_and_input_identity(scope):
    location = ir.SourceLoc(ir.SourceName("webgpu-terminal-sync"), 2, 1, 2, 32)
    barrier_id = ir.Var("barrier_id", "int32")
    count = ir.Var("count", "int32")
    call = ir.Call(
        "tl.storage_sync",
        [ir.StringImm(scope, loc=location), barrier_id, count],
        attrs={"diagnostic_tag": "legacy-webgpu-sync"},
        loc=location,
    )
    module = _module([ir.RegionStmt("tl.lexical_scope", [], [], {}, [ir.Evaluate(call)])], [barrier_id, count])
    before = ir.save_json(module)
    result = _lower_tilelang_sync(module)
    assert ir.save_json(module) == before
    assert not result.same_as(module)
    assert result["main"].params[0].same_as(barrier_id)
    assert result["main"].params[1].same_as(count)
    (actual,) = _objects(result, ir.Call)
    actual.validate()
    assert actual.op.name == "tirx.gpu_storage_sync"
    assert len(actual.args) == 1
    assert actual.args[0].value == "shared"
    for old, new in [
        (call.attrs, actual.attrs),
        (call.ty, actual.ty),
        (call.ty_args, actual.ty_args),
        (call.loc, actual.loc),
        (call.args[0].loc, actual.args[0].loc),
    ]:
        ir.assert_structural_equal(old, new)
    assert tirx.analysis.check_well_formed(result["main"])
    assert not tirx.analysis.undefined_vars(result["main"])
    assert _lower_tilelang_sync(result).same_as(result)


@pytest.mark.parametrize("scope", ["shared", "warp", "global"])
def test_native_scope_only_operation_is_not_rewritten(scope):
    call = ir.Call("tirx.gpu_storage_sync", [ir.StringImm(scope)])
    module = _module([ir.Evaluate(call)])
    assert _lower_tilelang_sync(module).same_as(module)
    assert _objects(module, ir.Call)[0].same_as(call)


@pytest.mark.parametrize("extra_args", [[1], [1, 32]])
def test_native_extra_arguments_are_rejected_without_being_erased(extra_args):
    call = ir.Call("tirx.gpu_storage_sync", [ir.StringImm("shared"), *[ir.IntImm("int32", value) for value in extra_args]], ty="void")
    module = _module([ir.Evaluate(call)])
    before = ir.save_json(module)
    with pytest.raises(TypeError, match="Call.args expected 1 argument"):
        build_webgpu(module, Target("webgpu"))
    assert ir.save_json(module) == before
    assert len(call.args) == 1 + len(extra_args)


@pytest.mark.parametrize("scope,values", [("shared", []), ("global", [1]), ("shared", [1, 32, 64])])
def test_provisional_invalid_tilelang_barrier_is_not_repaired(scope, values):
    call = ir.Call("tl.storage_sync", [ir.StringImm(scope), *[ir.IntImm("int32", value) for value in values]], ty="void")
    module = _module([ir.Evaluate(call)])
    before = ir.save_json(module)
    with pytest.raises(ValueError, match="expects|requires shared"):
        _lower_tilelang_sync(module)
    assert ir.save_json(module) == before


def test_provisional_nonvoid_tilelang_barrier_is_rejected():
    call = ir.Call("tl.storage_sync", [ir.StringImm("shared"), ir.IntImm("int32", 1)], ty="int32")
    with pytest.raises(TypeError, match="void result"):
        _lower_tilelang_sync(_module([ir.Evaluate(call)]))
