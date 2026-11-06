"""Register reallocation respects native role scopes and hardware budgets."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.ir.transform import PassContext
from tvm_tirx.target import Target

import tilelang
import tilelang.language as T
from tilelang.testing.ir import collect_calls, collect_nodes


def _register_kernel(producers, consumers, *, hints=None, disabled=False, simt=False, custom=False):
    @T.prim_func
    def kernel(A: T.Tensor((1024,), T.float32)):
        tx = T.launch_thread("threadIdx.x", producers + consumers)
        shared = T.alloc_shared((1024,), T.float32)
        if hints is not None:
            T.annotate_producer_reg_dealloc(hints[0])
            T.annotate_consumer_reg_alloc(hints[1])
        if disabled:
            T.disable_warp_group_reg_alloc()
        if custom:
            T.attr(0, "kCustomWarpSpecialization", 1)
        T.attr([producers, consumers], "kWarpSpecializationScope", 0)
        if tx < producers:
            if simt:
                shared[tx] = A[tx]
            else:
                T.evaluate(0)
        else:
            A[tx] = T.float32(1)

    return kernel


def _apply(function):
    module = ir.IRModule.from_expr(function.with_attr("global_symbol", "main"))
    result = tilelang.cuda.transform.AnnotateWarpGroupRegAlloc()(module)["main"]
    assert tirx.analysis.check_well_formed(result)
    assert tirx.analysis.check_well_formed(tirx.renew_def(result))
    return result


@pytest.mark.parametrize(
    "producers,consumers,decrease,increase",
    [(128, 128, 24, 240), (128, 256, 24, 240), (256, 256, 24, 224), (128, 384, 24, 160)],
)
def test_default_register_budget_fits_roles(producers, consumers, decrease, increase):
    result = _apply(_register_kernel(producers, consumers))
    calls = collect_calls(result, op="tl.set_max_nreg")
    assert [tuple(int(arg) for arg in call.args) for call in calls] == [(decrease, 0), (increase, 1)]
    assert producers * decrease + consumers * increase <= 64512
    branch = next(node for node in collect_nodes(result, ir.If) if node.else_case is not None)
    assert [int(call.args[1]) for call in collect_calls(branch.then_case, op="tl.set_max_nreg")] == [0]
    assert [int(call.args[1]) for call in collect_calls(branch.else_case, op="tl.set_max_nreg")] == [1]


@pytest.mark.parametrize("producers,consumers", [(384, 128), (256, 384), (512, 512)])
def test_unknown_layout_retains_original_register_budget(producers, consumers):
    result = _apply(_register_kernel(producers, consumers))
    assert not collect_calls(result, op="tl.set_max_nreg")


def test_simt_default_is_conservative_but_explicit_hints_are_preserved():
    assert not collect_calls(_apply(_register_kernel(128, 128, simt=True)), op="tl.set_max_nreg")
    result = _apply(_register_kernel(128, 128, simt=True, hints=(80, 232)))
    assert [tuple(int(arg) for arg in call.args) for call in collect_calls(result, op="tl.set_max_nreg")] == [
        (80, 0),
        (232, 1),
    ]
    assert not collect_calls(result, op="tl.annotate_producer_reg_dealloc")
    assert not collect_calls(result, op="tl.annotate_consumer_reg_alloc")


def test_disabled_reg_reallocation_consumes_annotations():
    result = _apply(_register_kernel(128, 256, hints=(40, 224), disabled=True))
    for name in ("tl.set_max_nreg", "tl.no_set_max_nreg", "tl.annotate_producer_reg_dealloc", "tl.annotate_consumer_reg_alloc"):
        assert not collect_calls(result, op=name)


@pytest.mark.parametrize("hints", [(23, 240), (24, 241)])
def test_invalid_hints_are_rejected(hints):
    with pytest.raises(ValueError, match="Invalid register hint"):
        _apply(_register_kernel(128, 256, hints=hints))


def test_explicit_schedule_is_not_reinterpreted_by_legacy_hint_pass():
    function = _register_kernel(128, 256, hints=(40, 224), custom=True).with_attr("global_symbol", "main")
    result = _apply(function)
    assert not collect_calls(result, op="tl.set_max_nreg")
    assert len(collect_nodes(result, ir.TensorStore)) == len(collect_nodes(function, ir.TensorStore))
    assert not any(region.op.name == "tl.attr" for region in collect_nodes(result, ir.RegionStmt))


def test_nested_manual_scopes_preserve_raw_calls_and_bind_definitions():
    @T.prim_func
    def kernel(A: T.Tensor((256,), T.int32)):
        tx = T.launch_thread("threadIdx.x", 256)
        T.attr(0, "kCustomWarpSpecialization", 1)
        T.attr([128, 128], "kWarpSpecializationScope", 0)
        if tx < 128:
            value = A[tx]
            T.set_max_nreg(80, 0)
            A[tx] = value + 1
        else:
            value = A[tx]
            T.set_max_nreg(232, 1)
            A[tx] = value + 2

    result = _apply(kernel)
    assert [tuple(int(arg) for arg in call.args) for call in collect_calls(result, op="tl.set_max_nreg")] == [
        (80, 0),
        (232, 1),
    ]
    assert len([bind for bind in collect_nodes(result, ir.Bind) if isinstance(bind.value, ir.TensorLoad)]) == 2
    # This small kernel has no remaining barrier/layout/launch metadata, so
    # finalization verifies that the register pass consumed its own markers.
    final = tilelang.transform.LowerSourceAnnotations()(ir.IRModule.from_expr(result))["main"]
    assert tirx.analysis.check_well_formed(final)


@pytest.mark.parametrize("arch,disabled", [("sm_80", False), ("sm_90", True)])
def test_pipeline_gate_disables_register_inference_but_consumes_its_hints(arch, disabled):
    from tilelang.cuda.pipeline import allow_warp_specialized

    function = _register_kernel(128, 128, hints=(40, 232))
    module = ir.IRModule.from_expr(function.with_attr("global_symbol", "main"))
    context = PassContext(config={"tl.disable_warp_specialized": disabled})
    enabled = allow_warp_specialized(pass_ctx=context, target=Target({"kind": "cuda", "arch": arch}))
    assert not enabled
    module = tilelang.cuda.transform.AnnotateWarpGroupRegAlloc(enable_register_inference=enabled)(module)
    assert not collect_calls(module["main"], op="tl.set_max_nreg")
    assert not collect_calls(module["main"], op="tl.annotate_producer_reg_dealloc")
    assert not collect_calls(module["main"], op="tl.annotate_consumer_reg_alloc")
    assert tirx.analysis.check_well_formed(module)


def test_register_pass_does_not_complete_an_unprocessed_pipeline():
    @T.prim_func
    def kernel(A: T.Tensor((4, 8), T.float32), B: T.Tensor((4, 8), T.float32)):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((8,), T.float32)
            for i in T.Pipelined(4, num_stages=2):
                T.copy(A[i, 0], shared)
                T.copy(shared, B[i, 0])

    function = kernel.with_attr("global_symbol", "main")
    result = _apply(function)
    assert tvm_ffi.structural_equal(result, function)
    assert not collect_calls(result, op="tl.set_max_nreg")
    assert len(collect_calls(result, op="tl.tileop.copy")) == 2
    assert any(int(loop.annotations.get("num_stages", 0)) == 2 for loop in collect_nodes(result, ir.For))
