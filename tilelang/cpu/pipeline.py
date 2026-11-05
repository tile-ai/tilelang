"""CPU lowering over native Function/TensorType/RegionStmt contracts."""

from __future__ import annotations

from tvm_tirx import IRModule, tirx
from tvm_tirx.target import Target

import tilelang
from tilelang.backend.pass_pipeline.pipeline_utils import (
    LayoutVisual,
    allow_vectorize,
    should_enable_race_check,
    should_force_let_inline,
)


def _should_enable_cpu_parallel(pass_ctx=None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return bool(pass_ctx and pass_ctx.config.get(tilelang.PassConfigKey.TL_CPU_PARALLEL, False))


def CPUPassPipelineBody(mod: IRModule, target: Target) -> IRModule:
    tr = tilelang.transform
    native = tirx.transform
    cpu = tilelang.cpu.transform
    pass_ctx = tr.get_pass_context()
    cpu_parallel = _should_enable_cpu_parallel(pass_ctx)

    # Dynamic frontend values must remain operands throughout type rebuilding.
    mod = tr.NormalizeAliasAnnotations()(mod)
    mod = tr.LowerTensorSpec()(mod)
    mod = native.BindTarget(target)(mod)
    mod = tr.MaterializeKernelLaunch(
        lower_thread_binding=False,
        lower_grid_binding=False,
        unsupported_annotations=["cluster_dims"],
    )(mod)
    mod = cpu.LowerCPUKernelLaunch()(mod)
    if should_force_let_inline(pass_ctx):
        mod = tr.LetInline()(mod)
    mod = tr.AddWrapperForSingleBufStore()(mod)
    mod = tr.LegalizeNegativeIndex()(mod)
    if should_enable_race_check(pass_ctx):
        mod = tr.VerifyParallelLoop()(mod)
    mod = tr.InjectAssumes()(mod)
    mod = tr.Simplify()(mod)

    # These semantic checks remain required for every compilation. A partial
    # development compiler missing a registration fails here, never silently
    # bypasses reducer lifetime or uninitialized-buffer diagnostics.
    mod = tr.CanonicalizeLegacyReducer()(mod)
    mod = tr.VerifyReducerEpoch()(mod)
    mod = tr.VerifyBufferInit()(mod)
    mod = tr.LayoutInference()(mod)
    mod = tr.ReducerPlanAndMaterialize()(mod)
    LayoutVisual(mod)
    if cpu_parallel:
        mod = cpu.MarkCPUAtomics()(mod)
    mod = tr.LowerTileOp()(mod)
    mod = tr.VerifyReducerConsumed()(mod)
    mod = cpu.LowerCPUAtomics()(mod)

    mod = tr.DecoupleTypeCast()(mod)
    mod = tr.LegalizeVectorizedLoop()(mod)
    mod = tr.LegalizeSafeMemoryAccess()(mod)
    mod = tr.LowerAccessPtr()(mod)
    mod = tr.Simplify()(mod)
    mod = tr.FinalizeAnnotations(target.kind.name)(mod)

    # SSA Binds and lexical regions express the old block/let scope directly.
    # Native tensor layout/flattening retains the original public signature;
    # native storage planning preserves allocation lifetime and typed views.
    mod = tr.LowerTensorLayouts()(mod)
    mod = native.FlattenTensor()(mod)
    mod = tr.ConfigIndexBitwidth()(mod)
    mod = native.StmtSimplify()(mod)
    mod = native.VectorizeLoop(enable_vectorize=allow_vectorize(pass_ctx))(mod)
    mod = tr.StorageRewrite()(mod)
    mod = tr.UnrollLoop()(mod)
    mod = native.RenormalizeSplitPattern()(mod)
    mod = native.StmtSimplify()(mod)
    mod = tilelang.transform.RemoveNoOp()(mod)
    mod = tilelang.transform.HoistIf()(mod)
    mod = tr.HoistNonRestrictParams()(mod)
    if cpu_parallel:
        mod = cpu.MaterializeCPUParallelGrid()(mod)
    mod = native.VerifyMemory()(mod)
    mod = native.AnnotateEntryFunc()(mod)

    mod = tr.AnnotateDeviceRegions()(mod)
    mod = tr.SplitHostDevice()(mod)
    mod = native.BF16ComputeLegalize()(mod)
    mod = native.FP8ComputeLegalize()(mod)
    mod = tr.MakePackedAPI()(mod)
    mod = native.FP8StorageLegalize()(mod)
    mod = native.BF16StorageLegalize()(mod)
    # Only now are these indices in the final native pointer signature.
    mod = tr.HoistNonRestrictParams(finalize=True)(mod)
    mod = tr.AnnotateReadOnlyParams()(mod)
    return mod
