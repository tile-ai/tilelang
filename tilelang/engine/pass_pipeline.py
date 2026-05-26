from __future__ import annotations

from tvm import IRModule, s_tir, tirx
from tvm.target import Target

import tilelang
from tilelang.transform import PassContext


def allow_vectorize(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    disable_vectorize = pass_ctx.config.get("tirx.disable_vectorize", False)
    return not disable_vectorize


def allow_global_thread_synchronization(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    enable_global_thread_sync = pass_ctx.config.get("tir.detect_global_barrier", False)
    return enable_global_thread_sync


def should_enable_aggressive_merge(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return bool(pass_ctx.config.get(tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE, False))


def should_force_let_inline(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return bool(pass_ctx and pass_ctx.config.get(tilelang.PassConfigKey.TL_FORCE_LET_INLINE, False))


def should_enable_ast_print(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return bool(pass_ctx and pass_ctx.config.get(tilelang.PassConfigKey.TL_AST_PRINT_ENABLE, False))


def should_enable_layout_visual(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return pass_ctx.config.get(tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE, False)


def should_enable_race_check(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return not pass_ctx.config.get(tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK, False)


def should_enable_prelower_semantic_check(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return not pass_ctx.config.get(tilelang.PassConfigKey.TL_DISABLE_PRELOWER_SEMANTIC_CHECK, False)


def get_layout_visual_formats(pass_ctx: PassContext | None = None) -> list[str]:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    formats_value = pass_ctx.config.get(tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_FORMATS, "")
    if not formats_value:
        return ["txt"]

    formats_str = formats_value.strip().lower()
    valid_formats = ["txt", "png", "pdf", "svg", "all"]

    if formats_str == "all":
        return ["txt", "png", "pdf", "svg"]

    if "," in formats_str:
        formats_list = [f.strip() for f in formats_str.split(",")]
    else:
        formats_list = [formats_str]

    invalid_formats = [f for f in formats_list if f not in valid_formats]
    if invalid_formats:
        raise ValueError(
            f"Invalid formats for TL_LAYOUT_VISUALIZATION_FORMATS: {invalid_formats}. "
            f"Valid formats are: {valid_formats}. "
            f"You can choose one of the valid formats or a comma-separated list of formats.(e.g., 'txt,png,pdf')"
        )
    return formats_list


def LayoutVisual(mod: IRModule) -> None:
    """Apply layout visualization pass if enabled."""
    if should_enable_layout_visual():
        formats = get_layout_visual_formats()
        tilelang.analysis.LayoutVisual(formats=formats)(mod)


def PreLowerSemanticCheck(mod: IRModule) -> None:
    """Run backend-independent validation before lowering."""

    if not should_enable_prelower_semantic_check():
        return

    if should_enable_ast_print():
        tilelang.analysis.ASTPrinter()(mod)
    tilelang.analysis.NestedLoopChecker()(mod)
    tilelang.analysis.FragmentLoopChecker()(mod)


def LowerCommon(mod: IRModule, target: Target) -> IRModule:
    """Common lowering pipeline for non-CUDA/HIP targets."""
    mod = tirx.transform.BindTarget(target)(mod)

    if should_force_let_inline():
        mod = tilelang.transform.LetInline()(mod)
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    if should_enable_race_check():
        mod = tilelang.transform.VerifyParallelLoop()(mod)
    mod = tilelang.transform.InjectAssumes()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LayoutReducer()(mod)
    mod = tilelang.transform.LayoutInference()(mod)
    LayoutVisual(mod)
    mod = tilelang.transform.LowerTileOp()(mod)
    mod = tilelang.transform.DecoupleTypeCast()(mod)
    mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
    mod = tilelang.transform.LowerAccessPtr()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.HoistNonRestrictParams()(mod)

    pass_ctx = tilelang.transform.get_pass_context()

    mod = tilelang.transform.LowerSharedTmem()(mod)
    mod = tilelang.transform.IfStmtBinding()(mod)
    mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    mod = tilelang.transform.LowerSharedBarrier()(mod)
    mod = tilelang.transform.HoistGlobalBufferAllocations()(mod)
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tirx.transform.NarrowDataType(32)(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    mod = tirx.transform.Simplify()(mod)
    mod = tilelang.transform.VectorizeLoop(enable_vectorize=allow_vectorize(pass_ctx=pass_ctx))(mod)
    mod = tilelang.transform.StorageRewrite()(mod)
    mod = tilelang.transform.LoopUnswitching()(mod)
    mod = tilelang.transform.UnrollLoop()(mod)
    mod = s_tir.transform.RenormalizeSplitPattern()(mod)
    mod = tirx.transform.Simplify()(mod)
    mod = tirx.transform.RemoveNoOp()(mod)
    mod = s_tir.transform.HoistIfThenElse()(mod)
    mod = tirx.transform.VerifyMemory()(mod)
    mod = tirx.transform.AnnotateEntryFunc()(mod)
    mod = s_tir.transform.InferFragment()(mod)
    mod = tilelang.transform.LowerThreadAllreduce()(mod)
    mod = tilelang.transform.LowerLDGSTG()(mod)
    if allow_global_thread_synchronization():
        mod = tilelang.transform.ThreadSync("global")(mod)
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)
    mod = tilelang.transform.MarkCudaSyncCalls(False)(mod)
    mod = tilelang.transform.AnnotateReadOnlyParams()(mod)
    mod = tilelang.transform.MergeSharedMemoryAllocations(enable_aggressive_merge=should_enable_aggressive_merge(pass_ctx=pass_ctx))(mod)
    mod = tilelang.transform.ThreadSync("shared")(mod)
    mod = tilelang.transform.MergeIfStmt()(mod)
    mod = tilelang.transform.MakePackedAPI()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LowerDeviceKernelLaunch()(mod)
    return mod
