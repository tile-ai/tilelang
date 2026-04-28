from __future__ import annotations
from tvm import tir, IRModule
from tvm.target import Target
import tilelang
from tilelang.backend.base import DeviceBackend
from tilelang.backend.registry import get_device_backend
from tilelang.transform import PassContext


def module_has_tma(mod: IRModule) -> bool:
    """Check if any function in the module was lowered with TMA operations.

    This reads the ``tl.has_tma`` attribute set by ``LowerTileOp`` during
    ``LowerAndLegalize``, which is the source of truth for whether TMA
    copies were actually generated.
    """
    return any(func.attrs and func.attrs.get("tl.has_tma", False) for _, func in mod.functions.items())


def allow_vectorize(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    disable_vectorize = pass_ctx.config.get("tir.disable_vectorize", False)
    return not disable_vectorize


def allow_global_thread_synchronization(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    enable_global_thread_sync = pass_ctx.config.get("tir.detect_global_barrier", False)
    return enable_global_thread_sync


def should_enable_aggressive_merge(
    pass_ctx: PassContext | None = None,
    target: Target | None = None,
    backend: DeviceBackend | None = None,
) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    enable_aggressive_merge = bool(pass_ctx.config.get(tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE, False))
    if backend is None and target is not None:
        backend = get_device_backend(target)
    if backend is not None and target is not None:
        enable_aggressive_merge = backend.pass_hooks.adjust_aggressive_shared_memory_merge(enable_aggressive_merge, target)
    return enable_aggressive_merge


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
    enabled = pass_ctx.config.get(tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE, False)
    return enabled


def should_enable_race_check(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    enabled = not pass_ctx.config.get(tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK, False)
    return enabled


def should_enable_prelower_semantic_check(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    enabled = not pass_ctx.config.get(tilelang.PassConfigKey.TL_DISABLE_PRELOWER_SEMANTIC_CHECK, False)
    return enabled


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
    """
    Check whether the module is valid before lowering. If not, raise a user-friendly error
    in Python side instead of letting the error dive into the complicated TVM/C++ stack.
    Note: This is a validation-only pipeline of passes and does not modify or return the module.
    """

    if not should_enable_prelower_semantic_check():
        return

    # Print AST for debugging purpose
    if should_enable_ast_print():
        tilelang.analysis.ASTPrinter()(mod)
    # Check if there are any invalid nested loops.
    tilelang.analysis.NestedLoopChecker()(mod)
    # Check if there are any invalid symbolic T.Parallel + fragment access.
    tilelang.analysis.FragmentLoopChecker()(mod)


def LowerAndLegalize(mod: IRModule, target: Target) -> IRModule:
    """
    Bind target information and progressively legalize and lower frontend Tile IR into a form suitable for downstream optimization and codegen.

    This pass pipeline:
    - Binds the provided target to the module.
    - Legalizes frontend Tile IR into TVM-compatible constructs.
    - Simplifies expressions.
    - Configures reducer layouts and performs layout inference for fragments and shared memory.
    - Lowers high-level tile operations and L2 persistent maps.
    - Legalizes vectorized loops and inserts safety checks for memory accesses.
    - Re-simplifies to remove redundancies introduced by safety checks.
    - Attempts loop vectorization for dynamic-shaped loops.

    Parameters:
        mod (IRModule): The input IR module containing frontend Tile IR.
        target (Target): Target device information to bind into the module.

    Returns:
        IRModule: The transformed module, ready for target-specific optimization passes.
    """
    backend = get_device_backend(target)
    # Bind the target device information to the module
    mod = tir.transform.BindTarget(target)(mod)

    if should_force_let_inline():
        # Force-let inline whenever the pass config requests it.
        mod = tilelang.transform.LetInline()(mod)
    # Add wrapper for single buf store
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    # Normalize negative indices to canonical non-negative form
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    # Verify parallel loop correctness
    if should_enable_race_check():
        mod = tilelang.transform.VerifyParallelLoop()(mod)
    # Inject assumes to speedup tvm prover
    mod = tilelang.transform.InjectAssumes()(mod)
    # Simplify the IR expressions
    mod = tilelang.transform.Simplify()(mod)
    # Set layouts for reducers
    mod = tilelang.transform.LayoutReducer()(mod)
    mod = backend.pass_hooks.pre_layout(mod, target)
    # Run pipeline planning and software-pipeline rewriting before layout
    # inference so inferred layouts see the final pipelined structure directly.
    mod = tilelang.transform.PipelinePlanning()(mod)
    mod = tilelang.transform.InjectSoftwarePipeline()(mod)
    mod = tilelang.transform.Simplify()(mod)
    # Infer memory layouts for fragments and shared memory
    mod = tilelang.transform.LayoutInference()(mod)
    # Visualize the layout
    LayoutVisual(mod)
    # Lower high-level tile operations to low-level operations
    mod = tilelang.transform.LowerTileOp()(mod)
    mod = backend.pass_hooks.post_tile_lowering(mod, target)
    # Decouple type cast vectorization constraints before vectorization
    mod = tilelang.transform.DecoupleTypeCast()(mod)
    # Legalize vectorized loops to ensure they are valid
    mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    # Add safety checks for memory accesses
    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
    # Lower frontend pointer metadata op to standard tvm_access_ptr
    mod = tilelang.transform.LowerAccessPtr()(mod)
    # Simplify again to clean up any duplicated conditions
    # that may have been introduced by safety checks
    # use an enhanced pass to simplify the dynamic symbolics
    # TODO(lei): return to tir pass when kSymbolicBound simplification
    # is merged into tvm.
    mod = tilelang.transform.Simplify()(mod)
    # Hoist any root-block annotations to PrimFunc attrs if pass is available
    mod = tilelang.transform.HoistNonRestrictParams()(mod)
    return mod


def OptimizeForTarget(mod: IRModule, target: Target) -> IRModule:
    backend = get_device_backend(target)
    pass_ctx = tilelang.transform.get_pass_context()
    mod = backend.pass_hooks.optimize_entry(mod, target)
    # which may be introduced by the LegalizeSafeMemoryAccess
    mod = tilelang.transform.IfStmtBinding()(mod)
    has_tma = module_has_tma(mod)
    # Pipeline barriers are now created at final expanded size by
    # InjectSoftwarePipeline, so no late MVB barrier fixup is needed.
    # Buffer allocation placement is handled uniformly for both paths.
    mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    mod = backend.pass_hooks.lower_shared_barrier(mod, target)
    mod = backend.pass_hooks.after_shared_barrier_lowering(mod, target, has_tma=has_tma)
    mod = tilelang.transform.HoistGlobalBufferAllocations()(mod)
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tir.transform.NarrowDataType(32)(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    # ConfigIndexBitwidth must be applied after FlattenBuffer
    # as it will flatten index computing
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tilelang.transform.VectorizeLoop(enable_vectorize=allow_vectorize(pass_ctx=pass_ctx))(mod)
    mod = tilelang.transform.StorageRewrite()(mod)
    mod = tilelang.transform.LoopUnswitching()(mod)
    mod = tilelang.transform.UnrollLoop()(mod)
    mod = tir.transform.RenormalizeSplitPattern()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tir.transform.RemoveNoOp()(mod)
    mod = tir.transform.HoistIfThenElse()(mod)

    mod = tir.transform.VerifyMemory()(mod)
    mod = tir.transform.AnnotateEntryFunc()(mod)
    # TODO(lei): This is a hack to make sure the
    # thread level allreduce pass can be applied
    # in TL. As Tl only use one thread dimension
    # the var binding information will be lost
    # in the lowering process with Legalization
    # and Simplify pass.
    # We can find a way better to create var instead
    # of putting the LowerThreadAllreduce before
    # the Legalization.
    mod = tir.transform.InferFragment()(mod)
    mod = tilelang.transform.LowerThreadAllreduce()(mod)
    mod = backend.pass_hooks.before_split_host_device(mod, target)
    # Global Barrier Synchronization must be applied before
    # SplitHostDevice pass, as the global barrier
    if allow_global_thread_synchronization():
        mod = tilelang.transform.ThreadSync("global")(mod)
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)

    mod = backend.pass_hooks.after_split_host_device(mod, target)

    mod = tilelang.transform.AnnotateReadOnlyParams()(mod)
    # MergeSharedMemoryAllocations must be applied after SplitHostDevice
    # because the merged allocation site is at the beginning of each device function
    enable_aggressive_merge = should_enable_aggressive_merge(pass_ctx=pass_ctx, target=target, backend=backend)
    mod = tilelang.transform.MergeSharedMemoryAllocations(enable_aggressive_merge=enable_aggressive_merge)(mod)
    mod = backend.pass_hooks.after_shared_memory_planning(mod, target)
    mod = tilelang.transform.ThreadSync("shared")(mod)
    mod = tilelang.transform.ThreadSync("shared.dyn")(mod)
    mod = backend.pass_hooks.after_shared_sync(mod, target)
    mod = tilelang.transform.MergeIfStmt()(mod)
    mod = tilelang.transform.MakePackedAPI()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LowerDeviceKernelLaunch()(mod)

    mod = backend.pass_hooks.before_device_codegen(mod, target)

    return mod
