"""Wrapping transformations."""
# pylint: disable=invalid-name, unsupported-binary-operation

from . import _ffi_api
from .pass_config import PassConfigKey  # noqa: F401
import tvm_tirx  # noqa: F401
from tvm_tirx.ir.transform import PassContext  # noqa: F401

_PYTHON_PASSES = {
    "Simplify": "simplify",
    "simplify_prim_func": "simplify",
    "LetInline": "simplify",
    "AddWrapperForSingleBufStore": "add_bufstore_wrapper",
    "HoistBroadcastValues": "hoist_broadcast_values",
    "DecoupleTypeCast": "decouple_type_cast",
}


def __getattr__(name):
    from importlib import import_module

    if name not in _PYTHON_PASSES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_PYTHON_PASSES[name]}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | _PYTHON_PASSES.keys())


def get_pass_context():
    """Get the current pass context"""
    return PassContext.current()


def RemoveNoOp():
    """Remove dead control flow and pure evaluations with TileLang store semantics.

    Element types, volatile memory and index side effects are preserved. This
    cleanup does not enable the optional arithmetic or Bind-inlining policies
    of :func:`Simplify`; it consumes the existing ``tirx.RemoveNoOp`` config.
    """
    return _ffi_api.RemoveNoOp()


def HoistIf():
    """Reuse native conditional hoisting with TileLang's typed store cleanup.

    The native factory is a Sequential whose last stage assumes arithmetic
    subtraction on all stored element types. Keep its hoisting and SSA stages,
    replacing only that cleanup; fail explicitly if an upstream rebase changes
    this internal composition.
    """
    from tvm_tirx import tirx
    from tvm_tirx.ir.transform import Sequential

    pipeline = tirx.transform.HoistIf()
    expected = ("tirx.InsertHoistedIf", "tirx.StmtSimplify", "tirx.RemoveNoOp")
    if not isinstance(pipeline, Sequential):
        raise RuntimeError("TIRx HoistIf is no longer a Sequential; review the TileLang cleanup boundary")
    stages = list(pipeline.passes)
    actual = tuple(stage.info.name for stage in stages)
    if actual != expected:
        raise RuntimeError(f"TIRx HoistIf stages changed: expected {expected}, got {actual}; review the TileLang cleanup boundary")
    return Sequential([*stages[:-1], RemoveNoOp()], opt_level=pipeline.info.opt_level, name="tl.HoistIf")


def LowerCustomDatatypes():
    """Lower host custom storage after packed ABI checks have been emitted.

    User arithmetic conversions use native Expr callbacks registered under
    ``tvm.datatype.lower.<target>.*``; storage conversion alone does not define
    custom arithmetic or change the logical input/output dtype checks.
    """
    return _ffi_api.LowerCustomDatatypes()


def PipelinePlanning():
    """infer the fragment/shared memory layout

    Returns
    -------
    fpass : tvm_tirx.transform.Pass
        The result pass
    """
    return _ffi_api.PipelinePlanning()  # type: ignore


def LayoutInference():
    """LayoutInference

    Returns
    -------
    fpass : tvm_tirx.transform.Pass
        The result pass
    """
    from tilelang.tileop import _register_builtin_callbacks

    _register_builtin_callbacks()
    return _ffi_api.LayoutInference()  # type: ignore


def LowerTileOp():
    """LowerTileOp

    Returns
    -------
    fpass : tvm_tirx.transform.Pass
        The result pass
    """
    from tilelang.tileop import _register_builtin_callbacks

    _register_builtin_callbacks()
    return _ffi_api.LowerTileOp()  # type: ignore


def InjectSoftwarePipeline():
    """InjectSoftwarePipeline

    Returns
    -------
    fpass : tvm_tirx.transform.Pass
        The result pass
    """
    return _ffi_api.InjectSoftwarePipeline()  # type: ignore


def LegalizeNegativeIndex():
    """Legalize negative indices in buffer loads.

    Returns
    -------
    fpass : tvm_tirx.transform.Pass
        The result pass
    """
    return _ffi_api.LegalizeNegativeIndex()  # type: ignore


def InjectAssumes():
    """Inject Assumes for natural shape boundary conditions. And convert Assumes in Evaluate(Call(...)) form
    (tvm builtin assume call) to AttrNode form.

    Returns:
    -------
    fpass : tvm_tirx.transform.Pass
        The result pass
    """
    return _ffi_api.InjectAssumes()


def VerifyParallelLoop():
    """VerifyParallelLoop

    Returns
    -------
    fpass : tvm_tirx.transform.Pass
        The result pass
    """
    return _ffi_api.VerifyParallelLoop()  # type: ignore


def VerifyBufferInit():
    """Warn when a non-global-scope buffer is read before anything writes it.

    Returns
    -------
    fpass : tvm_tirx.transform.Pass
        The registered pass. It inspects the IR and returns it unchanged.
    """
    return _ffi_api.VerifyBufferInit()  # type: ignore


def ThreadSync(storage_scope: str):
    """Insert sync between parallel read/write of shared buffers.

    Parameters
    ----------
    storage_scope: str
        The target storage scope.

    Returns
    -------
    fpass : tvm_tirx.transform.Pass
        The result pass
    """
    return _ffi_api.ThreadSync(storage_scope)  # type: ignore


def IfStmtBinding():
    """IfStmtBinding

    Returns
    -------
    fpass : tvm_tirx.transform.Pass
        The result pass
    """
    return _ffi_api.IfStmtBinding()  # type: ignore


def MergeIfStmt():
    """MergeIfStmt

    Returns
    -------
    fpass : tvm_tirx.transform.Pass
        The result pass
    """
    return _ffi_api.MergeIfStmt()  # type: ignore


def LoopUnswitching():
    """LoopUnswitching: Hoist loop-invariant if statements out of loops.

    Returns
    -------
    fpass : tvm_tirx.transform.Pass
        The result pass
    """
    return _ffi_api.LoopUnswitching()  # type: ignore


def LegalizeVectorizedLoop():
    """LegalizeLoopVectorize

    Returns
    -------
    fpass : tvm_tirx.transform.Pass
        The result pass
    """
    return _ffi_api.LegalizeVectorizedLoop()  # type: ignore


def LegalizeSafeMemoryAccess():
    """LegalizeLoopVectorize

    Returns
    -------
    fpass : tvm_tirx.transform.Pass
        The result pass
    """
    return _ffi_api.LegalizeSafeMemoryAccess()  # type: ignore


def LowerAccessPtr():
    """Lower TileLang frontend `tl.access_ptr` to `tir.builtin.tvm_access_ptr`.

    Returns
    -------
    fpass : tvm_tirx.transform.Pass
        The result pass
    """
    return _ffi_api.LowerAccessPtr()  # type: ignore


def LowerFrontendScopes():
    """Represent lexical allocation boundaries with native TIRx scope regions."""
    return _ffi_api.LowerFrontendScopes()


def LowerTensorSpec():
    """Materialize frontend tensor specifications as native typed SSA values."""
    return _ffi_api.LowerTensorSpec()


def LowerTensorDescriptors():
    """Resolve original tensor handles to packed slots before tensor lowering.

    Run after frontend specialization. Later signature changes must preserve
    these slots or remap them, as TileLang's parameter-pruning pass does.
    """
    return _ffi_api.LowerTensorDescriptors()


def MakePackedAPI():
    """MakePackedAPI

    Returns
    -------
    fpass : tvm_tirx.transform.Pass
        The result pass
    """
    return _ffi_api.MakePackedAPI()  # type: ignore


DEFAULT_SIMT_THREADS = 128


def MaterializeKernelLaunch(
    lower_thread_binding: bool = True,
    default_threads: int | list[int] | tuple | None = DEFAULT_SIMT_THREADS,
    unsupported_annotations: list[str] | tuple[str, ...] | None = None,
    *,
    lower_grid_binding: bool | None = None,
    launch_dim_tags: list[str] | tuple[str, ...] | None = None,
):
    """Materialize the target-neutral kernel launch nest emitted by T.Kernel
    into a backend-specific form. Each backend pipeline decides the mode for
    itself; this is where the target-dependent parts of a launch (whether a
    program-index space exists, whether threads exist and how many run by
    default) are decided.

    Parameters
    ----------
    lower_thread_binding : bool
        If True (SIMT backends, e.g. CUDA/ROCm/Metal), bind the thread
        placeholders as threadIdx.* thread_extent scopes.
        If False (backends without SIMT, e.g. CPU and Ascend), drop the thread
        placeholders. A body that references a thread index is rejected on such
        targets. Ascend pairs this with ``lower_grid_binding=True``: its NPU
        launch is a real 1-D core grid, while thread domains only exist inside
        ``T.SimtVF``, which emits its own thread scopes below the launch nest.
    default_threads : int | list[int] | tuple | None
        Thread-block extents used by SIMT backends when T.Kernel was called
        without ``threads=``. Ignored when ``lower_thread_binding`` is False.
        None means the backend has no default and ``threads=`` is required.
    unsupported_annotations : list[str] | None
        Launch annotations (keys on the ``tilelang_root`` block, e.g.
        ``cluster_dims``) that have no meaning on this backend. A launch
        carrying one is rejected here instead of being silently ignored by
        later passes.
    lower_grid_binding : bool | None
        Keyword-only. If True (targets with a real block/core-level launch,
        e.g. CUDA, Ascend), lower the launch loops (blockIdx.* grid axes and
        any tag in ``launch_dim_tags``) into thread_extent AttrStmts carrying
        each loop's own thread tag.
        If False (targets with no program-index space, e.g. CPU), lower those
        loops into plain serial For loops.
        If None (the default), follow ``lower_thread_binding``: a backend with
        SIMT threads has a program-index space too, and a backend without them
        has none. Pass this explicitly to decouple the two, as Ascend does.
    launch_dim_tags : list[str] | None
        Keyword-only. Extra thread_binding tags that belong to the launch
        nest rather than to the thread domain, so a backend can extend the
        launch vocabulary without this pass knowing about it. Ascend passes
        ``["cthread"]`` for ``T.MixedKernel``'s sub-block-id dimension.

    Returns
    -------
    fpass : tvm_tirx.transform.Pass
        The result pass
    """
    if lower_grid_binding is None:
        lower_grid_binding = lower_thread_binding
    if default_threads is not None:
        if isinstance(default_threads, int):
            default_threads = [default_threads, 1, 1]
        else:
            default_threads = list(default_threads) + [1] * (3 - len(default_threads))
    if unsupported_annotations is not None:
        unsupported_annotations = list(unsupported_annotations)
    if launch_dim_tags is not None:
        launch_dim_tags = list(launch_dim_tags)
    return _ffi_api.MaterializeKernelLaunch(  # type: ignore
        lower_grid_binding, lower_thread_binding, default_threads, unsupported_annotations, launch_dim_tags
    )


def AnnotateDeviceRegions():
    """AnnotateDeviceRegions

    Returns
    -------
    fpass : tvm_tirx.transform.Pass
        The result pass
    """
    return _ffi_api.AnnotateDeviceRegions()  # type: ignore


def SplitHostDevice():
    """Split host/device functions even for empty kernels.

    Returns
    -------
    fpass : tvm_tirx.transform.Pass
        The result pass
    """
    return _ffi_api.SplitHostDevice()  # type: ignore


def OutlineDeviceRegions():
    """Outline kernels before device storage, synchronization and launch lowering."""
    return _ffi_api.OutlineDeviceRegions()


def AnnotateReadOnlyParams():
    """Annotate read-only handle parameters for PrimFuncs.

    Adds attribute `tl.readonly_param_indices` listing param indices that are
    never written, enabling CUDA codegen to emit `const` qualifiers to unlock
    read-only cache loads.

    Returns
    -------
    fpass : tvm_tirx.transform.Pass
        The result pass
    """
    return _ffi_api.AnnotateReadOnlyParams()  # type: ignore


def VectorizeLoop(enable_vectorize: bool = True):
    """VectorizeLoop

    Returns
    -------
    fpass : tvm_tirx.transform.Pass
        The result pass
    """
    return _ffi_api.VectorizeLoop(enable_vectorize)  # type: ignore


def ConfigIndexBitwidth():
    """Config index bitwidth.

    Returns
    -------
    fpass : tvm_tirx.transform.Pass
        The result pass
    ----
    """
    return _ffi_api.ConfigIndexBitwidth()  # type: ignore


def PrepareTensorLayouts():
    """Materialize compact layouts before native definition-based lowering.

    The pass preserves the original Function and updates native definitions
    and uses together. Existing explicit layouts and view offsets are kept.
    """
    return _ffi_api.PrepareTensorLayouts()


def LowerTensorLayouts():
    """Lower physical tensor accesses while retaining the public parameter ABI.

    This includes layout preparation. Run LowerTensorSpec first to preserve
    explicit source stride declarations even when their addresses are compact.
    Implicit parameters and unmarked native default layouts retain compact
    validation; other stride/offset expressions remain in parameter types.
    This boundary consumes source declaration flags before host/device outlining.
    """
    return _ffi_api.LowerTensorLayouts()


def FlattenBuffer():
    """FlattenBuffer

    Returns
    -------
    fpass : tvm_tirx.transform.Pass
        The result pass
    """
    return _ffi_api.FlattenBuffer()  # type: ignore


def MergeSharedMemoryAllocations(enable_aggressive_merge: bool = False, align_bytes: int = 16, disable_reuse: bool = False):
    """MergeSharedMemoryAllocations

    Returns
    -------
    fpass : tvm_tirx.transform.Pass
        The result pass
    """
    return _ffi_api.MergeSharedMemoryAllocations(enable_aggressive_merge, align_bytes, disable_reuse)  # type: ignore


def PlanAndUpdateBufferAllocationLocation():
    """Plan and update buffer allocation locations within PrimFuncs.

    Returns
    -------
    fpass : tvm_tirx.transform.Pass
        The result pass
    """
    return _ffi_api.PlanAndUpdateBufferAllocationLocation()  # type: ignore


def HoistGlobalBufferAllocations():
    """Hoist global buffer allocations to the top of the block (host side).

    Returns
    -------
    fpass : tvm_tirx.transform.Pass
        The result pass
    """
    return _ffi_api.HoistGlobalBufferAllocations()  # type: ignore


def NormalizeAliasAnnotations():
    """Move signature alias annotations into explicit IR operands before rewriting."""
    return _ffi_api.NormalizeAliasAnnotations()


def HoistNonRestrictParams(finalize: bool = False):
    """Resolve alias exceptions by physical identity; finalize after ABI splitting."""
    return _ffi_api.HoistNonRestrictParams(finalize)


def StorageRewrite():
    """StorageRewrite

    Returns
    -------
    fpass : tvm_tirx.transform.Pass
        The result pass
    """
    return _ffi_api.StorageRewrite()  # type: ignore


def LowerSourceAnnotations():
    """Lower source scopes and hints, preserving metadata for its owning pass.

    Source blocks become lexical scopes. Codegen annotations retain their
    operands and suffix scope; compiler annotations remain for their consumers.
    Intrinsic metadata is normalized into native operands and static options.
    """
    return _ffi_api.LowerSourceAnnotations()


def LowerSourceAllocations():
    """Retire source allocation origins after backend lifetime preparation.

    CPU source pools must first be planned by ``LowerCPUKernelLaunch``.
    Other targets keep their native allocation scope and consume only the
    source-origin marker.
    """
    return _ffi_api.LowerSourceAllocations()


def VerifyLoweredAnnotations():
    """Reject unconsumed body metadata without rewriting the IR.

    Run after the packed ABI and alias consumers. Final Function attributes
    and archived source modules are outside this body-only verification.
    """
    return _ffi_api.VerifyLoweredAnnotations()


def InferWmmaFragmentMetadata():
    """Check whole-function WMMA shape/layout metadata and attach allocation hints.

    This preserves the old InferFragment consistency checks. It does not
    establish initialization dominance; a store can supply fragment metadata.
    """
    return _ffi_api.InferWmmaFragmentMetadata()


def LowerThreadAllreduce():
    """LowerThreadAllreduce"""
    return _ffi_api.LowerThreadAllreduce()  # type: ignore


def LowerIntrin():
    """LowerIntrin"""
    return _ffi_api.LowerIntrin()  # type: ignore


def LowerDeviceKernelLaunch():
    """
    Create and return a transform pass that lowers device kernel launch constructs to target-specific IR.

    This pass transforms high-level device kernel launch and related intrinsics into lower-level
    IR suitable for backend code generation and device-side lowering.

    Returns:
        tvm_tirx.transform.Pass: The transform pass that performs device kernel launch lowering.
    """
    return _ffi_api.LowerDeviceKernelLaunch()  # type: ignore


def CanonicalizeLegacyReducer():
    """Rewrite legacy (v1) reducer syntax into first-class reducer v2 ops.

    Deprecation shim: ``T.clear`` + read-modify-write stores + in-place
    ``T.finalize_reducer(acc)`` become ``reducer_init``/``reducer_update``/
    out-of-place finalize with a fresh destination fragment. Unrecognized
    access patterns are compile errors, never silently accepted.

    Returns:
        tvm_tirx.transform.Pass: The canonicalization pass.
    """
    return _ffi_api.CanonicalizeLegacyReducer()  # type: ignore


def VerifyReducerEpoch():
    """Verify lifecycle and access rules of reducer v2 epochs.

    Enforces that every ``T.alloc_reducer`` has exactly one
    ``T.reducer_init``, updates only inside ``T.Parallel`` between init and
    finalize, exactly one out-of-place ``T.finalize_reducer(acc, dst)``, and
    no ordinary reads/writes/aliasing of the reducer handle.

    Returns:
        tvm_tirx.transform.Pass: The verification pass.
    """
    return _ffi_api.VerifyReducerEpoch()  # type: ignore


def ReducerPlanAndMaterialize():
    """Plan physical storage/communication for reducer v2 epochs.

    Runs after LayoutInference (loop layouts are read-only inputs) and
    materializes the first-class reducer ops into ordinary fragment storage,
    guarded read-modify-write updates, and an explicit finalize plan.

    Returns:
        tvm_tirx.transform.Pass: The planning/materialization pass.
    """
    return _ffi_api.ReducerPlanAndMaterialize()  # type: ignore


def VerifyReducerConsumed():
    """Assert no reducer v2 construct survives past materialization.

    Returns:
        tvm_tirx.transform.Pass: The verification pass.
    """
    return _ffi_api.VerifyReducerConsumed()  # type: ignore


def UnrollLoop():
    """Unroll loops as in Halide pipeline.

    This pass unrolls loops based on configuration options including:
    - auto_max_step: Threshold of number of steps to be automatically unrolled
    - auto_max_depth: Maximum nested level of loops that can be automatically unrolled
    - auto_max_extent: Maximum extent of loop that will be unrolled
    - explicit_unroll: Whether to explicitly unroll instead of setting a pragma
    - unroll_local_access: Whether to always unroll local access

    Returns
    -------
    fpass : tvm_tirx.transform.Pass
        The result pass
    """
    return _ffi_api.UnrollLoop()  # type: ignore
