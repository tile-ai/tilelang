"""Auto-schedule semantic hints."""

from tilelang.utils.language import to_tile_region
from tvm import tirx
from tvm.tirx import Buffer, IntImm, StringImm, call_intrin, op
from tvm.tirx.script.builder.ir import attr

__all__ = ["assume_no_conflict", "PerCoreTask", "Stage", "Task"]


def PerCoreTask():
    """Represent one logical per-core task with guarded candidate sites.

    AutoSchedule treats the enclosed control flow as one logical task when it
    analyzes dependencies, estimates cost, and allocates flags. When
    synchronization is emitted, the same logical wait/set pair is placed next
    to every candidate access under its original control-flow guard.

    The compiler preserves explicit ``T.Task()`` candidates and infers markers
    for the remaining supported dependency-bearing candidate boundaries.
    Explicit and inferred candidates may therefore be mixed in one region.
    Every candidate must use the same single hardware pipe and access the same
    logical read/write buffers. Exactly one candidate must execute on each core
    in each dynamic invocation of the PerCoreTask; this dynamic property is a
    user control-flow contract.

    Scalar bindings may remain outside candidate markers when they do not
    access buffers and therefore cannot form buffer dependencies with statements
    outside the PerCoreTask. External synchronization is emitted only at
    ``T.Task`` candidate sites, so put any buffer access in a supported
    single-pipe candidate rather than an unmarked scalar binding.

    Explicit candidates also define the latency model. Statements inside one
    ``T.Task`` are serial, candidates before the same inter-core wait
    form one parallel phase whose cost is their maximum, and phases separated by
    waits are serial. The compiler wraps each unmarked inferred
    dependency-bearing leaf in an equivalent internal marker before applying
    the same latency and synchronization model.

    Example
    -------
    >>> with T.PerCoreTask():
    ...     if split_id == 0:
    ...         with T.Task():
    ...             T.copy(res, C[...])
    ...     T.ascend_sync_inter_arrive("PIPE_FIX", 0)
    ...     T.ascend_sync_inter_wait("PIPE_FIX", 0)
    ...     if split_id != 0:
    ...         with T.Task():
    ...             T.copy(res, C[...])
    """
    return attr(tirx.const(0, "int32"), "tl.ascend_per_core_task", tirx.const(1, "int32"))


def Task(latency: int | None = None, ii: int | None = None):
    """Group one or more serial statements into one scheduler task.

    AutoSchedule creates exactly one TaskNode for the complete marker body.
    Statements issue in source order without internal synchronization, so each
    preceding statement contributes its II and only the final statement exposes
    its latency. All statements must use one Ascend hardware pipe and one AIC/AIV
    core affinity; split cross-pipe or cross-core sequences into
    separate tasks. The marker does not select or launch a core by itself.

    ``latency`` is the number of cycles from task issue until its outputs are
    ready. ``ii`` is the minimum interval before the same hardware pipe may
    issue another instance. Task costs require ``latency >= ii``. Either value
    may be omitted, in which case the ``EstimateLatency`` pass supplies it and
    validates the final pair. Provided values are trusted exact overrides for
    the complete task body, not hints for the enclosing loop II. These are the
    only public task parameters. The compiler derives execution resources from
    the task body and determines core placement.

    Inside ``T.PerCoreTask()``, the same marker also denotes one concrete
    per-core candidate. Put the control flow that selects a core outside this
    scope so automatic synchronization remains under the same guard. Explicit
    markers may be mixed with unmarked candidates; the compiler infers markers
    for the remaining supported candidate boundaries. Do not nest
    ``T.PerCoreTask()`` inside this scope; put this ``T.Task()`` inside
    ``T.PerCoreTask()`` instead.

    Example
    -------
    >>> with T.Task():
    ...     T.copy(A, a_l1)
    ...     T.copy(B, b_l1)
    ...
    >>> with T.PerCoreTask():
    ...     if split_id == 0:
    ...         with T.Task():
    ...             T.copy(res, C[...])
    """
    if latency is not None and (isinstance(latency, bool) or not isinstance(latency, int) or latency < 0):
        raise ValueError(f"Task latency must be a non-negative integer, got {latency!r}")
    if ii is not None and (isinstance(ii, bool) or not isinstance(ii, int) or ii <= 0):
        raise ValueError(f"Task ii must be a positive integer, got {ii!r}")
    if latency is not None and ii is not None and latency < ii:
        raise ValueError(f"Task latency must be greater than or equal to ii, got latency={latency} and ii={ii}")

    annotations = {}
    if latency is not None:
        annotations["latency"] = IntImm("int64", latency)
    if ii is not None:
        annotations["ii"] = IntImm("int64", ii)
    return attr(annotations, "tl.ascend_task", tirx.const(1, "int32"))


def Stage(stage: int):
    """Assign a manual software-pipeline stage to enclosed scheduler tasks.

    If any direct child task or loop in a scheduled child list has a
    ``T.Stage`` annotation, AutoSchedule treats that child list as manual. One
    scope may contain several consecutive statements; every scheduler task
    materialized from those statements receives the same stage. Unannotated
    siblings default to stage 0. Within each hardware pipe, source order is the
    manual issue order; tasks on different pipes remain freely reorderable, so
    no separate order annotation is required. Nested child lists select
    automatic or manual scheduling independently.

    ``T.Stage`` must wrap complete scheduler tasks. It must not appear inside
    ``T.Task``, ``T.SimtVF``, ``T.SimdVF``, an SBlock, or a parallel loop. In
    Python's multi-context form, put ``T.Stage`` first so it is the outer scope:

    .. code-block:: python

        # Correct
        with T.Stage(1), T.SimtVF(threads=128):
            ...

        # Invalid: T.Stage is nested inside the SimtVF task boundary
        with T.SimtVF(threads=128), T.Stage(1):
            ...

    Non-zero stages require an enclosing loop with ``enable_offset=True``.
    ``T.Pipelined(..., num_stages=...)`` controls the automatic buffer-version
    search bound; it does not limit the values accepted by ``T.Stage``.

    Example
    -------
    >>> for i in T.Pipelined(8, annotations={"enable_offset": True}):
    ...     with T.Stage(0):
    ...         T.copy(A[i], temp)
    ...     with T.Stage(1):
    ...         T.copy(temp, B[i])
    """
    if isinstance(stage, bool) or not isinstance(stage, int) or stage < 0:
        raise ValueError(f"Stage must be a non-negative integer, got {stage!r}")
    return attr(IntImm("int64", stage), "tl.ascend_stage", tirx.const(1, "int32"))


def assume_no_conflict(a, b=None, *, level=None, cross=None, group=None):
    """Assert that two buffer regions do not conflict, suppressing a false
    dependency the auto-scheduler would otherwise conservatively insert.

    The auto-scheduler's dependency analysis cannot prove that dynamically
    indexed accesses (e.g. a permutation ``state_cache[perm[idx], ...]``) are
    disjoint across loop iterations, so it inserts redundant synchronization.
    This hint lets the user declare that the regions do not overlap. It is
    consumed by the ``NormalizeNoConflictHints`` pass before auto-scheduling; at
    runtime it is a no-op.

    Args:
        a: first region -- ``buffer[indices...]`` or a bare ``buffer`` (the
            whole buffer). A region must cover the same footprint as the actual
            access it refers to: use slices for the accessed extents (e.g.
            ``buf[i, 0:M, 0:N]``), since a fully scalar-indexed ``buf[i, 0, 0]``
            denotes only a 1-element region and will not match a wider copy.
        b: second region; defaults to ``a`` (a self-dependency).
        level: which enclosing loop the non-overlap is asserted at, counted from
            the outermost enclosing loop (``0`` = outermost, ``1`` = next inner,
            ...). ``None`` (default) asserts it at each enclosing loop
            independently. For a ``group``, ``level`` indexes the loops enclosing
            both halves (their common ancestors), still counted from the
            outermost.
        cross: ``None`` = both same- and cross-iteration are conflict-free
            (default); ``True`` = only cross-iteration; ``False`` = only
            same-iteration.
        group: string tag pairing two separately-written half-declarations whose
            regions live in different scopes (so they cannot be written in one
            call). Must appear exactly twice; the combined hint is attached to
            the loops enclosing both. Rarely needed.

    Example:
        >>> # self-dependency across iterations: A[perm[i]] written each
        >>> # iteration lands in a disjoint slot (perm is a permutation), so
        >>> # there is no cross-iteration conflict
        >>> T.assume_no_conflict(A[perm[i], 0:N], level=0, cross=True)
        >>> # two distinct regions that never overlap -- both within the same
        >>> # iteration and across iterations (cross=None, the default)
        >>> T.assume_no_conflict(A[lo, 0:N], A[hi, 0:N], level=0)
        >>> # whole buffer: drop every dependency involving B at every loop
        >>> # enclosing this statement (level=None broadcasts to all of them)
        >>> T.assume_no_conflict(B)
        >>> # group: pair two regions written in separate (sibling) scopes
        >>> T.assume_no_conflict(A[wperm[i], 0:N], group="perm")   # in one loop
        >>> T.assume_no_conflict(A[rperm[j], 0:N], group="perm")   # in another
    """
    if cross is True:
        cross_code = 1
    elif cross is False:
        cross_code = 0
    elif cross is None:
        cross_code = -1
    else:
        raise ValueError(f"cross must be None/True/False, got {cross!r}")
    if level is not None and (not isinstance(level, int) or level < 0):
        raise ValueError(f"level must be None or a non-negative int, got {level!r}")
    if group is not None and b is not None:
        raise ValueError(
            "assume_no_conflict: a group's two half-declarations each carry one region, so `b` must be None when `group` is set"
        )

    # A bare Buffer (no indexing) declares non-conflict for the whole buffer; it
    # is carried as its data Var (a handle PrimExpr) and matched by storage key
    # downstream. A concrete region is encoded as a `tl.region` Call (like
    # T.copy) -- NOT a bare BufferRegion -- so its min/extent are simplified/
    # inlined in lockstep with the real accesses (a frozen BufferRegion would
    # keep symbolic vars and never match a simplified access).
    def operand(x):
        return x.data if isinstance(x, Buffer) else to_tile_region(x, "rw")

    a_op = operand(a)
    b_op = operand(b) if b is not None else a_op
    # level None (all enclosing loops) is encoded as -1. The marker is a
    # statement-form Call (auto-evaluated by the eager builder, like T.copy).
    # Arg order matches the API: a, b, level, cross, group.
    level_imm = IntImm("int32", level if level is not None else -1)
    return call_intrin(
        "handle",
        op.Op.get("tl.assume_no_conflict"),
        a_op,
        b_op,
        level_imm,
        IntImm("int32", cross_code),
        StringImm(group or ""),
    )
