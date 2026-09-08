"""Ascend auto-schedule annotations: multi-buffer version control and memory
capacity overrides consumed by the Z3 scheduler."""

from tvm.tirx import IntImm
from tvm.tirx.script.builder.ir import sblock_attr

_VALID_MEMORY_SCOPES = ("shared", "shared.l1", "shared.l0c", "shared.l0a", "shared.l0b")

__all__ = [
    "annotate_buffer_versions",
    "annotate_manual_multi_buffer",
    "annotate_unlimit_memory",
]


_VALID_BUFFER_VERSION_MODES = ("auto", "iteration", "counter")


def annotate_buffer_versions(buffer_versions_map: dict):
    """Control automatic multi-buffer version counts and indexing modes.

    When using T.Persistent with num_stages > 1, the Z3 auto-scheduler computes
    optimal buffer version counts for each on-chip buffer. Each mapping value
    may be one of:

    - ``num_versions``: use a fixed positive version count and automatic mode;
    - ``(num_versions, mode)``: use a fixed count and explicit mode;
    - ``mode``: select a mode while leaving the version count to the scheduler.

    ``mode`` is ``"auto"``, ``"iteration"``, or ``"counter"``.
    ``"iteration"`` uses the affine flattened-loop index. ``"counter"`` uses
    a monotonic local counter that advances after a completed active buffer
    epoch. ``"auto"`` keeps the affine fast path when safe and selects a
    counter for conditionally executed or non-affine epochs.

    A fixed count is useful when you want to guarantee double-buffering for a
    critical buffer regardless of the Z3 solver's choice.

    AutoSchedule consumes this map and replaces it with the selected version
    counts, including versions chosen automatically by the solver.

    Example
    -------
    >>> @T.prim_func
    ... def my_kernel(x: T.Tensor((M, K), T.float16),
    ...               w: T.Tensor((N, K), T.float16),
    ...               y: T.Tensor((M, N), T.float16)):
    ...     with T.Kernel(T.ceildiv(N, block_N), threads=128) as bx:
    ...         fixed = T.alloc_shared((block_M, block_K), T.float16)
    ...         inferred = T.alloc_shared((block_M, block_K), T.float16)
    ...         T.annotate_buffer_versions({
    ...             fixed: (2, "counter"),
    ...             inferred: "iteration",
    ...         })
    ...         with T.Persistent(T.ceildiv(M, block_M), num_stages=4) as i:
    ...             ...
    """
    versions = {}
    modes = {}
    for buffer, specification in buffer_versions_map.items():
        if isinstance(specification, tuple):
            if len(specification) != 2:
                raise ValueError(f"buffer version tuple must be (num_versions, mode), got {specification!r} for buffer {buffer}")
            num_versions, mode = specification
            has_num_versions = True
            has_mode = True
        elif isinstance(specification, str):
            num_versions, mode = None, specification
            has_num_versions = False
            has_mode = True
        else:
            num_versions, mode = specification, None
            has_num_versions = True
            has_mode = False

        data = buffer.data
        if has_num_versions:
            if not isinstance(num_versions, int) or num_versions < 1:
                raise ValueError(f"num_versions must be a positive integer, got {num_versions} for buffer {buffer}")
            if data in versions and versions[data] != num_versions:
                raise ValueError(f"aliases of one buffer storage must use the same version count, got {versions[data]} and {num_versions}")
            versions[data] = num_versions

        if has_mode:
            if mode not in _VALID_BUFFER_VERSION_MODES:
                raise ValueError(f"buffer version mode must be one of {_VALID_BUFFER_VERSION_MODES}, got {mode!r} for buffer {buffer}")
            if data in modes and modes[data] != mode:
                raise ValueError(f"aliases of one buffer storage must use the same version mode, got {modes[data]!r} and {mode!r}")
            modes[data] = mode

    annotations = {}
    if versions:
        annotations["tl.buffer_versions_map"] = versions
    if modes:
        annotations["tl.buffer_version_mode"] = modes
    return sblock_attr(annotations)


def annotate_manual_multi_buffer(*items):
    """Declare buffers that the user has *manually* multi-buffered.

    Unlike :func:`annotate_buffer_versions` (which asks the Z3 auto-scheduler to
    version a buffer for you), this annotation tells AutoSchedule that the buffer
    is *already* multi-buffered by hand: the user writes the version index into
    every access themselves (e.g. ``buf[l % 2]`` / ``buf[(l + 1) % 2]``
    ping-pong), and the pass only needs to analyze dependencies and emit the
    ``set_flag`` / ``wait_flag`` synchronization.

    The per-buffer version count ``N`` is used as the ring size / the upper bound
    on the cross-iteration distance the analysis searches for — it does NOT
    reshape or reallocate the buffer and does NOT assume the version index lives
    in dimension 0. Each positional argument is either:

    - a bare buffer: ``N`` defaults to its leading dimension ``buffer.shape[0]``
      (which must then be a compile-time constant) — the common
      ``T.alloc_shared((N, ...))`` ping-pong case; or
    - a ``{buffer: N}`` dict: gives ``N`` explicitly, decoupled from the shape.
      Use this when the buffer has no dedicated leading version dim — e.g. a
      flattened buffer multi-buffered by disjoint sub-ranges — or when you want a
      version count that differs from ``shape[0]``.

    Example
    -------
    >>> @T.prim_func
    ... def my_kernel(...):
    ...     with T.Kernel(...) as core_id:
    ...         buf = T.alloc_shared((2, N), T.bfloat16)     # leading version dim
    ...         flat = T.alloc_shared((3 * N,), T.bfloat16)  # flat, sub-ranges
    ...         T.annotate_manual_multi_buffer(buf, {flat: 3})
    ...         for l in T.serial(L):
    ...             ...  # buf[l % 2] ping-pong; flat[(l % 3) * N : ...] triple-buffer
    """

    def _as_count(n, buffer):
        n = int(n.value if isinstance(n, IntImm) else n)
        if n < 1:
            raise ValueError(f"manual multi-buffer version count must be >= 1, got {n} for buffer {buffer}")
        return n

    _buffer_versions_map = {}
    for item in items:
        if isinstance(item, dict):
            for buffer, num_versions in item.items():
                _buffer_versions_map[buffer.data] = _as_count(num_versions, buffer)
            continue
        buffer = item
        shape0 = buffer.shape[0]
        if not isinstance(shape0, (int, IntImm)):
            raise ValueError(
                f"annotate_manual_multi_buffer needs a constant leading (version) dimension "
                f"to infer N for buffer {buffer}; pass {{buffer: N}} to specify it explicitly"
            )
        _buffer_versions_map[buffer.data] = _as_count(shape0, buffer)
    return sblock_attr({"tl.manual_multi_buffer": _buffer_versions_map})


def annotate_unlimit_memory(*scopes: str):
    """Remove the memory capacity limit for the given scopes during auto-scheduling.

    When unlimit'd, a scope's capacity is effectively unbounded so Z3 will not
    constrain multi-buffer versions based on that memory pool. Use this when you
    know a particular scope has spare headroom.

    Valid scopes: ``"shared"``, ``"shared.l1"``, ``"shared.l0c"``, ``"shared.l0a"``, ``"shared.l0b"``.

    Example
    -------
    >>> @T.prim_func
    ... def my_kernel(x: T.Tensor((M, K), T.float16),
    ...               w: T.Tensor((N, K), T.float16),
    ...               y: T.Tensor((M, N), T.float16)):
    ...     with T.Kernel(T.ceildiv(N, block_N), threads=128) as bx:
    ...         T.annotate_unlimit_memory("shared")
    ...         ...
    """
    valid = _VALID_MEMORY_SCOPES
    for scope in scopes:
        if scope not in valid:
            raise ValueError(f"Invalid memory scope {scope!r}, must be one of {valid}")
    return sblock_attr({"tl.unlimit_memory_scopes": list(scopes)})
