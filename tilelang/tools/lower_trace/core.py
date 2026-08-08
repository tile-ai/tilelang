"""IR lower trace — zero-intrusion debug tool for visualizing compilation passes.

Uses TVM's ``PassInstrument`` API and TileLang's explicit backend-pipeline
scope to capture IR before/after every pass and generate diff reports.  Final
codegen remains a separate FFI boundary because it is not a TVM pass.

This module reads its configuration (``TL_LOWER_TRACE`` / ``TL_LOWER_TRACE_DIR``)
through the centralized ``tilelang.env`` environment, or via values passed
programmatically to ``enable()``.

Usage::

    TL_LOWER_TRACE=1 python my_kernel.py        # HTML report
    TL_LOWER_TRACE=terminal python my_kernel.py  # terminal diff only
    TL_LOWER_TRACE=both python my_kernel.py      # both terminal and HTML
"""

from __future__ import annotations

import contextlib
import difflib
import functools
import os
import re
import shutil
import sys
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .diff import (
    _ANSI_BOLD,
    _ANSI_BLUE,
    _ANSI_CYAN,
    _ANSI_DIM,
    _ANSI_GREEN,
    _ANSI_RED,
    _ANSI_RESET,
    _ANSI_YELLOW,
)
from tilelang.env import env
from tilelang.utils.pass_events import (
    IncompletePass,
    PassEvent,
    PassEventObserver,
    StackedPassInstrument,
    active_stacked_pass_instruments,
    current_pass_phase,
    pass_phase,
    register_pass_instrument_provider,
    register_pipeline_scope_provider,
    unregister_pass_instrument_provider,
    unregister_pipeline_scope_provider,
)

if TYPE_CHECKING:
    from collections.abc import Callable

STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"
STATUS_CODEGEN = "codegen"


def _get_tvm_ffi():
    """FFI facade exposing ``get_global_func``/``register_global_func``.

    Prefers the new-style ``tvm.ffi`` (unified FFI); falls back to the legacy
    ``tvm._ffi`` shipped in ``3rdparty/tvm``, where the register entry is named
    ``register_func`` instead of ``register_global_func``.
    """
    try:
        import tvm.ffi as _ffi

        if hasattr(_ffi, "register_global_func") and hasattr(_ffi, "get_global_func"):
            return _ffi
    except ImportError:
        pass
    import tvm._ffi as _ffi

    class _LegacyFFI:
        """Adapter exposing the ``register_global_func``/``get_global_func`` API over the legacy ``tvm._ffi`` (where registration is named ``register_func``)."""

        get_global_func = staticmethod(_ffi.get_global_func)

        @staticmethod
        def register_global_func(name, func=None, override=False):
            """Register a global FFI function via the legacy ``tvm._ffi.register_func`` entry point."""
            return _ffi.register_func(name, func, override=override)

    return _LegacyFFI()


def _inspect_module_source(mod):
    """Return the source text of a ``tvm.runtime.Module``.

    Prefers the newer ``inspect_source``; falls back to the legacy ``get_source``
    shipped with ``3rdparty/tvm``. Returns ``""`` only when neither hook is
    available, so downstream string handling stays safe.

    Exceptions raised by an *available* hook propagate to the caller — this is
    required so codegen tracing can observe failures (e.g. a module whose
    ``inspect_source`` raises) and reset the current phase correctly. The
    fallback to ``get_source`` only happens when the preferred attribute is
    missing or not callable.
    """
    for _attr in ("inspect_source", "get_source"):
        _fn = getattr(mod, _attr, None)
        if callable(_fn):
            return _fn() or ""
    return ""


@dataclass
class LowerRecord:
    """Result of running a single pass."""

    phase: str
    name: str
    index: int
    before_text: str
    after_text: str
    changed: bool
    add_lines: int = 0
    del_lines: int = 0
    status: str = STATUS_COMPLETED
    error_msg: str = ""
    depth: int = 0
    parent_index: int | None = None


_records: list[LowerRecord] = []
_section_cache: dict[tuple[str, int], str] = {}
_original_codegen_ffis: dict[str, Callable] = {}

_CODEGEN_FFI_NAMES: list[str] = [
    "target.build.tilelang_cuda",
    "target.build.tilelang_cuda_without_compile",
    "target.build.tilelang_cutedsl",
    "target.build.tilelang_cutedsl_without_compile",
    "target.build.tilelang_hip",
    "target.build.tilelang_hip_without_compile",
    "target.build.tilelang_metal",
    "target.build.tilelang_c",
    "target.build.tilelang_c_host",
    "target.build.tilelang_ascend",
    "target.build.tilelang_ascend_pto",
    "target.build.llvm",
    "target.build.webgpu",
    "target.build.tilelang_cpp",
    "target.build.tilelang_webgpu",
]

# FFIs whose returned module is consumed *only* via ``get_source()`` — i.e.
# the module holds source text (C / WGSL / etc.), not a compiled binary, and is
# never passed to ``host_mod.import_module()``.  For these FFIs a real
# ``CSourceModule`` rebuilt from the user-edited source can be returned in
# place of the original module so that the downstream JIT adapter (NVRTC /
# Cython / CuTeDSL / bisheng) recompiles the user-edited source.
#
# Membership is determined by tracing the call sites in
# ``tilelang.engine.lower``:
# - ``*_without_compile`` FFIs are only called from ``device_codegen_without_compile``
#   (which sets ``enable_device_compile=False`` → ``enable_host_codegen=False``),
#   so the result is never passed to ``import_module``.
# - ``tilelang_cpp`` and ``tilelang_webgpu`` produce ``CSourceModule`` / ``WebGPUModule``
#   (source-only, no binary compilation step); they are likewise only called from
#   ``device_codegen_without_compile``.
# - ``tilelang_metal`` is dual-use: called from both ``device_codegen``
#   (full-compile → ``import_module``) and ``device_codegen_without_compile``
#   (source-only).  Since the wrapper cannot distinguish the two call sites
#   at runtime, it is conservatively excluded.
# - ``llvm``, ``tilelang_c_host`` are excluded because they may be reached
#   via ``host_codegen`` → ``import_module`` (the module's binary or source
#   is consumed by the host runtime module tree).
# - ``tilelang_cuda``, ``tilelang_hip``, ``tilelang_cutedsl`` (full-compile
#   variants) produce binary modules consumed via ``import_module``.
# - ``tilelang_ascend`` and ``tilelang_ascend_pto`` are called from
#   ``device_codegen`` (not ``device_codegen_without_compile``), but
#   ``BuildTileLangAscend`` returns a ``CSourceModuleCreate(code, "c", ...)``
#   — a *source-only* module, not a binary.  ``tilelang.engine.lower`` only
#   calls ``.get_source()`` on it and never ``import_module()``; the
#   ``CythonKernelAdapter`` recompiles that source string via the ``bisheng``
#   compiler.  Hence a ``CSourceModule`` rebuilt from the patched source is
#   safe and user edits get recompiled.
#
# New FFIs default to *not* being in this set (conservative: return real module
# + NOTE), and must be explicitly added here once their call chain is verified
# to be source-only.
_SOURCE_ONLY_CODEGEN_FFIS: frozenset[str] = frozenset(
    {
        "target.build.tilelang_cuda_without_compile",
        "target.build.tilelang_cutedsl_without_compile",
        "target.build.tilelang_hip_without_compile",
        "target.build.tilelang_c",
        "target.build.webgpu",
        "target.build.tilelang_cpp",
        "target.build.tilelang_webgpu",
        "target.build.tilelang_ascend",
        "target.build.tilelang_ascend_pto",
    }
)
_pass_index: int = 0
_script_dir: str | None = None
_run_dir: str | None = None
_lock = threading.RLock()
_run_counter: int = 0
_atexit_registered: bool = False
_default_pass_context = None
_default_pass_context_instruments: list[object] | None = None
_default_pass_instrument: StackedPassInstrument | None = None

_UNSET: object = object()
_mode_override: str | None | object = _UNSET
_trace_dir_override: str | None | object = _UNSET
_codegen_output_path_override: str | None | object = _UNSET

# Phase label used for passes that run outside any PassPipeline.lower window
# (e.g. pre-pipeline module passes and tvm.build postproc).
_UNSCOPED_PHASE = "unscoped"


def _parse_lower_trace_mode(value: str | None) -> str | None:
    """Parse a TL_LOWER_TRACE-style value into a mode string."""
    if value is None:
        return None
    v = value.lower().strip()
    if v in ("", "0", "false", "no", "off"):
        return None
    if v in ("1", "true", "yes", "on"):
        return "html"
    if v in ("terminal", "html", "both"):
        return v
    return "html"


def _get_mode() -> str | None:
    """Return the effective trace mode, preferring the programmatic override then the env var."""
    if _mode_override is not _UNSET:
        return _mode_override  # type: ignore[return-value]
    return env.get_lower_trace_mode()


def _is_trace_enabled() -> bool:
    """Return True when tracing is currently active (mode is not None)."""
    return _get_mode() is not None


def _should_print_terminal() -> bool:
    """Return True when the terminal diff output should be produced."""
    mode = _get_mode()
    return mode in ("terminal", "both")


def _should_gen_html() -> bool:
    """Return True when the HTML report should be produced."""
    mode = _get_mode()
    return mode in ("html", "both")


def _get_base_trace_dir() -> str:
    """Return the configured base trace directory (first level)."""
    if _trace_dir_override is not _UNSET and _trace_dir_override:
        return _trace_dir_override  # type: ignore[return-value]
    return env.get_lower_trace_dir()


def _ensure_script_dir() -> str:
    """Return ``<base_dir>/<script_name>/`` (created on first call, stable across runs)."""
    global _script_dir

    if _script_dir is not None:
        return _script_dir

    base_dir = _get_base_trace_dir()
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0] or "kernel"
    _script_dir = os.path.join(base_dir, script_name)

    os.makedirs(_script_dir, exist_ok=True)
    return _script_dir


def _ensure_run_dir() -> str:
    """Return ``<script_dir>/.run_records/run_<timestamp>_<pid>/`` (new per run)."""
    global _run_dir

    if _run_dir is not None:
        return _run_dir

    from datetime import datetime

    script_dir = _ensure_script_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    _run_dir = os.path.join(script_dir, ".run_records", f"run_{timestamp}_{os.getpid()}")

    os.makedirs(_run_dir, exist_ok=True)
    return _run_dir


def _update_html_symlink(run_html_path: str):
    """Create/refresh ``<script_dir>/report.html`` → ``run_html_path``.

    On platforms where ``os.symlink`` fails (e.g. Windows without privileges),
    falls back to copying the file and prints a one-time warning.
    """
    script_dir = _ensure_script_dir()
    link_path = os.path.join(script_dir, "report.html")
    try:
        if os.path.islink(link_path) or os.path.exists(link_path):
            os.remove(link_path)
        os.symlink(os.path.relpath(run_html_path, script_dir), link_path)
    except OSError:
        import shutil

        shutil.copyfile(run_html_path, link_path)


def _get_codegen_output_path() -> str | None:
    """Return the configured codegen source output path, or None when off.

    An explicit override (set via ``enable(codegen_output=...)``) always wins,
    including in terminal-only mode and an explicit ``None`` to suppress. When
    no override is set, the default ``<script_dir>/codegen.cpp`` is only
    produced for persistence modes (``html``/``both``) so ``terminal`` mode
    stays a pure no-write diff as documented.
    """
    if _codegen_output_path_override is not _UNSET:
        return _codegen_output_path_override
    if _should_gen_html():
        script_dir = _ensure_script_dir()
        return os.path.join(script_dir, "codegen.cpp")
    return None


def _safe_filename_component(name: str) -> str:
    """Sanitize a record-derived name for use as a path component.

    Replaces path separators and other filesystem-unsafe characters so that a
    custom pass/phase name cannot escape its phase subdirectory (CWE-22).
    """
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(name))


def _save_raw_files(record: LowerRecord):
    """Write before/after files to disk (phase subdirectory layout).

    For codegen records the *after* text is C++ source, so we write ``*.cpp``
    instead of ``*.tir``.

    The run-directory layout (``<script_dir>/.run_records/...``) is created in
    every active mode including ``terminal``, so that raw ``.tir`` dumps and the
    per-script directory remain available for inspection; only the HTML report
    itself is gated on a persistence mode (``html``/``both``) — see
    ``_final_report()`` and ``_incremental_flush_html()``.

    Persistence is best-effort: any filesystem failure (unwritable
    ``TL_LOWER_TRACE_DIR``, bad filename, transient error) degrades to a
    warning so the observational tracing flow never aborts compilation.
    """
    try:
        trace_dir = _ensure_run_dir()
        phase_dir = os.path.join(trace_dir, _safe_filename_component(record.phase))
        os.makedirs(phase_dir, exist_ok=True)

        prefix = f"{record.index:02d}_{_safe_filename_component(record.name)}"
        before_ext = ".tir"
        after_ext = ".cpp" if record.status == STATUS_CODEGEN else ".tir"
        with open(os.path.join(phase_dir, f"{prefix}_before{before_ext}"), "w", encoding="utf-8") as f:
            f.write(record.before_text)
        with open(os.path.join(phase_dir, f"{prefix}_after{after_ext}"), "w", encoding="utf-8") as f:
            f.write(record.after_text)
    except Exception as exc:
        print(f"  {_ANSI_RED}[lower_trace] WARNING: could not save raw trace files: {exc}{_ANSI_RESET}")


def _get_pass_display_name(pass_obj) -> str:
    """Extract display name from pass_info.name, e.g. 'tir.Simplify' -> 'Simplify'."""
    try:
        name = str(pass_obj.info.name)
        return name.split(".")[-1] if "." in name else name
    except Exception:
        return type(pass_obj).__name__


def _incremental_flush_html():
    """Write the current HTML report incrementally.

    Passes ``_section_cache`` to ``generate_html`` so that already-rendered
    pass sections (and their computed diffs) are reused instead of being
    recomputed on every flush.  Only newly recorded passes incur the diff
    cost, keeping the total tracing overhead O(n) rather than O(n^2).
    """
    if not _should_gen_html() or not _records or not _run_dir:
        return

    from .html import generate_html

    html_path = os.path.join(_run_dir, "report.html")
    generate_html(_records, html_path, section_cache=_section_cache)
    _update_html_symlink(html_path)


def _allocate_pass_index() -> int:
    """Allocate a process-wide trace index shared by passes and codegen."""
    global _pass_index

    with _lock:
        index = _pass_index
        _pass_index += 1
    return index


def _count_line_changes(before_text: str, after_text: str) -> tuple[int, int]:
    """Return inserted/deleted line counts for a before/after snapshot."""
    add_count = del_count = 0
    matcher = difflib.SequenceMatcher(None, before_text.splitlines(), after_text.splitlines())
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert":
            add_count += j2 - j1
        elif tag == "delete":
            del_count += i2 - i1
        elif tag == "replace":
            add_count += j2 - j1
            del_count += i2 - i1
    return add_count, del_count


def _instrument_phase() -> str:
    """Resolve the context-local pass-event phase."""
    return current_pass_phase() or _UNSCOPED_PHASE


def _compact_pass_name(name: str) -> str:
    """Match the compact display names historically used by lower trace."""
    return name.rsplit(".", 1)[-1]


class _LowerTraceObserver(PassEventObserver):
    """Turn shared PassInstrument callbacks into existing ``LowerRecord`` data."""

    def pass_started(self, mod, event: PassEvent) -> str:
        if _should_gen_html():
            _ensure_run_dir()
        # Serialize immediately: a pass may mutate the IRModule in place.
        return str(mod)

    def pass_finished(self, mod, event: PassEvent, before_text: str) -> None:
        after_text = str(mod)
        changed = before_text != after_text
        add_count, del_count = _count_line_changes(before_text, after_text) if changed else (0, 0)
        phase = event.phase or _UNSCOPED_PHASE
        pass_name = _compact_pass_name(event.name)

        with _lock:
            record = LowerRecord(
                phase=phase,
                name=pass_name,
                index=event.sequence,
                before_text=before_text,
                after_text=after_text,
                changed=changed,
                add_lines=add_count,
                del_lines=del_count,
                status=STATUS_COMPLETED,
                depth=event.depth,
                parent_index=event.parent_sequence,
            )
            _records.append(record)
            _records.sort(key=lambda item: item.index)
            _save_raw_files(record)
            tag = "CHANGED" if changed else "NO-OP"
            tag_color = _ANSI_GREEN if changed else _ANSI_DIM
            print(f"  [lower_trace] {phase}/{event.sequence:02d}_{pass_name}: {tag_color}{tag}{_ANSI_RESET}")

            if _should_gen_html():
                with contextlib.suppress(Exception):
                    _incremental_flush_html()

        if _should_print_terminal() and changed:
            from .diff import print_diff

            label = f"{phase}/{pass_name}"
            print_diff(before_text, after_text, f"{label} (before)", f"{label} (after)")

    def passes_incomplete(self, passes: list[IncompletePass], error: BaseException | None) -> None:
        error_msg = str(error) if error is not None else "pass did not complete"
        gen_html = _should_gen_html()

        with _lock:
            for item in passes:
                if item.event is None:
                    continue
                event = item.event
                phase = event.phase or _UNSCOPED_PHASE
                pass_name = _compact_pass_name(event.name)
                record = LowerRecord(
                    phase=phase,
                    name=pass_name,
                    index=event.sequence,
                    before_text=str(item.state) if item.state is not None else "",
                    after_text="",
                    changed=False,
                    status=STATUS_FAILED,
                    error_msg=error_msg,
                    depth=event.depth,
                    parent_index=event.parent_sequence,
                )
                _records.append(record)
                _save_raw_files(record)
                print(f"  {_ANSI_RED}[lower_trace] {phase}/{event.sequence:02d}_{pass_name}: FAILED ({error_msg}){_ANSI_RESET}")

            _records.sort(key=lambda record: record.index)
            if gen_html:
                with contextlib.suppress(Exception):
                    _incremental_flush_html()

    def callback_mismatch(self, actual: str, expected: str | None) -> None:
        print(f"  {_ANSI_RED}[lower_trace] WARNING: unmatched after-pass callback for {actual!r}; expected {expected!r}{_ANSI_RESET}")


def create_pass_instrument() -> StackedPassInstrument:
    """Create a context-local PassInstrument backed by lower-trace state."""
    return StackedPassInstrument(
        _LowerTraceObserver(),
        capture_nested=True,
        capture_predicate=lambda _name, _depth: _is_trace_enabled(),
        sequence_allocator=_allocate_pass_index,
        phase_provider=_instrument_phase,
    )


def _abort_active_instruments(error: BaseException) -> None:
    """Attach a pipeline exception to any active lower-trace pass frames."""
    for instrument in reversed(active_stacked_pass_instruments()):
        if isinstance(instrument.observer, _LowerTraceObserver) and instrument.pending_events:
            instrument.abort(error)


@contextlib.contextmanager
def _lower_trace_pipeline_scope(base_phase: str):
    """Provide run/phase lifecycle around an explicitly instrumented pipeline."""
    global _run_counter, _run_dir

    if not _is_trace_enabled():
        yield
        return

    with _lock:
        _run_counter += 1
        run_number = _run_counter
        if run_number > 1:
            _run_dir = None
        run_prefix = f"run{run_number}_" if run_number > 1 else ""
        phase_name = f"{run_prefix}{base_phase}"

    succeeded = False
    try:
        with pass_phase(phase_name):
            yield
        succeeded = True
    except Exception as error:
        _abort_active_instruments(error)
        print(f"  [lower_trace] EXCEPTION in {phase_name}: {error}")
        raise
    finally:
        with contextlib.suppress(Exception):
            _incremental_flush_html()

    if succeeded:
        print(f"  [lower_trace] run {run_number} ({phase_name}) complete: {len(_records)} total records")


def _make_patched_source_module(original_module, patched_source: str):
    """Build a real ``CSourceModule`` carrying ``patched_source``.

    Every codegen FFI in ``_CODEGEN_FFI_NAMES`` is invoked via
    ``tvm._ffi.get_global_func(name)(...)`` (see ``tilelang.engine.lower``
    ``device_codegen`` / ``device_codegen_without_compile``).  The wrapper is
    registered as a PackedFunc, so its return value is marshalled back through
    ``TVMCFuncSetReturn`` which only accepts TVM-recognised types (Module /
    Object / PackedFunc / scalars / str / ...).  A pure-Python proxy cannot
    cross that boundary (``TypeError: Don't know how to handle type ...``).

    We therefore construct a genuine ``CSourceModule`` — the same factory used
    by ``BuildTileLangAscend`` in C++ (``CSourceModuleCreate(code, "c",
    function_names)``) — so the patched source both crosses the FFI boundary
    and is returned by ``.get_source()``, which is the sole consumer in
    ``tilelang.engine.lower`` (line 233: ``codegen_mod.get_source()``).
    """
    import tvm.runtime._ffi_api as _ffi_api

    try:
        fmt = original_module.format
    except Exception:
        fmt = "c"
    return _ffi_api.CSourceModuleCreate(patched_source, fmt, [], None)


def _wrap_codegen_ffi(original_build, ffi_name=""):
    """Return a wrapper around a codegen FFI build function (``target.build.*``).

    Parameters
    ----------
    original_build : Callable
        The original codegen FFI function.
    ffi_name : str
        The registered FFI name (e.g. ``target.build.tilelang_cuda``).
        Used to decide whether a patched ``CSourceModule`` can be returned
        (safe for source-only FFIs) or the real module must be
        returned (required for full-compile FFIs whose binary is consumed
        downstream via ``host_mod.import_module``).

    The wrapper:
    1. Captures the final lowered TIR right before codegen runs (``str(mod)``).
    2. Enters a context-local ``codegen`` phase so that the internal
       ``tir.transform.Simplify()`` call in ``device_codegen`` is automatically
       attributed to the ``codegen`` phase.
    3. After codegen finishes, captures the generated source via
       ``result.inspect_source()`` and appends a ``STATUS_CODEGEN`` record.

    Codegen output handling (when ``codegen_output`` path is configured):

    Three files collaborate to disambiguate whether a content difference is
    caused by user edits, by a codegen change, or by both:
    - ``<path>``           — user-editable working copy.
    - ``<path>.original``  — baseline: the codegen snapshot the working copy
                             was last synced from (written only on init or
                             re-sync, never blindly overwritten).
    - ``<path>.latest``    — the actual codegen output of *this* run
                             (overwritten every run, for diff reference).

    On each run a three-way comparison (baseline / working / current codegen)
    decides:
    - neither changed            → use codegen as-is.
    - only codegen changed       → regenerate ``<path>`` and ``.original``
                                   from the new codegen.
    - only user edited           → inject the working copy (PATCHED).
    - both changed, working==    → user already synced manually; advance
      current                     baseline and use the working copy.
    - both changed, working!=    → CONFLICT: back up the user's working copy
      current                     to ``<path>.bak`` and the old baseline to
                                   ``<path>.original.bak``, then regenerate
                                   ``<path>`` and ``<path>.original`` from the new
                                   codegen and compile with it.  The user can
                                   recover their edits via
                                   ``diff(<path>.original.bak, <path>.bak)``.

    When the working copy is injected (PATCHED / SYNCED), the return value
    depends on whether the FFI is in ``_SOURCE_ONLY_CODEGEN_FFIS``:
    - Source-only FFIs (``*_without_compile``, ``tilelang_c``, ``webgpu``,
      ``tilelang_ascend``, ``tilelang_ascend_pto``)
      → return a ``CSourceModule`` rebuilt from the patched source (via
      ``_make_patched_source_module``) so it crosses the FFI boundary and
      the downstream JIT adapter (NVRTC / Cython / CuTeDSL / bisheng)
      recompiles the edited source.
    - Full-compile FFIs (``tilelang_cuda``, ``tilelang_hip``, ``tilelang_metal``,
      ``llvm``, ``tilelang_c_host``, …) → return the original ``result``
      (whose binary was compiled from TIR) and print a NOTE advising the
      user to switch to a source-compiling execution backend (e.g.
      ``nvrtc``) for edit-and-recompile support.
    """

    @functools.wraps(original_build)
    def wrapper(*args, **kwargs):
        """Run codegen under the trace: capture TIR-before/C++-after and emit a STATUS_CODEGEN record."""
        global _pass_index

        if not _is_trace_enabled():
            return original_build(*args, **kwargs)

        mod = args[0] if args else kwargs.get("mod")
        gen_html = _should_gen_html()
        if gen_html:
            _ensure_run_dir()

        before_text = str(mod)
        codegen_out_path = _get_codegen_output_path()

        try:
            with pass_phase("codegen"):
                result = original_build(*args, **kwargs)
        except Exception as e:
            _abort_active_instruments(e)
            with _lock:
                idx = _pass_index
                _pass_index += 1
                record = LowerRecord(
                    phase="codegen",
                    name=getattr(original_build, "__name__", "codegen"),
                    index=idx,
                    before_text=before_text,
                    after_text="",
                    changed=False,
                    status=STATUS_FAILED,
                    error_msg=str(e),
                )
                _records.append(record)
                _save_raw_files(record)
                print(f"  [lower_trace] codegen/{idx:02d}_codegen: FAILED ({e})")
            raise

        with _lock:
            idx = _pass_index
            _pass_index += 1
        patched_text = None
        codegen_text = ""
        after_text = ""
        try:
            codegen_text = _inspect_module_source(result)
            if codegen_out_path:
                original_path = codegen_out_path + ".original"
                latest_path = codegen_out_path + ".latest"
                try:
                    os.makedirs(os.path.dirname(os.path.abspath(codegen_out_path)), exist_ok=True)
                    with open(latest_path, "w", encoding="utf-8") as _f:
                        _f.write(codegen_text)
                    if not os.path.isfile(codegen_out_path) or not os.path.isfile(original_path):
                        if os.path.isfile(codegen_out_path):
                            shutil.copyfile(codegen_out_path, codegen_out_path + ".bak")
                            print(
                                f"  [lower_trace] codegen/{idx:02d}_codegen: INIT-BACKUP — {_ANSI_BOLD}{_ANSI_YELLOW}{codegen_out_path}{_ANSI_RESET} existed without baseline, backed up to {_ANSI_BOLD}{_ANSI_YELLOW}{codegen_out_path}.bak{_ANSI_RESET}"
                            )
                        with open(original_path, "w", encoding="utf-8") as _f:
                            _f.write(codegen_text)
                        shutil.copyfile(original_path, codegen_out_path)
                        print(f"  [lower_trace] codegen source initialized at: {_ANSI_GREEN}{codegen_out_path}{_ANSI_RESET}")
                    else:
                        with open(original_path, encoding="utf-8") as _f:
                            baseline_text = _f.read()
                        with open(codegen_out_path, encoding="utf-8") as _f:
                            working_text = _f.read()
                        user_edited = working_text.rstrip() != baseline_text.rstrip()
                        codegen_changed = codegen_text.rstrip() != baseline_text.rstrip()
                        if not user_edited and not codegen_changed:
                            patched_text = None
                        elif not user_edited and codegen_changed:
                            with open(original_path, "w", encoding="utf-8") as _f:
                                _f.write(codegen_text)
                            with open(codegen_out_path, "w", encoding="utf-8") as _f:
                                _f.write(codegen_text)
                            print(
                                f"  {_ANSI_CYAN}[lower_trace] codegen/{idx:02d}_codegen: REGENERATED (codegen changed, no user edits){_ANSI_RESET}"
                            )
                            patched_text = None
                        elif user_edited and not codegen_changed:
                            patched_text = working_text
                            print(f"  {_ANSI_BOLD}{_ANSI_GREEN}[lower_trace] codegen/{idx:02d}_codegen: PATCHED (user edits){_ANSI_RESET}")
                        else:
                            if working_text.rstrip() == codegen_text.rstrip():
                                with open(original_path, "w", encoding="utf-8") as _f:
                                    _f.write(codegen_text)
                                patched_text = working_text
                                print(
                                    f"  {_ANSI_BOLD}{_ANSI_GREEN}[lower_trace] codegen/{idx:02d}_codegen: SYNCED (user edits & codegen changed, but they are the same){_ANSI_RESET}"
                                )
                            else:
                                shutil.copyfile(codegen_out_path, codegen_out_path + ".bak")
                                shutil.copyfile(original_path, original_path + ".bak")
                                with open(original_path, "w", encoding="utf-8") as _f:
                                    _f.write(codegen_text)
                                with open(codegen_out_path, "w", encoding="utf-8") as _f:
                                    _f.write(codegen_text)
                                print(
                                    f"  {_ANSI_BOLD}{_ANSI_YELLOW}[lower_trace] codegen/{idx:02d}_codegen: CONFLICT (user edits & codegen changed, conflict with each other, codegen overwrites user edits). {_ANSI_RESET}"
                                )
                                patched_text = None
                except Exception as _exc:
                    print(f"  {_ANSI_RED}[lower_trace] WARNING: codegen file I/O failed: {_exc}{_ANSI_RESET}")
                    patched_text = None

            after_text = patched_text if patched_text is not None else codegen_text

            sm = difflib.SequenceMatcher(None, before_text.splitlines(), after_text.splitlines())
            add_count = del_count = 0
            for _tag, i1, i2, j1, j2 in sm.get_opcodes():
                if _tag == "insert":
                    add_count += j2 - j1
                elif _tag == "delete":
                    del_count += i2 - i1
                elif _tag == "replace":
                    add_count += j2 - j1
                    del_count += i2 - i1

            with _lock:
                record = LowerRecord(
                    phase="codegen",
                    name="codegen",
                    index=idx,
                    before_text=before_text,
                    after_text=after_text,
                    changed=True,
                    add_lines=add_count,
                    del_lines=del_count,
                    status=STATUS_CODEGEN,
                )
                _records.append(record)
                _save_raw_files(record)
                tag = "CODEGEN"
                path_suffix = f"  →  {_ANSI_BLUE}{codegen_out_path}{_ANSI_RESET}" if codegen_out_path else ""
                print(f"  [lower_trace] codegen/{idx:02d}_codegen: {tag} (+{add_count}/-{del_count}){path_suffix}")

                if gen_html:
                    with contextlib.suppress(Exception):
                        _incremental_flush_html()
        except Exception as exc:
            print(f"  {_ANSI_RED}[lower_trace] WARNING: post-codegen tracing failed: {exc}{_ANSI_RESET}")

        if _should_print_terminal():
            from .diff import print_diff

            print_diff(before_text, after_text, "codegen (TIR before)", "codegen (C++ after)")

        if patched_text is not None:
            if ffi_name in _SOURCE_ONLY_CODEGEN_FFIS:
                # Source-only FFIs produce modules whose sole consumer is
                # get_source()/inspect_source(); the downstream JIT adapter
                # (NVRTC/Cython/CuTeDSL/bisheng) recompiles the source string.
                # Return a real CSourceModule built from the patched source so
                # that (1) it crosses the TVM PackedFunc FFI return-value
                # boundary (a pure-Python proxy cannot) and (2) get_source()
                # yields the user-edited source for recompilation.
                return _make_patched_source_module(result, patched_text)
            else:
                # Full-compile FFIs return a module whose binary (PTX/hsaco)
                # was compiled from TIR and is consumed downstream via
                # host_mod.import_module().  A pure-Python proxy cannot be
                # used here (the FFI boundary requires a real Module handle),
                # so the unpatched module is returned.  Only the trace/display
                # text reflects the user's edits.
                if patched_text.rstrip() != codegen_text.rstrip():
                    target_kind = ""
                    _t = args[1] if len(args) > 1 else kwargs.get("target")
                    target_kind = getattr(getattr(_t, "kind", None), "name", "")
                    backend_hint = ""
                    if target_kind == "cuda":
                        backend_hint = " Use execution_backend='nvrtc' for edit-and-recompile support."
                    elif target_kind == "hip":
                        backend_hint = " Use execution_backend='cython' for edit-and-recompile support."
                    print(
                        f"  {_ANSI_YELLOW}[lower_trace] codegen/{idx:02d}_codegen: NOTE — "
                        f"user edits in {codegen_out_path} are recorded in the trace for diff "
                        f"viewing, but were NOT recompiled (the codegen FFI builds from TIR, "
                        f"not from C++ source). The compiled artifact reflects the unpatched "
                        f"codegen output.{backend_hint}{_ANSI_RESET}"
                    )

        return result

    return wrapper


def _register_atexit():
    """Register the final-report atexit handler (idempotent)."""
    global _atexit_registered
    if _atexit_registered:
        return
    import atexit

    atexit.register(_final_report)
    _atexit_registered = True


def _install_default_context_instrument() -> None:
    """Attach tracing to TVM's current context for direct ``tilelang.lower`` calls."""
    global _default_pass_context, _default_pass_context_instruments, _default_pass_instrument

    if _default_pass_instrument is not None:
        return

    from tvm.ir.transform import PassContext

    pass_context = PassContext.current()
    previous_instruments = list(pass_context.instruments)
    instrument = create_pass_instrument()
    pass_context.override_instruments([*previous_instruments, instrument])
    _default_pass_context = pass_context
    _default_pass_context_instruments = previous_instruments
    _default_pass_instrument = instrument


def _restore_default_context_instrument() -> None:
    """Restore the context instrument list saved by enable()."""
    global _default_pass_context, _default_pass_context_instruments, _default_pass_instrument

    if _default_pass_context is not None and _default_pass_context_instruments is not None:
        with contextlib.suppress(Exception):
            _default_pass_context.override_instruments(_default_pass_context_instruments)
    _default_pass_context = None
    _default_pass_context_instruments = None
    _default_pass_instrument = None


def enable(*, mode=_UNSET, trace_dir=_UNSET, codegen_output=_UNSET):
    """Enable IR pass tracing via TVM PassInstrument.

    Parameters
    ----------
    mode : str | None, optional
        Force a trace mode (``'terminal'``, ``'html'``, ``'both'``, or
        ``None`` to disable).  When omitted, the mode is read from the
        ``TL_LOWER_TRACE`` env var (via ``tilelang.env``) or a prior
        ``enable`` override.
    trace_dir : str | None, optional
        Force the trace output base directory.  When omitted, falls back to
        the ``TL_LOWER_TRACE_DIR`` env var, then
        ``./tmp/lower_trace_dir``.
    codegen_output : str | None, optional
        Path to save the codegen-generated C++/CUDA/etc. source code.  When
        omitted, defaults to ``<script_dir>/codegen.cpp`` (inside the
        per-script output directory, beside ``.run_records/``).  Pass ``None``
        explicitly to suppress all extra saves.  See ``_wrap_codegen_ffi``
        for the three-file (``<path>`` / ``<path>.original`` /
        ``<path>.latest``) patch-and-recompile workflow.
    """
    global _mode_override, _trace_dir_override, _codegen_output_path_override, _script_dir, _run_dir

    if mode is not _UNSET:
        _mode_override = _parse_lower_trace_mode(mode if mode is None else str(mode))
    if trace_dir is not _UNSET:
        new_trace_dir = trace_dir if trace_dir is None else str(trace_dir)
        if new_trace_dir != _trace_dir_override:
            _script_dir = None
            _run_dir = None
        _trace_dir_override = new_trace_dir
    if codegen_output is not _UNSET:
        _codegen_output_path_override = codegen_output if codegen_output is None else str(codegen_output)

    # Explicitly disabling (mode=None or an off-value): remove any providers a
    # prior enable() may have installed so global state is left unchanged,
    # then re-assert the None override so a stale TL_LOWER_TRACE env var
    # cannot silently re-enable tracing. The no-args case (mode unset) still
    # falls through to install providers and resolve the mode at runtime.
    if mode is not _UNSET and _mode_override is None:
        disable()
        _mode_override = None
        return

    global _atexit_registered

    register_pass_instrument_provider("lower_trace", create_pass_instrument)
    register_pipeline_scope_provider("lower_trace", _lower_trace_pipeline_scope)
    _install_default_context_instrument()

    if not _original_codegen_ffis:
        _ffi = _get_tvm_ffi()

        for ffi_name in _CODEGEN_FFI_NAMES:
            try:
                orig = _ffi.get_global_func(ffi_name)
                if orig is not None:
                    wrapped = _wrap_codegen_ffi(orig, ffi_name)
                    _original_codegen_ffis[ffi_name] = orig
                    _ffi.register_global_func(ffi_name, wrapped, override=True)
            except Exception as exc:
                print(f"[lower_trace] WARNING: could not wrap codegen FFI {ffi_name}: {exc}")

    _register_atexit()
    print("[lower_trace] IR pass tracing enabled (PassInstrument architecture). Set TL_LOWER_TRACE=1 to enable.")


def _final_report():
    """Generate final HTML report at process exit, covering all accumulated runs."""
    if not _should_gen_html() or not _records or not _run_dir:
        return
    try:
        from .html import generate_html

        html_path = os.path.join(_run_dir, "report.html")
        generate_html(_records, html_path, section_cache=_section_cache)
        _update_html_symlink(html_path)
        print(f"  [lower_trace] Final HTML report: {_ANSI_BLUE}{os.path.join(_script_dir, 'report.html')}{_ANSI_RESET}")
    except Exception as exc:
        print(f"  {_ANSI_RED}[lower_trace] WARNING: failed to generate final HTML report: {exc}{_ANSI_RESET}")


def disable():
    """Disable tracing providers and restore original runtime behavior."""
    global _atexit_registered, _run_counter
    global _mode_override, _trace_dir_override, _codegen_output_path_override, _script_dir, _run_dir

    unregister_pipeline_scope_provider("lower_trace")
    unregister_pass_instrument_provider("lower_trace")
    _restore_default_context_instrument()

    _ffi = _get_tvm_ffi()

    for ffi_name, orig in _original_codegen_ffis.items():
        with contextlib.suppress(Exception):
            _ffi.register_global_func(ffi_name, orig, override=True)
    _original_codegen_ffis.clear()

    _mode_override = _UNSET
    _trace_dir_override = _UNSET
    _codegen_output_path_override = _UNSET

    if _atexit_registered:
        import atexit

        atexit.unregister(_final_report)
        _atexit_registered = False

    _run_counter = 0
    _script_dir = None
    _run_dir = None
    reset()


def reset():
    """Clear collected records and section cache.

    ``_script_dir`` is preserved (stable across runs, holds codegen files +
    html symlink).  ``_run_dir`` is also preserved: clearing it here would
    split a single run into two directories, because pre-pipeline passes
    (which lazily create it via ``_ensure_run_dir``) may run before the explicit
    pipeline scope begins.  ``_lower_trace_pipeline_scope`` establishes a fresh
    directory for subsequent runs without clearing accumulated records.
    """
    global _records, _section_cache, _pass_index
    _records = []
    _section_cache = {}
    _pass_index = 0
