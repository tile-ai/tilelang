"""IR lower trace — zero-intrusion debug tool for visualizing compilation passes.

Monkey-patches ``tvm.ir.transform.Pass.__call__`` and ``PassPipeline.lower``
to automatically capture IR before/after every pass and generate diff reports.

This module has **no dependency on ``tilelang.env``**; configuration is read
from ``os.environ`` directly, or passed programmatically via ``enable()``.

Supports two architectures:
- New: ``PassPipeline.lower`` (each backend registers a pipeline object)
- Old: phase-based functions called from ``tilelang.engine.lower``

Usage::

    TL_LOWER_TRACE=1 python my_kernel.py        # HTML report
    TL_LOWER_TRACE=terminal python my_kernel.py  # terminal diff only
    TL_LOWER_TRACE=both python my_kernel.py      # both terminal and HTML
"""

from __future__ import annotations

import ast
import contextlib
import difflib
import dis
import functools
import inspect
import os
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

if TYPE_CHECKING:
    from collections.abc import Callable


STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"
STATUS_CODEGEN = "codegen"


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


_records: list[LowerRecord] = []
_section_cache: list[str] = []
_original_pass_call: Callable | None = None
_original_pipeline_lower: object | None = None
_original_codegen_ffis: dict[str, Callable] = {}
_legacy_patched: bool = False

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
]
_current_phase: str | None = None
_pass_index: int = 0
_auto_flush: bool = False
_script_dir: str | None = None
_run_dir: str | None = None
_lock = threading.RLock()
_run_counter: int = 0
_atexit_registered: bool = False

_UNSET: object = object()
_mode_override: str | None | object = _UNSET
_trace_dir_override: str | None | object = _UNSET
_codegen_output_path_override: str | None | object = _UNSET

# Phase label used for passes that run outside any PassPipeline.lower window
# (e.g. pre-pipeline module passes and tvm.build postproc), so they are still
# captured by the global Pass.__call__ hook.
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
    if _mode_override is not _UNSET:
        return _mode_override  # type: ignore[return-value]
    return _parse_lower_trace_mode(os.environ.get("TL_LOWER_TRACE"))


def _is_trace_enabled() -> bool:
    return _get_mode() is not None


def _should_print_terminal() -> bool:
    mode = _get_mode()
    return mode in ("terminal", "both")


def _should_gen_html() -> bool:
    mode = _get_mode()
    return mode in ("html", "both")


def _get_base_trace_dir() -> str:
    """Return the configured base trace directory (first level)."""
    if _trace_dir_override is not _UNSET and _trace_dir_override:
        return _trace_dir_override  # type: ignore[return-value]
    return os.environ.get("TL_LOWER_TRACE_DIR") or os.path.join(".", "tmp", "lower_trace_dir")


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
    """Return ``<script_dir>/run_records/run_<timestamp>_<pid>/`` (new per run)."""
    global _run_dir

    if _run_dir is not None:
        return _run_dir

    from datetime import datetime

    script_dir = _ensure_script_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    _run_dir = os.path.join(script_dir, "run_records", f"run_{timestamp}_{os.getpid()}")

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
    if _codegen_output_path_override is not _UNSET:
        return _codegen_output_path_override
    if _is_trace_enabled():
        script_dir = _ensure_script_dir()
        return os.path.join(script_dir, "codegen.cpp")
    return None


def _save_raw_files(record: LowerRecord):
    """Write before/after files to disk (phase subdirectory layout).

    For codegen records the *after* text is C++ source, so we write ``*.cpp``
    instead of ``*.tir``.
    """
    trace_dir = _ensure_run_dir()
    phase_dir = os.path.join(trace_dir, record.phase)
    os.makedirs(phase_dir, exist_ok=True)

    prefix = f"{record.index:02d}_{record.name}"
    before_ext = ".tir"
    after_ext = ".cpp" if record.status == STATUS_CODEGEN else ".tir"
    with open(os.path.join(phase_dir, f"{prefix}_before{before_ext}"), "w") as f:
        f.write(record.before_text)
    with open(os.path.join(phase_dir, f"{prefix}_after{after_ext}"), "w") as f:
        f.write(record.after_text)


def _get_pass_display_name(pass_obj) -> str:
    """Extract display name from pass_info.name, e.g. 'tir.Simplify' -> 'Simplify'."""
    try:
        name = str(pass_obj.info.name)
        return name.split(".")[-1] if "." in name else name
    except Exception:
        return type(pass_obj).__name__


def _incremental_flush_html():
    """Write the current HTML report incrementally.

    Uses _section_cache to avoid re-rendering previously completed sections.
    Total cost is O(n) instead of O(n^2) for full rewrites.
    """
    if not _records or not _run_dir:
        return

    from .html import generate_html

    html_path = os.path.join(_run_dir, "report.html")
    generate_html(_records, html_path)
    _update_html_symlink(html_path)


def _traced_pass_call(self, mod):
    """Intercept all Pass.__call__ invocations to record before/after IR.

    Captures every pass invocation globally (matching pass_diff's hook),
    including those outside any PassPipeline.lower window (pre-pipeline module
    passes, tvm.build postproc).  Records are appended at runtime with the
    pass's actual display name, eliminating the prior index-based pre-registration
    that could drift when conditional passes (e.g. LetInline) were skipped.
    """
    global _pass_index

    if not _is_trace_enabled():
        return _original_pass_call(self, mod)

    phase = _current_phase or _UNSCOPED_PHASE
    gen_html = _should_gen_html()
    if gen_html:
        _ensure_run_dir()
    before_text = str(mod)

    with _lock:
        idx = _pass_index
        _pass_index += 1

    try:
        result = _original_pass_call(self, mod)
    except Exception as e:
        with _lock:
            record = LowerRecord(
                phase=phase,
                name=_get_pass_display_name(self),
                index=idx,
                before_text=before_text,
                after_text="",
                changed=False,
                add_lines=0,
                del_lines=0,
                status=STATUS_FAILED,
                error_msg=str(e),
            )
            _records.append(record)
            _save_raw_files(record)
            print(f"  {_ANSI_RED}[lower_trace] {phase}/{idx:02d}_{record.name}: FAILED ({e}){_ANSI_RESET}")
        raise

    after_text = str(result)
    changed = before_text != after_text

    pass_name = _get_pass_display_name(self)

    add_count = del_count = 0
    if changed:
        sm = difflib.SequenceMatcher(None, before_text.splitlines(), after_text.splitlines())
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "insert":
                add_count += j2 - j1
            elif tag == "delete":
                del_count += i2 - i1
            elif tag == "replace":
                add_count += j2 - j1
                del_count += i2 - i1

    with _lock:
        record = LowerRecord(
            phase=phase,
            name=pass_name,
            index=idx,
            before_text=before_text,
            after_text=after_text,
            changed=changed,
            add_lines=add_count,
            del_lines=del_count,
            status=STATUS_COMPLETED,
        )
        _records.append(record)
        _save_raw_files(record)
        tag = "CHANGED" if changed else "NO-OP"
        tag_color = _ANSI_GREEN if changed else _ANSI_DIM
        print(f"  [lower_trace] {phase}/{idx:02d}_{pass_name}: {tag_color}{tag}{_ANSI_RESET}")

        if gen_html:
            with contextlib.suppress(Exception):
                _incremental_flush_html()

    if _should_print_terminal() and changed:
        from .diff import print_diff

        label = f"{phase}/{pass_name}"
        print_diff(before_text, after_text, f"{label} (before)", f"{label} (after)")

    return result


def _extract_pass_name_from_attr_chain(node: ast.expr) -> str | None:
    """Walk an attribute chain (e.g. tilelang.transform.Simplify) and extract pass name.

    Returns the pass name (e.g. 'Simplify') if the chain contains a 'transform' segment
    followed by an uppercase CamelCase name. Returns None otherwise.
    """
    if not isinstance(node, ast.Attribute):
        return None
    names = []
    cur = node
    while isinstance(cur, ast.Attribute):
        names.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        names.append(cur.id)
    names.reverse()
    try:
        transform_idx = names.index("transform")
    except ValueError:
        return None
    if transform_idx + 1 >= len(names):
        return None
    pass_name = names[transform_idx + 1]
    if not pass_name or not pass_name[0].isupper():
        return None
    return pass_name


def _discover_passes(phase_func) -> list[str]:
    """Extract pass names from a phase function's source code via AST parsing."""
    try:
        source = inspect.getsource(phase_func)
    except (OSError, TypeError):
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    passes = []
    seen_calls: set = set()

    class _PassVisitor(ast.NodeVisitor):
        def visit_Call(self, node):
            func = node.func
            found_in_nested = False
            while isinstance(func, ast.Call):
                if id(func) not in seen_calls:
                    seen_calls.add(id(func))
                    name = _extract_pass_name_from_attr_chain(func.func)
                    if name:
                        passes.append(name)
                        found_in_nested = True
                func = func.func

            if not found_in_nested and id(node) not in seen_calls:
                seen_calls.add(id(node))
                name = _extract_pass_name_from_attr_chain(func)
                if name:
                    passes.append(name)

            self.generic_visit(node)

    _PassVisitor().visit(tree)
    return passes


def _discover_passes_recursive(phase_func) -> list[str]:
    """Extract pass names, following local helper calls in the same module."""
    passes = []
    visited = set()
    seen_calls: set = set()

    def _visit(func):
        func_id = id(func)
        if func_id in visited:
            return
        visited.add(func_id)

        try:
            source = inspect.getsource(func)
        except (OSError, TypeError):
            return

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return

        func_module = inspect.getmodule(func)
        local_ns = {}
        if func_module:
            local_ns.update(vars(func_module))
        if hasattr(func, "__globals__"):
            local_ns.update(func.__globals__)

        class _PassVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                call_func = node.func

                found_in_nested = False
                while isinstance(call_func, ast.Call):
                    if id(call_func) not in seen_calls:
                        seen_calls.add(id(call_func))
                        name = _extract_pass_name_from_attr_chain(call_func.func)
                        if name:
                            passes.append(name)
                            found_in_nested = True
                    call_func = call_func.func

                if not found_in_nested and id(node) not in seen_calls:
                    seen_calls.add(id(node))
                    if isinstance(call_func, ast.Attribute):
                        name = _extract_pass_name_from_attr_chain(call_func)
                        if name:
                            passes.append(name)

                    elif isinstance(call_func, ast.Name):
                        name = call_func.id
                        resolved = local_ns.get(name)
                        if (
                            resolved is not None
                            and callable(resolved)
                            and not isinstance(resolved, type)
                            and not inspect.isbuiltin(resolved)
                        ):
                            resolved_module = getattr(resolved, "__module__", None)
                            func_module_name = getattr(func, "__module__", None)
                            if resolved_module == func_module_name:
                                _visit(resolved)

                self.generic_visit(node)

        _PassVisitor().visit(tree)

    _visit(phase_func)
    return passes


def _discover_phases(lower_func) -> list:
    """Discover phase functions from the old architecture via bytecode scanning."""
    try:
        from tilelang.engine import phase as phase_module
    except ImportError:
        return []

    phase_funcs = []
    seen_names = set()
    try:
        for instr in dis.get_instructions(lower_func):
            if instr.opname == "LOAD_GLOBAL" and instr.argval not in seen_names:
                name = instr.argval
                seen_names.add(name)
                func = getattr(phase_module, name, None)
                if func is not None and callable(func):
                    phase_funcs.append(func)
    except (TypeError, OSError):
        pass

    if not phase_funcs:
        phase_funcs = [
            getattr(phase_module, name)
            for name in sorted(dir(phase_module))
            if not name.startswith("_") and callable(getattr(phase_module, name, None))
        ]

    def _src_line(f):
        try:
            return inspect.getsourcelines(f)[1]
        except (OSError, TypeError):
            return 999999

    phase_funcs.sort(key=_src_line)
    return phase_funcs


def _wrap_phase(original_func, phase_index, total_phases):
    """Wrap a phase function to set tracing context (legacy architecture).

    Phase context is set so that passes invoked within this window are tagged
    with the phase label.  Pass records are appended at runtime by
    ``_traced_pass_call``; no pre-registration is performed.
    """
    base_phase_name = f"phase{phase_index}_{original_func.__name__}"

    @functools.wraps(original_func)
    def wrapper(*args, **kwargs):
        global _run_counter, _current_phase, _auto_flush, _run_dir

        with _lock:
            if phase_index == 1:
                _run_counter += 1
                if _run_counter == 1:
                    reset()
                else:
                    _run_dir = None

            run_prefix = f"run{_run_counter}_" if _run_counter > 1 else ""
            phase_name = f"{run_prefix}{base_phase_name}"

            _current_phase = phase_name
            _auto_flush = _should_gen_html()

        try:
            result = original_func(*args, **kwargs)
        except Exception as e:
            with _lock:
                _auto_flush = False
                _current_phase = None
                print(f"  [lower_trace] EXCEPTION in {phase_name}: {e}")

            with contextlib.suppress(Exception):
                _incremental_flush_html()

            raise

        with _lock:
            _auto_flush = False
            _current_phase = None

            if phase_index == total_phases:
                print(f"  [lower_trace] run {_run_counter} ({phase_name}) complete: {len(_records)} total records")

        with contextlib.suppress(Exception):
            _incremental_flush_html()

        return result

    return wrapper


def _traced_pipeline_lower(self, mod, target):
    """Intercept PassPipeline.lower to set phase context for pass tracing (new architecture).

    Only sets ``_current_phase`` so that passes invoked within this window are
    tagged with the pipeline label.  Pass records are appended at runtime by
    ``_traced_pass_call``; no pre-registration is performed, so conditional
    passes (e.g. LetInline) that are skipped at runtime simply do not appear,
    matching the behaviour of ``pass_diff``.
    """
    global _run_counter, _current_phase, _auto_flush, _run_dir

    with _lock:
        _run_counter += 1
        if _run_counter == 1:
            reset()
        else:
            _run_dir = None

        run_prefix = f"run{_run_counter}_" if _run_counter > 1 else ""
        phase_name = f"{run_prefix}pipeline_{self.name}"
        _current_phase = phase_name
        _auto_flush = _should_gen_html()

    try:
        result = _original_pipeline_lower(self, mod, target)
    except Exception as e:
        with _lock:
            _auto_flush = False
            _current_phase = None
            print(f"  [lower_trace] EXCEPTION in {phase_name}: {e}")

        with contextlib.suppress(Exception):
            _incremental_flush_html()

        raise

    with _lock:
        _auto_flush = False
        _current_phase = None

    with contextlib.suppress(Exception):
        _incremental_flush_html()

    print(f"  [lower_trace] run {_run_counter} ({phase_name}) complete: {len(_records)} total records")

    return result


class _CodegenSourceProxy:
    """Proxy returned when codegen is skipped (source loaded from file)."""

    def __init__(self, source: str):
        self._source = source

    def inspect_source(self) -> str:
        return self._source

    def get_source(self) -> str:
        return self._source


def _wrap_codegen_ffi(original_build):
    """Return a wrapper around a codegen FFI build function (``target.build.*``).

    The wrapper:
    1. Captures the final lowered TIR right before codegen runs (``str(mod)``).
    2. Temporarily sets ``_current_phase = 'codegen'`` so that the internal
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
                                   ``<path>`` and ``.original`` from the new
                                   codegen and compile with it.  The user can
                                   recover their edits via
                                   ``diff(<path>.original.bak, <path>.bak)``.
    """

    @functools.wraps(original_build)
    def wrapper(*args, **kwargs):
        global _pass_index, _current_phase

        if not _is_trace_enabled():
            return original_build(*args, **kwargs)

        mod = args[0] if args else kwargs.get("mod")
        gen_html = _should_gen_html()
        if gen_html:
            _ensure_run_dir()

        before_text = str(mod)
        codegen_out_path = _get_codegen_output_path()

        with _lock:
            idx = _pass_index
            _pass_index += 1
            _current_phase = "codegen"

        try:
            result = original_build(*args, **kwargs)
        except Exception as e:
            with _lock:
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
                _current_phase = None
                print(f"  [lower_trace] codegen/{idx:02d}_codegen: FAILED ({e})")
            raise

        codegen_text = result.inspect_source()

        patched_text = None
        if codegen_out_path:
            original_path = codegen_out_path + ".original"
            latest_path = codegen_out_path + ".latest"
            try:
                os.makedirs(os.path.dirname(os.path.abspath(codegen_out_path)), exist_ok=True)
                with open(latest_path, "w") as _f:
                    _f.write(codegen_text)
                if not os.path.isfile(codegen_out_path) or not os.path.isfile(original_path):
                    if os.path.isfile(codegen_out_path):
                        shutil.copyfile(codegen_out_path, codegen_out_path + ".bak")
                        print(
                            f"  {_ANSI_BOLD}{_ANSI_YELLOW}[lower_trace] codegen/{idx:02d}_codegen: INIT-BACKUP — {codegen_out_path} existed without baseline, backed up to {codegen_out_path}.bak{_ANSI_RESET}"
                        )
                    with open(original_path, "w") as _f:
                        _f.write(codegen_text)
                    shutil.copyfile(original_path, codegen_out_path)
                    print(f"  {_ANSI_GREEN}[lower_trace] codegen source initialized at: {codegen_out_path}{_ANSI_RESET}")
                else:
                    with open(original_path) as _f:
                        baseline_text = _f.read()
                    with open(codegen_out_path) as _f:
                        working_text = _f.read()
                    user_edited = working_text.rstrip() != baseline_text.rstrip()
                    codegen_changed = codegen_text.rstrip() != baseline_text.rstrip()
                    if not user_edited and not codegen_changed:
                        patched_text = None
                    elif not user_edited and codegen_changed:
                        with open(original_path, "w") as _f:
                            _f.write(codegen_text)
                        with open(codegen_out_path, "w") as _f:
                            _f.write(codegen_text)
                        print(
                            f"  {_ANSI_CYAN}[lower_trace] codegen/{idx:02d}_codegen: REGENERATED (codegen changed, no user edits){_ANSI_RESET}"
                        )
                        patched_text = None
                    elif user_edited and not codegen_changed:
                        patched_text = working_text
                        print(
                            f"  {_ANSI_BOLD}{_ANSI_GREEN}[lower_trace] codegen/{idx:02d}_codegen: PATCHED from {codegen_out_path}{_ANSI_RESET}"
                        )
                    else:
                        if working_text.rstrip() == codegen_text.rstrip():
                            with open(original_path, "w") as _f:
                                _f.write(codegen_text)
                            patched_text = working_text
                            print(
                                f"  {_ANSI_BOLD}{_ANSI_GREEN}[lower_trace] codegen/{idx:02d}_codegen: SYNCED (user edits match codegen, baseline advanced){_ANSI_RESET}"
                            )
                        else:
                            shutil.copyfile(codegen_out_path, codegen_out_path + ".bak")
                            shutil.copyfile(original_path, original_path + ".bak")
                            with open(original_path, "w") as _f:
                                _f.write(codegen_text)
                            with open(codegen_out_path, "w") as _f:
                                _f.write(codegen_text)
                            print(
                                f"  {_ANSI_BOLD}{_ANSI_YELLOW}[lower_trace] codegen/{idx:02d}_codegen: CONFLICT "
                                f"— {codegen_out_path} had user edits AND codegen changed; "
                                f"backed up to {codegen_out_path}.bak / {original_path}.bak, "
                                f"regenerated from new codegen.{_ANSI_RESET}"
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
            print(f"  [lower_trace] codegen/{idx:02d}_codegen: {tag} (+{add_count}/−{del_count}){path_suffix}")
            _current_phase = None

            if gen_html:
                with contextlib.suppress(Exception):
                    _incremental_flush_html()

        if _should_print_terminal():
            from .diff import print_diff

            print_diff(before_text, after_text, "codegen (TIR before)", "codegen (C++ after)")

        if patched_text is not None:
            return _CodegenSourceProxy(patched_text)
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


def enable(*, mode=_UNSET, trace_dir=_UNSET, codegen_output=_UNSET):
    """Enable IR pass tracing via monkey-patching.

    Parameters
    ----------
    mode : str | None, optional
        Force a trace mode (``'terminal'``, ``'html'``, ``'both'``, or
        ``None`` to disable).  When omitted, the mode is read from the
        ``TL_LOWER_TRACE`` env var (or a prior ``enable`` override),
        keeping this module free of any ``tilelang.env`` dependency.
    trace_dir : str | None, optional
        Force the trace output base directory.  When omitted, falls back to
        the ``TL_LOWER_TRACE_DIR`` env var, then
        ``./tmp/lower_trace_dir``.
    codegen_output : str | None, optional
        Path to save the codegen-generated C++/CUDA/etc. source code.  When
        omitted, defaults to ``<script_dir>/codegen.cpp`` (inside the
        per-script output directory, beside ``run_records/``).  Pass ``None``
        explicitly to suppress all extra saves.  See ``_wrap_codegen_ffi``
        for the three-file (``<path>`` / ``<path>.original`` /
        ``<path>.latest``) patch-and-recompile workflow.
    """
    global _mode_override, _trace_dir_override, _codegen_output_path_override

    if mode is not _UNSET:
        _mode_override = _parse_lower_trace_mode(mode if mode is None else str(mode))
    if trace_dir is not _UNSET:
        _trace_dir_override = trace_dir if trace_dir is None else str(trace_dir)
    if codegen_output is not _UNSET:
        _codegen_output_path_override = codegen_output if codegen_output is None else str(codegen_output)

    from tvm.ir.transform import Pass

    global _original_pass_call, _original_pipeline_lower, _atexit_registered, _legacy_patched
    if _original_pass_call is None:
        _original_pass_call = Pass.__call__
        Pass.__call__ = _traced_pass_call

    if not _original_codegen_ffis:
        import tvm.ffi

        for ffi_name in _CODEGEN_FFI_NAMES:
            try:
                orig = tvm.ffi.get_global_func(ffi_name)
                if orig is not None:
                    wrapped = _wrap_codegen_ffi(orig)
                    wrapped._original_ffi_name = ffi_name
                    _original_codegen_ffis[ffi_name] = orig
                    tvm.ffi.register_global_func(ffi_name, wrapped, override=True)
            except Exception:
                pass

    _register_atexit()

    if _original_pipeline_lower is not None or _legacy_patched:
        return

    try:
        from tilelang.backend.pass_pipeline import PassPipeline

        _original_pipeline_lower = PassPipeline.lower
        PassPipeline.lower = _traced_pipeline_lower
        print("[lower_trace] IR pass tracing enabled (PassPipeline architecture). Set TL_LOWER_TRACE=1 to enable.")
        return
    except ImportError:
        pass

    try:
        import tilelang.engine.lower as lower_mod

        lower_func = lower_mod.lower
        patch_mod = lower_mod
    except (ImportError, AttributeError):
        try:
            from tilelang.engine import lower as lower_func

            import tilelang.engine as patch_mod
        except (ImportError, AttributeError) as e:
            print(f"[lower_trace] WARNING: could not enable tracing — {e}")
            return

    phase_funcs = _discover_phases(lower_func)
    for i, phase_func in enumerate(phase_funcs):
        wrapped = _wrap_phase(phase_func, i + 1, len(phase_funcs))
        setattr(patch_mod, phase_func.__name__, wrapped)
        try:
            from tilelang.engine import phase as phase_module

            if hasattr(phase_module, phase_func.__name__):
                setattr(phase_module, phase_func.__name__, wrapped)
        except ImportError:
            pass
        if phase_func.__name__ in getattr(lower_func, "__globals__", {}):
            lower_func.__globals__[phase_func.__name__] = wrapped

    _legacy_patched = True
    print(f"[lower_trace] IR pass tracing enabled (phase-based architecture, {len(phase_funcs)} phases). Set TL_LOWER_TRACE=1 to enable.")


def _final_report():
    """Generate final HTML report at process exit, covering all accumulated runs."""
    if not _records or not _run_dir:
        return
    try:
        from .html import generate_html

        html_path = os.path.join(_run_dir, "report.html")
        generate_html(_records, html_path)
        _update_html_symlink(html_path)
        print(f"  [lower_trace] Final HTML report: {_ANSI_BLUE}{os.path.join(_script_dir, 'report.html')}{_ANSI_RESET}")
    except Exception as exc:
        print(f"  {_ANSI_RED}[lower_trace] WARNING: failed to generate final HTML report: {exc}{_ANSI_RESET}")


def disable():
    """Remove the pass tracing hook and restore original behavior."""
    global _original_pass_call, _original_pipeline_lower, _atexit_registered, _run_counter, _legacy_patched
    global _mode_override, _trace_dir_override, _codegen_output_path_override, _script_dir, _run_dir

    if _original_pass_call is not None:
        from tvm.ir.transform import Pass

        Pass.__call__ = _original_pass_call
        _original_pass_call = None

    if _original_pipeline_lower is not None:
        from tilelang.backend.pass_pipeline import PassPipeline

        PassPipeline.lower = _original_pipeline_lower

    _original_pipeline_lower = None
    _legacy_patched = False

    import tvm.ffi

    for ffi_name, orig in _original_codegen_ffis.items():
        with contextlib.suppress(Exception):
            tvm.ffi.register_global_func(ffi_name, orig, override=True)
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
    (which lazily create it via ``_ensure_run_dir``) run before
    ``PassPipeline.lower``/``_wrap_phase`` invokes ``reset`` on its first run.
    A fresh ``_run_dir`` for each subsequent run is instead established
    directly in ``_traced_pipeline_lower`` / ``_wrap_phase`` (when
    ``_run_counter > 1``), without calling ``reset`` so that records keep
    accumulating across runs.
    """
    global _records, _section_cache, _current_phase, _pass_index, _auto_flush
    _records = []
    _section_cache = []
    _current_phase = None
    _pass_index = 0
    _auto_flush = False
