"""IR lower tracing implemented as a per-compilation instrumentation tool.

``enable()`` only registers configuration for future compilations.  Every
compiler front door snapshots that configuration into a distinct
``LowerTraceSession`` which owns its records, pass numbering, pipeline scopes,
and report lifecycle.  No TVM global function or default ``PassContext`` is
patched.
"""

from __future__ import annotations

import contextlib
import difflib
import os
import re
import shutil
import sys
import threading
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from tilelang.env import env
from tilelang.instrumentation import (
    CodegenEvent,
    IncompletePass,
    PassEvent,
    PassEventObserver,
    PassInstrumentationTool,
    StackedPassInstrument,
    active_stacked_pass_instruments,
    current_compile_pass_instrumentation,
    current_pass_phase,
    pass_phase,
    register_pass_instrumentation_tool,
    unregister_pass_instrumentation_tool,
)

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
    from collections.abc import Callable, Generator


STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"
STATUS_CODEGEN = "codegen"

_UNSCOPED_PHASE = "unscoped"
_UNSET: object = object()
_mode_override: str | None | object = _UNSET
_trace_dir_override: str | None | object = _UNSET
_codegen_output_path_override: str | None | object = _UNSET
_config_lock = threading.RLock()
_last_session: ContextVar[LowerTraceSession | None] = ContextVar("tilelang_last_lower_trace_session", default=None)
# Explicit ``codegen_output`` paths may intentionally be shared across compile
# sessions. Serialize only their three-file read/modify/write transaction; pass
# records and callback state remain entirely session-local.
_codegen_path_locks_guard = threading.Lock()
_codegen_path_locks: dict[str, threading.Lock] = {}


# FFIs whose returned module is consumed as source rather than as an already
# compiled binary.  For these entries, a user-edited working copy can safely be
# returned as a new CSourceModule and compiled by the downstream adapter.
_SOURCE_ONLY_CODEGEN_FFIS: frozenset[str] = frozenset(
    {
        "target.build.tilelang_cuda_without_compile",
        "target.build.tilelang_cutedsl_without_compile",
        "target.build.tilelang_hip_without_compile",
        "target.build.tilelang_metal_without_compile",
        "target.build.tilelang_c",
        "target.build.webgpu",
        "target.build.tilelang_cpp",
        "target.build.tilelang_webgpu",
        "target.build.tilelang_ascend",
        "target.build.tilelang_ascend_pto",
    }
)


@dataclass
class LowerRecord:
    """Result of running a single compiler pass or codegen boundary."""

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


@dataclass(frozen=True)
class _LowerTraceConfig:
    mode: str | None
    trace_dir: str
    codegen_output: str | None | object


def _parse_lower_trace_mode(value: str | None) -> str | None:
    """Parse a ``TL_LOWER_TRACE``-style value into a mode string."""
    if value is None:
        return None
    normalized = value.lower().strip()
    if normalized in ("", "0", "false", "no", "off"):
        return None
    if normalized in ("1", "true", "yes", "on"):
        return "html"
    if normalized in ("terminal", "html", "both"):
        return normalized
    return "html"


def _get_mode() -> str | None:
    """Return the configuration used by future compile sessions."""
    with _config_lock:
        override = _mode_override
    if override is not _UNSET:
        return override  # type: ignore[return-value]
    return env.get_lower_trace_mode()


def _is_trace_enabled() -> bool:
    """Return whether future compilation sessions have tracing enabled."""
    return _get_mode() is not None


def _safe_filename_component(name: str) -> str:
    """Make an event-derived name safe to use as one path component."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(name))


def _get_pass_display_name(pass_obj) -> str:
    """Extract a compact display name from ``pass_info.name``."""
    try:
        name = str(pass_obj.info.name)
        return name.split(".")[-1] if "." in name else name
    except Exception:
        return type(pass_obj).__name__


def _count_line_changes(before_text: str, after_text: str) -> tuple[int, int]:
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


def _compact_pass_name(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def _inspect_module_source(mod) -> str:
    """Return source from a runtime module without hiding hook failures."""
    for attribute in ("inspect_source", "get_source"):
        inspect = getattr(mod, attribute, None)
        if callable(inspect):
            return inspect() or ""
    return ""


def _make_patched_source_module(original_module, patched_source: str):
    """Build a real TVM ``CSourceModule`` carrying user-edited source."""
    import tvm.runtime._ffi_api as _ffi_api

    try:
        module_format = original_module.format
    except Exception:
        module_format = "c"
    return _ffi_api.CSourceModuleCreate(patched_source, module_format, [], None)


def _codegen_path_lock(path: str) -> threading.Lock:
    normalized_path = os.path.realpath(path)
    with _codegen_path_locks_guard:
        return _codegen_path_locks.setdefault(normalized_path, threading.Lock())


class LowerTraceSession(PassInstrumentationTool):
    """All mutable lower-trace state for one logical compilation."""

    def __init__(
        self,
        *,
        mode: str | None,
        trace_dir: str,
        codegen_output: str | None | object = _UNSET,
    ) -> None:
        self.mode = mode
        self.trace_dir = trace_dir
        self.codegen_output = codegen_output
        self.records: list[LowerRecord] = []
        self.section_cache: dict[tuple[str, int], str] = {}
        self.pass_index = 0
        self.pipeline_count = 0
        self.script_dir: str | None = None
        self.run_dir: str | None = None
        self._finished = False

    @property
    def enabled(self) -> bool:
        return self.mode is not None

    @property
    def should_print_terminal(self) -> bool:
        return self.mode in ("terminal", "both")

    @property
    def should_generate_html(self) -> bool:
        return self.mode in ("html", "both")

    def ensure_script_dir(self) -> str:
        if self.script_dir is None:
            script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0] or "kernel"
            self.script_dir = os.path.join(self.trace_dir, script_name)
            os.makedirs(self.script_dir, exist_ok=True)
        return self.script_dir

    def ensure_run_dir(self) -> str:
        if self.run_dir is None:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            run_name = f"run_{timestamp}_{os.getpid()}_{uuid4().hex[:8]}"
            self.run_dir = os.path.join(self.ensure_script_dir(), ".run_records", run_name)
            os.makedirs(self.run_dir, exist_ok=True)
        return self.run_dir

    def get_codegen_output_path(self) -> str | None:
        if self.codegen_output is not _UNSET:
            return self.codegen_output  # type: ignore[return-value]
        if self.should_generate_html:
            return os.path.join(self.ensure_script_dir(), "codegen.cpp")
        return None

    def allocate_index(self) -> int:
        index = self.pass_index
        self.pass_index += 1
        return index

    def append_record(self, record: LowerRecord) -> None:
        self.records.append(record)
        # Nested pass callbacks finish inside-out, while indices are allocated
        # at entry.  Keep persisted/report order aligned with execution start.
        self.records.sort(key=lambda item: item.index)
        self.save_raw_files(record)

    def save_raw_files(self, record: LowerRecord) -> None:
        """Persist snapshots without allowing tracing I/O to fail compilation."""
        try:
            phase_dir = os.path.join(self.ensure_run_dir(), _safe_filename_component(record.phase))
            os.makedirs(phase_dir, exist_ok=True)
            prefix = f"{record.index:02d}_{_safe_filename_component(record.name)}"
            after_ext = ".cpp" if record.status == STATUS_CODEGEN else ".tir"
            before_path = os.path.join(phase_dir, f"{prefix}_before.tir")
            after_path = os.path.join(phase_dir, f"{prefix}_after{after_ext}")
            with open(before_path, "w", encoding="utf-8") as before_file:
                before_file.write(record.before_text)
            with open(after_path, "w", encoding="utf-8") as after_file:
                after_file.write(record.after_text)
        except Exception as exc:
            print(f"  {_ANSI_RED}[lower_trace] WARNING: could not save raw trace files: {exc}{_ANSI_RESET}")

    def update_html_link(self, run_html_path: str) -> None:
        script_dir = self.ensure_script_dir()
        link_path = os.path.join(script_dir, "report.html")
        temporary_path = f"{link_path}.tmp.{os.getpid()}.{uuid4().hex}"
        try:
            os.symlink(os.path.relpath(run_html_path, script_dir), temporary_path)
            os.replace(temporary_path, link_path)
        except OSError:
            with contextlib.suppress(OSError):
                os.remove(temporary_path)
            shutil.copyfile(run_html_path, temporary_path)
            os.replace(temporary_path, link_path)
        finally:
            with contextlib.suppress(OSError):
                os.remove(temporary_path)

    def flush_html(self) -> None:
        if not self.should_generate_html or not self.records or self.run_dir is None:
            return
        from .html import generate_html

        html_path = os.path.join(self.run_dir, "report.html")
        generate_html(self.records, html_path, section_cache=self.section_cache)
        self.update_html_link(html_path)

    def create_pass_instrument(self) -> StackedPassInstrument | None:
        if not self.enabled:
            return None
        return StackedPassInstrument(
            _LowerTraceObserver(self),
            capture_nested=True,
            sequence_allocator=self.allocate_index,
            phase_provider=lambda: current_pass_phase() or _UNSCOPED_PHASE,
        )

    @contextlib.contextmanager
    def pipeline_scope(self, base_phase: str) -> Generator[None, None, None]:
        if not self.enabled:
            yield
            return

        self.pipeline_count += 1
        run_number = self.pipeline_count
        prefix = f"run{run_number}_" if run_number > 1 else ""
        phase_name = f"{prefix}{base_phase}"
        succeeded = False
        try:
            with pass_phase(phase_name):
                yield
            succeeded = True
        except Exception as error:
            self.abort_active_instruments(error)
            print(f"  [lower_trace] EXCEPTION in {phase_name}: {error}")
            raise
        finally:
            with contextlib.suppress(Exception):
                self.flush_html()

        if succeeded:
            print(f"  [lower_trace] pipeline {run_number} ({phase_name}) complete: {len(self.records)} total records")

    def abort_active_instruments(self, error: BaseException) -> None:
        for instrument in reversed(active_stacked_pass_instruments()):
            observer = instrument.observer
            if isinstance(observer, _LowerTraceObserver) and observer.session is self and instrument.pending_events:
                instrument.abort(error)

    def _resolve_codegen_edit(self, codegen_text: str, output_path: str, index: int) -> str | None:
        """Apply the baseline/working/current three-way source workflow."""
        with _codegen_path_lock(output_path):
            return self._resolve_codegen_edit_locked(codegen_text, output_path, index)

    def _resolve_codegen_edit_locked(self, codegen_text: str, output_path: str, index: int) -> str | None:
        """Resolve one codegen edit while holding the output-path lock."""
        original_path = output_path + ".original"
        latest_path = output_path + ".latest"
        try:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(latest_path, "w", encoding="utf-8") as latest_file:
                latest_file.write(codegen_text)

            if not os.path.isfile(output_path) or not os.path.isfile(original_path):
                if os.path.isfile(output_path):
                    shutil.copyfile(output_path, output_path + ".bak")
                    print(
                        f"  [lower_trace] codegen/{index:02d}_codegen: INIT-BACKUP — "
                        f"{_ANSI_BOLD}{_ANSI_YELLOW}{output_path}{_ANSI_RESET} existed without baseline, "
                        f"backed up to {_ANSI_BOLD}{_ANSI_YELLOW}{output_path}.bak{_ANSI_RESET}"
                    )
                with open(original_path, "w", encoding="utf-8") as original_file:
                    original_file.write(codegen_text)
                shutil.copyfile(original_path, output_path)
                print(f"  [lower_trace] codegen source initialized at: {_ANSI_GREEN}{output_path}{_ANSI_RESET}")
                return None

            with open(original_path, encoding="utf-8") as original_file:
                baseline_text = original_file.read()
            with open(output_path, encoding="utf-8") as working_file:
                working_text = working_file.read()

            user_edited = working_text.rstrip() != baseline_text.rstrip()
            codegen_changed = codegen_text.rstrip() != baseline_text.rstrip()
            if not user_edited and not codegen_changed:
                return None
            if not user_edited and codegen_changed:
                with open(original_path, "w", encoding="utf-8") as original_file:
                    original_file.write(codegen_text)
                with open(output_path, "w", encoding="utf-8") as working_file:
                    working_file.write(codegen_text)
                print(f"  {_ANSI_CYAN}[lower_trace] codegen/{index:02d}_codegen: REGENERATED (codegen changed, no user edits){_ANSI_RESET}")
                return None
            if user_edited and not codegen_changed:
                print(f"  {_ANSI_BOLD}{_ANSI_GREEN}[lower_trace] codegen/{index:02d}_codegen: PATCHED (user edits){_ANSI_RESET}")
                return working_text
            if working_text.rstrip() == codegen_text.rstrip():
                with open(original_path, "w", encoding="utf-8") as original_file:
                    original_file.write(codegen_text)
                print(
                    f"  {_ANSI_BOLD}{_ANSI_GREEN}[lower_trace] codegen/{index:02d}_codegen: "
                    f"SYNCED (user edits and codegen changed to the same source){_ANSI_RESET}"
                )
                return working_text

            shutil.copyfile(output_path, output_path + ".bak")
            shutil.copyfile(original_path, original_path + ".bak")
            with open(original_path, "w", encoding="utf-8") as original_file:
                original_file.write(codegen_text)
            with open(output_path, "w", encoding="utf-8") as working_file:
                working_file.write(codegen_text)
            print(
                f"  {_ANSI_BOLD}{_ANSI_YELLOW}[lower_trace] codegen/{index:02d}_codegen: "
                f"CONFLICT (codegen restored; user edits saved in .bak){_ANSI_RESET}"
            )
        except Exception as exc:
            print(f"  {_ANSI_RED}[lower_trace] WARNING: codegen file I/O failed: {exc}{_ANSI_RESET}")
        return None

    def run_codegen(self, event: CodegenEvent, next_call: Callable[[], Any]) -> Any:
        if not self.enabled:
            return next_call()

        if self.should_generate_html:
            self.ensure_run_dir()
        before_text = str(event.mod)
        output_path = self.get_codegen_output_path()

        try:
            with pass_phase("codegen"):
                result = next_call()
        except Exception as error:
            self.abort_active_instruments(error)
            index = self.allocate_index()
            self.append_record(
                LowerRecord(
                    phase="codegen",
                    name="codegen",
                    index=index,
                    before_text=before_text,
                    after_text="",
                    changed=False,
                    status=STATUS_FAILED,
                    error_msg=str(error),
                )
            )
            print(f"  [lower_trace] codegen/{index:02d}_codegen: FAILED ({error})")
            raise

        # Allocate after the backend call: passes invoked by codegen must sort
        # before the enclosing codegen record.
        index = self.allocate_index()
        patched_text: str | None = None
        codegen_text = ""
        after_text = ""
        try:
            codegen_text = _inspect_module_source(result)
            if output_path:
                patched_text = self._resolve_codegen_edit(codegen_text, output_path, index)
            after_text = patched_text if patched_text is not None else codegen_text
            add_count, del_count = _count_line_changes(before_text, after_text)
            self.append_record(
                LowerRecord(
                    phase="codegen",
                    name="codegen",
                    index=index,
                    before_text=before_text,
                    after_text=after_text,
                    changed=True,
                    add_lines=add_count,
                    del_lines=del_count,
                    status=STATUS_CODEGEN,
                )
            )
            path_suffix = f"  →  {_ANSI_BLUE}{output_path}{_ANSI_RESET}" if output_path else ""
            print(f"  [lower_trace] codegen/{index:02d}_codegen: CODEGEN (+{add_count}/-{del_count}){path_suffix}")
            if self.should_generate_html:
                with contextlib.suppress(Exception):
                    self.flush_html()
        except Exception as exc:
            print(f"  {_ANSI_RED}[lower_trace] WARNING: post-codegen tracing failed: {exc}{_ANSI_RESET}")

        if self.should_print_terminal:
            from .diff import print_diff

            print_diff(before_text, after_text, "codegen (TIR before)", "codegen (source after)")

        if patched_text is None:
            return result
        if event.name in _SOURCE_ONLY_CODEGEN_FFIS:
            return _make_patched_source_module(result, patched_text)

        if patched_text.rstrip() != codegen_text.rstrip():
            target_kind = getattr(getattr(event.target, "kind", None), "name", "")
            backend_hint = ""
            if target_kind == "cuda":
                backend_hint = " Use execution_backend='nvrtc' for edit-and-recompile support."
            elif target_kind == "hip":
                backend_hint = " Use execution_backend='cython' for edit-and-recompile support."
            print(
                f"  {_ANSI_YELLOW}[lower_trace] codegen/{index:02d}_codegen: NOTE — user edits in "
                f"{output_path} are recorded for diff viewing, but were NOT recompiled because this "
                f"backend already built a binary from TIR.{backend_hint}{_ANSI_RESET}"
            )
        return result

    def finish(self, error: BaseException | None) -> None:
        if self._finished:
            return
        self._finished = True
        with contextlib.suppress(Exception):
            self.flush_html()
        if self.should_generate_html and self.records and self.script_dir is not None:
            print(f"  [lower_trace] Final HTML report: {_ANSI_BLUE}{os.path.join(self.script_dir, 'report.html')}{_ANSI_RESET}")

    def reset(self) -> None:
        """Clear this session's records; primarily useful to unit tests."""
        self.records.clear()
        self.section_cache.clear()
        self.pass_index = 0
        self.pipeline_count = 0


class _LowerTraceObserver(PassEventObserver):
    """Translate normalized pass events into one session's trace records."""

    def __init__(self, session: LowerTraceSession) -> None:
        self.session = session

    def pass_started(self, mod, event: PassEvent) -> str:
        if self.session.should_generate_html:
            self.session.ensure_run_dir()
        return str(mod)

    def pass_finished(self, mod, event: PassEvent, before_text: str) -> None:
        after_text = str(mod)
        changed = before_text != after_text
        add_count, del_count = _count_line_changes(before_text, after_text) if changed else (0, 0)
        phase = event.phase or _UNSCOPED_PHASE
        pass_name = _compact_pass_name(event.name)
        self.session.append_record(
            LowerRecord(
                phase=phase,
                name=pass_name,
                index=event.sequence,
                before_text=before_text,
                after_text=after_text,
                changed=changed,
                add_lines=add_count,
                del_lines=del_count,
                depth=event.depth,
                parent_index=event.parent_sequence,
            )
        )
        tag = "CHANGED" if changed else "NO-OP"
        color = _ANSI_GREEN if changed else _ANSI_DIM
        print(f"  [lower_trace] {phase}/{event.sequence:02d}_{pass_name}: {color}{tag}{_ANSI_RESET}")

        if self.session.should_generate_html:
            with contextlib.suppress(Exception):
                self.session.flush_html()
        if self.session.should_print_terminal and changed:
            from .diff import print_diff

            label = f"{phase}/{pass_name}"
            print_diff(before_text, after_text, f"{label} (before)", f"{label} (after)")

    def passes_incomplete(self, passes: list[IncompletePass], error: BaseException | None) -> None:
        error_msg = str(error) if error is not None else "pass did not complete"
        for item in passes:
            if item.event is None:
                continue
            event = item.event
            phase = event.phase or _UNSCOPED_PHASE
            pass_name = _compact_pass_name(event.name)
            self.session.append_record(
                LowerRecord(
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
            )
            print(f"  {_ANSI_RED}[lower_trace] {phase}/{event.sequence:02d}_{pass_name}: FAILED ({error_msg}){_ANSI_RESET}")
        if self.session.should_generate_html:
            with contextlib.suppress(Exception):
                self.session.flush_html()

    def callback_mismatch(self, actual: str, expected: str | None) -> None:
        print(f"  {_ANSI_RED}[lower_trace] WARNING: unmatched after-pass callback for {actual!r}; expected {expected!r}{_ANSI_RESET}")


def _remember_session(session: LowerTraceSession | None) -> None:
    _last_session.set(session)


def get_last_session() -> LowerTraceSession | None:
    """Return the most recent session in the current execution context."""
    return _last_session.get()


def current_lower_trace_session() -> LowerTraceSession | None:
    """Return lower trace state for the current compile context, if present."""
    compile_session = current_compile_pass_instrumentation()
    if compile_session is None:
        return None
    return compile_session.find_tool(LowerTraceSession)


def _snapshot_config() -> _LowerTraceConfig:
    """Atomically resolve configuration for future compile sessions."""
    with _config_lock:
        mode = _mode_override
        trace_dir = _trace_dir_override
        codegen_output = _codegen_output_path_override
        resolved_mode = env.get_lower_trace_mode() if mode is _UNSET else mode
        resolved_trace_dir = env.get_lower_trace_dir() if trace_dir is _UNSET or not trace_dir else trace_dir
        return _LowerTraceConfig(
            mode=resolved_mode,  # type: ignore[arg-type]
            trace_dir=str(resolved_trace_dir),
            codegen_output=codegen_output,
        )


def _create_lower_trace_session(config: _LowerTraceConfig | None = None) -> LowerTraceSession:
    """Create an isolated tool instance from one immutable config snapshot."""
    config = config or _snapshot_config()
    session = LowerTraceSession(
        mode=config.mode,
        trace_dir=config.trace_dir,
        codegen_output=config.codegen_output,
    )
    _remember_session(session)
    return session


def create_pass_instrument() -> StackedPassInstrument:
    """Create an instrument for the active session (standalone if necessary)."""
    session = current_lower_trace_session() or _create_lower_trace_session()
    instrument = session.create_pass_instrument()
    if instrument is None:
        # Preserve the historical helper contract even when disabled; the
        # predicate suppresses all captures.
        return StackedPassInstrument(
            _LowerTraceObserver(session),
            capture_predicate=lambda _name, _depth: False,
            sequence_allocator=session.allocate_index,
        )
    return instrument


def enable(*, mode=_UNSET, trace_dir=_UNSET, codegen_output=_UNSET) -> None:
    """Enable lower tracing for future compile sessions.

    An already-running compilation keeps the immutable configuration snapshot
    with which it started.  Calling ``enable`` or ``disable`` therefore never
    mutates another kernel's active instrumentation.
    """
    global _mode_override, _trace_dir_override, _codegen_output_path_override

    parsed_mode = _UNSET
    if mode is not _UNSET:
        parsed_mode = _parse_lower_trace_mode(mode if mode is None else str(mode))
        if parsed_mode is None:
            disable()
            with _config_lock:
                _mode_override = None
            return

    with _config_lock:
        if parsed_mode is not _UNSET:
            _mode_override = parsed_mode
        if trace_dir is not _UNSET:
            _trace_dir_override = trace_dir if trace_dir is None else str(trace_dir)
        if codegen_output is not _UNSET:
            _codegen_output_path_override = codegen_output if codegen_output is None else str(codegen_output)
        config = _snapshot_config()
        register_pass_instrumentation_tool("lower_trace", lambda: _create_lower_trace_session(config))
    print("[lower_trace] IR pass tracing enabled with per-compilation PassInstrument sessions.")


def disable() -> None:
    """Stop adding lower trace to future compilations."""
    global _mode_override, _trace_dir_override, _codegen_output_path_override

    with _config_lock:
        unregister_pass_instrumentation_tool("lower_trace")
        _mode_override = _UNSET
        _trace_dir_override = _UNSET
        _codegen_output_path_override = _UNSET
    _remember_session(None)


def reset() -> None:
    """Reset only the compile session active in the current execution context."""
    session = current_lower_trace_session()
    if session is not None:
        session.reset()


__all__ = [
    "LowerRecord",
    "LowerTraceSession",
    "STATUS_CODEGEN",
    "STATUS_COMPLETED",
    "STATUS_FAILED",
    "STATUS_SKIPPED",
    "create_pass_instrument",
    "current_lower_trace_session",
    "disable",
    "enable",
    "get_last_session",
    "reset",
]
