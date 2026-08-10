"""Shared pass-instrument event plumbing for TileLang developer tools.

This module owns the mechanics that are common to pass-oriented tools:
pairing nested ``run_before_pass``/``run_after_pass`` callbacks, assigning a
stable execution order, tracking parent/depth metadata, and draining incomplete
frames when a pass fails.  Consumers remain responsible for choosing what to
snapshot and how to render or persist the resulting data.

The implementation deliberately has no dependency on lower-trace, pass
visualizer, timing, HTML, or filesystem code.  Keeping this layer neutral lets
those tools share TVM's :class:`PassInstrument` API without coupling their
different output models.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Generator, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, TypeVar

from tvm.ir.instrument import pass_instrument


_current_pass_phase: ContextVar[str | None] = ContextVar("tilelang_pass_phase", default=None)
_active_stacked_instruments: ContextVar[tuple[StackedPassInstrument, ...]] = ContextVar(
    "tilelang_active_stacked_pass_instruments",
    default=(),
)
_current_compile_instrumentation: ContextVar[CompilePassInstrumentationSession | None] = ContextVar(
    "tilelang_compile_pass_instrumentation",
    default=None,
)
_current_pass_instrument_context: ContextVar[str | None] = ContextVar(
    "tilelang_pass_instrument_context",
    default=None,
)
# This lock protects only registration and the short factory snapshot at
# session creation. Pass callbacks, pipeline scopes, and codegen never hold it.
_tool_registry_lock = threading.RLock()
_tool_factories: dict[str, Callable[[], PassInstrumentationTool]] = {}


@contextmanager
def pass_phase(name: str | None) -> Generator[None, None, None]:
    """Attach a phase label to pass events emitted inside this scope."""
    token = _current_pass_phase.set(None if name is None else str(name))
    try:
        yield
    finally:
        _current_pass_phase.reset(token)


def current_pass_phase() -> str | None:
    """Return the phase label active in the current execution context."""
    return _current_pass_phase.get()


def active_stacked_pass_instruments() -> tuple[StackedPassInstrument, ...]:
    """Return shared stack instruments active in the current PassContext."""
    return _active_stacked_instruments.get()


@dataclass(frozen=True)
class CodegenEvent:
    """One explicit backend code-generation call within a compile session."""

    name: str
    mod: Any
    target: Any


class PassInstrumentationTool:
    """Per-compile extension point for pass and codegen developer tools."""

    pass_instrument_priority = 0
    """Ordering key for PassContext callbacks; lower values run first."""

    def create_pass_instrument(self) -> object | None:
        """Create fresh callback state for one TVM PassContext."""
        return None

    def pipeline_scope(self, base_phase: str) -> AbstractContextManager[None]:
        """Return the scope entered around one backend pass pipeline."""
        return nullcontext()

    def run_codegen(self, event: CodegenEvent, next_call: Callable[[], Any]) -> Any:
        """Observe or wrap codegen, delegating to ``next_call`` exactly once."""
        return next_call()

    def finish(self, error: BaseException | None) -> None:
        """Finalize this tool after its owning compile session exits."""


_ToolT = TypeVar("_ToolT", bound=PassInstrumentationTool)


def _bind_codegen_tool(
    tool: PassInstrumentationTool,
    event: CodegenEvent,
    next_call: Callable[[], Any],
) -> Callable[[], Any]:
    """Bind one tool around the next codegen callback in the chain."""

    def run() -> Any:
        return tool.run_codegen(event, next_call)

    return run


class CompilePassInstrumentationSession:
    """Concrete pass-tool state owned by one logical compilation."""

    def __init__(self, tools: Sequence[PassInstrumentationTool], *, name: str | None = None) -> None:
        self.name = name
        self.tools = tuple(tools)

    def create_pass_instruments(self, *, context: str | None = None) -> list[object]:
        """Create independent, ordered instruments for one TVM PassContext."""
        token = _current_pass_instrument_context.set(None if context is None else str(context))
        try:
            instruments = []
            for position, tool in enumerate(self.tools):
                instrument = tool.create_pass_instrument()
                if instrument is not None:
                    instruments.append((tool.pass_instrument_priority, position, instrument))
        finally:
            _current_pass_instrument_context.reset(token)
        instruments.sort(key=lambda item: (item[0], item[1]))
        return [instrument for _, _, instrument in instruments]

    @contextmanager
    def pipeline_scope(self, base_phase: str) -> Generator[None, None, None]:
        """Enter every tool scope for one backend pass pipeline."""
        with ExitStack() as stack:
            for tool in self.tools:
                stack.enter_context(tool.pipeline_scope(base_phase))
            yield

    def run_codegen(self, event: CodegenEvent, call: Callable[[], Any]) -> Any:
        """Compose tool codegen middleware in registration order."""
        wrapped = call
        for tool in reversed(self.tools):
            wrapped = _bind_codegen_tool(tool, event, wrapped)
        return wrapped()

    def finish(self, error: BaseException | None) -> None:
        """Finalize tools in reverse registration order."""
        finish_error = None
        for tool in reversed(self.tools):
            try:
                tool.finish(error)
            except BaseException as exc:
                if finish_error is None:
                    finish_error = exc
        # Never hide the compilation failure that tools were asked to observe.
        if error is None and finish_error is not None:
            raise finish_error

    def find_tool(self, tool_type: type[_ToolT]) -> _ToolT | None:
        """Return the first tool matching ``tool_type``, if present."""
        return next((tool for tool in self.tools if isinstance(tool, tool_type)), None)


def register_pass_instrumentation_tool(
    name: str,
    factory: Callable[[], PassInstrumentationTool],
) -> None:
    """Register or replace a factory used by future compile sessions."""
    with _tool_registry_lock:
        _tool_factories[str(name)] = factory


def unregister_pass_instrumentation_tool(name: str) -> None:
    """Stop adding a tool to future compile sessions."""
    with _tool_registry_lock:
        _tool_factories.pop(str(name), None)


def current_compile_pass_instrumentation() -> CompilePassInstrumentationSession | None:
    """Return the compile session active in this execution context."""
    return _current_compile_instrumentation.get()


def current_pass_instrument_context() -> str | None:
    """Return metadata for the PassContext whose instruments are being created."""
    return _current_pass_instrument_context.get()


def _create_default_tools() -> list[PassInstrumentationTool]:
    """Instantiate one immutable snapshot of the global tool configuration."""
    with _tool_registry_lock:
        factories = tuple(_tool_factories.values())
    return [factory() for factory in factories]


@contextmanager
def compile_pass_instrumentation(
    name: str | None = None,
    *,
    tools: Sequence[PassInstrumentationTool] = (),
    include_default_tools: bool = True,
    reuse_existing: bool = True,
) -> Generator[CompilePassInstrumentationSession, None, None]:
    """Own pass-tool state for one logical compilation.

    Nested compiler helpers reuse the active session by default, so every
    PassContext involved in one compilation writes to the same tool state.
    Independent tools such as Pass Visualizer can request a fresh session by
    setting ``reuse_existing=False`` and ``include_default_tools=False``.
    """
    current = current_compile_pass_instrumentation()
    if current is not None and reuse_existing:
        if tools or not include_default_tools:
            raise ValueError("cannot add tools or exclude defaults when reusing an active compile session")
        yield current
        return

    session_tools = _create_default_tools() if include_default_tools else []
    session_tools.extend(tools)
    session = CompilePassInstrumentationSession(session_tools, name=name)
    token = _current_compile_instrumentation.set(session)
    error = None
    try:
        yield session
    except BaseException as exc:
        error = exc
        raise
    finally:
        try:
            session.finish(error)
        finally:
            _current_compile_instrumentation.reset(token)


def create_pass_instruments(*, context: str | None = None) -> list[object]:
    """Create instruments for the active compile session, if any.

    ``context`` is descriptive metadata for this particular TVM PassContext.
    Tools can snapshot it during callback creation without changing their
    ``create_pass_instrument`` interface.
    """
    session = current_compile_pass_instrumentation()
    return session.create_pass_instruments(context=context) if session is not None else []


@contextmanager
def instrument_current_pass_context() -> Generator[None, None, None]:
    """Temporarily add active-session instruments to TVM's current context.

    Compiler front doors use this only when they created the compile session
    themselves.  Nested helpers normally run inside a caller-owned
    ``PassContext`` whose instruments were created from the same session, so
    they must not attach a duplicate set.
    """
    session = current_compile_pass_instrumentation()
    if session is None:
        yield
        return

    instruments = session.create_pass_instruments()
    if not instruments:
        yield
        return

    from tvm.ir.transform import PassContext

    pass_context = PassContext.current()
    previous = list(pass_context.instruments)
    try:
        pass_context.override_instruments([*previous, *instruments])
    except BaseException:
        # ``override_instruments`` exits the previous callbacks before entering
        # the replacements, so restore the caller's context on partial entry.
        pass_context.override_instruments(previous)
        raise
    try:
        yield
    finally:
        pass_context.override_instruments(previous)


_CodegenResult = TypeVar("_CodegenResult")


def instrument_codegen(
    name: str,
    mod: Any,
    target: Any,
    call: Callable[[], _CodegenResult],
) -> _CodegenResult:
    """Run an explicit backend codegen call through the active session."""
    session = current_compile_pass_instrumentation()
    if session is None:
        return call()
    return session.run_codegen(CodegenEvent(name=name, mod=mod, target=target), call)


@contextmanager
def pass_pipeline(name: str) -> Generator[None, None, None]:
    """Run a backend pipeline under its phase and compile-session tools."""
    base_phase = f"pipeline_{name}"
    session = current_compile_pass_instrumentation()
    with pass_phase(base_phase):
        if session is None:
            yield
        else:
            with session.pipeline_scope(base_phase):
                yield


@dataclass(frozen=True)
class PassEvent:
    """Identity and execution metadata for one observed compiler pass."""

    name: str
    sequence: int
    depth: int
    parent_sequence: int | None
    phase: str | None


@dataclass(frozen=True)
class IncompletePass:
    """A pass frame that received no matching after-pass callback."""

    name: str
    depth: int
    event: PassEvent | None
    state: Any


class PassEventObserver:
    """No-op observer interface consumed by :class:`StackedPassInstrument`."""

    def enter_pass_context(self) -> None:
        """Handle entry into a TVM PassContext."""

    def exit_pass_context(self) -> None:
        """Handle normal exit from a TVM PassContext."""

    def pass_started(self, mod, event: PassEvent):
        """Capture consumer state before a pass and return it."""
        return None

    def pass_finished(self, mod, event: PassEvent, state: Any) -> None:
        """Consume the module and state after a pass completes."""

    def passes_incomplete(self, passes: Sequence[IncompletePass], error: BaseException | None) -> None:
        """Handle passes that never received an after-pass callback."""

    def callback_mismatch(self, actual: str, expected: str | None) -> None:
        """Handle an unmatched or out-of-order after-pass callback."""


@dataclass
class _ActivePass:
    name: str
    depth: int
    event: PassEvent | None
    state: Any = None


CapturePredicate = Callable[[str, int], bool]
SequenceAllocator = Callable[[], int]


@pass_instrument
class StackedPassInstrument:
    """Pair nested TVM pass callbacks and dispatch normalized events.

    Parameters
    ----------
    observer:
        Consumer that snapshots and records the pass-specific data.
    capture_nested:
        When false, nested callbacks are still tracked for correct pairing but
        only depth-zero passes are emitted to the observer.
    capture_predicate:
        Optional finer-grained predicate receiving ``(name, depth)``.  It is
        applied after ``capture_nested`` filtering.
    sequence_allocator:
        Optional allocator for consumers that need sequence numbers shared
        across multiple PassContexts.  Without one, numbering restarts at zero
        whenever this instrument enters a context.
    phase_provider:
        Supplies the phase attached to each event.  Defaults to
        :func:`current_pass_phase`.
    """

    def __init__(
        self,
        observer: PassEventObserver,
        *,
        capture_nested: bool = True,
        capture_predicate: CapturePredicate | None = None,
        sequence_allocator: SequenceAllocator | None = None,
        phase_provider: Callable[[], str | None] = current_pass_phase,
    ) -> None:
        self.observer = observer
        self.capture_nested = capture_nested
        self.capture_predicate = capture_predicate
        self.sequence_allocator = sequence_allocator
        self.phase_provider = phase_provider
        self._stack: list[_ActivePass] = []
        self._next_sequence = 0

    def _deactivate(self) -> None:
        """Remove this instance without assuming instrument exit order.

        TVM currently calls ``exit_pass_ctx`` in registration order, rather
        than the reverse order used by nested Python context managers.  Token
        reset would therefore resurrect an instrument that already exited.
        Removing by identity works for both FIFO and LIFO callback orders.
        """
        active = list(_active_stacked_instruments.get())
        for index in range(len(active) - 1, -1, -1):
            if active[index] is self:
                del active[index]
                _active_stacked_instruments.set(tuple(active))
                return

    @property
    def pending_events(self) -> tuple[PassEvent, ...]:
        """Return currently open observed events, ordered outermost first."""
        return tuple(frame.event for frame in self._stack if frame.event is not None)

    def _allocate_sequence(self) -> int:
        if self.sequence_allocator is not None:
            return self.sequence_allocator()
        sequence = self._next_sequence
        self._next_sequence += 1
        return sequence

    def _should_capture(self, name: str, depth: int) -> bool:
        if depth > 0 and not self.capture_nested:
            return False
        if self.capture_predicate is not None:
            return bool(self.capture_predicate(name, depth))
        return True

    def enter_pass_ctx(self) -> None:
        self._stack.clear()
        self._next_sequence = 0
        _active_stacked_instruments.set((*_active_stacked_instruments.get(), self))
        try:
            self.observer.enter_pass_context()
        except Exception:
            self._deactivate()
            raise

    def exit_pass_ctx(self) -> None:
        try:
            if self._stack:
                self.abort()
            self.observer.exit_pass_context()
        finally:
            self._deactivate()

    def run_before_pass(self, mod, info) -> None:
        name = str(info.name)
        depth = len(self._stack)
        event = None
        state = None

        if self._should_capture(name, depth):
            parent_sequence = next(
                (frame.event.sequence for frame in reversed(self._stack) if frame.event is not None),
                None,
            )
            event = PassEvent(
                name=name,
                sequence=self._allocate_sequence(),
                depth=depth,
                parent_sequence=parent_sequence,
                phase=self.phase_provider(),
            )
            state = self.observer.pass_started(mod, event)

        self._stack.append(_ActivePass(name=name, depth=depth, event=event, state=state))

    def run_after_pass(self, mod, info) -> None:
        name = str(info.name)
        if not self._stack or self._stack[-1].name != name:
            expected = self._stack[-1].name if self._stack else None
            self.observer.callback_mismatch(name, expected)
            self.abort(RuntimeError(f"unmatched after-pass callback for {name!r}; expected {expected!r}"))
            return

        frame = self._stack.pop()
        if frame.event is not None:
            self.observer.pass_finished(mod, frame.event, frame.state)

    def abort(self, error: BaseException | None = None) -> tuple[IncompletePass, ...]:
        """Drain open frames and notify the observer of incomplete passes.

        The returned tuple is ordered outermost first.  The deepest entry is
        therefore the pass nearest to the original failure, while earlier
        entries are incomplete ancestors.
        """
        incomplete = tuple(
            IncompletePass(
                name=frame.name,
                depth=frame.depth,
                event=frame.event,
                state=frame.state,
            )
            for frame in self._stack
        )
        self._stack.clear()
        if incomplete:
            self.observer.passes_incomplete(incomplete, error)
        return incomplete


__all__ = [
    "CodegenEvent",
    "CompilePassInstrumentationSession",
    "IncompletePass",
    "PassEvent",
    "PassEventObserver",
    "PassInstrumentationTool",
    "StackedPassInstrument",
    "active_stacked_pass_instruments",
    "compile_pass_instrumentation",
    "create_pass_instruments",
    "instrument_current_pass_context",
    "current_compile_pass_instrumentation",
    "current_pass_instrument_context",
    "current_pass_phase",
    "instrument_codegen",
    "pass_phase",
    "pass_pipeline",
    "register_pass_instrumentation_tool",
    "unregister_pass_instrumentation_tool",
]
