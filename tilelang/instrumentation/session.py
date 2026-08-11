"""Compile-scoped orchestration for TileLang instrumentation tools."""

from __future__ import annotations

import threading
from collections.abc import Callable, Generator, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, TypeVar

from .events import pass_phase


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
    themselves. Nested helpers normally run inside a caller-owned
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


def run_codegen_with_instrumentation(
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


__all__ = [
    "CodegenEvent",
    "CompilePassInstrumentationSession",
    "PassInstrumentationTool",
    "compile_pass_instrumentation",
    "create_pass_instruments",
    "current_compile_pass_instrumentation",
    "current_pass_instrument_context",
    "instrument_current_pass_context",
    "pass_pipeline",
    "register_pass_instrumentation_tool",
    "run_codegen_with_instrumentation",
    "unregister_pass_instrumentation_tool",
]
