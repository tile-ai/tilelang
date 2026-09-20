"""Normalized TVM pass events shared by TileLang instrumentation tools.

This module pairs nested ``run_before_pass``/``run_after_pass`` callbacks,
assigns stable execution order and parent/depth metadata, and drains incomplete
frames when a pass fails. Consumers remain responsible for choosing what to
snapshot and how to render or persist the resulting data.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from tvm.ir.instrument import pass_instrument


_current_pass_phase: ContextVar[str | None] = ContextVar("tilelang_pass_phase", default=None)
_active_stacked_instruments: ContextVar[tuple[StackedPassInstrument, ...]] = ContextVar(
    "tilelang_active_stacked_pass_instruments",
    default=(),
)


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
        Optional finer-grained predicate receiving ``(name, depth)``. It is
        applied after ``capture_nested`` filtering.
    sequence_allocator:
        Optional allocator for consumers that need sequence numbers shared
        across multiple PassContexts. Without one, numbering restarts at zero
        whenever this instrument enters a context.
    phase_provider:
        Supplies the phase attached to each event. Defaults to
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
        than the reverse order used by nested Python context managers. Token
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

        The returned tuple is ordered outermost first. The deepest entry is
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
    "CapturePredicate",
    "IncompletePass",
    "PassEvent",
    "PassEventObserver",
    "SequenceAllocator",
    "StackedPassInstrument",
    "active_stacked_pass_instruments",
    "current_pass_phase",
    "pass_phase",
]
