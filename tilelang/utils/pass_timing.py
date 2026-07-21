"""Per-pass timing instrumentation for tilelang compilation pipelines.

Records inclusive and self wall-clock duration for each pass using
``time.monotonic()``. Data persists after PassContext exit for
post-compilation reporting.

Enabled via TILELANG_PASS_PROFILE=1 env var or
PassConfigKey.TL_PASS_PROFILE pass config.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

from tvm.ir import _ffi_instrument_api

logger = logging.getLogger("tilelang.pass_timing")


@dataclass
class PassTimingRecord:
    """Single completed pass execution record."""

    name: str
    duration_s: float
    depth: int  # nesting depth (0 = top-level)
    self_duration_s: float = 0.0
    sequence: int = 0


@dataclass
class _ActivePass:
    name: str
    start_time: float
    depth: int
    sequence: int
    child_duration_s: float = 0.0


class _PassTimingState:
    """Mutable callback state owned by the underlying TVM instrument."""

    def __init__(self):
        self.records: list[PassTimingRecord] = []
        self.stack: list[_ActivePass] = []
        self.next_sequence = 0

    def enter_pass_ctx(self):
        self.records.clear()
        self.stack.clear()
        self.next_sequence = 0

    def exit_pass_ctx(self):
        if self.stack:
            logger.warning(
                "Discarding %d incomplete pass timing frame(s): %s",
                len(self.stack),
                ", ".join(frame.name for frame in self.stack),
            )
            self.stack.clear()

    def run_before_pass(self, _mod, info):
        self.stack.append(
            _ActivePass(
                name=info.name,
                start_time=time.monotonic(),
                depth=len(self.stack),
                sequence=self.next_sequence,
            )
        )
        self.next_sequence += 1

    def run_after_pass(self, _mod, info):
        if not self.stack or self.stack[-1].name != info.name:
            expected = self.stack[-1].name if self.stack else "<none>"
            logger.warning(
                "Ignoring unmatched pass timing callback for %s (expected %s)",
                info.name,
                expected,
            )
            self.stack.clear()
            return

        frame = self.stack.pop()
        duration_s = time.monotonic() - frame.start_time
        self_duration_s = max(0.0, duration_s - frame.child_duration_s)
        self.records.append(
            PassTimingRecord(
                name=frame.name,
                duration_s=duration_s,
                self_duration_s=self_duration_s,
                depth=frame.depth,
                sequence=frame.sequence,
            )
        )
        if self.stack:
            self.stack[-1].child_duration_s += duration_s


class TileLangPassTimingInstrument:
    """Per-pass timing instrument for tilelang compilation pipelines.

    This is a plain-Python wrapper that creates a TVM ``PassInstrument``
    internally. Callback state is kept separate so the native instrument does
    not retain this wrapper.

    Parameters
    ----------
    threshold_ms : float
        Only show passes whose inclusive duration meets this threshold (ms).
        0.0 means show all passes.
    """

    def __init__(self, threshold_ms: float = 0.0):
        self._state = _PassTimingState()
        self._records = self._state.records
        self._stack = self._state.stack
        self._threshold_ms = threshold_ms
        self._instrument = self._create_tvm_instrument(self._state)

    def _enter_pass_ctx(self):
        self._state.enter_pass_ctx()

    def _exit_pass_ctx(self):
        self._state.exit_pass_ctx()

    def _run_before_pass(self, info):
        self._state.run_before_pass(None, info)

    def _run_after_pass(self, info):
        self._state.run_after_pass(None, info)

    @staticmethod
    def _create_tvm_instrument(state: _PassTimingState):
        """Create an instrument whose callbacks do not retain the wrapper."""
        return _ffi_instrument_api.PassInstrument(
            "TileLangPassTimingInstrument",
            state.enter_pass_ctx,
            state.exit_pass_ctx,
            None,
            state.run_before_pass,
            state.run_after_pass,
        )

    @property
    def instrument(self):
        """The underlying TVM PassInstrument to add to ``PassContext``."""
        return self._instrument

    @property
    def records(self) -> list[PassTimingRecord]:
        return sorted(self._records, key=lambda record: record.sequence)

    @property
    def total_duration_s(self) -> float:
        # Sum only top-level passes to avoid double-counting nested passes.
        return sum(record.duration_s for record in self._records if record.depth == 0)

    def report(self, context: str | None = None) -> str:
        """Generate a formatted timing report string."""
        if not self._records:
            message = "[tilelang pass timing] No passes recorded."
            return f"{message} Context: {context}" if context else message

        threshold_s = self._threshold_ms / 1000.0
        total = self.total_duration_s
        lines = []

        lines.append("=" * 96)
        lines.append("TileLang Pass Timing Report")
        if context:
            lines.append(f"Context: {context}")
        lines.append("=" * 96)
        lines.append(f"Total: {total:.4f}s ({len(self._records)} passes)")
        if self._threshold_ms > 0:
            lines.append(f"(inclusive threshold: {self._threshold_ms:.1f}ms)")
        lines.append("-" * 96)
        lines.append(f"{'#':>3}  {'Pass Name':<43} {'Inclusive':>10} {'Self':>10}  {'Incl %':>7} {'Self %':>7}")
        lines.append("-" * 96)

        skipped = 0
        for index, record in enumerate(self.records, 1):
            if record.duration_s < threshold_s:
                skipped += 1
                continue
            inclusive_pct = (record.duration_s / total * 100) if total > 0 else 0
            self_pct = (record.self_duration_s / total * 100) if total > 0 else 0
            name_col = f"{'  ' * record.depth}{record.name}"
            lines.append(
                f"{index:>3}  {name_col:<43} {record.duration_s:>9.4f}s "
                f"{record.self_duration_s:>9.4f}s {inclusive_pct:>6.1f}% {self_pct:>6.1f}%"
            )

        lines.append("-" * 96)
        sorted_records = sorted(
            (record for record in self._records if record.duration_s >= threshold_s),
            key=lambda record: record.duration_s,
            reverse=True,
        )
        top_n = min(10, len(sorted_records))
        if top_n > 0:
            lines.append(f"Top {top_n} Slowest Passes by Inclusive Time:")
            for rank, record in enumerate(sorted_records[:top_n], 1):
                inclusive_pct = (record.duration_s / total * 100) if total > 0 else 0
                lines.append(f"  {rank:>2}. {record.name:<45} {record.duration_s:>9.4f}s {inclusive_pct:>5.1f}%")

        if skipped > 0:
            lines.append(f"\n({skipped} passes skipped - under {self._threshold_ms:.1f}ms inclusive threshold)")

        lines.append("=" * 96)
        return "\n".join(lines)

    def print_report(self, context: str | None = None):
        """Print the timing report to the logger."""
        logger.info("\n%s", self.report(context=context))


def build_pass_instruments(
    base_instruments: Sequence[object], threshold_ms: float | None
) -> tuple[list[object], TileLangPassTimingInstrument | None]:
    """Build instruments with timing first so later after-pass callbacks are excluded."""
    instruments = list(base_instruments)
    if threshold_ms is None:
        return instruments, None

    timing_instrument = TileLangPassTimingInstrument(threshold_ms=threshold_ms)
    instruments.insert(0, timing_instrument.instrument)
    return instruments, timing_instrument


@contextmanager
def report_pass_timing_on_exit(timing_instrument: TileLangPassTimingInstrument | None, context: str) -> Iterator[None]:
    """Emit a timing report after PassContext exits, including on failure."""
    try:
        yield
    finally:
        if timing_instrument is not None:
            timing_instrument.print_report(context=context)
