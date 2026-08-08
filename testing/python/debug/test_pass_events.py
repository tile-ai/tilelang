"""Tests for the shared nested-pass event instrumentation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import threading
from types import SimpleNamespace

from tilelang.utils.pass_events import (
    IncompletePass,
    PassEvent,
    PassEventObserver,
    StackedPassInstrument,
    active_stacked_pass_instruments,
    create_registered_pass_instruments,
    current_pass_phase,
    pass_phase,
    pass_pipeline,
    register_pass_instrument_provider,
    register_pipeline_scope_provider,
    unregister_pass_instrument_provider,
    unregister_pipeline_scope_provider,
)


class _RecordingObserver(PassEventObserver):
    def __init__(self):
        self.started: list[PassEvent] = []
        self.finished: list[PassEvent] = []
        self.incomplete: list[IncompletePass] = []
        self.errors: list[BaseException | None] = []
        self.mismatches: list[tuple[str, str | None]] = []
        self.context_entries = 0
        self.context_exits = 0

    def enter_pass_context(self):
        self.context_entries += 1

    def exit_pass_context(self):
        self.context_exits += 1

    def pass_started(self, mod, event):
        self.started.append(event)
        return f"before:{mod}"

    def pass_finished(self, mod, event, state):
        assert state.startswith("before:")
        self.finished.append(event)

    def passes_incomplete(self, passes, error):
        self.incomplete.extend(passes)
        self.errors.append(error)

    def callback_mismatch(self, actual, expected):
        self.mismatches.append((actual, expected))


def _info(name: str):
    return SimpleNamespace(name=name)


def test_stacked_instrument_records_nested_parentage_and_start_order():
    observer = _RecordingObserver()
    instrument = StackedPassInstrument(observer)

    instrument.enter_pass_ctx()
    instrument.run_before_pass("m0", _info("test.Outer"))
    instrument.run_before_pass("m1", _info("test.Inner"))
    instrument.run_after_pass("m2", _info("test.Inner"))
    instrument.run_after_pass("m3", _info("test.Outer"))
    instrument.exit_pass_ctx()

    assert [event.name for event in observer.started] == ["test.Outer", "test.Inner"]
    assert [event.name for event in observer.finished] == ["test.Inner", "test.Outer"]
    outer, inner = observer.started
    assert (outer.sequence, outer.depth, outer.parent_sequence) == (0, 0, None)
    assert (inner.sequence, inner.depth, inner.parent_sequence) == (1, 1, 0)
    assert observer.context_entries == 1
    assert observer.context_exits == 1


def test_stacked_instrument_can_emit_only_top_level_passes():
    observer = _RecordingObserver()
    instrument = StackedPassInstrument(observer, capture_nested=False)

    instrument.enter_pass_ctx()
    instrument.run_before_pass("m0", _info("test.Outer"))
    instrument.run_before_pass("m1", _info("test.Inner"))
    instrument.run_after_pass("m2", _info("test.Inner"))
    instrument.run_after_pass("m3", _info("test.Outer"))
    instrument.exit_pass_ctx()

    assert [event.name for event in observer.started] == ["test.Outer"]
    assert [event.name for event in observer.finished] == ["test.Outer"]
    assert observer.started[0].sequence == 0


def test_stacked_instrument_reports_incomplete_passes_with_error():
    observer = _RecordingObserver()
    instrument = StackedPassInstrument(observer)
    error = RuntimeError("boom")

    instrument.enter_pass_ctx()
    instrument.run_before_pass("m0", _info("test.Outer"))
    instrument.run_before_pass("m1", _info("test.Inner"))
    incomplete = instrument.abort(error)
    instrument.exit_pass_ctx()

    assert [item.event.name for item in incomplete] == ["test.Outer", "test.Inner"]
    assert [item.event.name for item in observer.incomplete] == ["test.Outer", "test.Inner"]
    assert observer.errors == [error]
    assert instrument.pending_events == ()


def test_stacked_instrument_recovers_from_callback_mismatch():
    observer = _RecordingObserver()
    instrument = StackedPassInstrument(observer)

    instrument.enter_pass_ctx()
    instrument.run_before_pass("m0", _info("test.Expected"))
    instrument.run_after_pass("m1", _info("test.Actual"))
    instrument.exit_pass_ctx()

    assert observer.mismatches == [("test.Actual", "test.Expected")]
    assert [item.event.name for item in observer.incomplete] == ["test.Expected"]
    assert isinstance(observer.errors[0], RuntimeError)
    assert instrument.pending_events == ()


def test_pass_phase_is_nested_and_restored():
    assert current_pass_phase() is None
    with pass_phase("pipeline_cuda"):
        assert current_pass_phase() == "pipeline_cuda"
        with pass_phase("codegen"):
            assert current_pass_phase() == "codegen"
        assert current_pass_phase() == "pipeline_cuda"
    assert current_pass_phase() is None


def test_registered_instruments_and_pipeline_scopes_are_composable():
    events = []

    @contextmanager
    def pipeline_scope(base_phase):
        with pass_phase(f"run2_{base_phase}"):
            events.append(("enter", current_pass_phase()))
            yield
            events.append(("exit", current_pass_phase()))

    marker = object()
    register_pass_instrument_provider("test", lambda: marker)
    register_pipeline_scope_provider("test", pipeline_scope)
    try:
        assert create_registered_pass_instruments() == [marker]
        with pass_pipeline("cuda"):
            events.append(("body", current_pass_phase()))
    finally:
        unregister_pipeline_scope_provider("test")
        unregister_pass_instrument_provider("test")

    assert events == [
        ("enter", "run2_pipeline_cuda"),
        ("body", "run2_pipeline_cuda"),
        ("exit", "run2_pipeline_cuda"),
    ]
    assert current_pass_phase() is None


def test_active_instruments_follow_nested_context_lifecycle():
    outer = StackedPassInstrument(_RecordingObserver())
    inner = StackedPassInstrument(_RecordingObserver())

    outer.enter_pass_ctx()
    assert active_stacked_pass_instruments() == (outer,)
    inner.enter_pass_ctx()
    assert active_stacked_pass_instruments() == (outer, inner)
    inner.exit_pass_ctx()
    assert active_stacked_pass_instruments() == (outer,)
    outer.exit_pass_ctx()
    assert active_stacked_pass_instruments() == ()


def test_active_instruments_support_tvm_fifo_exit_order():
    first = StackedPassInstrument(_RecordingObserver())
    second = StackedPassInstrument(_RecordingObserver())

    first.enter_pass_ctx()
    second.enter_pass_ctx()
    first.exit_pass_ctx()
    assert active_stacked_pass_instruments() == (second,)
    second.exit_pass_ctx()
    assert active_stacked_pass_instruments() == ()


def test_pass_phase_is_isolated_between_threads():
    barrier = threading.Barrier(2)

    def read_phase(name):
        with pass_phase(name):
            barrier.wait()
            return current_pass_phase()

    with ThreadPoolExecutor(max_workers=2) as pool:
        phases = list(pool.map(read_phase, ("pipeline_cuda", "pipeline_c")))

    assert phases == ["pipeline_cuda", "pipeline_c"]
    assert current_pass_phase() is None
