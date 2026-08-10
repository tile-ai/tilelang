"""Tests for the shared nested-pass event instrumentation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import threading
from types import SimpleNamespace

from tilelang.utils.pass_events import (
    CodegenEvent,
    IncompletePass,
    PassEvent,
    PassEventObserver,
    PassInstrumentationTool,
    StackedPassInstrument,
    active_stacked_pass_instruments,
    compile_pass_instrumentation,
    create_pass_instruments,
    current_compile_pass_instrumentation,
    current_pass_instrument_context,
    current_pass_phase,
    instrument_codegen,
    pass_phase,
    pass_pipeline,
    register_pass_instrumentation_tool,
    unregister_pass_instrumentation_tool,
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


class _RecordingTool(PassInstrumentationTool):
    def __init__(self, label: str, events: list[tuple]):
        self.label = label
        self.events = events
        self.instruments = []

    def create_pass_instrument(self):
        marker = object()
        self.instruments.append(marker)
        return marker

    @contextmanager
    def pipeline_scope(self, base_phase):
        with pass_phase(f"{self.label}_{base_phase}"):
            self.events.append(("enter", self.label, current_pass_phase()))
            yield
            self.events.append(("exit", self.label, current_pass_phase()))

    def run_codegen(self, event: CodegenEvent, next_call):
        self.events.append(("codegen", self.label, event.name))
        return next_call()

    def finish(self, error):
        self.events.append(("finish", self.label, error))


def test_registered_tools_are_snapshotted_and_composable():
    events = []
    instances = []

    def factory():
        tool = _RecordingTool("tool", events)
        instances.append(tool)
        return tool

    register_pass_instrumentation_tool("test", factory)
    try:
        with compile_pass_instrumentation(name="kernel") as session:
            assert session is current_compile_pass_instrumentation()
            first = create_pass_instruments()
            second = create_pass_instruments()
            assert first != second
            assert instances[0].instruments == [*first, *second]
            with pass_pipeline("cuda"):
                events.append(("body", current_pass_phase()))
            assert instrument_codegen("target.build.test", "mod", "target", lambda: "result") == "result"
    finally:
        unregister_pass_instrumentation_tool("test")

    assert events == [
        ("enter", "tool", "tool_pipeline_cuda"),
        ("body", "tool_pipeline_cuda"),
        ("exit", "tool", "tool_pipeline_cuda"),
        ("codegen", "tool", "target.build.test"),
        ("finish", "tool", None),
    ]
    assert current_pass_phase() is None
    assert current_compile_pass_instrumentation() is None


def test_pass_instrument_context_and_priority_are_applied_at_creation():
    contexts = []

    class ContextTool(_RecordingTool):
        def __init__(self, label, priority):
            super().__init__(label, [])
            self.pass_instrument_priority = priority

        def create_pass_instrument(self):
            contexts.append((self.label, current_pass_instrument_context()))
            return super().create_pass_instrument()

    later = ContextTool("later", 10)
    earlier = ContextTool("earlier", -10)
    with compile_pass_instrumentation(
        name="ordered",
        tools=[later, earlier],
        include_default_tools=False,
    ):
        instruments = create_pass_instruments(context="stage=lower")

    assert contexts == [("later", "stage=lower"), ("earlier", "stage=lower")]
    assert instruments == [earlier.instruments[0], later.instruments[0]]
    assert current_pass_instrument_context() is None


def test_nested_helpers_reuse_the_owning_compile_session():
    outer_tool = _RecordingTool("outer", [])
    with (
        compile_pass_instrumentation(
            name="outer",
            tools=[outer_tool],
            include_default_tools=False,
        ) as outer,
        compile_pass_instrumentation(name="nested") as nested,
    ):
        assert nested is outer


def test_registry_changes_only_affect_future_sessions():
    events = []
    register_pass_instrumentation_tool("snapshot-test", lambda: _RecordingTool("snapshot", events))
    try:
        with compile_pass_instrumentation(name="active") as active:
            unregister_pass_instrumentation_tool("snapshot-test")
            assert active.find_tool(_RecordingTool) is not None

        with compile_pass_instrumentation(name="future") as future:
            assert future.find_tool(_RecordingTool) is None
    finally:
        unregister_pass_instrumentation_tool("snapshot-test")


def test_compile_sessions_are_isolated_between_threads():
    barrier = threading.Barrier(2)

    def compile_one(label):
        tool = _RecordingTool(label, [])
        with compile_pass_instrumentation(
            name=label,
            tools=[tool],
            include_default_tools=False,
        ) as session:
            marker = create_pass_instruments()[0]
            barrier.wait()
            assert current_compile_pass_instrumentation() is session
            return session, tool, marker

    with ThreadPoolExecutor(max_workers=2) as pool:
        left, right = list(pool.map(compile_one, ("left", "right")))

    assert left[0] is not right[0]
    assert left[1] is not right[1]
    assert left[2] is not right[2]
    assert current_compile_pass_instrumentation() is None


def test_active_instruments_support_tvm_fifo_exit_order():
    first = StackedPassInstrument(_RecordingObserver())
    second = StackedPassInstrument(_RecordingObserver())

    first.enter_pass_ctx()
    assert active_stacked_pass_instruments() == (first,)
    second.enter_pass_ctx()
    assert active_stacked_pass_instruments() == (first, second)
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
