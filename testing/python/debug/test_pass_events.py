"""Tests for the shared nested-pass event instrumentation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import threading
from types import SimpleNamespace

from tilelang.instrumentation import (
    CodegenEvent,
    PassEvent,
    PassEventObserver,
    PassInstrumentationTool,
    StackedPassInstrument,
    active_stacked_pass_instruments,
    compile_pass_instrumentation,
    create_pass_instruments,
    current_compile_pass_instrumentation,
    current_pass_phase,
    run_codegen_with_instrumentation,
    pass_phase,
    pass_pipeline,
    register_pass_instrumentation_tool,
    unregister_pass_instrumentation_tool,
)


class _RecordingObserver(PassEventObserver):
    def __init__(self):
        self.started: list[PassEvent] = []
        self.finished: list[PassEvent] = []
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
            with compile_pass_instrumentation(name="nested") as nested:
                assert nested is session
            first = create_pass_instruments()
            second = create_pass_instruments()
            assert first != second
            assert instances[0].instruments == [*first, *second]
            with pass_pipeline("cuda"):
                events.append(("body", current_pass_phase()))
            assert run_codegen_with_instrumentation("target.build.test", "mod", "target", lambda: "result") == "result"
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
