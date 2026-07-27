"""Tests for pass timing instrumentation."""

import gc
import logging
import weakref
from types import SimpleNamespace

import pytest

import tilelang.language as T
from tilelang import tvm
from tilelang.utils.pass_timing import (
    PassTimingRecord,
    TileLangPassTimingInstrument,
    build_pass_instruments,
    report_pass_timing_on_exit,
)


def _simple_module():
    @T.prim_func
    def program(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
        with T.Kernel(threads=16):
            tid = T.get_thread_binding()
            B[tid] = A[tid] + 1.0

    return tvm.IRModule({"main": program})


def test_pass_timing_records_simple_pass():
    timing = TileLangPassTimingInstrument()

    with tvm.transform.PassContext(instruments=[timing.instrument]):
        tvm.tirx.transform.Simplify()(_simple_module())

    assert timing.records
    assert any(record.name.endswith(".Simplify") for record in timing.records)
    assert all(record.duration_s >= 0 for record in timing.records)
    assert all(0 <= record.self_duration_s <= record.duration_s for record in timing.records)


def test_pass_timing_wrapper_can_be_collected():
    timing = TileLangPassTimingInstrument()
    timing_ref = weakref.ref(timing)
    state_ref = weakref.ref(timing._state)

    del timing
    gc.collect()

    assert timing_ref() is None
    assert state_ref() is None


def test_build_pass_instruments_prepends_timing():
    base_instrument = object()

    instruments, timing = build_pass_instruments([base_instrument], threshold_ms=0.0)

    assert timing is not None
    assert instruments[0] is timing.instrument
    assert instruments[1] is base_instrument


def test_build_pass_instruments_without_profile_preserves_base_instruments():
    base_instrument = object()

    instruments, timing = build_pass_instruments([base_instrument], threshold_ms=None)

    assert timing is None
    assert instruments == [base_instrument]


def test_pass_timing_excludes_later_after_pass_callbacks(monkeypatch):
    clock = [0.0]

    @tvm.ir.instrument.pass_instrument
    class AdvanceClockAfterPass:
        def run_after_pass(self, mod, info):
            clock[0] += 1.0

    instruments, timing = build_pass_instruments([AdvanceClockAfterPass()], threshold_ms=0.0)
    monkeypatch.setattr("tilelang.utils.pass_timing.time.monotonic", lambda: clock[0])

    with tvm.transform.PassContext(instruments=instruments):
        tvm.tirx.transform.Simplify()(_simple_module())

    assert timing is not None
    assert timing.records[0].duration_s == 0.0
    assert clock[0] == 1.0


def test_pass_timing_calculates_nested_self_time(monkeypatch):
    timing = TileLangPassTimingInstrument()
    timestamps = iter([0.0, 1.0, 3.0, 5.0])
    monkeypatch.setattr("tilelang.utils.pass_timing.time.monotonic", lambda: next(timestamps))
    parent = SimpleNamespace(name="parent")
    child = SimpleNamespace(name="child")

    timing._enter_pass_ctx()
    timing._run_before_pass(parent)
    timing._run_before_pass(child)
    timing._run_after_pass(child)
    timing._run_after_pass(parent)

    parent_record, child_record = timing.records
    assert parent_record.duration_s == pytest.approx(5.0)
    assert parent_record.self_duration_s == pytest.approx(3.0)
    assert child_record.duration_s == pytest.approx(2.0)
    assert child_record.self_duration_s == pytest.approx(2.0)
    assert timing.total_duration_s == pytest.approx(5.0)


def test_pass_timing_report_filters_by_inclusive_threshold():
    timing = TileLangPassTimingInstrument(threshold_ms=10.0)
    timing._records.extend(
        [
            PassTimingRecord("slow", 0.020, 0, self_duration_s=0.015, sequence=0),
            PassTimingRecord("fast", 0.005, 0, self_duration_s=0.005, sequence=1),
        ]
    )

    report = timing.report()

    assert "slow" in report
    assert "fast" not in report
    assert "1 passes skipped" in report
    assert "Inclusive" in report
    assert "Self" in report


def test_pass_timing_report_includes_context():
    timing = TileLangPassTimingInstrument()
    timing._records.append(PassTimingRecord("pass", 0.001, 0))

    report = timing.report(context="stage=grouped-host, config=3, kernel=main_gc_3")

    assert "Context: stage=grouped-host, config=3, kernel=main_gc_3" in report


def test_pass_timing_report_is_emitted_on_failure(monkeypatch):
    timing = TileLangPassTimingInstrument()
    contexts = []
    monkeypatch.setattr(timing, "print_report", lambda context=None: contexts.append(context))

    with (
        pytest.raises(RuntimeError, match="expected failure"),
        report_pass_timing_on_exit(timing, context="stage=jit-lower, kernel=main"),
    ):
        raise RuntimeError("expected failure")

    assert contexts == ["stage=jit-lower, kernel=main"]


def test_pass_timing_cleans_incomplete_frames_after_pass_failure(caplog):
    timing = TileLangPassTimingInstrument()

    @tvm.transform.module_pass(opt_level=0, name="FailingPass")
    def failing_pass(mod, ctx):
        raise RuntimeError("expected failure")

    caplog.set_level(logging.WARNING, logger="tilelang.pass_timing")
    with pytest.raises(RuntimeError, match="expected failure"), tvm.transform.PassContext(instruments=[timing.instrument]):
        failing_pass(_simple_module())

    assert not timing._stack
    assert "Discarding 1 incomplete pass timing frame" in caplog.text


def test_pass_timing_ignores_unmatched_after_callback(caplog):
    timing = TileLangPassTimingInstrument()
    caplog.set_level(logging.WARNING, logger="tilelang.pass_timing")

    timing._enter_pass_ctx()
    timing._run_after_pass(SimpleNamespace(name="unexpected"))

    assert not timing.records
    assert not timing._stack
    assert "Ignoring unmatched pass timing callback" in caplog.text
