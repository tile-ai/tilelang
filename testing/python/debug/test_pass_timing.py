"""Tests for pass timing instrumentation."""

import gc
import weakref
from types import SimpleNamespace

import pytest

import tilelang.language as T
from tilelang import tvm
from tilelang.transform import PassConfigKey
from tilelang.instrumentation import (
    compile_pass_instrumentation,
    create_pass_instruments,
    current_compile_pass_instrumentation,
)
from tilelang.tools.pass_timing import (
    PassTimingRecord,
    PassTimingTool,
    TileLangPassTimingInstrument,
    _extract_kernel_label,
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


def test_pass_timing_tool_creates_fresh_instruments_with_context():
    timing_tool = PassTimingTool()
    base_instrument = object()

    with compile_pass_instrumentation(
        name="timing-test",
        tools=[timing_tool],
        include_default_tools=False,
    ):
        first = [
            *create_pass_instruments(context="stage=first"),
            base_instrument,
        ]
        second = create_pass_instruments(context="stage=second")

    assert first[0] is timing_tool.timings[0].instrument
    assert first[1] is base_instrument
    assert second[0] is timing_tool.timings[1].instrument
    assert first[0] is not second[0]
    assert timing_tool.contexts == ("stage=first", "stage=second")


def test_jit_compile_session_owns_configured_timing_tool(monkeypatch):
    from tilelang.jit.kernel import JITKernel

    kernel = object.__new__(JITKernel)
    kernel.pass_configs = {
        PassConfigKey.TL_PASS_PROFILE: True,
        PassConfigKey.TL_PASS_PROFILE_THRESHOLD_MS: 3.0,
    }
    kernel.compile_flags = None
    kernel.verbose = False
    kernel.target = tvm.target.Target("c")
    kernel.target_host = None
    kernel.execution_backend = "cython"
    artifact = object()
    result = object()
    observed = []

    def observe_session(stage):
        session = current_compile_pass_instrumentation()
        timing_tool = session.find_tool(PassTimingTool) if session is not None else None
        observed.append((stage, session, timing_tool.threshold_ms if timing_tool is not None else None))

    def compile_artifact(_self, _func, _pass_configs, _phase_context):
        observe_session("artifact")
        return artifact

    def create_adapter(_self, _func, _out_idx, actual_artifact, _pass_configs, _phase_context):
        assert actual_artifact is artifact
        observe_session("adapter")
        return result

    monkeypatch.setattr(JITKernel, "_compile_artifact", compile_artifact)
    monkeypatch.setattr(JITKernel, "_create_adapter_from_artifact", create_adapter)
    func = next(iter(_simple_module().functions.items()))[1]

    assert kernel._compile_and_create_adapter(func, []) is result
    assert kernel.artifact is artifact
    assert [item[0] for item in observed] == ["artifact", "adapter"]
    assert observed[0][1] is observed[1][1]
    assert [item[2] for item in observed] == [3.0, 3.0]


def test_pass_timing_excludes_later_after_pass_callbacks(monkeypatch):
    clock = [0.0]
    timing_tool = PassTimingTool()

    @tvm.ir.instrument.pass_instrument
    class AdvanceClockAfterPass:
        def run_after_pass(self, mod, info):
            clock[0] += 1.0

    monkeypatch.setattr("tilelang.tools.pass_timing.time.monotonic", lambda: clock[0])

    with compile_pass_instrumentation(
        name="timing-order",
        tools=[timing_tool],
        include_default_tools=False,
    ):
        instruments = [
            *create_pass_instruments(context="stage=test"),
            AdvanceClockAfterPass(),
        ]
        with tvm.transform.PassContext(instruments=instruments):
            tvm.tirx.transform.Simplify()(_simple_module())

    timing = timing_tool.timings[0]
    assert timing.records[0].duration_s == 0.0
    assert clock[0] == 1.0


def test_pass_timing_calculates_nested_self_time(monkeypatch):
    timing = TileLangPassTimingInstrument()
    timestamps = iter([0.0, 1.0, 3.0, 5.0])
    monkeypatch.setattr("tilelang.tools.pass_timing.time.monotonic", lambda: next(timestamps))
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


def test_pass_timing_tool_reports_all_contexts_on_session_failure(monkeypatch):
    timing_tool = PassTimingTool()
    reports = []
    monkeypatch.setattr(timing_tool, "print_report", lambda: reports.append(timing_tool.contexts))

    with (
        pytest.raises(RuntimeError, match="expected failure"),
        compile_pass_instrumentation(
            name="failing-compile",
            tools=[timing_tool],
            include_default_tools=False,
        ),
    ):
        create_pass_instruments(context="stage=jit-lower, kernel=main")
        raise RuntimeError("expected failure")

    assert reports == [("stage=jit-lower, kernel=main",)]


def test_pass_timing_cleans_incomplete_frames_after_pass_failure(monkeypatch):
    timing = TileLangPassTimingInstrument()
    warnings = []
    monkeypatch.setattr(
        "tilelang.tools.pass_timing.logger.warning",
        lambda message, *args: warnings.append(message % args),
    )

    @tvm.transform.module_pass(opt_level=0, name="FailingPass")
    def failing_pass(mod, ctx):
        raise RuntimeError("expected failure")

    with pytest.raises(RuntimeError, match="expected failure"), tvm.transform.PassContext(instruments=[timing.instrument]):
        failing_pass(_simple_module())

    assert not timing._stack
    assert any("Discarding 1 incomplete pass timing frame" in warning for warning in warnings)


def test_pass_timing_ignores_unmatched_after_callback(monkeypatch):
    timing = TileLangPassTimingInstrument()
    warnings = []
    monkeypatch.setattr(
        "tilelang.tools.pass_timing.logger.warning",
        lambda message, *args: warnings.append(message % args),
    )

    timing._enter_pass_ctx()
    timing._run_after_pass(SimpleNamespace(name="unexpected"))

    assert not timing.records
    assert not timing._stack
    assert any("Ignoring unmatched pass timing callback" in warning for warning in warnings)


def test_pass_timing_captures_kernel_name_from_module():
    timing = TileLangPassTimingInstrument()

    with tvm.transform.PassContext(instruments=[timing.instrument]):
        tvm.tirx.transform.Simplify()(_simple_module())

    # The kernel name follows the PrimFunc's global_symbol attribute (which in
    # the real compilation flow is also used as the IRModule key).
    assert any(record.kernel == "program" for record in timing.records)
    report = timing.report()
    assert "Simplify(program)" in report


def test_pass_timing_record_name_unaffected_by_kernel():
    timing = TileLangPassTimingInstrument()

    with tvm.transform.PassContext(instruments=[timing.instrument]):
        tvm.tirx.transform.Simplify()(_simple_module())

    record = timing.records[0]
    assert record.name.endswith(".Simplify")
    assert record.kernel == "program"


def test_extract_kernel_label_none_or_empty():
    assert _extract_kernel_label(None) == ""
    assert _extract_kernel_label(tvm.IRModule()) == ""


def test_extract_kernel_label_single_function():
    assert _extract_kernel_label(_simple_module()) == "program"


def test_extract_kernel_label_multiple_functions_shows_count():
    @T.prim_func
    def func_a(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
        with T.Kernel(threads=16):
            tid = T.get_thread_binding()
            B[tid] = A[tid] + 1.0

    @T.prim_func
    def func_b(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
        with T.Kernel(threads=16):
            tid = T.get_thread_binding()
            B[tid] = A[tid] + 2.0

    mod = tvm.IRModule({"kernel_a": func_a, "kernel_b": func_b})

    label = _extract_kernel_label(mod)

    assert label == "func_a +1 more"


def test_pass_timing_report_kernel_suffix_when_present():
    timing = TileLangPassTimingInstrument()
    timing._records.append(PassTimingRecord("tirx.Simplify", 0.001, 0, kernel="matmul"))

    report = timing.report()

    assert "tirx.Simplify(matmul)" in report


def test_pass_timing_report_omits_suffix_when_kernel_empty():
    timing = TileLangPassTimingInstrument()
    timing._records.append(PassTimingRecord("tirx.Simplify", 0.001, 0, kernel=""))

    report = timing.report()

    assert "tirx.Simplify(matmul)" not in report
    assert "tirx.Simplify" in report
