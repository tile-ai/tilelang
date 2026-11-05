"""Diagnostic consumers retain native graphs that TVMScript cannot print."""

import importlib
import json

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

import tilelang
from tilelang.backend.module import create_backend_context
from tilelang.engine.lower import host_codegen
from tilelang.instrumentation import CodegenEvent
from tilelang.tools.lower_trace import core, lower_trace
from tilelang.utils.pass_diff import _get_script, pass_diff
from tilelang.utils import pass_diff_hook


JSON_MARKER = "# Snapshot format: native IR JSON (not TVMScript)\n"


@pytest.fixture(autouse=True)
def _isolate_diagnostic_hooks(monkeypatch):
    monkeypatch.setenv("TILELANG_PASS_DIFF", "0")
    monkeypatch.delenv("TILELANG_PASS_DIFF_OUTPUT", raising=False)
    pass_diff_hook.uninstall_pass_diff_hook()
    yield
    pass_diff_hook.uninstall_pass_diff_hook()


def _source_host_graph():
    target = Target("c", Target("c"))
    helper = ir.GlobalVar("helper")
    argument, value = ir.Var("argument", "int32"), ir.Var("value", "int32")
    leaf = tirx.Function([value], [ir.Return(value + 3)], value.ty).with_attr("target", target)
    caller = tirx.Function(
        [argument],
        [ir.Evaluate(ir.Call(helper, [argument], ty=argument.ty)), ir.Return(ir.Tuple([]))],
        ir.TupleType([]),
    ).with_attr({"global_symbol": "main", "target": target})
    module = ir.IRModule({ir.GlobalVar("main"): caller, helper: leaf})
    return tilelang.transform.MakePackedAPI()(module)


def _assert_snapshot(text, expected):
    assert "TVMScript capture failed" in text
    assert "Back optional access" in text
    assert JSON_MARKER in text
    serialized = text.split(JSON_MARKER, 1)[1]
    json.loads(serialized)
    restored = ir.load_json(serialized)
    ir.assert_structural_equal(restored, expected)
    ir.assert_structural_equal(expected, restored)
    assert tvm_ffi.structural_hash(restored) == tvm_ffi.structural_hash(expected)


@ir.transform.module_pass(opt_level=0, name="CaptureMarker")
def _mark_module(module, _context):
    return module.with_attr("capture.marker", 7)


@ir.transform.module_pass(opt_level=0, name="CaptureFailure")
def _fail_pass(module, _context):
    raise RuntimeError("original pass failure")


def test_printable_ir_capture_keeps_the_original_script():
    module = ir.IRModule({"main": tirx.Function([], [ir.Return(ir.Tuple([]))], ir.TupleType([]))})
    before = ir.save_json(module)
    assert _get_script(module) == module.script()
    assert ir.save_json(module) == before


def test_native_metadata_capture_preserves_the_complete_graph():
    module = _source_host_graph()
    before = ir.save_json(module)
    _assert_snapshot(_get_script(module), module)
    assert ir.save_json(module) == before


@pytest.mark.parametrize("tool", [pass_diff, lower_trace])
def test_diagnostic_pass_chains_keep_nested_metadata(tool, tmp_path):
    module = _source_host_graph()
    before = ir.save_json(module)
    report = tmp_path / "capture.html"
    (step,) = tool(module, _mark_module, mode="html", html_path=str(report))
    _assert_snapshot(step["before_script"], module)
    _assert_snapshot(step["after_script"], module.with_attr("capture.marker", 7))
    assert step["changed"]
    assert "capture.marker" in report.read_text()
    assert ir.save_json(module) == before


@pytest.mark.parametrize("tool", [pass_diff, lower_trace])
def test_diagnostic_capture_does_not_replace_a_real_pass_error(tool, tmp_path):
    module = _source_host_graph()
    before = ir.save_json(module)
    with pytest.raises(RuntimeError, match="original pass failure"):
        tool(module, _fail_pass, mode="html", html_path=str(tmp_path / "failure.html"))
    assert ir.save_json(module) == before


def test_automatic_pass_diff_captures_native_metadata(monkeypatch, tmp_path):
    module = _source_host_graph()
    monkeypatch.setenv("TILELANG_PASS_DIFF", "html")
    monkeypatch.setenv("TILELANG_PASS_DIFF_OUTPUT", str(tmp_path))
    pass_diff_hook.install_pass_diff_hook()
    result = _mark_module(module)
    (step,) = pass_diff_hook._html_steps
    _assert_snapshot(step["before_script"], module)
    _assert_snapshot(step["after_script"], result)
    assert step["changed"]


def test_pass_instrument_captures_content_instead_of_an_object_address(tmp_path):
    module = _source_host_graph()
    trace = core.LowerTraceSession(mode="html", trace_dir=str(tmp_path), codegen_output=None)
    with ir.transform.PassContext(instruments=[trace.create_pass_instrument()]):
        result = _mark_module(module)
    (record,) = trace.records
    _assert_snapshot(record.before_text, module)
    _assert_snapshot(record.after_text, result)
    assert record.changed and record.status == core.STATUS_COMPLETED


def test_codegen_trace_keeps_native_metadata_and_real_source(tmp_path):
    module = _source_host_graph()
    trace = core.LowerTraceSession(mode="html", trace_dir=str(tmp_path), codegen_output=None)
    target = Target("c")
    context = create_backend_context(target, target, "tvm_ffi")
    output = trace.run_codegen(
        CodegenEvent(name="tl.codegen.build_c_host", mod=module, target=target, role="host"),
        lambda: host_codegen(module, context),
    )
    (record,) = trace.records
    _assert_snapshot(record.before_text, module)
    assert "main" in output.inspect_source()
    assert "main" in record.after_text
    assert record.status == core.STATUS_CODEGEN


def test_capture_reports_both_diagnostic_failures(monkeypatch):
    module = _source_host_graph()

    def unavailable_json(_value):
        raise RuntimeError("JSON encoder unavailable")

    monkeypatch.setattr(importlib.import_module("tvm_tirx.ir"), "save_json", unavailable_json)
    text = _get_script(module)
    assert "Back optional access" in text
    assert "Native IR JSON capture also failed" in text
    assert "JSON encoder unavailable" in text
