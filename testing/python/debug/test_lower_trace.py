# type: ignore
"""Tests for the lower_trace debugging feature."""

import os
import pytest
import tempfile

import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang import tvm
from tilelang.tools.lower_trace import core as _core


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    monkeypatch.delenv("TL_LOWER_TRACE", raising=False)
    monkeypatch.delenv("TL_LOWER_TRACE_DIR", raising=False)
    monkeypatch.setattr(_core, "_mode_override", _core._UNSET)
    yield
    monkeypatch.delenv("TL_LOWER_TRACE", raising=False)
    monkeypatch.delenv("TL_LOWER_TRACE_DIR", raising=False)
    monkeypatch.setattr(_core, "_mode_override", _core._UNSET)


def _simple_program():
    @T.prim_func
    def program(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        with T.Kernel(threads=128):
            tid = T.get_thread_binding()
            B[tid] = A[tid] + 1.0

    return program


def _noop_pass():
    return tvm.tirx.transform.Simplify()


def test_env_default_off():
    assert _core._get_mode() is None


def test_env_off_values(monkeypatch):
    for v in ("0", "off", "false", "no", ""):
        monkeypatch.setenv("TL_LOWER_TRACE", v)
        assert _core._get_mode() is None, f"Expected None for {v!r}"


def test_env_truthy_maps_to_html(monkeypatch):
    for v in ("1", "on", "true", "yes"):
        monkeypatch.setenv("TL_LOWER_TRACE", v)
        assert _core._get_mode() == "html", f"Expected 'html' for {v!r}"


def test_env_explicit_modes(monkeypatch):
    monkeypatch.setenv("TL_LOWER_TRACE", "terminal")
    assert _core._get_mode() == "terminal"

    monkeypatch.setenv("TL_LOWER_TRACE", "html")
    assert _core._get_mode() == "html"

    monkeypatch.setenv("TL_LOWER_TRACE", "both")
    assert _core._get_mode() == "both"


def test_lower_trace_api_single_pass(capsys):
    from tilelang.tools.lower_trace import lower_trace

    program = _simple_program()
    results = lower_trace(program, _noop_pass(), mode="terminal")
    assert len(results) == 1
    assert "name" in results[0]
    assert "changed" in results[0]
    captured = capsys.readouterr()
    assert "Pass 1" in captured.out


def test_lower_trace_api_chain():
    from tilelang.tools.lower_trace import lower_trace

    program = _simple_program()
    passes = [
        ("Simplify1", _noop_pass()),
        ("Simplify2", _noop_pass()),
    ]
    results = lower_trace(program, passes, mode="terminal")
    assert len(results) == 2
    assert results[0]["name"] == "Simplify1"
    assert results[1]["name"] == "Simplify2"


def test_enable_disable():
    from tilelang.tools.lower_trace import enable, disable

    enable()
    disable()


def test_lower_trace_html():
    from tilelang.tools.lower_trace import lower_trace

    program = _simple_program()
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        html_path = f.name

    try:
        results = lower_trace(program, _noop_pass(), mode="html", html_path=html_path)
        assert len(results) == 1
        assert os.path.exists(html_path)
        with open(html_path) as f:
            content = f.read()
        assert "TileLang" in content or "pass" in content.lower()
    finally:
        os.unlink(html_path)


def test_discover_passes():
    from tilelang.tools.lower_trace.core import _discover_passes
    from tilelang.cpu.pipeline import CPUPassPipelineBody

    pass_names = _discover_passes(CPUPassPipelineBody)
    assert len(pass_names) > 10, f"Expected >10 passes, got {len(pass_names)}"
    assert "Simplify" in pass_names
    assert "LayoutInference" in pass_names
    assert "BindTarget" in pass_names


def test_lower_trace_dark_theme():
    from tilelang.tools.lower_trace import lower_trace

    program = _simple_program()
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        html_path = f.name

    try:
        lower_trace(program, _noop_pass(), mode="html", html_path=html_path)
        with open(html_path) as f:
            content = f.read()

        assert 'id="theme-btn"' in content, "Theme toggle button missing"
        assert "toggleTheme" in content, "toggleTheme JS function missing"
        assert "--bg:" in content, "CSS variable --bg missing"
        assert '[data-theme="dark"]' in content, "Dark theme CSS override missing"
        assert "localStorage" in content, "localStorage persistence missing"
        assert "lower-trace-theme" in content, "Theme localStorage key missing"
    finally:
        os.unlink(html_path)


def test_multi_run_accumulation(monkeypatch):
    from tilelang.tools.lower_trace import enable, disable
    from tilelang.tools.lower_trace import core as _core
    from tilelang.backend.pass_pipeline import resolve_pipeline
    import tilelang.language as T

    monkeypatch.setenv("TL_LOWER_TRACE", "both")
    monkeypatch.setenv("TL_LOWER_TRACE_DIR", tempfile.mkdtemp(prefix="lt_test_"))

    disable()
    enable()

    @T.prim_func
    def tiny(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
        with T.Kernel(32):
            tid = T.get_thread_binding()
            B[tid] = A[tid] + 1.0

    mod = tvm.IRModule({"main": tiny})
    target = tvm.target.Target("c")
    pipeline = resolve_pipeline(target)

    assert _core._run_counter == 0, f"Expected run_counter=0 before enable, got {_core._run_counter}"

    pipeline.lower(mod, target)
    run1_count = len(_core._records)
    assert _core._run_counter == 1, f"Expected run_counter=1 after first run, got {_core._run_counter}"
    assert run1_count > 0, "First run should produce records"

    pipeline.lower(mod, target)
    total_count = len(_core._records)
    assert _core._run_counter == 2, f"Expected run_counter=2 after second run, got {_core._run_counter}"
    assert total_count > run1_count, f"Second run should accumulate records: total={total_count}, run1={run1_count}"

    phases = {rec.phase for rec in _core._records}
    assert "pipeline_c" in phases, "First run should have phase 'pipeline_c'"
    assert "run2_pipeline_c" in phases, "Second run should have phase 'run2_pipeline_c'"

    disable()
    monkeypatch.delenv("TL_LOWER_TRACE", raising=False)
    monkeypatch.delenv("TL_LOWER_TRACE_DIR", raising=False)


def test_diff_html_line_numbers_monotone():
    import re
    from tilelang.tools.lower_trace.diff import _make_diff_html

    # Whitespace-variant duplicates land inside a single replace hunk: top-level
    # difflib (full-line) won't pre-match them as equal, but the strip-level
    # pairing inside the hunk used to greedily pair a later left line to an
    # earlier right line, making the right column render out of order.
    before = "\n".join([" A", "A", " B"])
    after = "\n".join(["C", "A ", "B"])
    html = _make_diff_html(before, after, context=3)

    left, right = [], []
    for row in re.finditer(r"<tr[^>]*>(.*?)</tr>", html, re.S):
        for side, txt in re.findall(r'<td class="ln[^"]*"\s+data-side="([lr])"[^>]*>(\d*)</td>', row.group(1)):
            (left if side == "l" else right).append(int(txt) if txt.strip() else None)

    assert left and right, f"no line-number cells parsed:\n{html}"
    for name, col in (("left", left), ("right", right)):
        nums = [n for n in col if n is not None]
        assert nums == sorted(nums), f"{name} column line numbers not ascending: {nums}"


def test_no_skipped_phantom_records(monkeypatch):
    """Pre-registration is gone: no SKIPPED records, indices global-monotonic."""
    from tilelang.tools.lower_trace import enable, disable
    from tilelang.tools.lower_trace import core as _core
    from tilelang.tools.lower_trace.core import STATUS_SKIPPED
    from tilelang.backend.pass_pipeline import resolve_pipeline
    import tilelang.language as T

    monkeypatch.setenv("TL_LOWER_TRACE", "both")
    monkeypatch.setenv("TL_LOWER_TRACE_DIR", tempfile.mkdtemp(prefix="lt_test_"))

    disable()
    enable()

    @T.prim_func
    def tiny(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
        with T.Kernel(32):
            tid = T.get_thread_binding()
            B[tid] = A[tid] + 1.0

    mod = tvm.IRModule({"main": tiny})
    target = tvm.target.Target("c")
    pipeline = resolve_pipeline(target)
    pipeline.lower(mod, target)

    # No phantom/skipped records remain — every record is COMPLETED or FAILED
    skipped = [r for r in _core._records if r.status == STATUS_SKIPPED]
    assert not skipped, f"Found {len(skipped)} SKIPPED records (pre-registration not removed)"

    # Indices are strictly increasing across all records (global-monotonic)
    indices = [r.index for r in _core._records]
    assert indices == sorted(indices), f"Indices not ascending: {indices}"
    assert len(indices) == len(set(indices)), f"Duplicate indices: {indices}"

    # No phantom LetInline slot when should_force_let_inline() is False
    letinline = [r for r in _core._records if "LetInline" in r.name]
    assert not letinline, f"Phantom LetInline records found: {letinline}"

    disable()
    monkeypatch.delenv("TL_LOWER_TRACE", raising=False)
    monkeypatch.delenv("TL_LOWER_TRACE_DIR", raising=False)


if __name__ == "__main__":
    tilelang.testing.main()
