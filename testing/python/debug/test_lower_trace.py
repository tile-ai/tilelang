# type: ignore
"""Tests for the lower_trace debugging feature."""

import contextlib
import os
import pytest
import tempfile

# Clear any inherited TL_LOWER_TRACE* env before importing tilelang so the
# import-time activation hook in ``tilelang/__init__.py`` cannot fire during
# pytest collection. The autouse ``_isolate_env`` fixture below handles
# per-test isolation; this guard prevents leaks across collected modules.
# The caller's original env values are restored immediately after import so
# other collected modules or debugging runs still see their own settings.
_TRACE_ENV_KEYS = ("TL_LOWER_TRACE", "TL_LOWER_TRACE_DIR")
_saved_trace_env = {key: os.environ.get(key) for key in _TRACE_ENV_KEYS}
try:
    for key in _TRACE_ENV_KEYS:
        os.environ.pop(key, None)

    import tilelang
    import tilelang.testing
    import tilelang.language as T
    from tilelang import tvm
    from tilelang.tools.lower_trace import core as _core
finally:
    for key, value in _saved_trace_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    del _saved_trace_env


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Ensure each test starts with tracing disabled and env vars cleared (before and after)."""
    monkeypatch.delenv("TL_LOWER_TRACE", raising=False)
    monkeypatch.delenv("TL_LOWER_TRACE_DIR", raising=False)
    _core.disable()
    yield
    _core.disable()
    monkeypatch.delenv("TL_LOWER_TRACE", raising=False)
    monkeypatch.delenv("TL_LOWER_TRACE_DIR", raising=False)


def _simple_program():
    """Return a trivial elementwise-add prim_func used as trace input."""

    @T.prim_func
    def program(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        with T.Kernel(threads=128):
            tid = T.get_thread_binding()
            B[tid] = A[tid] + 1.0

    return program


def _noop_pass():
    """Return a Simplify pass (typically a no-op on the simple test program)."""
    return tvm.tirx.transform.Simplify()


def test_env_default_off():
    """With no TL_LOWER_TRACE set, tracing mode resolves to None (off)."""
    assert _core._get_mode() is None


def test_env_off_values(monkeypatch):
    """Falsish env values ('0','off','false','no','') all disable tracing."""
    for v in ("0", "off", "false", "no", ""):
        monkeypatch.setenv("TL_LOWER_TRACE", v)
        assert _core._get_mode() is None, f"Expected None for {v!r}"


def test_env_truthy_maps_to_html(monkeypatch):
    """Truthy shorthand values ('1','on','true','yes') map to 'html' mode."""
    for v in ("1", "on", "true", "yes"):
        monkeypatch.setenv("TL_LOWER_TRACE", v)
        assert _core._get_mode() == "html", f"Expected 'html' for {v!r}"


def test_env_explicit_modes(monkeypatch):
    """Explicit mode names ('terminal','html','both') are passed through verbatim."""
    monkeypatch.setenv("TL_LOWER_TRACE", "terminal")
    assert _core._get_mode() == "terminal"

    monkeypatch.setenv("TL_LOWER_TRACE", "html")
    assert _core._get_mode() == "html"

    monkeypatch.setenv("TL_LOWER_TRACE", "both")
    assert _core._get_mode() == "both"


def test_lower_trace_api_single_pass(capsys):
    """lower_trace() with a single pass returns one result and prints a 'Pass 1' header."""
    from tilelang.tools.lower_trace import lower_trace

    program = _simple_program()
    results = lower_trace(program, _noop_pass(), mode="terminal")
    assert len(results) == 1
    assert "name" in results[0]
    assert "changed" in results[0]
    captured = capsys.readouterr()
    assert "Pass 1" in captured.out


def test_lower_trace_api_chain():
    """lower_trace() applies a named pass chain in order and preserves names."""
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
    """enable()/disable() round-trip installs and removes instrumentation cleanly."""
    from tilelang.tools.lower_trace import enable, disable

    enable()
    disable()


def test_lower_trace_html():
    """lower_trace() in html mode writes a report file containing pass content."""
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


def test_lower_trace_dark_theme():
    """The HTML report embeds theme toggle CSS/JS and a localStorage key."""
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


def test_multiple_pipelines_share_one_compile_session(monkeypatch, tmp_path):
    """Multiple PassContexts in one compile share state without leaking globally."""
    from tilelang.tools.lower_trace import enable, disable
    from tilelang.tools.lower_trace import core as _core
    from tilelang.backend import create_backend_context
    from tilelang.utils.pass_events import compile_pass_instrumentation, create_pass_instruments
    import tilelang.language as T

    monkeypatch.setenv("TL_LOWER_TRACE", "both")
    monkeypatch.setenv("TL_LOWER_TRACE_DIR", str(tmp_path))

    disable()
    enable()

    @T.prim_func
    def tiny(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
        with T.Kernel(32):
            tid = T.get_thread_binding()
            B[tid] = A[tid] + 1.0

    mod = tvm.IRModule({"main": tiny})
    context = create_backend_context("c", "c", "cython")

    with compile_pass_instrumentation(name="two-pipelines") as compile_session:
        trace = compile_session.find_tool(_core.LowerTraceSession)
        assert trace is not None
        assert trace.pipeline_count == 0
        with tvm.transform.PassContext(instruments=create_pass_instruments()):
            context.lower(mod)
        run1_count = len(trace.records)
        assert trace.pipeline_count == 1
        assert run1_count > 0

        with tvm.transform.PassContext(instruments=create_pass_instruments()):
            context.lower(mod)
        assert trace.pipeline_count == 2
        assert len(trace.records) > run1_count

    phases = {record.phase for record in trace.records}
    assert "pipeline_c" in phases, "First run should have phase 'pipeline_c'"
    assert "run2_pipeline_c" in phases, "Second run should have phase 'run2_pipeline_c'"

    disable()
    monkeypatch.delenv("TL_LOWER_TRACE", raising=False)
    monkeypatch.delenv("TL_LOWER_TRACE_DIR", raising=False)


def test_diff_html_line_numbers_monotone():
    """Diff row line-number columns must stay ascending after strip-level pairing."""
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


def test_diff_html_trailing_newline_only_difference():
    """Only-trailing-newline diffs must not be misreported as 'No differences.'.

    Regression guard: ``splitlines()`` normalises ``"x\\n"`` and ``"x"`` to the
    same content, so a sole EOF-newline change used to short-circuit to the
    no-op message.  The compromise fix renders an explicit trailing-newline
    notice instead.
    """
    from tilelang.tools.lower_trace.diff import _make_diff_html

    html_present = _make_diff_html("x\n", "x", context=3)
    assert "No differences." not in html_present
    assert "trailing newline" in html_present

    html_absent = _make_diff_html("x", "x\n", context=3)
    assert "No differences." not in html_absent
    assert "trailing newline" in html_absent

    # Identical content *and* identical trailing newline → still no-op.
    assert _make_diff_html("x\n", "x\n", context=3) == '<p class="noop-msg">No differences.</p>'
    assert _make_diff_html("x", "x", context=3) == '<p class="noop-msg">No differences.</p>'


def test_no_skipped_phantom_records(monkeypatch, tmp_path):
    """Pre-registration is gone: no SKIPPED records, indices global-monotonic."""
    from tilelang.tools.lower_trace import enable, disable
    from tilelang.tools.lower_trace import core as _core
    from tilelang.tools.lower_trace.core import STATUS_SKIPPED
    import tilelang.language as T

    monkeypatch.setenv("TL_LOWER_TRACE", "both")
    monkeypatch.setenv("TL_LOWER_TRACE_DIR", str(tmp_path))

    disable()
    enable()

    @T.prim_func
    def tiny(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
        with T.Kernel(32):
            tid = T.get_thread_binding()
            B[tid] = A[tid] + 1.0

    tilelang.lower(tiny, target="c")

    trace = _core.get_last_session()
    assert trace is not None

    # No phantom/skipped records remain — every record is COMPLETED or FAILED
    skipped = [r for r in trace.records if r.status == STATUS_SKIPPED]
    assert not skipped, f"Found {len(skipped)} SKIPPED records (pre-registration not removed)"

    # Indices are strictly increasing across all records (global-monotonic)
    indices = [r.index for r in trace.records]
    assert indices == sorted(indices), f"Indices not ascending: {indices}"
    assert len(indices) == len(set(indices)), f"Duplicate indices: {indices}"

    # No phantom LetInline slot when should_force_let_inline() is False
    letinline = [r for r in trace.records if "LetInline" in r.name]
    assert not letinline, f"Phantom LetInline records found: {letinline}"

    disable()
    monkeypatch.delenv("TL_LOWER_TRACE", raising=False)
    monkeypatch.delenv("TL_LOWER_TRACE_DIR", raising=False)


def test_terminal_mode_no_html(monkeypatch, tmp_path):
    """TL_LOWER_TRACE=terminal must not produce an HTML report on finalization.

    Raw snapshot persistence creates a run directory even in terminal-only
    mode; finalizing that compile session must still not emit ``report.html``.
    """
    from tilelang.tools.lower_trace import enable, disable
    import tilelang.language as T

    monkeypatch.setenv("TL_LOWER_TRACE", "terminal")
    monkeypatch.setenv("TL_LOWER_TRACE_DIR", str(tmp_path))

    disable()
    enable()

    try:

        @T.prim_func
        def tiny(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
            with T.Kernel(32):
                tid = T.get_thread_binding()
                B[tid] = A[tid] + 1.0

        tilelang.lower(tiny, target="c")

        trace = _core.get_last_session()
        assert trace is not None
        trace.flush_html()
        script_dir = trace.script_dir
        assert script_dir is not None, "script_dir should be set after a run"
        symlink_report = os.path.join(script_dir, "report.html")
        assert not os.path.exists(symlink_report), f"terminal mode must not write report.html symlink at {symlink_report}"
        if trace.run_dir is not None:
            run_report = os.path.join(trace.run_dir, "report.html")
            assert not os.path.exists(run_report), f"terminal mode must not write report.html at {run_report}"
    finally:
        disable()


def test_lower_trace_html_on_failure(tmp_path):
    """lower_trace() flushes a partial HTML report even when a pass raises.

    Regression guard: previously an exception from ``p(mod)`` aborted the
    function before ``generate_html()`` ran, so ``mode='html'``/``'both'``
    lost the partial trace of completed passes.
    """
    from tilelang.tools.lower_trace import lower_trace

    program = _simple_program()
    html_path = str(tmp_path / "partial.html")

    class _BoomPass:
        def __call__(self, mod):
            """Always raise so the traced pass loop hits its failure path."""
            raise RuntimeError("intentional boom")

    passes = [
        ("Simplify", _noop_pass()),
        ("Boom", _BoomPass()),
    ]

    with pytest.raises(RuntimeError, match="intentional boom"):
        lower_trace(program, passes, mode="html", html_path=html_path)

    assert os.path.exists(html_path), "partial HTML report must be flushed on failure"
    with open(html_path) as f:
        content = f.read()
    assert "Simplify" in content, "completed pass must still appear in partial report"
    assert "Boom" in content, "failing pass name must appear in partial report"
    assert "FAILED" in content, "failing step must be marked FAILED"


# ---------------------------------------------------------------------------
# Codegen edit-and-recompile (Phase 1: _make_patched_source_module for _without_compile)
# ---------------------------------------------------------------------------


class _MockCodegenModule:
    """Minimal stand-in for a TVM runtime.Module returned by codegen FFIs."""

    def __init__(self, source: str):
        """Store the source string to return from inspect_source/get_source."""
        self._source = source

    def inspect_source(self) -> str:
        """Return the source string captured at construction."""
        return self._source

    def get_source(self) -> str:
        """Return the source string captured at construction."""
        return self._source


def _make_mock_build(source: str):
    """Return a mock codegen FFI that always produces *source*."""

    def mock_build(*args, **kwargs):
        """Pretend to compile and return a _MockCodegenModule holding *source*."""
        return _MockCodegenModule(source)

    return mock_build


class _MockPatchedModule:
    """Stand-in for the CSourceModule returned by _make_patched_source_module."""

    def __init__(self, source: str):
        """Store the patched source string to return from get_source."""
        self._source = source

    def get_source(self) -> str:
        """Return the patched source string."""
        return self._source

    def inspect_source(self) -> str:
        """Return the patched source string."""
        return self._source


def _patched_module_factory(original_module, patched_source):
    """Test replacement for _make_patched_source_module (avoids real TVM C++ FFI)."""
    return _MockPatchedModule(patched_source)


def _make_trace_session(tmp_path, mode="terminal"):
    """Create isolated lower-trace state for one unit-test compilation."""
    return _core.LowerTraceSession(
        mode=mode,
        trace_dir=str(tmp_path),
        codegen_output=str(tmp_path / "codegen.cpp"),
    )


def _run_codegen(session, build, ffi_name, mod="fake_mod", target=None):
    """Exercise the same explicit codegen middleware used by backend registries."""
    from tilelang.utils.pass_events import CodegenEvent

    return session.run_codegen(
        CodegenEvent(name=ffi_name, mod=mod, target=target),
        lambda: build(mod) if target is None else build(mod, target),
    )


def test_pass_instrument_captures_nested_tvm_passes(tmp_path):
    """The instrument backend records C++-nested passes with parent metadata."""

    @tvm.transform.module_pass(opt_level=0, name="test.Outer")
    def outer_pass(mod, _ctx):
        return tvm.tirx.transform.Simplify()(mod)

    program = _simple_program()
    mod = tvm.IRModule({"main": program})
    trace = _make_trace_session(tmp_path, mode="html")
    instrument = trace.create_pass_instrument()
    assert instrument is not None
    with tvm.transform.PassContext(instruments=[instrument]):
        outer_pass(mod)

    records = [record for record in trace.records if record.name in ("Outer", "Simplify")]
    assert [record.name for record in records] == ["Outer", "Simplify"]
    assert [(record.depth, record.parent_index) for record in records] == [(0, None), (1, records[0].index)]

    assert trace.run_dir is not None
    report_path = os.path.join(trace.run_dir, "report.html")
    with open(report_path) as report_file:
        report = report_file.read()
    assert 'data-depth="1"' in report
    assert f'data-parent-index="{records[0].index}"' in report


def test_lower_trace_and_pass_visualizer_share_event_infrastructure(tmp_path):
    """Both tools can observe one PassContext with their distinct policies."""
    from tilelang.tools.pass_visualizer import StructureTreePassInstrument

    @tvm.transform.module_pass(opt_level=0, name="test.Outer")
    def outer_pass(mod, _ctx):
        return tvm.tirx.transform.Simplify()(mod)

    mod = tvm.IRModule({"main": _simple_program()})
    trace = _make_trace_session(tmp_path)
    lower_trace_instrument = trace.create_pass_instrument()
    assert lower_trace_instrument is not None
    visualizer_instrument = StructureTreePassInstrument()
    with tvm.transform.PassContext(instruments=[lower_trace_instrument, visualizer_instrument]):
        outer_pass(mod)

    trace_records = [record for record in trace.records if record.name in ("Outer", "Simplify")]
    assert [record.name for record in trace_records] == ["Outer", "Simplify"]
    assert [record.name for record in visualizer_instrument.ordered_records()] == ["test.Outer"]


def test_direct_tilelang_lower_owns_an_instrumentation_session(tmp_path):
    """Programmatic enable traces direct lower() without mutating a global context."""
    from tilelang.tools.lower_trace import enable

    enable(mode="terminal", trace_dir=str(tmp_path), codegen_output=None)
    artifact = tilelang.lower(_simple_program(), target="c")

    assert artifact.kernel_source
    trace = _core.get_last_session()
    assert trace is not None
    assert any(record.phase == "pipeline_c" for record in trace.records)
    assert any(record.status == _core.STATUS_CODEGEN for record in trace.records)


def test_jit_compile_reuses_one_session_through_adapter_creation(monkeypatch, tmp_path):
    """The JIT lower/codegen/adapter path remains one logical compilation."""
    from tilelang.tools.lower_trace import enable

    monkeypatch.setenv("TILELANG_DISABLE_CACHE", "1")
    enable(mode="terminal", trace_dir=str(tmp_path), codegen_output=None)
    compiled = tilelang.compile(_simple_program(), target="c")

    assert compiled is not None
    trace = _core.get_last_session()
    assert trace is not None
    assert trace.pipeline_count == 1
    indices = [record.index for record in trace.records]
    assert indices == sorted(indices)
    assert len(indices) == len(set(indices))
    assert any(record.status == _core.STATUS_CODEGEN for record in trace.records)


def test_concurrent_sessions_serialize_a_shared_codegen_path(tmp_path):
    """An explicit shared edit path is transactional across compile sessions."""
    from concurrent.futures import ThreadPoolExecutor
    import threading

    output_path = tmp_path / "shared.cpp"
    barrier = threading.Barrier(2)
    source = "// identical generated source\n"

    def run_one(label):
        trace = _core.LowerTraceSession(
            mode="terminal",
            trace_dir=str(tmp_path / label),
            codegen_output=str(output_path),
        )

        def build(_mod):
            barrier.wait()
            return _MockCodegenModule(source)

        _run_codegen(trace, build, "target.build.tilelang_c")
        return trace

    with ThreadPoolExecutor(max_workers=2) as pool:
        left, right = pool.map(run_one, ("left", "right"))

    assert output_path.read_text() == source
    assert (tmp_path / "shared.cpp.original").read_text() == source
    assert (tmp_path / "shared.cpp.latest").read_text() == source
    assert [record.index for record in left.records] == [0]
    assert [record.index for record in right.records] == [0]


def test_backend_codegen_uses_explicit_session_hook(monkeypatch, tmp_path):
    """Backend registry codegen is observed without replacing a global FFI."""
    from tilelang.backend import device_codegen as device_codegen_registry
    from tilelang.utils.pass_events import compile_pass_instrumentation

    result_module = _MockCodegenModule("// explicit hook output\n")
    monkeypatch.setattr(
        device_codegen_registry.tvm.ffi,
        "get_global_func",
        lambda _name: lambda _mod, _target: result_module,
    )
    build = device_codegen_registry.global_func_device_codegen("target.build.test")
    trace = _core.LowerTraceSession(mode="terminal", trace_dir=str(tmp_path), codegen_output=None)

    with compile_pass_instrumentation(
        name="codegen-hook",
        tools=[trace],
        include_default_tools=False,
    ):
        assert build("fake_mod", "fake_target") is result_module

    assert [record.status for record in trace.records] == [_core.STATUS_CODEGEN]


@contextlib.contextmanager
def _patch_make_patched_source_module():
    """Patch _make_patched_source_module so tests avoid the real TVM C++ FFI."""
    from unittest.mock import patch

    with patch("tilelang.tools.lower_trace.core._make_patched_source_module", side_effect=_patched_module_factory) as mock:
        yield mock


def test_codegen_proxy_for_without_compile(tmp_path):
    """*_without_compile FFIs return a patched module when user edits codegen.cpp."""
    source_v1 = "// generated kernel v1\n"
    mock_build = _make_mock_build(source_v1)
    ffi_name = "target.build.tilelang_cuda_without_compile"
    trace = _make_trace_session(tmp_path)
    codegen_path = trace.get_codegen_output_path()
    assert codegen_path is not None

    with _patch_make_patched_source_module() as mock_factory:
        result1 = _run_codegen(trace, mock_build, ffi_name)
        assert result1.inspect_source() == source_v1
        edited = "// edited by user\n"
        with open(codegen_path, "w") as f:
            f.write(edited)

        result2 = _run_codegen(trace, mock_build, ffi_name)
        assert mock_factory.called, "_make_patched_source_module should be called for _without_compile FFI"
        assert result2.get_source() == edited, "Patched module should return the user-edited source"


def test_codegen_proxy_for_source_only_ffi(tmp_path):
    """Source-only FFIs without a _without_compile suffix (tilelang_c, webgpu) also return patched module."""
    source_v1 = "// generated C kernel v1\n"
    mock_build = _make_mock_build(source_v1)
    ffi_name = "target.build.tilelang_c"
    trace = _make_trace_session(tmp_path)
    codegen_path = trace.get_codegen_output_path()
    assert codegen_path is not None

    with _patch_make_patched_source_module() as mock_factory:
        _run_codegen(trace, mock_build, ffi_name)
        edited = "// edited C kernel\n"
        with open(codegen_path, "w") as f:
            f.write(edited)

        result2 = _run_codegen(trace, mock_build, ffi_name)
        assert mock_factory.called, "Expected patched module for source-only FFI tilelang_c"
        assert result2.get_source() == edited


def test_codegen_no_proxy_for_full_compile(tmp_path, capsys):
    """Full-compile FFIs return the real module (not patched) + NOTE when user edits codegen.cpp."""
    source_v1 = "// generated kernel v1\n"
    mock_build = _make_mock_build(source_v1)
    ffi_name = "target.build.tilelang_cuda"
    trace = _make_trace_session(tmp_path)
    codegen_path = trace.get_codegen_output_path()
    assert codegen_path is not None
    target = tvm.target.Target("cuda")

    with _patch_make_patched_source_module() as mock_factory:
        result1 = _run_codegen(trace, mock_build, ffi_name, target=target)
        assert result1.inspect_source() == source_v1
        edited = "// edited by user\n"
        with open(codegen_path, "w") as f:
            f.write(edited)

        capsys.readouterr()
        result2 = _run_codegen(trace, mock_build, ffi_name, target=target)
        assert not mock_factory.called
        assert result2.inspect_source() == source_v1
        captured = capsys.readouterr()
        assert "NOT recompiled" in captured.out
        assert "nvrtc" in captured.out


def test_codegen_conflict_backup(tmp_path):
    """CONFLICT: both user edited and codegen changed → backup + regenerate."""
    source_v1 = "// generated kernel v1\n"
    mock_build = _make_mock_build(source_v1)
    ffi_name = "target.build.tilelang_cuda_without_compile"
    trace = _make_trace_session(tmp_path)
    codegen_path = trace.get_codegen_output_path()
    assert codegen_path is not None
    original_path = codegen_path + ".original"

    with _patch_make_patched_source_module() as mock_factory:
        _run_codegen(trace, mock_build, ffi_name)
        with open(codegen_path, "w") as f:
            f.write("// user edit\n")

        source_v2 = "// new codegen output v2\n"
        result2 = _run_codegen(trace, _make_mock_build(source_v2), ffi_name)
        assert os.path.exists(codegen_path + ".bak"), "User working copy not backed up"
        assert os.path.exists(original_path + ".bak"), "Old baseline not backed up"
        with open(codegen_path) as f:
            assert f.read() == source_v2
        with open(original_path) as f:
            assert f.read() == source_v2
        assert not mock_factory.called, "CONFLICT must not call _make_patched_source_module"
        assert result2.inspect_source() == source_v2


def test_codegen_synced(tmp_path):
    """SYNCED: user edits match new codegen output → baseline advances, patched module returned."""
    source_v1 = "// generated kernel v1\n"
    mock_build = _make_mock_build(source_v1)
    ffi_name = "target.build.tilelang_cuda_without_compile"
    trace = _make_trace_session(tmp_path)
    codegen_path = trace.get_codegen_output_path()
    assert codegen_path is not None
    original_path = codegen_path + ".original"

    with _patch_make_patched_source_module() as mock_factory:
        _run_codegen(trace, mock_build, ffi_name)
        source_v2 = "// new codegen output v2\n"
        with open(codegen_path, "w") as f:
            f.write(source_v2)

        result2 = _run_codegen(trace, _make_mock_build(source_v2), ffi_name)
        with open(original_path) as f:
            assert f.read() == source_v2
        assert mock_factory.called, "SYNCED should call _make_patched_source_module for _without_compile FFI"
        assert result2.get_source() == source_v2


def test_codegen_phase_reset_on_inspect_source_failure(tmp_path):
    """The context-local phase must reset even if post-codegen tracing raises.

    Regression guard: previously an exception in the codegen post-processing
    (inspect_source / file I/O / diff) left the phase stuck at "codegen",
    misattributing later records.  The exception is now caught and warned
    (does not propagate), but the phase must still be restored.
    """

    class _ExplodingModule:
        def inspect_source(self):
            """Raise to exercise the post-codegen error-handling path."""
            raise RuntimeError("inspect_source blew up")

    def mock_build(*args, **kwargs):
        """Return an _ExplodingModule to trigger the inspect_source failure path."""
        return _ExplodingModule()

    trace = _make_trace_session(tmp_path)
    _run_codegen(trace, mock_build, "target.build.tilelang_cuda_without_compile")

    from tilelang.utils.pass_events import current_pass_phase

    assert current_pass_phase() is None, "phase must be reset after inspect_source failure"


def test_codegen_failure_is_recorded_in_the_owning_session(tmp_path):
    trace = _make_trace_session(tmp_path)

    def failing_build(_mod):
        raise RuntimeError("codegen failed")

    with pytest.raises(RuntimeError, match="codegen failed"):
        _run_codegen(trace, failing_build, "target.build.tilelang_c")

    assert len(trace.records) == 1
    assert trace.records[0].status == _core.STATUS_FAILED
    assert trace.records[0].phase == "codegen"
    assert trace.records[0].error_msg == "codegen failed"


def test_codegen_restores_outer_phase(tmp_path):
    """codegen nested in an active pipeline phase must restore it, not clear to None."""
    from tilelang.utils.pass_events import current_pass_phase, pass_phase

    source_v1 = "// generated kernel v1\n"
    trace = _make_trace_session(tmp_path)
    with pass_phase("pipeline_test"):
        _run_codegen(trace, _make_mock_build(source_v1), "target.build.tilelang_cuda_without_compile")
        assert current_pass_phase() == "pipeline_test", "outer phase must be restored after codegen"


def test_codegen_record_index_after_nested_pass(tmp_path):
    """codegen record index must come after any pass invoked inside original_build.

    Regression guard: pre-allocating the codegen idx before ``original_build`` ran
    let an internal traced pass (e.g. ``tir.transform.Simplify``) grab a later
    index, so records could appear as N+1 before N.  The idx is now allocated
    immediately before appending the codegen record.
    """
    from tilelang.tools.lower_trace.core import LowerRecord, STATUS_COMPLETED

    nested_idx = []
    trace = _make_trace_session(tmp_path)

    def mock_build(*args, **kwargs):
        """Append an internal traced pass then return a normal mock module."""
        nested_idx.append(trace.allocate_index())
        trace.append_record(
            LowerRecord(
                phase="codegen",
                name="internal_simplify",
                index=nested_idx[0],
                before_text="",
                after_text="",
                changed=False,
                add_lines=0,
                del_lines=0,
                status=STATUS_COMPLETED,
            )
        )
        return _MockCodegenModule("// generated\n")

    _run_codegen(trace, mock_build, "target.build.tilelang_cuda_without_compile")

    codegen_records = [record for record in trace.records if record.name == "codegen"]
    assert codegen_records, "no codegen record found"
    assert codegen_records[-1].index > nested_idx[0], "codegen index not after nested pass"
    indices = [record.index for record in trace.records]
    assert indices == sorted(indices), f"records not in ascending index order: {indices}"


def test_import_time_activation(tmp_path):
    """The env hook in tilelang/__init__.py must activate tracing on first import.

    This module clears TL_LOWER_TRACE before importing tilelang, so the rest of
    the suite never exercises the import-time activation path. Use a subprocess
    to set the env var *before* the first import and verify tracing is on.
    """
    import subprocess
    import sys

    env = dict(os.environ)
    env["TL_LOWER_TRACE"] = "1"
    env["TL_LOWER_TRACE_DIR"] = str(tmp_path)

    code = (
        "import tilelang\n"
        "from tilelang.tools.lower_trace import core as _core\n"
        "assert _core._is_trace_enabled(), 'tracing not enabled at import time'\n"
        "assert _core._get_mode() == 'html', 'expected html mode for TL_LOWER_TRACE=1'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"subprocess failed (rc={result.returncode}):\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"


if __name__ == "__main__":
    tilelang.testing.main()
