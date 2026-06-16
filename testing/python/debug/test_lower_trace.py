# type: ignore
"""Tests for the lower_trace debugging feature."""

import os
import pytest
import tempfile

# Clear any inherited TL_LOWER_TRACE* env before importing tilelang so the
# import-time activation hook in ``tilelang/__init__.py`` cannot fire during
# pytest collection. The autouse ``_isolate_env`` fixture below handles
# per-test isolation; this guard prevents leaks across collected modules.
os.environ.pop("TL_LOWER_TRACE", None)
os.environ.pop("TL_LOWER_TRACE_DIR", None)

import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang import tvm
from tilelang.tools.lower_trace import core as _core


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


def test_multi_run_accumulation(monkeypatch, tmp_path):
    from tilelang.tools.lower_trace import enable, disable
    from tilelang.tools.lower_trace import core as _core
    from tilelang.backend.pass_pipeline import resolve_pipeline
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
    from tilelang.backend.pass_pipeline import resolve_pipeline
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


# ---------------------------------------------------------------------------
# Codegen edit-and-recompile (Phase 1: _CodegenSourceProxy for _without_compile)
# ---------------------------------------------------------------------------


class _MockCodegenModule:
    """Minimal stand-in for a TVM runtime.Module returned by codegen FFIs."""

    def __init__(self, source: str):
        self._source = source

    def inspect_source(self) -> str:
        return self._source


def _make_mock_build(source: str):
    """Return a mock codegen FFI that always produces *source*."""

    def mock_build(*args, **kwargs):
        return _MockCodegenModule(source)

    return mock_build


def _setup_trace_overrides(tmp_path, mode="terminal"):
    """Set lower_trace overrides for unit testing.

    The autouse ``_isolate_env`` fixture calls ``disable()`` before each test,
    so overrides can be set safely inside the test body.
    """
    _core._mode_override = mode
    _core._trace_dir_override = str(tmp_path)
    _core._codegen_output_path_override = str(tmp_path / "codegen.cpp")
    _core.reset()


def _clear_trace_overrides():
    """Reset overrides (also done by the autouse fixture's ``disable()``)."""
    _core._mode_override = _core._UNSET
    _core._trace_dir_override = _core._UNSET
    _core._codegen_output_path_override = _core._UNSET
    _core.reset()


def test_codegen_proxy_for_without_compile(tmp_path):
    """*_without_compile FFIs return _CodegenSourceProxy when user edits codegen.cpp."""
    from tilelang.tools.lower_trace.core import _wrap_codegen_ffi, _CodegenSourceProxy

    source_v1 = "// generated kernel v1\n"
    mock_build = _make_mock_build(source_v1)
    wrapper = _wrap_codegen_ffi(mock_build, "target.build.tilelang_cuda_without_compile")

    _setup_trace_overrides(tmp_path)
    codegen_path = _core._codegen_output_path_override

    try:
        # Run 1: initializes codegen.cpp + .original from codegen output
        result1 = wrapper("fake_mod")
        assert result1.inspect_source() == source_v1

        # Edit codegen.cpp (user edit)
        edited = "// edited by user\n"
        with open(codegen_path, "w") as f:
            f.write(edited)

        # Run 2: user edited, codegen unchanged → PATCHED → proxy returned
        result2 = wrapper("fake_mod")
        assert isinstance(result2, _CodegenSourceProxy), "Expected _CodegenSourceProxy for _without_compile FFI"
        assert result2.inspect_source() == edited, "Proxy should return the user-edited source"
    finally:
        _clear_trace_overrides()


def test_codegen_proxy_for_source_only_ffi(tmp_path):
    """Source-only FFIs without a _without_compile suffix (tilelang_c, webgpu) also return proxy."""
    from tilelang.tools.lower_trace.core import _wrap_codegen_ffi, _CodegenSourceProxy

    source_v1 = "// generated C kernel v1\n"
    mock_build = _make_mock_build(source_v1)
    wrapper = _wrap_codegen_ffi(mock_build, "target.build.tilelang_c")

    _setup_trace_overrides(tmp_path)
    codegen_path = _core._codegen_output_path_override

    try:
        # Run 1: initializes codegen.cpp + .original
        wrapper("fake_mod")

        # Edit codegen.cpp
        edited = "// edited C kernel\n"
        with open(codegen_path, "w") as f:
            f.write(edited)

        # Run 2: PATCHED → proxy returned (tilelang_c is in _SOURCE_ONLY_CODEGEN_FFIS)
        result2 = wrapper("fake_mod")
        assert isinstance(result2, _CodegenSourceProxy), "Expected _CodegenSourceProxy for source-only FFI tilelang_c"
        assert result2.inspect_source() == edited
    finally:
        _clear_trace_overrides()


def test_codegen_no_proxy_for_full_compile(tmp_path, capsys):
    """Full-compile FFIs return the real module (not proxy) + NOTE when user edits codegen.cpp."""
    from tilelang.tools.lower_trace.core import _wrap_codegen_ffi, _CodegenSourceProxy

    source_v1 = "// generated kernel v1\n"
    mock_build = _make_mock_build(source_v1)
    wrapper = _wrap_codegen_ffi(mock_build, "target.build.tilelang_cuda")

    _setup_trace_overrides(tmp_path)
    codegen_path = _core._codegen_output_path_override
    target = tvm.target.Target("cuda")

    try:
        # Run 1: initializes codegen.cpp + .original
        result1 = wrapper("fake_mod", target)
        assert result1.inspect_source() == source_v1

        # Edit codegen.cpp
        edited = "// edited by user\n"
        with open(codegen_path, "w") as f:
            f.write(edited)

        # Run 2: user edited, codegen unchanged → PATCHED
        # But full-compile FFI → return real module, NOT proxy
        capsys.readouterr()  # clear prior output
        result2 = wrapper("fake_mod", target)
        assert not isinstance(result2, _CodegenSourceProxy), "Full-compile FFI must NOT return proxy (would crash tvm_ffi backend)"
        assert result2.inspect_source() == source_v1, "Full-compile FFI should return original (unpatched) module"

        captured = capsys.readouterr()
        assert "NOT recompiled" in captured.out
        assert "nvrtc" in captured.out  # backend hint
    finally:
        _clear_trace_overrides()


def test_codegen_conflict_backup(tmp_path):
    """CONFLICT: both user edited and codegen changed → backup + regenerate."""
    from tilelang.tools.lower_trace.core import _wrap_codegen_ffi

    source_v1 = "// generated kernel v1\n"
    mock_build = _make_mock_build(source_v1)
    wrapper = _wrap_codegen_ffi(mock_build, "target.build.tilelang_cuda_without_compile")

    _setup_trace_overrides(tmp_path)
    codegen_path = _core._codegen_output_path_override
    original_path = codegen_path + ".original"

    try:
        # Run 1: init
        wrapper("fake_mod")

        # Edit codegen.cpp (user edit)
        with open(codegen_path, "w") as f:
            f.write("// user edit\n")

        # Change codegen output (new wrapper with different source)
        source_v2 = "// new codegen output v2\n"
        wrapper = _wrap_codegen_ffi(_make_mock_build(source_v2), "target.build.tilelang_cuda_without_compile")

        # Run 2: CONFLICT — working != current
        result2 = wrapper("fake_mod")

        # .bak files created
        assert os.path.exists(codegen_path + ".bak"), "User working copy not backed up"
        assert os.path.exists(original_path + ".bak"), "Old baseline not backed up"

        # codegen.cpp regenerated from new codegen
        with open(codegen_path) as f:
            assert f.read() == source_v2

        # .original advanced to new codegen
        with open(original_path) as f:
            assert f.read() == source_v2

        # No proxy returned (regenerated from new codegen, patched_text=None)
        assert result2.inspect_source() == source_v2
    finally:
        _clear_trace_overrides()


def test_codegen_synced(tmp_path):
    """SYNCED: user edits match new codegen output → baseline advances, proxy returned."""
    from tilelang.tools.lower_trace.core import _wrap_codegen_ffi, _CodegenSourceProxy

    source_v1 = "// generated kernel v1\n"
    mock_build = _make_mock_build(source_v1)
    wrapper = _wrap_codegen_ffi(mock_build, "target.build.tilelang_cuda_without_compile")

    _setup_trace_overrides(tmp_path)
    codegen_path = _core._codegen_output_path_override
    original_path = codegen_path + ".original"

    try:
        # Run 1: init
        wrapper("fake_mod")

        # Edit codegen.cpp to match what the new codegen will produce
        source_v2 = "// new codegen output v2\n"
        with open(codegen_path, "w") as f:
            f.write(source_v2)

        # Change codegen output to the same value
        wrapper = _wrap_codegen_ffi(_make_mock_build(source_v2), "target.build.tilelang_cuda_without_compile")

        # Run 2: SYNCED
        result2 = wrapper("fake_mod")

        # .original advanced to new codegen
        with open(original_path) as f:
            assert f.read() == source_v2

        # Proxy returned (patched_text = working_text = source_v2)
        assert isinstance(result2, _CodegenSourceProxy), "SYNCED should return proxy for _without_compile FFI"
        assert result2.inspect_source() == source_v2
    finally:
        _clear_trace_overrides()


def test_codegen_phase_reset_on_inspect_source_failure(tmp_path):
    """_current_phase must be reset to None even if inspect_source() raises.

    Regression guard: previously an exception in the codegen post-processing
    (inspect_source / file I/O / diff) left _current_phase stuck at "codegen",
    misattributing later records.
    """
    from tilelang.tools.lower_trace.core import _wrap_codegen_ffi

    class _ExplodingModule:
        def inspect_source(self):
            raise RuntimeError("inspect_source blew up")

    def mock_build(*args, **kwargs):
        return _ExplodingModule()

    wrapper = _wrap_codegen_ffi(mock_build, "target.build.tilelang_cuda_without_compile")

    _setup_trace_overrides(tmp_path)
    try:
        import pytest

        with pytest.raises(RuntimeError, match="inspect_source blew up"):
            wrapper("fake_mod")

        assert _core._current_phase is None, "_current_phase must be reset after inspect_source failure"
    finally:
        _clear_trace_overrides()


def test_codegen_restores_outer_phase(tmp_path):
    """codegen nested in an active pipeline phase must restore it, not clear to None."""
    from tilelang.tools.lower_trace.core import _wrap_codegen_ffi

    source_v1 = "// generated kernel v1\n"
    wrapper = _wrap_codegen_ffi(_make_mock_build(source_v1), "target.build.tilelang_cuda_without_compile")

    _setup_trace_overrides(tmp_path)
    try:
        _core._current_phase = "pipeline_test"
        wrapper("fake_mod")
        assert _core._current_phase == "pipeline_test", "outer phase must be restored after codegen"
    finally:
        _core._current_phase = None
        _clear_trace_overrides()


def test_codegen_record_index_after_nested_pass(tmp_path):
    """codegen record index must come after any pass invoked inside original_build.

    Regression guard: pre-allocating the codegen idx before ``original_build`` ran
    let an internal traced pass (e.g. ``tir.transform.Simplify``) grab a later
    index, so records could appear as N+1 before N.  The idx is now allocated
    immediately before appending the codegen record.
    """
    from tilelang.tools.lower_trace.core import _wrap_codegen_ffi, LowerRecord, STATUS_COMPLETED

    nested_idx = []

    def mock_build(*args, **kwargs):
        with _core._lock:
            nested_idx.append(_core._pass_index)
            _core._pass_index += 1
            _core._records.append(
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

    wrapper = _wrap_codegen_ffi(mock_build, "target.build.tilelang_cuda_without_compile")

    _setup_trace_overrides(tmp_path)
    try:
        wrapper("fake_mod")

        codegen_records = [r for r in _core._records if r.name == "codegen"]
        assert codegen_records, "no codegen record found"
        assert codegen_records[-1].index > nested_idx[0], "codegen index not after nested pass"

        indices = [r.index for r in _core._records]
        assert indices == sorted(indices), f"records not in ascending index order: {indices}"
    finally:
        _clear_trace_overrides()


if __name__ == "__main__":
    tilelang.testing.main()
