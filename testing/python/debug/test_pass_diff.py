# type: ignore
"""Tests for the pass_diff debugging feature.

Covers:
- Environment variable parsing (get_pass_diff_mode)
- Core diff computation (_compute_diff, _count_changes)
- Programmatic pass_diff() API (single pass, chain, named, HTML output)
- Hook install / uninstall lifecycle
"""

import os

import pytest

import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang import tvm
from tilelang.env import env


@pytest.fixture(autouse=True)
def _isolate_pass_diff_state(monkeypatch):
    try:
        from tilelang.utils import pass_diff_hook
    except ImportError:
        yield
        return

    monkeypatch.setenv("TILELANG_PASS_DIFF", "0")
    monkeypatch.delenv("TILELANG_PASS_DIFF_OUTPUT", raising=False)
    pass_diff_hook.uninstall_pass_diff_hook()
    yield
    pass_diff_hook.uninstall_pass_diff_hook()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_program():
    """A minimal PrimFunc that passes can transform."""

    @T.prim_func
    def program(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        with T.Kernel(threads=128):
            tid = T.get_thread_binding()
            B[tid] = A[tid] + 1.0

    return program


def _noop_pass():
    """A TVM pass that is unlikely to modify a simple IR."""
    return tvm.tirx.transform.Simplify()


def _transforming_pass():
    """A TVM pass that modifies the IR."""
    return tvm.tirx.transform.UnrollLoop()


# ---------------------------------------------------------------------------
# get_pass_diff_mode — env var parsing
# ---------------------------------------------------------------------------


class TestGetPassDiffMode:
    @pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
    def test_off_values(self, monkeypatch, value):
        monkeypatch.setenv("TILELANG_PASS_DIFF", value)
        assert env.get_pass_diff_mode() is None

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on", "terminal"])
    def test_terminal_values(self, monkeypatch, value):
        monkeypatch.setenv("TILELANG_PASS_DIFF", value)
        assert env.get_pass_diff_mode() == "terminal"

    def test_html(self, monkeypatch):
        monkeypatch.setenv("TILELANG_PASS_DIFF", "html")
        assert env.get_pass_diff_mode() == "html"

    def test_both(self, monkeypatch):
        monkeypatch.setenv("TILELANG_PASS_DIFF", "both")
        assert env.get_pass_diff_mode() == "both"

    def test_unknown_truthy_falls_back_to_terminal(self, monkeypatch):
        monkeypatch.setenv("TILELANG_PASS_DIFF", "something_random")
        assert env.get_pass_diff_mode() == "terminal"


# ---------------------------------------------------------------------------
# Core diff utilities
# ---------------------------------------------------------------------------


class TestComputeDiff:
    def test_no_changes(self):
        from tilelang.utils.pass_diff import _compute_diff

        lines = ["line 1", "line 2", "line 3"]
        diff = _compute_diff(lines, lines, context=3)
        assert diff == []

    def test_with_changes(self):
        from tilelang.utils.pass_diff import _compute_diff

        before = ["line 1", "line 2", "line 3"]
        after = ["line 1", "line 2 modified", "line 3"]
        diff = _compute_diff(before, after, context=3)
        assert len(diff) > 0
        # Should contain at least one deletion and one insertion
        assert any(line.startswith("-") and not line.startswith("---") for line in diff)
        assert any(line.startswith("+") and not line.startswith("+++") for line in diff)

    def test_addition_only(self):
        from tilelang.utils.pass_diff import _compute_diff

        before = ["line 1"]
        after = ["line 1", "line 2 new"]
        diff = _compute_diff(before, after, context=3)
        assert any(line.startswith("+") and not line.startswith("+++") for line in diff)

    def test_deletion_only(self):
        from tilelang.utils.pass_diff import _compute_diff

        before = ["line 1", "line 2"]
        after = ["line 1"]
        diff = _compute_diff(before, after, context=3)
        assert any(line.startswith("-") and not line.startswith("---") for line in diff)


class TestSplitRowsByHunk:
    def test_empty(self):
        from tilelang.utils.pass_diff import _split_rows_by_hunk

        assert _split_rows_by_hunk([]) == []

    def test_single_hunk(self):
        from tilelang.utils.pass_diff import _split_rows_by_hunk

        rows = [
            {"type": "hunk", "left": "@@ -1,3 +1,3 @@", "right": "", "left_num": None, "right_num": None},
            {"type": "del", "left": "old", "right": "", "left_num": 1, "right_num": None},
            {"type": "add", "left": "", "right": "new", "left_num": None, "right_num": 1},
        ]
        hunks = _split_rows_by_hunk(rows)
        assert len(hunks) == 1
        assert len(hunks[0]) == 3

    def test_multiple_hunks(self):
        from tilelang.utils.pass_diff import _split_rows_by_hunk

        rows = [
            {"type": "hunk", "left": "@@ -1 +1 @@", "right": "", "left_num": None, "right_num": None},
            {"type": "del", "left": "a", "right": "", "left_num": 1, "right_num": None},
            {"type": "hunk", "left": "@@ -10 +10 @@", "right": "", "left_num": None, "right_num": None},
            {"type": "add", "left": "", "right": "b", "left_num": None, "right_num": 10},
            {"type": "hunk", "left": "@@ -20 +20 @@", "right": "", "left_num": None, "right_num": None},
            {"type": "context", "left": "c", "right": "c", "left_num": 20, "right_num": 20},
        ]
        hunks = _split_rows_by_hunk(rows)
        assert len(hunks) == 3
        # Each hunk should start with its @@ header
        assert hunks[0][0]["left"] == "@@ -1 +1 @@"
        assert hunks[1][0]["left"] == "@@ -10 +10 @@"
        assert hunks[2][0]["left"] == "@@ -20 +20 @@"


class TestRenderExpandDetails:
    def test_empty(self):
        from tilelang.utils.pass_diff import _render_expand_details

        assert _render_expand_details([]) == {}

    def test_groups_by_hunk_idx(self):
        from tilelang.utils.pass_diff import _render_expand_details

        expand_sections = [
            {
                "hunk_idx": 0,
                "position": "before",
                "direction": "up",
                "count": 5,
                "lines": [{"left": "l1", "right": "l1", "left_num": 1, "right_num": 1}],
            },
            {
                "hunk_idx": 0,
                "position": "after",
                "direction": "down",
                "count": 3,
                "lines": [{"left": "l2", "right": "l2", "left_num": 10, "right_num": 10}],
            },
            {
                "hunk_idx": 1,
                "position": "before",
                "direction": "up",
                "count": 2,
                "lines": [{"left": "l3", "right": "l3", "left_num": 15, "right_num": 15}],
            },
            {
                "hunk_idx": 2,
                "position": "after",
                "direction": "down",
                "count": 4,
                "lines": [{"left": "l4", "right": "l4", "left_num": 25, "right_num": 25}],
            },
        ]
        result = _render_expand_details(expand_sections)
        assert set(result.keys()) == {0, 1, 2}

        # Hunk 0 has both before and after
        before_0, after_0 = result[0]
        assert "Show 5 lines above" in before_0
        assert "Show 3 lines below" in after_0

        # Hunk 1 has only before
        before_1, after_1 = result[1]
        assert "Show 2 lines above" in before_1
        assert after_1 == ""

        # Hunk 2 has only after
        before_2, after_2 = result[2]
        assert before_2 == ""
        assert "Show 4 lines below" in after_2


class TestCountChanges:
    def test_empty_diff(self):
        from tilelang.utils.pass_diff import _count_changes

        ins, dels = _count_changes([])
        assert ins == 0
        assert dels == 0

    def test_mixed_changes(self):
        from tilelang.utils.pass_diff import _count_changes

        diff = [
            "--- before",
            "+++ after",
            "@@ -1,3 +1,3 @@",
            " context",
            "-old line",
            "-another old",
            "+new line",
        ]
        ins, dels = _count_changes(diff)
        assert ins == 1
        assert dels == 2

    def test_header_lines_not_counted(self):
        from tilelang.utils.pass_diff import _count_changes

        diff = ["--- before", "+++ after", " context"]
        ins, dels = _count_changes(diff)
        assert ins == 0
        assert dels == 0


# ---------------------------------------------------------------------------
# pass_diff() — programmatic API
# ---------------------------------------------------------------------------


class TestPassDiffAPI:
    def test_single_pass_returns_results(self):
        from tilelang.utils.pass_diff import pass_diff

        program = _simple_program()
        results = pass_diff(program, _noop_pass(), mode="terminal")

        assert isinstance(results, list)
        assert len(results) == 1

        step = results[0]
        assert "name" in step
        assert "before_script" in step
        assert "after_script" in step
        assert "diff_lines" in step
        assert "insertions" in step
        assert "deletions" in step
        assert "changed" in step

    def test_pass_chain(self):
        from tilelang.utils.pass_diff import pass_diff

        program = _simple_program()
        passes = [
            ("Simplify", _noop_pass()),
            ("LoopPartition", _transforming_pass()),
        ]
        results = pass_diff(program, passes, mode="terminal")

        assert len(results) == 2
        assert results[0]["name"] == "Simplify"
        assert results[1]["name"] == "LoopPartition"

    def test_unnamed_passes(self):
        from tilelang.utils.pass_diff import pass_diff

        program = _simple_program()
        results = pass_diff(program, [_noop_pass(), _transforming_pass()], mode="terminal")

        assert len(results) == 2
        # Names should be derived from class names
        assert results[0]["name"]  # non-empty
        assert results[1]["name"]  # non-empty

    def test_html_output(self, tmp_path):
        from tilelang.utils.pass_diff import pass_diff

        program = _simple_program()
        html_file = str(tmp_path / "test_report.html")

        pass_diff(program, _noop_pass(), mode="html", html_path=html_file)

        assert os.path.isfile(html_file)

        with open(html_file) as f:
            content = f.read()
        # Should contain basic HTML structure
        assert "<!DOCTYPE html>" in content
        assert "Pass Diff Report" in content
        # Should contain at least one pass section
        assert "pass-section" in content

    def test_both_mode(self, tmp_path, capsys):
        from tilelang.utils.pass_diff import pass_diff

        program = _simple_program()
        html_file = str(tmp_path / "test_both.html")

        pass_diff(program, _noop_pass(), mode="both", html_path=html_file)

        # Terminal output should have been printed
        captured = capsys.readouterr()
        assert "Pass 1" in captured.out

        # HTML file should exist
        assert os.path.isfile(html_file)

    def test_unchanged_pass_reports_no_changes(self):
        from tilelang.utils.pass_diff import pass_diff

        program = _simple_program()
        # Simplify on a simple program should produce no diff
        results = pass_diff(program, _noop_pass(), mode="terminal")

        step = results[0]
        assert step["changed"] is False
        assert step["insertions"] == 0
        assert step["deletions"] == 0

    def test_changed_pass_detected(self):
        from tilelang.utils.pass_diff import pass_diff

        program = _shared_rw_program()
        results = pass_diff(program, tilelang.transform.ThreadSync("shared"), mode="terminal")

        step = results[0]
        assert step["changed"] is True
        assert step["insertions"] > 0
        assert "tvm_storage_sync" in step["after_script"]

    def test_context_parameter(self):
        from tilelang.utils.pass_diff import pass_diff

        program = _simple_program()
        # With context=0, only changed lines should appear
        results_c0 = pass_diff(program, _transforming_pass(), mode="terminal", context=0)
        # With context=10, more surrounding lines
        results_c10 = pass_diff(program, _transforming_pass(), mode="terminal", context=10)

        # More context → more or equal diff lines
        assert len(results_c10[0]["diff_lines"]) >= len(results_c0[0]["diff_lines"])


# ---------------------------------------------------------------------------
# HTML report content validation
# ---------------------------------------------------------------------------


class TestHTMLReport:
    def test_html_contains_toolbar(self, tmp_path):
        from tilelang.utils.pass_diff import pass_diff

        program = _simple_program()
        html_file = str(tmp_path / "toolbar_test.html")
        pass_diff(program, _noop_pass(), mode="html", html_path=html_file)

        with open(html_file) as f:
            content = f.read()
        assert "toolbar" in content
        assert "Expand All" in content
        assert "Collapse All" in content

    def test_html_contains_theme_toggle(self, tmp_path):
        from tilelang.utils.pass_diff import pass_diff

        program = _simple_program()
        html_file = str(tmp_path / "theme_test.html")
        pass_diff(program, _noop_pass(), mode="html", html_path=html_file)

        with open(html_file) as f:
            content = f.read()
        assert "theme-btn" in content
        assert "Light" in content

    def test_html_multiple_passes(self, tmp_path):
        from tilelang.utils.pass_diff import pass_diff

        program = _simple_program()
        html_file = str(tmp_path / "multi_pass.html")
        passes = [
            ("Pass_A", _noop_pass()),
            ("Pass_B", _transforming_pass()),
        ]
        pass_diff(program, passes, mode="html", html_path=html_file)

        with open(html_file) as f:
            content = f.read()
        assert "Pass_A" in content
        assert "Pass_B" in content
        assert "2 passes" in content or "passes" in content

    def test_html_statistics(self, tmp_path):
        from tilelang.utils.pass_diff import pass_diff

        program = _simple_program()
        html_file = str(tmp_path / "stats_test.html")
        passes = [
            ("Simplify", _noop_pass()),
            ("LoopPartition", _transforming_pass()),
        ]
        pass_diff(program, passes, mode="html", html_path=html_file)

        with open(html_file) as f:
            content = f.read()
        # Toolbar should show pass count
        assert "passes" in content

    def test_html_multi_hunk_expand_ordering(self, tmp_path):
        """Verify expand blocks are interleaved per-hunk, not clustered."""
        from tilelang.utils.pass_diff import _generate_html
        import difflib

        # Construct a 50-line script with 3 well-separated changes to produce 3 hunks
        before_lines = [f"line_{i:03d}" for i in range(1, 51)]
        after_lines = list(before_lines)
        after_lines[4] = "CHANGED_005"
        after_lines[24] = "CHANGED_025"
        after_lines[44] = "CHANGED_045"

        diff_lines = list(difflib.unified_diff(before_lines, after_lines, lineterm="", n=3))

        step = {
            "name": "MultiHunk",
            "before_script": "\n".join(before_lines),
            "after_script": "\n".join(after_lines),
            "diff_lines": diff_lines,
            "insertions": 3,
            "deletions": 3,
            "changed": True,
        }

        html_file = str(tmp_path / "hunk_order.html")
        _generate_html([step], html_file)

        with open(html_file) as f:
            content = f.read()

        import re

        expand_positions = [m.start() for m in re.finditer(r'<details class="expand-details">', content)]
        table_positions = [m.start() for m in re.finditer(r'<table class="diff-table">', content)]

        assert len(expand_positions) >= 2, f"Expected >=2 expand blocks, got {len(expand_positions)}"
        assert len(table_positions) >= 4, f"Expected >=4 table blocks, got {len(table_positions)}"

        # Key assertion: some expand blocks should be BETWEEN tables (not all clustered)
        first_table = min(table_positions)
        last_table = max(table_positions)
        between_expands = [p for p in expand_positions if first_table < p < last_table]
        assert len(between_expands) > 0, f"Expand blocks clustered, not interleaved. Expand: {expand_positions}, Table: {table_positions}"


# ---------------------------------------------------------------------------
# ThreadSync (auto-insert sync) pass via pass_diff
# ---------------------------------------------------------------------------


def _shared_rw_program():
    """A low-level TIR program with a shared-memory write→read dependency.

    Thread X writes to shared buffer B, then reads from B with a different
    index pattern.  ThreadSync("shared") should insert tvm_storage_sync
    between the write and the read.
    """
    from tvm.script import tirx as TS

    @TS.prim_func(private=True)
    def func(A: TS.Buffer((16,), "float32"), E: TS.Buffer((16,), "float32")):
        _block = TS.launch_thread("blockIdx.x", 1)
        B = TS.alloc_buffer((16,), "float32", scope="shared")
        C = TS.alloc_buffer((1,), "float32", scope="local")
        threadIdx_x = TS.launch_thread("threadIdx.x", 16)
        _ty = TS.launch_thread("threadIdx.y", 1)
        _tz = TS.launch_thread("threadIdx.z", 1)
        B_1 = TS.decl_buffer((16,), data=B.data, scope="shared")
        A_1 = TS.decl_buffer((16,), data=A.data)
        # Write: each thread writes to a distinct location
        B_1[threadIdx_x] = A_1[threadIdx_x]
        C_1 = TS.decl_buffer((1,), data=C.data, scope="local")
        # Read: each thread reads from a DIFFERENT location (shifted)
        # This creates a cross-thread dependency → sync needed
        C_1[0] = B_1[(threadIdx_x + 1) % 16]
        E_1 = TS.decl_buffer((16,), data=E.data)
        E_1[threadIdx_x] = C_1[0]

    return func


def _atomic_no_sync_program():
    """A TIR program with only atomic writes — no sync should be inserted."""
    from tvm.script import tirx as TS

    @TS.prim_func(private=True)
    def func():
        A_shared = TS.alloc_buffer((128,), dtype="float32", scope="shared")
        tx = TS.launch_thread("threadIdx.x", 128)
        _ty = TS.launch_thread("threadIdx.y", 1)
        _tz = TS.launch_thread("threadIdx.z", 1)
        TS.evaluate(
            TS.call_intrin(
                "float32",
                tvm.tirx.op.Op.get("tl.atomic_add_elem_op"),
                TS.tvm_access_ptr(
                    TS.type_annotation("float32"),
                    A_shared.data,
                    tx,
                    1,
                    3,
                ),
                TS.float32(1),
                TS.int32(0),
            )
        )

    return func


class TestThreadSyncPassDiff:
    """Verify pass_diff works with tilelang.transform.ThreadSync.

    ThreadSync is a pure IR transformation pass — it does not require a GPU.
    """

    def test_thread_sync_detects_changes(self):
        """ThreadSync should modify the IR when shared-memory dependencies exist."""
        from tilelang.utils.pass_diff import pass_diff

        func = _shared_rw_program()
        results = pass_diff(func, tilelang.transform.ThreadSync("shared"), mode="terminal")

        assert len(results) == 1
        step = results[0]
        assert step["changed"] is True
        assert step["insertions"] > 0
        # The inserted line should contain tvm_storage_sync
        inserted_lines = [line for line in step["diff_lines"] if line.startswith("+") and not line.startswith("+++")]
        assert any("tvm_storage_sync" in line for line in inserted_lines), f"Expected tvm_storage_sync in insertions, got: {inserted_lines}"

    def test_thread_sync_after_script_contains_sync(self):
        """The after_script should contain the inserted tvm_storage_sync call."""
        from tilelang.utils.pass_diff import pass_diff

        func = _shared_rw_program()
        results = pass_diff(func, tilelang.transform.ThreadSync("shared"), mode="terminal")

        step = results[0]
        assert "tvm_storage_sync" in step["after_script"]
        # The before_script should NOT have it
        assert "tvm_storage_sync" not in step["before_script"]

    def test_thread_sync_no_sync_for_atomics(self):
        """Atomic WAW/RMW should not trigger sync insertion — pass_diff reports no change."""
        from tilelang.utils.pass_diff import pass_diff

        func = _atomic_no_sync_program()
        results = pass_diff(func, tilelang.transform.ThreadSync("shared"), mode="terminal")

        step = results[0]
        assert step["changed"] is False
        assert step["insertions"] == 0
        assert step["deletions"] == 0
        assert "tvm_storage_sync" not in step["after_script"]

    def test_thread_sync_named_pass(self):
        """ThreadSync in a named (label, pass) tuple should appear with the given name."""
        from tilelang.utils.pass_diff import pass_diff

        func = _shared_rw_program()
        results = pass_diff(
            func,
            [("InsertSharedSync", tilelang.transform.ThreadSync("shared"))],
            mode="terminal",
        )

        assert results[0]["name"] == "InsertSharedSync"
        assert results[0]["changed"] is True

    def test_thread_sync_pass_chain(self):
        """Run a realistic pass chain: AnnotateDeviceRegions → SplitHostDevice → ThreadSync."""
        from tilelang.utils.pass_diff import pass_diff
        from tvm.script import tirx as TS

        @TS.prim_func(private=True)
        def func(A: TS.Buffer((16,), "float32"), E: TS.Buffer((16,), "float32")):
            _block = TS.launch_thread("blockIdx.x", 1)
            B = TS.alloc_buffer((16,), "float32", scope="shared")
            C = TS.alloc_buffer((1,), "float32", scope="local")
            threadIdx_x = TS.launch_thread("threadIdx.x", 16)
            _ty = TS.launch_thread("threadIdx.y", 1)
            _tz = TS.launch_thread("threadIdx.z", 1)
            B_1 = TS.decl_buffer((16,), data=B.data, scope="shared")
            A_1 = TS.decl_buffer((16,), data=A.data)
            B_1[threadIdx_x] = A_1[threadIdx_x]
            C_1 = TS.decl_buffer((1,), data=C.data, scope="local")
            C_1[0] = B_1[(threadIdx_x + 1) % 16]
            E_1 = TS.decl_buffer((16,), data=E.data)
            E_1[threadIdx_x] = C_1[0]

        cuda_target = tvm.target.Target("cuda", host="llvm")
        mod = tvm.IRModule({"main": func})
        mod = tvm.tirx.transform.Apply(lambda f: f.with_attr({"global_symbol": "test", "target": cuda_target}))(mod)

        passes = [
            ("AnnotateDeviceRegions", tvm.tirx.transform.AnnotateDeviceRegions()),
            ("SplitHostDevice", tvm.tirx.transform.SplitHostDevice()),
            ("ThreadSync", tilelang.transform.ThreadSync("shared")),
        ]
        results = pass_diff(mod, passes, mode="terminal")

        assert len(results) == 3
        assert results[0]["name"] == "AnnotateDeviceRegions"
        assert results[1]["name"] == "SplitHostDevice"
        assert results[2]["name"] == "ThreadSync"
        # ThreadSync is the one that should produce changes
        assert results[2]["changed"] is True
        assert "tvm_storage_sync" in results[2]["after_script"]

    def test_thread_sync_html_report(self, tmp_path):
        """HTML report from ThreadSync should contain sync-related diff content."""
        from tilelang.utils.pass_diff import pass_diff

        func = _shared_rw_program()
        html_file = str(tmp_path / "thread_sync_report.html")

        results = pass_diff(
            func,
            tilelang.transform.ThreadSync("shared"),
            mode="html",
            html_path=html_file,
        )

        assert os.path.isfile(html_file)

        with open(html_file) as f:
            content = f.read()

        # The HTML should reference the sync insertion
        assert "tvm_storage_sync" in content
        # Should have at least one insertion (the sync call)
        assert results[0]["insertions"] > 0

    def test_thread_sync_shared_dyn(self):
        """ThreadSync('shared.dyn') should also work via pass_diff."""
        from tilelang.utils.pass_diff import pass_diff
        from tvm.script import tirx as TS

        @TS.prim_func(private=True)
        def func():
            buf_dyn = TS.alloc_buffer((256,), "float32", scope="shared.dyn")
            local_buf = TS.alloc_buffer((4,), "float32", scope="local")
            tx = TS.launch_thread("threadIdx.x", 128)
            _ty = TS.launch_thread("threadIdx.y", 1)
            _tz = TS.launch_thread("threadIdx.z", 1)
            buf_dyn_1 = TS.decl_buffer((256,), data=buf_dyn.data, scope="shared.dyn")
            local_1 = TS.decl_buffer((4,), data=local_buf.data, scope="local")
            # Write to shared.dyn
            buf_dyn_1[tx] = TS.float32(1.0)
            # Read from a different offset → needs sync
            local_1[0] = buf_dyn_1[(tx + 1) % 256]

        results = pass_diff(func, tilelang.transform.ThreadSync("shared.dyn"), mode="terminal")

        step = results[0]
        assert step["changed"] is True
        assert 'tvm_storage_sync("shared.dyn")' in step["after_script"]


# ---------------------------------------------------------------------------
# Hook install / uninstall lifecycle
# ---------------------------------------------------------------------------


class TestHookLifecycle:
    def test_hook_disabled_by_default(self, monkeypatch):
        from tilelang.utils import pass_diff_hook

        # Ensure clean state
        pass_diff_hook.uninstall_pass_diff_hook()
        monkeypatch.setenv("TILELANG_PASS_DIFF", "0")

        pass_diff_hook.install_pass_diff_hook()
        assert pass_diff_hook._original_call is None

        # Cleanup
        pass_diff_hook.uninstall_pass_diff_hook()

    def test_hook_install_and_uninstall(self, monkeypatch):
        from tilelang.utils import pass_diff_hook

        # Ensure clean state
        pass_diff_hook.uninstall_pass_diff_hook()
        monkeypatch.setenv("TILELANG_PASS_DIFF", "terminal")

        pass_diff_hook.install_pass_diff_hook()
        assert pass_diff_hook._original_call is not None

        # Install again should be idempotent
        original_call_ref = pass_diff_hook._original_call
        pass_diff_hook.install_pass_diff_hook()
        assert pass_diff_hook._original_call is original_call_ref

        # Uninstall
        pass_diff_hook.uninstall_pass_diff_hook()
        assert pass_diff_hook._original_call is None

    def test_hook_html_creates_output_dir(self, monkeypatch, tmp_path):
        from tilelang.utils import pass_diff_hook

        # Ensure clean state
        pass_diff_hook.uninstall_pass_diff_hook()

        output_dir = str(tmp_path / "custom_output")
        monkeypatch.setenv("TILELANG_PASS_DIFF", "html")
        monkeypatch.setenv("TILELANG_PASS_DIFF_OUTPUT", output_dir)

        pass_diff_hook.install_pass_diff_hook()

        # Output directory should be created
        assert os.path.isdir(output_dir)

        # HTML path should be set
        assert pass_diff_hook._html_path is not None
        assert pass_diff_hook._html_path.startswith(output_dir)
        assert "pass_diff_" in pass_diff_hook._html_path

        # Cleanup
        pass_diff_hook.uninstall_pass_diff_hook()

    def test_hook_uninstall_when_not_installed(self):
        from tilelang.utils import pass_diff_hook

        # Should not raise
        pass_diff_hook.uninstall_pass_diff_hook()
        assert pass_diff_hook._original_call is None


# ---------------------------------------------------------------------------
# Env var TILELANG_PASS_DIFF_OUTPUT
# ---------------------------------------------------------------------------


class TestPassDiffOutput:
    def test_default_output_dir(self, monkeypatch):
        monkeypatch.delenv("TILELANG_PASS_DIFF_OUTPUT", raising=False)
        assert str(env.TILELANG_PASS_DIFF_OUTPUT) == "tmp/pass_diff_output"

    def test_custom_output_dir(self, monkeypatch):
        monkeypatch.setenv("TILELANG_PASS_DIFF_OUTPUT", "my/custom/dir")
        assert str(env.TILELANG_PASS_DIFF_OUTPUT) == "my/custom/dir"


if __name__ == "__main__":
    tilelang.testing.main()
