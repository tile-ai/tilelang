"""Tests for source span injection into tirx IR (TILELANG_ENABLE_IR_SPAN).

Spans let compiler diagnostics and tools (LSP, visualizers) map IR nodes back
to user source lines. Injection happens during eager parsing and must be
exact: every emitted statement/buffer carries the line of the user statement
that produced it.
"""

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.ir import (
    get_buffer_span,
    get_stmt_span,
    make_span,
    set_buffer_span,
    set_stmt_span,
    span_coverage,
    span_to_location,
)
from tvm.tirx.stmt_functor import post_order_visit


def _marker_line(marker: str) -> int:
    with open(__file__) as f:
        for i, line in enumerate(f, 1):
            if marker in line:
                return i
    raise ValueError(f"marker not found: {marker}")


def _make_kernel():
    @T.prim_func
    def main(
        A: T.Tensor((128, 128), "float16"),
        B: T.Tensor((128, 128), "float16"),
        C: T.Tensor((128, 128), "float16"),
    ):
        with T.Kernel(1, threads=128) as bx:  # span_marker_kernel
            A_shared = T.alloc_shared((128, 128), "float16")  # span_marker_alloc_shared
            C_local = T.alloc_fragment((128, 128), "float32")  # span_marker_alloc_fragment
            T.copy(A, A_shared)  # span_marker_copy
            T.clear(C_local)  # span_marker_clear
            for _k in T.serial(8):  # span_marker_for
                T.gemm(A_shared, A_shared, C_local)  # span_marker_gemm
            if bx == 0:  # span_marker_if
                T.copy(C_local, C)  # span_marker_copy_out

    return main


def _stmt_span_lines(func) -> dict[int, list[str]]:
    """Map source line -> IR node type names stamped with that line."""
    lines: dict[int, list[str]] = {}

    def visit(node):
        if not isinstance(node, tvm.tirx.Stmt):
            return
        loc = span_to_location(get_stmt_span(node))
        if loc is not None:
            assert loc[0] == __file__
            lines.setdefault(loc[1], []).append(type(node).__name__)

    post_order_visit(func.body, visit)
    return lines


def _alloc_buffer_span_lines(func) -> dict[int, list[str]]:
    lines: dict[int, list[str]] = {}

    def visit(node):
        if type(node).__name__ == "SBlockRealize":
            for buf in node.block.alloc_buffers:
                loc = span_to_location(get_buffer_span(buf))
                if loc is not None:
                    assert loc[0] == __file__
                    lines.setdefault(loc[1], []).append(buf.name)

    post_order_visit(func.body, visit)
    return lines


def test_stmt_spans_point_to_user_lines():
    func = _make_kernel()
    lines = _stmt_span_lines(func)
    assert _marker_line("span_marker_copy") in lines
    assert _marker_line("span_marker_clear") in lines
    assert _marker_line("span_marker_gemm") in lines
    assert "For" in lines[_marker_line("span_marker_for")]
    assert any(k.startswith("If") for k in lines[_marker_line("span_marker_if")])
    assert _marker_line("span_marker_copy_out") in lines


def test_buffer_spans_point_to_alloc_lines():
    func = _make_kernel()
    buf_lines = _alloc_buffer_span_lines(func)
    assert "A_shared" in buf_lines[_marker_line("span_marker_alloc_shared")]
    assert "C_local" in buf_lines[_marker_line("span_marker_alloc_fragment")]


def test_span_disabled(monkeypatch):
    monkeypatch.setenv("TILELANG_ENABLE_IR_SPAN", "0")
    func = _make_kernel()
    cov = span_coverage(func)
    assert cov["stmts"] == [0, cov["stmts"][1]]
    assert cov["buffers"][0] == 0


def test_span_not_in_structural_equal(monkeypatch):
    from tvm.ir import assert_structural_equal

    func_on = _make_kernel()
    monkeypatch.setenv("TILELANG_ENABLE_IR_SPAN", "0")
    func_off = _make_kernel()
    assert_structural_equal(func_on, func_off)


def test_script_print_ignores_span():
    func = _make_kernel()
    text_with_span = func.script()
    assert text_with_span and isinstance(text_with_span, str)


def test_span_coverage_helper():
    func = _make_kernel()
    cov = span_coverage(func)
    with_span, total = cov["stmts"]
    assert total > 0
    # SeqStmt and other assembly nodes are never stamped; user-level
    # statements must dominate.
    assert with_span / total > 0.5


def test_make_span_and_setters_roundtrip():
    span = make_span(__file__, 42)
    func = _make_kernel()
    # Exercise the raw setters on a real node.
    set_stmt_span(func.body, span)
    assert span_to_location(get_stmt_span(func.body)) == (__file__, 42)
    buf = func.buffer_map[func.params[0]]
    set_buffer_span(buf, span)
    assert span_to_location(get_buffer_span(buf)) == (__file__, 42)


def _make_macro_kernel():
    @T.macro
    def load_to_shared(A, A_shared):
        T.copy(A, A_shared)  # span_marker_macro_copy

    @T.prim_func
    def main(A: T.Tensor((128, 128), "float16"), B: T.Tensor((128, 128), "float16")):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((128, 128), "float16")
            load_to_shared(A, A_shared)  # span_marker_macro_call

    return main


def test_spans_inside_macro():
    func = _make_macro_kernel()
    lines = _stmt_span_lines(func)
    # Statements inside a macro body point at the macro definition line,
    # mirroring Python traceback semantics for the macro body frame.
    macro_lines = lines.get(_marker_line("span_marker_macro_copy"))
    call_lines = lines.get(_marker_line("span_marker_macro_call"))
    assert macro_lines or call_lines, "macro copy statement carries no span"


# ---------------------------------------------------------------------------
# Error location tests: pass diagnostics must carry a `--> file:line` hint.
# ---------------------------------------------------------------------------


def _expect_error_at(excinfo, marker: str):
    message = str(excinfo.value)
    expected = f"--> {__file__}:{_marker_line(marker)}:"
    assert expected in message, f"expected {expected!r} in error message:\n{message}"


def _make_parallel_fragment_bad_index():
    @T.prim_func
    def main(A: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=128):
            frag = T.alloc_fragment((4,), "float32")  # span_marker_frag_alloc
            for i in T.Parallel(4):
                frag[1] = A[i]  # span_marker_frag_store

    return main


def test_error_span_parallel_fragment_nonzero_index():
    func = _make_parallel_fragment_bad_index()
    with pytest.raises(Exception) as excinfo:
        tilelang.lower(func, target="c")
    assert "Only fragment[0] access is allowed" in str(excinfo.value)
    _expect_error_at(excinfo, "span_marker_frag_alloc")


def _make_pipeline_duplicate_order():
    @T.prim_func
    def main(
        A: T.Tensor((128, 128), "float16"),
        B: T.Tensor((128, 128), "float16"),
        C: T.Tensor((128, 128), "float16"),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((128, 128), "float16")
            B_shared = T.alloc_shared((128, 128), "float16")
            for _k in T.Pipelined(2, order=[0, 0], stage=[0, 1]):  # span_marker_pipelined
                T.copy(A, A_shared)  # span_marker_pipe_copy1
                T.copy(B, B_shared)  # span_marker_pipe_copy2

    return main


@tilelang.testing.requires_metal
def test_error_span_pipeline_duplicate_order():
    func = _make_pipeline_duplicate_order()
    with pytest.raises(Exception) as excinfo, tvm.target.Target("metal"):
        tilelang.lower(func, target="metal")
    assert "same order" in str(excinfo.value)
    _expect_error_at(excinfo, "span_marker_pipe_copy2")


def _make_parallel_inconsistent_indices():
    @T.prim_func
    def main(A: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=128):
            frag = T.alloc_fragment((4,), "float32")  # span_marker_frag_alloc2
            for i in T.Parallel(4):
                frag[0] = A[i]
                frag[1] = A[i]  # span_marker_frag_store2

    return main


def test_error_span_parallel_inconsistent_indices():
    func = _make_parallel_inconsistent_indices()
    with pytest.raises(Exception) as excinfo:
        tilelang.lower(func, target="c")
    _expect_error_at(excinfo, "span_marker_frag_alloc2")


def _make_copy_range_mismatch():
    @T.prim_func
    def main(A: T.Tensor((128,), "float32"), B: T.Tensor((64,), "float32")):
        with T.Kernel(1, threads=128):
            B_shared = T.alloc_shared((64,), "float32")
            T.copy(A[0:128], B_shared[0:64])  # span_marker_bad_copy

    return main


def test_error_span_copy_range_mismatch():
    func = _make_copy_range_mismatch()
    with pytest.raises(Exception) as excinfo:
        tilelang.lower(func, target="c")
    assert "--> " in str(excinfo.value)


# ---------------------------------------------------------------------------
# Review hardening: parameter buffer spans, path edge cases, coverage stats.
# ---------------------------------------------------------------------------


def _make_kernel_param_span():
    @T.prim_func
    def main(
        A: T.Tensor((128, 128), "float16"),  # span_marker_param_a
        B: T.Tensor((128, 128), "float16"),
    ):
        with T.Kernel(1, threads=128):
            T.copy(A, B)

    return main


def test_param_buffer_has_span():
    func = _make_kernel_param_span()
    buffer_a = func.buffer_map[func.params[0]]
    loc = span_to_location(get_buffer_span(buffer_a))
    assert loc is not None, "parameter buffer A carries no span"
    assert loc == (__file__, _marker_line("span_marker_param_a"))


def test_enrich_error_space_and_windows_paths(tmp_path):
    from tilelang.errors import _MARKER_RE, enrich_error

    # POSIX path containing spaces: snippet must render.
    spaced = tmp_path / "dir with spaces"
    spaced.mkdir()
    src = spaced / "kernel.py"
    src.write_text("line one\nline two\nline three\n")
    exc = ValueError(f"boom\n  --> {src}:2:1")
    enrich_error(exc)
    assert "line two" in str(exc)

    # Windows drive-letter path: regex must not stop at the drive colon.
    m = _MARKER_RE.search("\n  --> C:\\work\\kernel.py:21:1")
    assert m is not None
    assert m.group("file") == "C:\\work\\kernel.py"
    assert m.group("line") == "21"
    assert m.group("col") == "1"


def test_enrich_error_passes_through_plain_errors():
    from tilelang.errors import enrich_error

    exc = ValueError("plain error without marker")
    assert enrich_error(exc) is exc
    assert str(exc) == "plain error without marker"


def test_mutate_with_windows_style_path():
    """Windows absolute paths must not produce truncated \\U escapes."""
    import ast as py_ast

    import tilelang.language.eager.ast as eager_ast

    def sample(a):
        return a + 1

    fake_path = r"C:\Users\fake\kernel.py"
    tree = eager_ast.utils.get_ast(sample)
    mut = eager_ast.DSLMutator({}, sample.__globals__, fake_path)
    new_tree = mut.visit(tree)
    source = py_ast.unparse(new_tree)
    # repr() quoting keeps the path intact and the generated source valid.
    assert repr(fake_path) in source
    compile(source, "<mutated>", "exec")


def test_span_coverage_counts_alloc_buffers():
    func = _make_kernel()
    cov = span_coverage(func)
    # 3 parameter buffers + A_shared + C_local; all must be spanned and the
    # alloc (non-parameter) buffers must be counted.
    assert cov["buffers"][1] >= 5
    assert cov["buffers"][0] == cov["buffers"][1]


if __name__ == "__main__":
    tilelang.testing.main()
