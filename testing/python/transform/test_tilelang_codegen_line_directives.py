"""Tests for opt-in ``#line`` directive emission in C-family codegen.

With the ``tl.emit_line_directives`` pass config enabled, codegen maps every
generated statement that carries a TIR span back to its Python source line via
``#line N "file"``. These tests pin the emission on the CPU ``c`` target
(no GPU / compiler toolchain needed) and lock the no-deduplication semantics:
after ``#line N`` each emitted line advances the logical line, so a second
statement from the same Python line must re-emit ``#line N`` (dedup would
drift). See docs/issue_2913/codegen_final.md for the design.
"""

import re

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm

N = 128


@T.prim_func
def vec_add(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32")):
    for i in T.Parallel(N):
        B[i] = A[i] + 1.0  # line_marker_store


@T.prim_func
def vec_add_cuda(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32")):
    with T.Kernel(1, threads=N) as bx:
        for i in T.Parallel(N):
            B[bx * N + i] = A[bx * N + i] + 1.0  # line_marker_store_cuda


def _lowered_source(func, emit: bool) -> str:
    # target is not passed to lower() explicitly: it resolves through
    # determine_target("auto") reading the ambient Target("c") context.
    config = {tilelang.PassConfigKey.TL_EMIT_LINE_DIRECTIVES: emit}
    with tvm.target.Target("c"), tvm.transform.PassContext(opt_level=3, config=config):
        artifact = tilelang.lower(func)
    source = artifact.kernel_source
    assert source is not None, "CPU C codegen produced no kernel source"
    return source


def _marker_line(marker: str) -> int:
    with open(__file__) as f:
        for i, line in enumerate(f, 1):
            if marker in line:
                return i
    raise ValueError(f"marker not found: {marker}")


def _line_directives(source: str) -> list[tuple[int, str]]:
    return [(int(num), fname) for num, fname in re.findall(r'^#line (\d+) "(.*)"$', source, re.M)]


def test_line_directives_emitted_when_enabled():
    source = _lowered_source(vec_add, emit=True)
    directives = _line_directives(source)
    assert directives, f"no #line directives emitted:\n{source}"

    # Directives must point back at this test file (span SourceName).
    files = {fname for _, fname in directives}
    assert __file__ in files, f"expected {__file__} among {files}:\n{source}"

    # The function entry must be anchored to the user's def line, and the
    # store statement to its actual source line.
    def_line = _marker_line("def vec_add")
    assert (def_line, __file__) in directives, (
        f"function entry line {def_line} not mapped (PrimFunc span lost?); directives: {directives}\n{source}"
    )
    store_line = _marker_line("line_marker_store")
    assert (store_line, __file__) in directives, f"store line {store_line} not mapped; directives: {directives}\n{source}"


def test_line_directives_disabled_by_default():
    source = _lowered_source(vec_add, emit=False)
    assert "#line" not in source, source


def test_line_directives_disabled_when_config_absent():
    """The default path (config key entirely absent) must not emit either."""
    with tvm.target.Target("c"), tvm.transform.PassContext(opt_level=3):
        artifact = tilelang.lower(vec_add)
    assert artifact.kernel_source is not None
    assert "#line" not in artifact.kernel_source, artifact.kernel_source


@tilelang.testing.requires_cuda
def test_line_directives_cuda_source():
    """CUDA source (compile-only path, no GPU/nvcc) also maps statements."""
    target = {"kind": "cuda"}
    config = {tilelang.PassConfigKey.TL_EMIT_LINE_DIRECTIVES: True}
    with tvm.transform.PassContext(opt_level=3, config=config), tvm.target.Target(target):
        artifact = tilelang.lower(vec_add_cuda, target=target)
    source = artifact.kernel_source
    assert source is not None, "CUDA codegen produced no kernel source"
    directives = _line_directives(source)
    store_line = _marker_line("line_marker_store_cuda")
    assert (store_line, __file__) in directives, f"store line {store_line} not mapped; directives: {directives}\n{source}"


def test_same_line_statements_reemit_directive():
    """Statements lowered from one Python line must each re-emit ``#line N``.

    The single store ``B[i] = A[i] + 1.0`` lowers to several statements that
    share its span (the hoisted broadcast declaration plus the store itself),
    so its line must appear as several directives. This locks the no-dedup
    semantics: with (file, line) deduplication only the first would be emitted
    and the following statements would drift to logical lines N+1, N+2, ...
    """
    source = _lowered_source(vec_add, emit=True)
    directives = _line_directives(source)
    store_line = _marker_line("line_marker_store")
    same_line = [d for d in directives if d == (store_line, __file__)]
    assert len(same_line) >= 2, (
        f"expected >=2 re-emitted directives for line {store_line}, got {same_line}; all directives: {directives}\n{source}"
    )


if __name__ == "__main__":
    tilelang.testing.main()
