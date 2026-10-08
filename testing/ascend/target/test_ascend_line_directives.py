"""Test ``#line`` directive emission in the AscendC codegen."""

import re

import tilelang
import tilelang.ascend.language as T
from tilelang import tvm

N = 64


@T.prim_func
def vec_add(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32")):
    with T.Kernel(1), T.SimtVF(threads=N):
        for i in T.Parallel(N):
            B[i] = A[i] + 1.0  # line_marker_store


def _lowered_source(emit: bool) -> str:
    config = {tilelang.PassConfigKey.TL_EMIT_LINE_DIRECTIVES: emit}
    with tvm.target.Target("ascend"), tvm.transform.PassContext(opt_level=3, config=config):
        artifact = tilelang.lower(vec_add, target="ascend")
    source = artifact.kernel_source
    assert source is not None, "Ascend codegen produced no kernel source"
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
    source = _lowered_source(emit=True)
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
    source = _lowered_source(emit=False)
    assert "#line" not in source, source
