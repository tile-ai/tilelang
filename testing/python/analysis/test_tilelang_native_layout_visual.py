"""Layout visualization reads native metadata and draws actual Layout objects."""

import importlib.util

import tvm_ffi
from tvm_tirx import ir

import tilelang.language as T
from tilelang.analysis import LayoutVisual
from tilelang.layout import Fragment, PartialFragment
from tilelang.tools.plot_layout import plot_layout, plot_fragment_tv


def test_layout_visual_reads_local_native_annotation_and_labels_partials(capsys):
    layout = Fragment((4, 8), forward_thread_fn=lambda i, j: i * 8 + j, forward_index_fn=lambda i, j: 0)
    partial = PartialFragment.from_fragment(layout.replicate(2))

    @T.prim_func
    def kernel(A: T.Tensor((4, 8), "float32")):
        fragment = T.alloc_fragment((4, 8), "float32")
        T.annotate_layout({fragment: partial})
        T.annotate_layout({fragment: partial})
        T.copy(A, fragment)

    original = ir.IRModule.from_expr(kernel)
    result = LayoutVisual("txt")(original)
    assert tvm_ffi.structural_equal(original, result)
    text = capsys.readouterr().out
    assert text.count("inferred layout:") == 1
    assert "(partials)" in text
    for field in ("Shape:", "Thread:", "Index:", "Replicate:"):
        assert field in text


def test_layout_visual_emits_native_fragment_svg(tmp_path, monkeypatch):
    assert importlib.util.find_spec("matplotlib") is not None, "matplotlib is required for the actual layout plot regression"
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.chdir(tmp_path)
    layout = Fragment((4, 8), forward_thread_fn=lambda i, j: i * 8 + j, forward_index_fn=lambda i, j: 0)

    @T.prim_func
    def kernel(A: T.Tensor((4, 8), "float32")):
        T.annotate_layout({A: layout})
        A[0, 0] = 1.0

    LayoutVisual("svg")(ir.IRModule.from_expr(kernel))
    output = tmp_path / "tmp/A_layout.svg"
    assert output.exists()
    assert "<svg" in output.read_text()


def test_native_plain_layout_and_fragment_tv_plots(tmp_path, monkeypatch):
    assert importlib.util.find_spec("matplotlib") is not None, "matplotlib is required for the actual layout plot regression"
    monkeypatch.setenv("MPLBACKEND", "Agg")
    padded = T.Layout((2, 3), lambda i, j: i * 4 + j)
    for i in range(2):
        for j in range(3):
            assert int(padded.map_forward_index([i, j])[0]) == i * 4 + j
    plot_layout(padded, tmp_path, name="padded", formats="svg", view="output")
    fragment = Fragment((2, 2), forward_thread_fn=lambda i, j: i * 2 + j, forward_index_fn=lambda i, j: 0)
    plot_fragment_tv(fragment, tmp_path, name="threads", formats="svg")
    assert "<svg" in (tmp_path / "padded.svg").read_text()
    assert "<svg" in (tmp_path / "threads.svg").read_text()
