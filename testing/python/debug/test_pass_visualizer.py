# type: ignore
"""Tests for the pass_visualizer debugging tool.

Covers:
- Core helpers: build_module, build_pass_stages, inspect_structure
- Structure-tree capture and per-pass diffing (build_pass_data via viewer)
- HTML / text emission (emit_html, emit_txt)

These run the CUDA lowering prologue on a small kernel. An explicit cuda target
is passed so the pipeline does not depend on auto target detection.
"""

import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang import tvm

from tilelang.utils.pass_visualizer import (
    build_module,
    build_pass_stages,
    inspect_structure,
)
from tilelang.utils.pass_visualizer.viewer import (
    _capture_tree,
    build_pass_data,
    emit_html,
    emit_txt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gemm_relu_kernel():
    """A small fused GEMM + bias + ReLU @tilelang.jit kernel."""

    @tilelang.jit(out_idx=[-1])
    def gemm_relu(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):

        @T.prim_func
        def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            bias: T.Tensor((N,), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_K, block_N), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)

                for i, j in T.Parallel(block_M, block_N):
                    C_local[i, j] = T.max(C_local[i, j] + bias[bx * block_N + j], 0)

                T.copy(C_local, C[by * block_M, bx * block_N])

        return main

    return gemm_relu


def _build_small_module():
    """Build an IRModule + resolved cuda target for the small kernel."""
    kernel = _gemm_relu_kernel()
    func = kernel.get_tir(M=128, N=128, K=128, block_M=64, block_N=64, block_K=32)
    cuda_target = tvm.target.Target("cuda", host="llvm")
    return build_module(func, target=cuda_target)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


@tilelang.testing.requires_cuda
def test_build_pass_stages_nonempty():
    """build_pass_stages returns an ordered list of (name, transform) pairs."""
    _mod, target = _build_small_module()
    stages = build_pass_stages(target)

    assert len(stages) > 0
    names = [name for name, _ in stages]
    # The prologue must run LayoutInference and LowerTileOp.
    assert "LayoutInference" in names
    assert "LowerTileOp" in names
    for name, transform in stages:
        assert isinstance(name, str)
        assert callable(transform)


@tilelang.testing.requires_cuda
def test_inspect_structure_renders_tree(capsys):
    """inspect_structure prints the SBlock tree with tile ops expanded by field."""
    mod, _target = _build_small_module()
    inspect_structure(mod)
    out = capsys.readouterr().out

    assert "PrimFunc" in out
    assert "SBlock" in out
    # gemm is expanded by field name, not printed as one positional line.
    assert "T.gemm" in out
    assert "a_region" in out


@tilelang.testing.requires_cuda
def test_capture_tree_returns_lines():
    """_capture_tree turns inspect_structure output into a list of text lines."""
    mod, _target = _build_small_module()
    lines = _capture_tree(mod)

    assert isinstance(lines, list)
    assert len(lines) > 0
    assert any("PrimFunc" in ln for ln in lines)


# ---------------------------------------------------------------------------
# build_pass_data + emission (file-driven)
# ---------------------------------------------------------------------------


@tilelang.testing.requires_cuda
def test_build_pass_data_and_emit(tmp_path):
    """End-to-end: run the example kernel through the pipeline and emit HTML/txt."""
    import os

    kernel_path = os.path.join(
        os.path.dirname(tilelang.utils.pass_visualizer.__file__),
        "examples",
        "gemm_relu.py",
    )
    with open(kernel_path) as f:
        source = f.read()

    kwargs = {"M": 128, "N": 128, "K": 128, "block_M": 64, "block_N": 64, "block_K": 32}
    name, stages = build_pass_data(kernel_path, None, "auto", kwargs, source)

    assert name == "gemm_relu"
    # source + (input) + at least one pass.
    assert len(stages) >= 3
    for st in stages:
        assert "name" in st and "flag" in st and "rows" in st
    flags = {st["flag"] for st in stages}
    assert "source" in flags
    assert "input" in flags

    html = emit_html(name, stages)
    assert "Pass browser" in html
    assert "gemm_relu" in html

    txt = emit_txt(name, stages)
    assert "kernel: gemm_relu" in txt
    assert "T.gemm" in txt


if __name__ == "__main__":
    tilelang.testing.main()
