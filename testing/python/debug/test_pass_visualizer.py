# type: ignore
"""Tests for the pass_visualizer debugging tool.

Covers:
- Core helpers: build_module, inspect_structure, StructureTreePassInstrument
- Real-pipeline structure capture and per-pass diffing (build_pass_data via viewer)
- HTML / text emission (emit_html, emit_txt)

These run the CUDA lowering prologue on a small kernel. An explicit cuda target
is passed so the pipeline does not depend on auto target detection.
"""

import pytest

import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang import tvm

from tilelang.tools.pass_visualizer import (
    build_module,
    capture_structure,
    inspect_structure,
    PassStructureRecord,
    StructureTreePassInstrument,
)
from tilelang.tools.pass_visualizer.viewer import (
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
def test_structure_tree_instrument_records_real_passes():
    """The instrument records top-level passes using their PassInfo names."""
    mod, target = _build_small_module()
    instrument = StructureTreePassInstrument()

    with tvm.transform.PassContext(instruments=[instrument]), target:
        mod = tvm.tirx.transform.BindTarget(target)(mod)
        tilelang.transform.MaterializeKernelLaunch()(mod)

    records = instrument.ordered_records()
    assert [record.name for record in records] == ["tirx.BindTarget", "tl.MaterializeKernelLaunch"]
    assert instrument.input_lines
    assert any("PrimFunc" in line for line in instrument.input_lines)


def test_structure_tree_instrument_ignores_nested_passes():
    """Nested implementation passes do not duplicate linear browser stages."""
    mod, _target = _build_small_module()
    instrument = StructureTreePassInstrument()

    @tvm.transform.module_pass(opt_level=0, name="test.Outer")
    def outer_pass(mod, _ctx):
        return tvm.tirx.transform.Simplify()(mod)

    with tvm.transform.PassContext(instruments=[instrument]):
        outer_pass(mod)

    records = instrument.ordered_records()
    assert [record.name for record in records] == ["test.Outer"]


def test_pass_structure_record_changed_uses_structure_snapshot():
    """The changed flag describes the structure-tree diff shown by the viewer."""
    unchanged = PassStructureRecord(
        name="test.Unchanged",
        sequence=0,
        before_lines=["PrimFunc", "  body"],
        after_lines=["PrimFunc", "  body"],
    )
    changed = PassStructureRecord(
        name="test.Changed",
        sequence=1,
        before_lines=["PrimFunc", "  body"],
        after_lines=["PrimFunc", "  changed body"],
    )

    assert not unchanged.changed
    assert changed.changed


def test_structure_tree_instrument_cleans_up_after_failure():
    """A missing after-pass callback is retained as a diagnostic, not stale state."""
    mod, _target = _build_small_module()
    instrument = StructureTreePassInstrument()

    @tvm.transform.module_pass(opt_level=0, name="test.Fail")
    def failing_pass(_mod, _ctx):
        raise RuntimeError("expected failure")

    with pytest.raises(RuntimeError, match="expected failure"), tvm.transform.PassContext(instruments=[instrument]):
        failing_pass(mod)

    assert instrument.records == []
    assert instrument.incomplete_passes == ["test.Fail"]

    # Entering a new context resets the incomplete frame and records normally.
    with tvm.transform.PassContext(instruments=[instrument]):
        tvm.tirx.transform.Simplify()(mod)
    assert [record.name for record in instrument.ordered_records()] == ["tirx.Simplify"]
    assert instrument.incomplete_passes == []


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
def test_capture_structure_returns_lines():
    """capture_structure turns inspect_structure output into text lines."""
    mod, _target = _build_small_module()
    lines = capture_structure(mod)

    assert isinstance(lines, list)
    assert len(lines) > 0
    assert any("PrimFunc" in ln for ln in lines)


# ---------------------------------------------------------------------------
# build_pass_data + emission (file-driven)
# ---------------------------------------------------------------------------


@tilelang.testing.requires_cuda
def test_build_pass_data_and_emit(tmp_path):
    """End-to-end: observe the real prologue and emit HTML/txt."""
    import os

    kernel_path = os.path.join(
        os.path.dirname(tilelang.tools.pass_visualizer.__file__),
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
    stage_names = [st["name"].split("] ", 1)[-1] for st in stages[2:]]
    assert "LayoutInference" in stage_names
    assert "LowerTileOp" in stage_names
    assert "AddWrapperForSingleBufStore" in stage_names
    assert "DecoupleTypeCast" in stage_names
    assert "pass_fn" not in stage_names

    html = emit_html(name, stages)
    assert "Pass browser" in html
    assert "gemm_relu" in html

    txt = emit_txt(name, stages)
    assert "kernel: gemm_relu" in txt
    assert "T.gemm" in txt


def test_build_pass_data_rejects_non_cuda_target():
    """The focused viewer rejects backends whose pass pipeline it cannot dispatch."""
    import os

    kernel_path = os.path.join(
        os.path.dirname(tilelang.tools.pass_visualizer.__file__),
        "examples",
        "gemm_relu.py",
    )
    with open(kernel_path) as f:
        source = f.read()

    kwargs = {"M": 128, "N": 128, "K": 128, "block_M": 64, "block_N": 64, "block_K": 32}
    with pytest.raises(ValueError, match="currently supports only CUDA targets"):
        build_pass_data(kernel_path, None, "llvm", kwargs, source)


@tilelang.testing.requires_cuda
def test_build_pass_data_observes_canonical_prologue(monkeypatch):
    """A newly introduced real pipeline pass appears without a viewer pass list."""
    import os

    kernel_path = os.path.join(
        os.path.dirname(tilelang.tools.pass_visualizer.__file__),
        "examples",
        "gemm_relu.py",
    )
    with open(kernel_path) as f:
        source = f.read()

    @tvm.transform.module_pass(opt_level=0, name="test.InjectedPipelinePass")
    def injected_pass(mod, _ctx):
        return mod

    def injected_prologue(mod, _target):
        return injected_pass(mod)

    monkeypatch.setattr("tilelang.tools.pass_visualizer.viewer.CUDAPassPipelineBodyPrologue", injected_prologue)
    kwargs = {"M": 128, "N": 128, "K": 128, "block_M": 64, "block_N": 64, "block_K": 32}
    _name, stages = build_pass_data(kernel_path, None, "cuda", kwargs, source)

    assert [stage["name"] for stage in stages] == ["source code", "(input)", "[01] InjectedPipelinePass"]


@tilelang.testing.requires_cuda
def test_build_pass_data_respects_jit_pass_configs(tmp_path):
    """Conditional stages use the analyzed JITImpl's real PassContext config."""
    import os

    kernel_path = os.path.join(
        os.path.dirname(tilelang.tools.pass_visualizer.__file__),
        "examples",
        "gemm_relu.py",
    )
    with open(kernel_path) as f:
        source = f.read()
    source = source.replace(
        "@tilelang.jit(out_idx=[-1])",
        '@tilelang.jit(out_idx=[-1], pass_configs={"tl.disable_warp_specialized": True})',
    )
    configured_path = tmp_path / "gemm_relu_no_ws.py"
    configured_path.write_text(source)

    kwargs = {"M": 128, "N": 128, "K": 128, "block_M": 64, "block_N": 64, "block_K": 32}
    _name, stages = build_pass_data(str(configured_path), None, "cuda", kwargs, source)

    stage_names = [st["name"] for st in stages]
    assert not any("ProducerConsumerWarpSpecialized" in name for name in stage_names)


if __name__ == "__main__":
    tilelang.testing.main()
