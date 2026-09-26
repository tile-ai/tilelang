"""Shared helpers for the CUDA Tile IR backend test suite."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tilelang import tvm as tvm
from tilelang.tileir import checks
from tvm import tirx
from tilelang.jit.adapter.tileir.adapter import TileIRKernelAdapter
from tilelang.tileir.artifact import TileIRLoweringResult
from tilelang.tileir import lowering as tileir_lowering
from tilelang.tileir.lowering import lower_primfunc_to_tileir
from tilelang.tileir import tir_analysis as tileir_tir_analysis
from tilelang.backend.target import determine_target

Range = tvm.ir.Range
structural_equal = tvm.ir.structural_equal
Target = tvm.target.Target

skip_no_cuda_tile = pytest.mark.skipif(
    not checks._module_exists(checks.CUDA_TILE_IR_MLIR_MODULE),
    reason="cuda_tile MLIR bindings unavailable",
)


def _cuda_target_for_test() -> Target:
    return determine_target({"kind": "cuda", "arch": "sm_120"}, return_object=True)


def _call_ops(prim_func: tirx.PrimFunc) -> set[str]:
    ops: set[str] = set()

    def visit(node):
        if isinstance(node, tirx.Call):
            ops.add(tileir_tir_analysis._op_name(node))

    tirx.stmt_functor.post_order_visit(prim_func.body, visit)
    return ops


def _buffer_argument_names_for_test(prim_func: tirx.PrimFunc) -> tuple[str, ...]:
    return tuple(prim_func.buffer_map[param].name for param in prim_func.params if param in prim_func.buffer_map)


def _prepared_tileir_kernel_for_test(jit_kernel, *args):
    target = determine_target("tileir -arch=sm_120", return_object=True)
    prim_func = jit_kernel.get_tir(*args)
    _, prepared = TileIRKernelAdapter._prepare_device_module(prim_func, target)
    semantic_program = tileir_lowering._extract_semantic_program_for_lowering(prepared, target)
    return target, prepared, semantic_program


def _build_tileir_module_for_test(prim_func, target, *, options=None):
    """Build MLIR through the public single-kernel pipeline for unit tests."""
    from tilelang.tileir.pipeline import build_tileir_module

    attrs = getattr(prim_func, "attrs", None)
    global_symbol = attrs.get("global_symbol", "") if attrs is not None else ""
    kernel_name = str(global_symbol) if global_symbol else None

    target_arch = getattr(target, "arch", None)
    if target_arch is None:
        target_attrs = getattr(target, "attrs", None)
        if target_attrs is not None and "arch" in target_attrs:
            target_arch = target_attrs["arch"]

    return build_tileir_module(
        prim_func,
        kernel_name=kernel_name,
        arch=str(target_arch) if target_arch else None,
        fast_math=bool(getattr(options, "fast_math", False)),
        disable_tma=bool(getattr(options, "disable_tma", False)),
    )


def _tileir_source_for_test(jit_kernel, *args) -> str:
    """Lower a JIT kernel to MLIR source via the TileIR pipeline.

    Routes through ``build_tileir_module`` (TIR → SemanticIR → TileIR Block →
    passes → MLIR) — the same path used by ``lower_primfunc_to_tileir``.
    """
    from tilelang.tileir.pipeline import build_tileir_module as _new_build

    target, prepared, _semantic_program = _prepared_tileir_kernel_for_test(jit_kernel, *args)
    attrs = getattr(prepared, "attrs", None)
    kernel_name = str(attrs.get("global_symbol", "")) if attrs else ""
    kernel_name = kernel_name or "kernel"
    # Extract the architecture used by the TileIR pipeline.
    arch: str | None = None
    target_arch_raw = getattr(target, "arch", None)
    if target_arch_raw is None:
        target_attrs = getattr(target, "attrs", None)
        if target_attrs is not None and "arch" in target_attrs:
            target_arch_raw = str(target_attrs["arch"])
    if target_arch_raw:
        arch = str(target_arch_raw)
    module = _new_build(prepared, kernel_name=kernel_name, arch=arch)
    return str(module)


def _semantic_kinds(stmt) -> set[str]:
    kinds = {stmt.kind}
    for child in stmt.children:
        kinds.update(_semantic_kinds(child))
    return kinds


def _semantic_tile_ops(stmt) -> set[str]:
    ops = set()
    if stmt.kind == "tile_op":
        ops.update(value for key, value in stmt.attrs if key == "op")
    for child in stmt.children:
        ops.update(_semantic_tile_ops(child))
    return ops


def _semantic_stmts(stmt):
    yield stmt
    for child in stmt.children:
        yield from _semantic_stmts(child)


def _load_flash_decode_example():
    path = Path(__file__).parents[3] / "examples" / "flash_decoding" / "example_gqa_decode.py"
    spec = importlib.util.spec_from_file_location("tilelang_test_example_gqa_decode", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_mla_paged_example():
    path = Path(__file__).parents[3] / "examples" / "deepseek_mla" / "example_mla_decode_paged.py"
    spec = importlib.util.spec_from_file_location("tilelang_test_example_mla_decode_paged", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_mla_decode_example():
    path = Path(__file__).parents[3] / "examples" / "deepseek_mla" / "example_mla_decode.py"
    spec = importlib.util.spec_from_file_location("tilelang_test_example_mla_decode", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_mla_kv_fp8_example():
    path = Path(__file__).parents[3] / "examples" / "deepseek_mla" / "experimental" / "example_mla_decode_kv_fp8.py"
    spec = importlib.util.spec_from_file_location("tilelang_test_example_mla_decode_kv_fp8", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_mla_persistent_example():
    path = Path(__file__).parents[3] / "examples" / "deepseek_mla" / "example_mla_decode_persistent.py"
    spec = importlib.util.spec_from_file_location("tilelang_test_example_mla_decode_persistent", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_mhc_pre_example():
    path = Path(__file__).parents[3] / "examples" / "deepseek_mhc" / "example_mhc_pre.py"
    spec = importlib.util.spec_from_file_location("tilelang_test_example_mhc_pre", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_gemv_example():
    path = Path(__file__).parents[3] / "examples" / "gemv" / "example_gemv.py"
    spec = importlib.util.spec_from_file_location("tilelang_test_example_gemv", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_dequant_gemv_example():
    path = Path(__file__).parents[3] / "examples" / "dequantize_gemm" / "example_dequant_gemv_fp16xint4.py"
    spec = importlib.util.spec_from_file_location("tilelang_test_example_dequant_gemv_fp16xint4", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(path.parent))
        spec.loader.exec_module(module)
    return module


def _load_deepseek_v4_act_quant_example():
    path = Path(__file__).parents[3] / "examples" / "deepseek_v4" / "act_quant.py"
    spec = importlib.util.spec_from_file_location("tilelang_test_deepseek_v4_act_quant", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_minference_example():
    path = Path(__file__).parents[3] / "examples" / "minference" / "example_vertical_slash_sparse_attn.py"
    spec = importlib.util.spec_from_file_location("tilelang_test_minference_vs_sparse_attn", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _skip_if_tileir_toolchain_unavailable():
    try:
        checks.check_tileir_available()
    except checks.TileIRDependencyError as exc:
        pytest.skip(f"CUDA TileIR toolchain unavailable: {exc}")


def _setup_gpu():
    """Return torch and the active CUDA target, or skip when unavailable."""

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("No CUDA GPU available")
    major, minor = torch.cuda.get_device_capability()
    return torch, major, minor, f"tileir -arch=sm_{major}{minor}"


def _lower_tileir_primfunc_for_test(prim_func: tirx.PrimFunc) -> TileIRLoweringResult:
    try:
        toolchain = checks.check_tileir_available()
    except checks.TileIRDependencyError as exc:
        pytest.skip(f"CUDA TileIR toolchain unavailable: {exc}")
    return lower_primfunc_to_tileir(prim_func, _cuda_target_for_test(), toolchain)


def _enable_tileir_runtime(monkeypatch):
    torch = pytest.importorskip("torch")
    major, minor = torch.cuda.get_device_capability()
    monkeypatch.setenv("TILELANG_DISABLE_CACHE", "1")
    monkeypatch.setenv("TILELANG_EXECUTION_BACKEND", "tileir")
    monkeypatch.setenv("TILELANG_TARGET", f"tileir -arch=sm_{major}{minor}")
    return torch
