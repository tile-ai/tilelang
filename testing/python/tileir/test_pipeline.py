"""End-to-end pipeline tests.

Tests ``tilelang.tileir.pipeline.build_tileir_module``:

1. Import smoke-test.
2. Fill kernel: single GLOBAL buffer + Fill op → MLIR module produced with
   non-empty MLIR text and correct structure.
3. Assembly: the module assembles to a valid cubin (requires tileiras toolchain
   and a GPU).
4. Entry-arg alignment: ``root.params`` count matches ``program.params`` count.
5. Kernel-name derivation: ``global_symbol`` attribute becomes the entry name.

SHARED/REGISTER alloc_buffer kernels:
--------------------------------------------------------------
- copy_kernel (GLOBAL→SHARED→GLOBAL via T.copy): SHARED buffers are
  materialized as zero-tiles in _tile_map.
- tiny_gemm (GLOBAL→SHARED, T.gemm, REGISTER accumulator, SHARED→GLOBAL):
  REGISTER buffers are also in _tile_map.
- gemv_kernel (f16 inputs + f32 accumulator + f16 output with ftof cast):
  _cast_tile must pass the loc= argument.

Coverage table (per representative kernel → stage reached + next failing
construct): see test_build_tileir_module_gap_table for the programmatic version.
"""

from __future__ import annotations

import pytest

import tilelang
from tilelang import language as T
from tilelang.tileir.checks import has_cuda_tile_ir_bindings

# ---------------------------------------------------------------------------
# Skip guard — needs cuda_tile MLIR bindings
# ---------------------------------------------------------------------------

_HAS_CUDA_TILE = has_cuda_tile_ir_bindings()

skip_no_cuda_tile = pytest.mark.skipif(
    not _HAS_CUDA_TILE,
    reason="cuda_tile MLIR bindings unavailable",
)


# ---------------------------------------------------------------------------
# Kernel factories (defined at module scope — avoids inspect.getsource issues)
# ---------------------------------------------------------------------------


@tilelang.jit
def _fill_kernel(a):
    a: T.Tensor((32, 64), T.float16)
    with T.Kernel(1, threads=128):
        T.fill(a, 0.0)


def _get_fill_prim_func():
    return _fill_kernel.get_tir(None)


# ---------------------------------------------------------------------------
# Test 1: import smoke
# ---------------------------------------------------------------------------


def test_pipeline_imports():
    """build_tileir_module must be importable without errors."""
    from tilelang.tileir.pipeline import build_tileir_module  # noqa: F401

    assert callable(build_tileir_module)


def test_build_tileir_module_rejects_multi_kernel_program(monkeypatch):
    """The single-module API must not silently discard later kernels."""
    import tilelang.tileir.pipeline as pipeline
    from tilelang.tileir.semantic import SemanticKernel, SemanticProgram, SemanticStmt

    body = SemanticStmt(kind="seq")
    program = SemanticProgram(
        name="multi",
        params=(),
        global_alloc_buffers=(),
        kernels=(
            SemanticKernel("first", ("1", "1", "1"), ("1", "1", "1"), (), body),
            SemanticKernel("second", ("1", "1", "1"), ("1", "1", "1"), (), body),
        ),
    )
    monkeypatch.setattr(pipeline, "tir_to_sem", lambda _prim_func: program)
    monkeypatch.setattr(
        pipeline,
        "lower_kernel",
        lambda *_args, **_kwargs: pytest.fail("multi-kernel input reached single-kernel lowering"),
    )

    with pytest.raises(ValueError, match="exactly one kernel.*found 2"):
        pipeline.build_tileir_module(object())


# ---------------------------------------------------------------------------
# Test 2: fill kernel produces a non-empty MLIR module
# ---------------------------------------------------------------------------


@skip_no_cuda_tile
def test_build_tileir_module_fill_kernel_produces_module():
    """build_tileir_module(fill_kernel) returns a non-None module with non-empty MLIR text."""
    from tilelang.tileir.pipeline import build_tileir_module

    prim_func = _get_fill_prim_func()
    module = build_tileir_module(prim_func)

    assert module is not None, "Expected a non-None MLIR module"
    text = str(module)
    assert text, "Expected non-empty MLIR text"
    # The module should contain the cuda_tile.module wrapper and the entry function.
    assert "cuda_tile.module" in text, f"Expected 'cuda_tile.module' in:\n{text[:400]}"
    assert "entry" in text, f"Expected 'entry' function in:\n{text[:400]}"


# ---------------------------------------------------------------------------
# Test 3: kernel name from global_symbol
# ---------------------------------------------------------------------------


@skip_no_cuda_tile
def test_build_tileir_module_kernel_name_from_global_symbol():
    """The entry function name comes from prim_func.attrs['global_symbol']."""
    from tilelang.tileir.pipeline import build_tileir_module

    prim_func = _get_fill_prim_func()
    module = build_tileir_module(prim_func)
    text = str(module)

    # global_symbol on the fill JIT function is "_fill_kernel"
    assert "_fill_kernel" in text, f"Expected '_fill_kernel' as entry name in MLIR text:\n{text[:400]}"


# ---------------------------------------------------------------------------
# Test 4: explicit kernel_name override
# ---------------------------------------------------------------------------


@skip_no_cuda_tile
def test_build_tileir_module_explicit_kernel_name():
    """Passing kernel_name= overrides the global_symbol."""
    from tilelang.tileir.pipeline import build_tileir_module

    prim_func = _get_fill_prim_func()
    module = build_tileir_module(prim_func, kernel_name="my_fill")
    text = str(module)

    assert "my_fill" in text, f"Expected 'my_fill' entry in:\n{text[:400]}"


# ---------------------------------------------------------------------------
# Test 5: MLIR text contains expected buffer flattening
# ---------------------------------------------------------------------------


@skip_no_cuda_tile
def test_build_tileir_module_buffer_arg_flattening():
    """A 2-D fp16 GLOBAL buffer is flattened to ptr + 2×shape + 2×stride = 5 args."""
    from tilelang.tileir.pipeline import build_tileir_module

    prim_func = _get_fill_prim_func()
    module = build_tileir_module(prim_func)
    text = str(module)

    assert "tile<ptr<f16>>" in text, f"Expected ptr<f16> arg in:\n{text[:400]}"
    # 2 shape + 2 stride → at least 4 tile<i32>
    i32_count = text.count("tile<i32>")
    assert i32_count >= 4, f"Expected >= 4 tile<i32> args (shape+stride) but found {i32_count} in:\n{text[:400]}"


# ---------------------------------------------------------------------------
# Test 6: entry_args count matches program.params
# ---------------------------------------------------------------------------


@skip_no_cuda_tile
def test_build_tileir_module_entry_args_alignment():
    """root.params count must equal program.params count after lower_kernel."""
    from tilelang.tileir.lowering.tir_to_sem import tir_to_sem
    from tilelang.tileir.lowering.sem_to_ir import lower_kernel
    from tilelang.tileir.ir.builder import IRBuilder

    prim_func = _get_fill_prim_func()
    program = tir_to_sem(prim_func)
    builder = IRBuilder()
    root = lower_kernel(program.kernels[0], builder, program=program)

    assert len(root.params) == len(program.params), f"root.params ({len(root.params)}) != program.params ({len(program.params)})"
    # All param Values must be GLOBAL buffers
    from tilelang.tileir.ir.types import MemSpace

    for v in root.params:
        assert v.type.space == MemSpace.GLOBAL, f"Expected GLOBAL space for param {v.name!r}, got {v.type.space}"


# ---------------------------------------------------------------------------
# Test 7: assembly (requires tileiras toolchain)
# ---------------------------------------------------------------------------


@skip_no_cuda_tile
def test_build_tileir_module_assembles():
    """The fill kernel module assembles to a valid cubin via tileiras."""
    from tilelang.tileir.pipeline import build_tileir_module
    from tilelang.tileir.assembly import assemble_tileir_module
    from tilelang.tileir.checks import TileIRDependencyError, check_tileir_available
    from tilelang.backend.target import determine_target

    try:
        toolchain = check_tileir_available()
    except TileIRDependencyError as exc:
        pytest.skip(f"tileiras toolchain unavailable: {exc}")

    # Detect GPU arch
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        cap_text = result.stdout.strip().splitlines()[0]
        if len(cap_text.split(".")) != 2 or not all(part.isdigit() for part in cap_text.split(".")):
            raise ValueError(f"unexpected compute capability: {cap_text!r}")
        cap = cap_text.replace(".", "")
        arch = f"sm_{cap}"
    except Exception:
        pytest.skip("Could not detect GPU architecture")

    prim_func = _get_fill_prim_func()
    module = build_tileir_module(prim_func, arch=arch)
    target = determine_target({"kind": "cuda", "arch": arch}, return_object=True)

    result = assemble_tileir_module(
        module,
        kernel_name="_fill_kernel",
        target=target,
        toolchain=toolchain,
    )
    assert result.cubin, "Expected non-empty cubin from assembly"
    assert len(result.cubin) > 0, "Expected cubin with bytes"


# ===========================================================================
# SHARED/REGISTER buffer kernels
# ===========================================================================

# ---------------------------------------------------------------------------
# Kernel factories (module-scope to avoid getsource issues)
# ---------------------------------------------------------------------------


@tilelang.jit
def _copy_kernel(a, b):
    a: T.Tensor((32, 64), T.float16)
    b: T.Tensor((32, 64), T.float16)
    with T.Kernel(1, threads=128):
        shared = T.alloc_shared([32, 64], T.float16)
        T.copy(a, shared)
        T.copy(shared, b)


def _get_copy_prim_func():
    return _copy_kernel.get_tir(None)


@tilelang.jit
def _tiny_gemm(A, B, C):
    A: T.Tensor((64, 64), T.float16)
    B: T.Tensor((64, 64), T.float16)
    C: T.Tensor((64, 64), T.float32)
    with T.Kernel(1, threads=128):
        sa = T.alloc_shared([64, 64], T.float16)
        sb = T.alloc_shared([64, 64], T.float16)
        acc = T.alloc_fragment([64, 64], T.float32)
        T.copy(A, sa)
        T.copy(B, sb)
        T.fill(acc, 0.0)
        T.gemm(sa, sb, acc)
        T.copy(acc, C)


def _get_tiny_gemm_prim_func():
    return _tiny_gemm.get_tir(None)


@tilelang.jit
def _skinny_gemm(A, B, C):
    A: T.Tensor((16, 64), T.float16)
    B: T.Tensor((64, 128), T.float16)
    C: T.Tensor((16, 128), T.float32)
    with T.Kernel(1, threads=128):
        sa = T.alloc_shared([16, 64], T.float16)
        sb = T.alloc_shared([64, 128], T.float16)
        acc = T.alloc_fragment([16, 128], T.float32)
        T.copy(A, sa)
        T.copy(B, sb)
        T.fill(acc, 0.0)
        T.gemm(sa, sb, acc)
        T.copy(acc, C)


def _get_skinny_gemm_prim_func():
    return _skinny_gemm.get_tir(None)


@tilelang.jit
def _gemv_kernel(A, B, C):
    A: T.Tensor((64, 64), T.float16)
    B: T.Tensor((64, 64), T.float16)
    C: T.Tensor((64, 64), T.float16)
    with T.Kernel(1, threads=128):
        sa = T.alloc_shared([64, 64], T.float16)
        sb = T.alloc_shared([64, 64], T.float16)
        acc = T.alloc_fragment([64, 64], T.float32)
        T.copy(A, sa)
        T.copy(B, sb)
        T.fill(acc, 0.0)
        T.gemm(sa, sb, acc, transpose_B=True)
        T.copy(acc, C)


def _get_gemv_prim_func():
    return _gemv_kernel.get_tir(None)


# ---------------------------------------------------------------------------
# Helper: detect GPU arch and skip if unavailable
# ---------------------------------------------------------------------------


def _detect_arch_and_toolchain():
    """Return (arch, target, toolchain) or raise pytest.skip."""
    from tilelang.tileir.checks import TileIRDependencyError, check_tileir_available
    from tilelang.backend.target import determine_target
    import subprocess

    try:
        toolchain = check_tileir_available()
    except TileIRDependencyError as exc:
        pytest.skip(f"tileiras toolchain unavailable: {exc}")

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        cap_text = result.stdout.strip().splitlines()[0]
        if len(cap_text.split(".")) != 2 or not all(part.isdigit() for part in cap_text.split(".")):
            raise ValueError(f"unexpected compute capability: {cap_text!r}")
        cap = cap_text.replace(".", "")
        arch = f"sm_{cap}"
    except Exception:
        pytest.skip("Could not detect GPU architecture")

    target = determine_target({"kind": "cuda", "arch": arch}, return_object=True)
    return arch, target, toolchain


# ---------------------------------------------------------------------------
# Test 8: copy_kernel builds (SHARED buffer materialized in _tile_map)
# ---------------------------------------------------------------------------


@skip_no_cuda_tile
def test_copy_kernel_builds_module():
    """copy_kernel (GLOBAL→SHARED→GLOBAL) must emit a non-empty MLIR module.

    SHARED alloc_buffers are materialized as zero-constant tiles in
    EmitContext._tile_map so Copy.emit_mlir can resolve them without
    hitting a KeyError in get_buffer_info().
    """
    from tilelang.tileir.pipeline import build_tileir_module

    prim_func = _get_copy_prim_func()
    module = build_tileir_module(prim_func)

    assert module is not None, "Expected a non-None module from copy_kernel"
    text = str(module)
    assert text, "Expected non-empty MLIR text for copy_kernel"
    assert "cuda_tile.module" in text, f"Expected cuda_tile.module in:\n{text[:400]}"
    assert "constant" in text, f"Expected zero-constant shared tile init in:\n{text[:400]}"


# ---------------------------------------------------------------------------
# Test 9: copy_kernel assembles
# ---------------------------------------------------------------------------


@skip_no_cuda_tile
def test_copy_kernel_assembles():
    """copy_kernel assembles to a valid cubin with > 0 bytes.

    End-to-end: GLOBAL→SHARED→GLOBAL copy emits MLIR that tileiras accepts.
    """
    from tilelang.tileir.pipeline import build_tileir_module
    from tilelang.tileir.assembly import assemble_tileir_module

    arch, target, toolchain = _detect_arch_and_toolchain()

    prim_func = _get_copy_prim_func()
    module = build_tileir_module(prim_func, arch=arch)
    result = assemble_tileir_module(
        module,
        kernel_name="_copy_kernel",
        target=target,
        toolchain=toolchain,
    )
    assert result.cubin, "Expected non-empty cubin from copy_kernel"
    assert len(result.cubin) > 0, "Expected cubin bytes > 0 for copy_kernel"


# ---------------------------------------------------------------------------
# Test 10: copy_kernel alloc_buffers on root block
# ---------------------------------------------------------------------------


@skip_no_cuda_tile
def test_copy_kernel_alloc_buffers_in_root_block():
    """root.alloc_buffers must contain the SHARED buffer Value for copy_kernel.

    lower_kernel populates block.alloc_buffers with non-GLOBAL
    (SHARED/REGISTER) buffer Values so emit_module can materialize
    them as zero-constant tiles.
    """
    from tilelang.tileir.lowering.tir_to_sem import tir_to_sem
    from tilelang.tileir.lowering.sem_to_ir import lower_kernel
    from tilelang.tileir.ir.builder import IRBuilder
    from tilelang.tileir.ir.types import MemSpace

    prim_func = _get_copy_prim_func()
    program = tir_to_sem(prim_func)
    builder = IRBuilder()
    root = lower_kernel(program.kernels[0], builder, program=program)

    assert hasattr(root, "alloc_buffers"), "Block must have alloc_buffers attribute"
    assert len(root.alloc_buffers) == 1, f"Expected 1 SHARED alloc_buffer for copy_kernel, got {len(root.alloc_buffers)}"
    shared_val = root.alloc_buffers[0]
    assert shared_val.name == "shared", f"Expected alloc_buffer name 'shared', got {shared_val.name!r}"
    assert shared_val.type.space == MemSpace.SHARED, f"Expected SHARED space, got {shared_val.type.space}"


# ---------------------------------------------------------------------------
# Test 11: tiny_gemm (SHARED + REGISTER accumulator) builds + assembles
# ---------------------------------------------------------------------------


@skip_no_cuda_tile
def test_tiny_gemm_builds_and_assembles():
    """tiny_gemm with SHARED operands and REGISTER accumulator builds + assembles.

    REGISTER alloc_fragment buffers live in _tile_map alongside
    SHARED buffers, so Gemm.emit_mlir resolves the accumulator via
    _load_buffer_tile / _store_buffer_tile without hitting _buffer_map.
    """
    from tilelang.tileir.pipeline import build_tileir_module
    from tilelang.tileir.assembly import assemble_tileir_module

    arch, target, toolchain = _detect_arch_and_toolchain()

    prim_func = _get_tiny_gemm_prim_func()
    module = build_tileir_module(prim_func, arch=arch)

    assert module is not None
    text = str(module)
    assert "mmaf" in text or "mma" in text, f"Expected MMA op in tiny_gemm MLIR:\n{text[:600]}"

    result = assemble_tileir_module(
        module,
        kernel_name="_tiny_gemm",
        target=target,
        toolchain=toolchain,
    )
    assert result.cubin and len(result.cubin) > 0, "Expected non-empty cubin from tiny_gemm"


@skip_no_cuda_tile
def test_pipeline_selects_skinny_gemm_orientation_only_on_hopper():
    """The architecture-aware pass must be wired into build_tileir_module."""
    from tilelang.tileir.pipeline import build_tileir_module

    prim_func = _get_skinny_gemm_prim_func()
    hopper = str(build_tileir_module(prim_func, arch="sm_90"))
    blackwell = str(build_tileir_module(prim_func, arch="sm_100"))

    assert hopper.count("permute") == 4, hopper
    assert "tile<128x64xf16>" in hopper, hopper
    assert "tile<64x16xf16>" in hopper, hopper
    assert blackwell.count("permute") == 0, blackwell
    assert "tile<16x64xf16>, tile<64x128xf16>" in blackwell, blackwell


# ---------------------------------------------------------------------------
# Test 12: gemv (cross-dtype copy: f32 REGISTER → f16 GLOBAL) assembles
# ---------------------------------------------------------------------------


@skip_no_cuda_tile
def test_gemv_cross_dtype_copy_assembles():
    """A GEMV with f32 accumulation and f16 output assembles successfully."""
    from tilelang.tileir.pipeline import build_tileir_module
    from tilelang.tileir.assembly import assemble_tileir_module

    arch, target, toolchain = _detect_arch_and_toolchain()

    prim_func = _get_gemv_prim_func()
    module = build_tileir_module(prim_func, arch=arch)

    assert module is not None
    text = str(module)
    assert "ftof" in text, f"Expected ftof (f32→f16) cast in gemv MLIR:\n{text[:600]}"

    result = assemble_tileir_module(
        module,
        kernel_name="_gemv_kernel",
        target=target,
        toolchain=toolchain,
    )
    assert result.cubin and len(result.cubin) > 0, "Expected non-empty cubin from gemv_kernel"
