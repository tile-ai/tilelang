"""Semantic IR extraction and structured-lowering shape tests."""

from __future__ import annotations


from dataclasses import fields

import pytest

import tilelang
import tilelang.testing
from tilelang import language as T
from tilelang import tvm as tvm
from tilelang.tileir import checks
from tvm import tirx
from tilelang.jit.adapter.tileir.adapter import TileIRKernelAdapter
from tilelang.tileir.artifact import TileIRLoweringResult
from tilelang.tileir.errors import TileIRLoweringError, TileIRLoweringNotImplementedError
from tilelang.tileir.launch import _split_host_orchestrated_primfunc, extract_launch_metadata
from tilelang.tileir.lowering import lower_primfunc_to_tileir
from tilelang.tileir.semantic import (
    SemanticKernel,
    SemanticProgram,
    SemanticRegion,
    SemanticStmt,
    TileLangSemanticError,
    extract_semantic_program,
)
from tilelang.tileir import tir_analysis as tileir_tir_analysis
from tilelang.backend.target import determine_target

Range = tvm.ir.Range
structural_equal = tvm.ir.structural_equal
Target = tvm.target.Target


from tileir_test_utils import (
    _cuda_target_for_test,
    _call_ops,
    _buffer_argument_names_for_test,
    _tileir_source_for_test,
    _semantic_kinds,
    _semantic_tile_ops,
    _semantic_stmts,
    _load_flash_decode_example,
    _load_mla_paged_example,
)


@tilelang.testing.requires_cuda
def test_tileir_semantic_copy_preserves_frontend_region_slices():
    @T.prim_func
    def kernel(A: T.Tensor((16, 32), T.float32), B: T.Tensor((16, 32), T.float32)):
        with T.Kernel(1, threads=32):
            T.copy(A[4:12, 8:24], B[2:10, 0:16])

    program = extract_semantic_program(kernel)
    copy = next(stmt for stmt in _semantic_stmts(program.kernels[0].body) if dict(stmt.attrs).get("op") == "tl.tileop.copy")

    assert copy.regions == (
        SemanticRegion(buffer="A", access="1", indices=("4", "8"), shape=(8, 16)),
        SemanticRegion(buffer="B", access="2", indices=("2", "0"), shape=(8, 16)),
    )


def test_tileir_semantic_region_rejects_unrelated_call():
    from tilelang.tileir.semantic import _semantic_region

    with pytest.raises(TileLangSemanticError, match="Expected TileLang tile region"):
        _semantic_region(tirx.call_extern("handle", "not_a_region"))


def test_semantic_program_has_no_whole_tir_statement_backrefs_and_preserves_param_order():
    a_handle = tirx.Var("a_handle", "handle")
    scale = tirx.Var("scale", "float32")
    b_handle = tirx.Var("b_handle", "handle")
    a = tirx.decl_buffer((1,), "float32", name="A")
    b = tirx.decl_buffer((1,), "float32", name="B")
    block_x = tirx.IterVar(
        Range(tirx.IntImm("int32", 0), tirx.IntImm("int32", 1)),
        tirx.Var("block_x", "int32"),
        tirx.IterVar.ThreadIndex,
        "blockIdx.x",
    )
    body = tirx.AttrStmt(
        block_x,
        "thread_extent",
        tirx.IntImm("int32", 1),
        tirx.BufferStore(b, tirx.BufferLoad(a, [0]) * scale, [0]),
    )
    prim_func = tirx.PrimFunc(
        [a_handle, scale, b_handle],
        body,
        buffer_map={a_handle: a, b_handle: b},
    ).with_attr("global_symbol", "ordered_params")

    program = extract_semantic_program(prim_func)

    assert "source" not in {field.name for field in fields(SemanticProgram)}
    assert "source" not in {field.name for field in fields(SemanticKernel)}
    assert "source" not in {field.name for field in fields(SemanticStmt)}
    assert program.param_order == ("A", "scale", "B")
    store = next(stmt for stmt in _semantic_stmts(program.kernels[0].body) if stmt.kind == "buffer_store")
    assert store.value is not None
    assert tuple(str(index) for index in store.indices) == ("0",)


def test_tileir_lowering_boundary_reports_structured_coverage_gap(tmp_path):
    toolchain = checks.TileIRToolchain(
        cuda_tile_ir_module=checks.CUDA_TILE_IR_MLIR_MODULE,
        tileiras_path=tmp_path / "tileiras",
        tileiras_version="tileiras 13.4",
    )
    block_x = tirx.IterVar(
        Range(tirx.IntImm("int32", 0), tirx.IntImm("int32", 1)),
        tirx.Var("block_x", "int32"),
        tirx.IterVar.ThreadIndex,
        "blockIdx.x",
    )
    prim_func = tirx.PrimFunc(
        [],
        tirx.AttrStmt(block_x, "thread_extent", tirx.IntImm("int32", 1), tirx.Evaluate(0)),
    ).with_attr("global_symbol", "noop")

    with pytest.raises(TileIRLoweringNotImplementedError, match="Semantic IR boundary") as exc_info:
        lower_primfunc_to_tileir(prim_func, _cuda_target_for_test(), toolchain)
    assert "TIR nodes:" in str(exc_info.value)


def test_tileir_multi_kernel_split_rejects_non_launch_root_statement():
    def launch(name: str):
        block_x = tirx.IterVar(
            Range(tirx.IntImm("int32", 0), tirx.IntImm("int32", 1)),
            tirx.Var(name, "int32"),
            tirx.IterVar.ThreadIndex,
            "blockIdx.x",
        )
        return tirx.AttrStmt(block_x, "thread_extent", tirx.IntImm("int32", 1), tirx.Evaluate(0))

    root = tirx.SBlock(
        [],
        [],
        [],
        "root",
        tirx.SeqStmt(
            [
                launch("first"),
                tirx.Evaluate(tirx.call_extern("handle", "host_bookkeeping")),
                launch("second"),
            ]
        ),
    )
    prim_func = tirx.PrimFunc([], tirx.SBlockRealize([], tirx.const(True, "bool"), root)).with_attr("global_symbol", "main")

    with pytest.raises(TileIRLoweringError, match="non-kernel root statement"):
        _split_host_orchestrated_primfunc(prim_func)


def test_tileir_semantic_discovery_rejects_non_launch_root_statement():
    block_x = tirx.IterVar(
        Range(tirx.IntImm("int32", 0), tirx.IntImm("int32", 1)),
        tirx.Var("block_x", "int32"),
        tirx.IterVar.ThreadIndex,
        "blockIdx.x",
    )
    launch = tirx.AttrStmt(block_x, "thread_extent", tirx.IntImm("int32", 1), tirx.Evaluate(0))
    root = tirx.SBlock(
        [],
        [],
        [],
        "root",
        tirx.SeqStmt([launch, tirx.Evaluate(tirx.call_extern("handle", "host_bookkeeping"))]),
    )
    prim_func = tirx.PrimFunc([], tirx.SBlockRealize([], tirx.const(True, "bool"), root)).with_attr("global_symbol", "main")

    with pytest.raises(TileIRLoweringError, match="non-kernel root statement"):
        extract_semantic_program(prim_func)


def test_tileir_extract_launch_metadata_uses_tile_block_grid():
    prim_func = (
        tirx.PrimFunc([], tirx.Evaluate(0))
        .with_attr(
            "thread_extent",
            {
                "blockIdx.x": tirx.IntImm("int32", 4),
                "blockIdx.y": tirx.IntImm("int32", 3),
                "threadIdx.x": tirx.IntImm("int32", 128),
                "threadIdx.y": tirx.IntImm("int32", 2),
            },
        )
        .with_attr("dyn_shared_memory_buf", tirx.IntImm("int32", 256))
    )

    metadata = extract_launch_metadata(prim_func)

    assert metadata.grid == (4, 3, 1)
    assert metadata.block == (1, 1, 1)
    assert metadata.dynamic_smem_bytes == 256


def test_tileir_extract_launch_metadata_keeps_dynamic_grid_expr():
    prim_func = tirx.PrimFunc([], tirx.Evaluate(0)).with_attr("thread_extent", {"blockIdx.x": tirx.Var("n", "int32")})

    metadata = extract_launch_metadata(prim_func)

    assert isinstance(metadata.grid[0], tirx.Var)
    assert metadata.grid[0].name == "n"
    assert metadata.grid[1:] == (1, 1)


def test_tileir_extract_launch_metadata_from_tilelang_kernel_body():
    from testing.python.jit.test_tilelang_jit_gemm import matmul_kernel_jit

    prim_func = matmul_kernel_jit.get_tir(
        128,
        128,
        64,
        64,
        64,
        32,
        False,
        False,
        T.float16,
        T.float16,
        T.float32,
        1,
        128,
    )

    metadata = extract_launch_metadata(prim_func)

    assert metadata.grid == (2, 2, 1)
    assert metadata.block == (1, 1, 1)


def test_tileir_prepare_keeps_high_level_tile_ops_before_cuda_lowering():
    from testing.python.jit.test_tilelang_jit_gemm import matmul_kernel_jit

    prim_func = matmul_kernel_jit.get_tir(
        128,
        128,
        64,
        64,
        64,
        32,
        False,
        False,
        T.float16,
        T.float16,
        T.float32,
        1,
        128,
    )
    _, prepared = TileIRKernelAdapter._prepare_device_module(prim_func, determine_target("tileir -arch=sm_120", return_object=True))

    ops = _call_ops(prepared)

    assert {"tl.tileop.copy", "tl.tileop.gemm"}.issubset(ops)
    assert "tir.ptx_mma" not in ops
    assert "tl.ptx_ldmatrix" not in ops
    assert "tl.tma_load" not in ops
    assert "tir.ptx_commit_group" not in ops
    assert "tir.ptx_wait_group" not in ops


def test_tileir_prepare_flash_decode_no_split_stops_before_cuda_intrinsics():
    example = _load_flash_decode_example()
    prim_func = example.flashattn.jit_impl.get_tir(1, 32, 8, 256, 128, 128, 64, 1, 2, 128)

    _, prepared = TileIRKernelAdapter._prepare_device_module(
        prim_func,
        determine_target("tileir -arch=sm_120", return_object=True),
        example.get_pass_configs(),
    )
    ops = _call_ops(prepared)

    assert {"tl.tileop.copy", "tl.tileop.gemm", "tl.tileop.reduce"}.issubset(ops)
    assert "tir.ptx_mma" not in ops
    assert "tl.ptx_ldmatrix" not in ops
    assert "tl.tma_load" not in ops
    assert "tir.ptx_commit_group" not in ops
    assert "tir.ptx_wait_group" not in ops


def test_tileir_prepare_accepts_flash_decode_multi_kernel_split_before_cuda_intrinsics():
    example = _load_flash_decode_example()
    prim_func = example.flashattn.jit_impl.get_tir(1, 32, 8, 256, 128, 128, 64, 2, 2, 128)

    _, prepared = TileIRKernelAdapter._prepare_device_module(
        prim_func,
        determine_target("tileir -arch=sm_120", return_object=True),
        example.get_pass_configs(),
    )
    ops = _call_ops(prepared)

    assert {"tl.tileop.copy", "tl.tileop.gemm", "tl.tileop.reduce"}.issubset(ops)
    assert "tir.ptx_mma" not in ops
    assert "tl.ptx_ldmatrix" not in ops


def test_tileir_lowering_splits_flash_decode_host_orchestration(monkeypatch, tmp_path):
    import tilelang.tileir.pipeline as _pipeline_module
    from tilelang.tileir.semantic import extract_semantic_program

    example = _load_flash_decode_example()
    prim_func = example.flashattn.jit_impl.get_tir(1, 32, 8, 256, 128, 128, 64, 2, 2, 128)
    toolchain = checks.TileIRToolchain(
        cuda_tile_ir_module=checks.CUDA_TILE_IR_MLIR_MODULE,
        tileiras_path=tmp_path / "tileiras",
        tileiras_version="tileiras 13.4",
    )
    lowered_symbols = []

    def fake_lower_single_kernel(sub_func, target, toolchain=None, pass_configs=None):
        del target
        del pass_configs
        symbol = str(sub_func.attrs["global_symbol"])
        # Verify the sub-func has exactly one kernel in its semantic program.
        sub_program = extract_semantic_program(sub_func)
        assert sub_program.kernels[0].name == symbol
        assert len(sub_program.kernels) == 1
        lowered_symbols.append(symbol)
        return TileIRLoweringResult(
            kernel_name=symbol,
            cubin=symbol.encode(),
            tileir_source=f"module @{symbol}",
            launch_metadata=extract_launch_metadata(sub_func),
            argument_names=tuple(sub_func.buffer_map[param].name for param in sub_func.params if param in sub_func.buffer_map),
        )

    monkeypatch.setattr(_pipeline_module, "lower_single_kernel_to_tileir", fake_lower_single_kernel)

    result = lower_primfunc_to_tileir(prim_func, _cuda_target_for_test(), toolchain)

    assert result.kernel_name == "flashattn_gqa_decode_split"
    assert lowered_symbols == ["flashattn_gqa_decode_split_0", "flashattn_gqa_decode_split_1"]
    assert [buffer.name for buffer in result.temporary_buffers] == ["glse", "Output_partial"]
    assert result.temporary_buffers[0].shape == (1, 32, 2)
    assert result.temporary_buffers[1].shape == (1, 32, 2, 128)
    assert len(result.kernels) == 2
    assert result.kernels[0].launch_metadata.grid == (1, 8, 2)
    assert result.kernels[1].launch_metadata.grid == (32, 1, 1)
    assert result.kernels[0].argument_names == ("Q", "K", "V", "mask", "glse", "Output_partial")
    assert result.kernels[1].argument_names == ("Output", "glse", "Output_partial")


def test_tileir_lowering_splits_grid_sync_into_ordered_host_launches(monkeypatch, tmp_path):
    @tilelang.jit
    def sync_kernel(n: int):
        @T.prim_func
        def main(A: T.Tensor([n], T.float32)):
            with T.Kernel(1, threads=1) as block_id:
                for w in T.serial(1):
                    tile_id = block_id + w
                    bid = tile_id // 1
                    if bid < n:
                        A[0] = 1.0

                T.sync_grid()

                for w in T.serial(1):
                    tile_id = block_id + w
                    hid = tile_id // n
                    bid = tile_id % 1
                    if bid < n and hid < n:
                        A[bid] = 2.0

        return main

    prim_func = sync_kernel.get_tir(4)
    toolchain = checks.TileIRToolchain(
        cuda_tile_ir_module=checks.CUDA_TILE_IR_MLIR_MODULE,
        tileiras_path=tmp_path / "tileiras",
        tileiras_version="tileiras 13.4",
    )
    lowered_symbols = []

    def fake_lower_single_kernel(sub_func, target, toolchain=None, pass_configs=None):
        del target
        del toolchain
        del pass_configs
        assert "tl.sync_grid" not in _call_ops(sub_func)
        symbol = str(sub_func.attrs["global_symbol"])
        lowered_symbols.append(symbol)
        return TileIRLoweringResult(
            kernel_name=symbol,
            cubin=b"cubin",
            tileir_source=f"module @{symbol}",
            launch_metadata=extract_launch_metadata(sub_func),
            argument_names=_buffer_argument_names_for_test(sub_func),
        )

    import tilelang.tileir.pipeline as _pipeline_module

    monkeypatch.setattr(_pipeline_module, "lower_single_kernel_to_tileir", fake_lower_single_kernel)

    target = _cuda_target_for_test()
    result = lower_primfunc_to_tileir(prim_func, target, toolchain)
    _, prepared = TileIRKernelAdapter._prepare_device_module(prim_func, target)
    cached_artifact = TileIRKernelAdapter._attach_argument_metadata(result, prepared)
    TileIRKernelAdapter._validate_cached_artifact_abi(cached_artifact, prepared)

    assert lowered_symbols == ["main_0", "main_1"]
    assert len(result.kernels) == 2
    assert result.kernels[0].launch_metadata.grid == (1, 1, 1)
    assert result.kernels[1].launch_metadata.grid == (1, 1, 1)
    assert result.temporary_buffers == ()


def test_tileir_semantic_ir_extracts_flash_decode_split_contract():
    example = _load_flash_decode_example()
    prim_func = example.flashattn.jit_impl.get_tir(1, 32, 8, 256, 128, 128, 64, 2, 2, 128)

    program = extract_semantic_program(prim_func)

    assert program.name == "flashattn_gqa_decode_split"
    assert [buffer.name for buffer in program.global_alloc_buffers] == ["glse", "Output_partial"]
    assert len(program.kernels) == 2
    assert program.kernels[0].grid == ("1", "8", "2")
    assert program.kernels[1].grid == ("32", "1", "1")
    assert {"tl.tileop.copy", "tl.tileop.gemm", "tl.tileop.reduce"}.issubset(_semantic_tile_ops(program.kernels[0].body))
    assert {"for", "thread_extent", "block"}.issubset(_semantic_kinds(program.kernels[0].body))
    assert {"shared.dyn", "local.fragment"}.issubset({buffer.scope for buffer in program.kernels[0].alloc_buffers})
    loop_attrs = [dict(stmt.attrs) for stmt in _semantic_stmts(program.kernels[0].body) if stmt.kind == "for"]
    assert any(attrs.get("kind") == "pipelined" and attrs.get("annotation.num_stages") == "2" for attrs in loop_attrs)
    gemm_regions = [
        stmt.regions
        for stmt in _semantic_stmts(program.kernels[0].body)
        if stmt.kind == "tile_op" and dict(stmt.attrs).get("op") == "tl.tileop.gemm"
    ]
    assert any([region.buffer for region in regions] == ["Q_shared", "K_shared", "acc_s"] for regions in gemm_regions)


def test_tileir_semantic_ir_extracts_gemm_contract_attrs():
    @tilelang.jit
    def gemm_contract_kernel(a, b, c):
        a: T.Tensor((32, 64), T.float16)
        b: T.Tensor((64, 32), T.float16)
        c: T.Tensor((32, 32), T.float32)

        with T.Kernel(1):
            a_shared = T.alloc_shared((32, 64), T.float16)
            b_shared = T.alloc_shared((64, 32), T.float16)
            acc = T.alloc_fragment((32, 32), T.float32)
            T.copy(a, a_shared)
            T.copy(b, b_shared)
            T.gemm(
                a_shared,
                b_shared,
                acc,
                policy=T.GemmWarpPolicy.FullRow,
                clear_accum=True,
            )
            T.copy(acc, c)

    prim_func = gemm_contract_kernel.get_tir(None, None, None)
    program = extract_semantic_program(prim_func)
    gemm_attrs = [
        dict(stmt.attrs)
        for stmt in _semantic_stmts(program.kernels[0].body)
        if stmt.kind == "tile_op" and dict(stmt.attrs).get("op") == "tl.tileop.gemm"
    ]

    assert len(gemm_attrs) == 1
    assert gemm_attrs[0]["policy"] == "1"
    assert gemm_attrs[0]["clear_accum"] == "1"
    assert gemm_attrs[0]["M"] == "32"
    assert gemm_attrs[0]["N"] == "32"
    assert gemm_attrs[0]["K"] == "64"
    assert gemm_attrs[0]["transpose_A"] == "0"
    assert gemm_attrs[0]["transpose_B"] == "0"


def test_tileir_semantic_ir_extracts_mla_paged_control_flow_contract():
    example = _load_mla_paged_example()
    prim_func = example.mla_decode_tilelang.get_tir(
        1,
        32,
        1,
        1024,
        512,
        64,
        64,
        16,
        1,
        64,
        None,
    )

    program = extract_semantic_program(prim_func)

    assert program.name == "main_no_split"
    assert len(program.kernels) == 1
    assert program.kernels[0].grid == ("1", "2", "1")
    kinds = _semantic_kinds(program.kernels[0].body)
    assert {"for", "let", "if", "block", "thread_extent"}.issubset(kinds)
    assert {"tl.tileop.copy", "tl.tileop.gemm", "tl.tileop.reduce"}.issubset(_semantic_tile_ops(program.kernels[0].body))
    loop_attrs = [dict(stmt.attrs) for stmt in _semantic_stmts(program.kernels[0].body) if stmt.kind == "for"]
    assert any(attrs.get("kind") == "pipelined" and attrs.get("annotation.num_stages") == "2" for attrs in loop_attrs)


def test_tileir_semantic_ir_rejects_unsupported_evaluate_call():
    block_x = tirx.IterVar(
        Range(tirx.IntImm("int32", 0), tirx.IntImm("int32", 1)),
        tirx.Var("block_x", "int32"),
        tirx.IterVar.ThreadIndex,
        "blockIdx.x",
    )
    prim_func = tirx.PrimFunc(
        [],
        tirx.AttrStmt(
            block_x,
            "thread_extent",
            tirx.IntImm("int32", 1),
            tirx.Evaluate(tirx.call_intrin("handle", "tirx.tvm_call_packed", "unsupported")),
        ),
    ).with_attr("global_symbol", "unsupported_call")

    with pytest.raises(TileLangSemanticError, match="Unsupported TileLang tile operation"):
        extract_semantic_program(prim_func)


def _prim_func_with_evaluate_call(call, name):
    block_x = tirx.IterVar(
        Range(tirx.IntImm("int32", 0), tirx.IntImm("int32", 1)),
        tirx.Var("block_x", "int32"),
        tirx.IterVar.ThreadIndex,
        "blockIdx.x",
    )
    return tirx.PrimFunc(
        [],
        tirx.AttrStmt(block_x, "thread_extent", tirx.IntImm("int32", 1), tirx.Evaluate(call)),
    ).with_attr("global_symbol", name)


def test_tileir_semantic_ir_rejects_gemm_sp_with_sparse_mma_reason():
    """`tl.tileop.gemm_sp` (2:4 structured-sparse GEMM) must be rejected with
    a message explaining WHY: the cuda_tile dialect has no sparse MMA op, and
    a dense fallback would be numerically wrong (B is 2:4-compressed)."""
    prim_func = _prim_func_with_evaluate_call(
        tirx.Call("handle", tvm.ir.Op.get("tl.tileop.gemm_sp"), []),
        "gemm_sp_kernel",
    )
    with pytest.raises(TileLangSemanticError, match="no counterpart in CUDA Tile IR"):
        extract_semantic_program(prim_func)


def test_tileir_semantic_ir_names_unknown_call_extern():
    """An unknown `tir.call_extern` must be rejected with the extern function
    NAME in the message (not just the generic `tir.call_extern` op name)."""
    prim_func = _prim_func_with_evaluate_call(
        tirx.call_extern("handle", "my_mystery_cuda_helper"),
        "extern_kernel",
    )
    with pytest.raises(TileLangSemanticError, match="my_mystery_cuda_helper"):
        extract_semantic_program(prim_func)


def test_tileir_static_extent_proof_keeps_non_affine_let_vars_symbolic():
    tile_id = tirx.Var("tile_id", "int32")
    sid = tirx.Var("sid", "int32")
    k = tirx.Var("k", "int32")
    kv_start = tirx.Var("kv_start", "int32")
    kv_end = tirx.Var("kv_end", "int32")
    bindings = {
        "sid": tile_id % 2,
        "kv_start": sid * 64 + k * 64,
        "kv_end": sid * 64 + (k + 1) * 64,
    }

    assert tileir_tir_analysis._static_int_expr_value(kv_end - kv_start, bindings) == 64


def test_tileir_partition_index_exact_division_simplifies_tile_offsets():
    block = tirx.Var("block", "int32")
    loop = tirx.Var("loop", "int32")

    assert structural_equal(tileir_tir_analysis._divide_tir_expr_if_exact(block * 64, 64), block)
    assert structural_equal(tileir_tir_analysis._divide_tir_expr_if_exact((block * 64) + (loop * 64), 64), block + loop)
    assert tileir_tir_analysis._divide_tir_expr_if_exact(block + 1, 64) is None


def test_tilelang_math_frontend_emits_expected_tir_ops():
    x = tirx.Var("x", "float32")
    y = tirx.Var("y", "float32")
    i = tirx.Var("i", "int32")

    assert _call_ops(tirx.PrimFunc([], tirx.Evaluate(T.abs(x)))) == {"tir.fabs"}
    assert isinstance(T.abs(i), tirx.Select)
    assert _call_ops(tirx.PrimFunc([], tirx.Evaluate(T.pow(x, y)))) == {"tir.pow"}
    assert _call_ops(tirx.PrimFunc([], tirx.Evaluate(T.pow(x, 2)))) == {"tl.pow_of_int"}


def test_tileir_semantic_ir_lowers_warp_specialize_as_sequential_passthrough():
    """warp_specialize (T.ws) lowers as a transparent sequential pass-through.

    The body of each ``T.ws`` block is lowered in program order while the
    downstream assembler owns warp-group routing. The producer/consumer
    sections therefore form a valid sequential kernel here: ``src = a``
    followed by ``b = src``.
    """
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def warp_specialize_kernel(a, b):
        a: T.Tensor((32,), T.float32)
        b: T.Tensor((32,), T.float32)

        with T.Kernel(1, threads=256):
            src = T.alloc_fragment((32,), T.float32)
            with T.ws(0):
                T.copy(a, src)
            with T.ws(1):
                T.copy(src, b)

    # No rejection: the two ws sections lower in order to a valid kernel.
    source = _tileir_source_for_test(warp_specialize_kernel, None, None)
    assert "cuda_tile" in source
    # Both copies are emitted (a -> src load, src -> b store).
    assert "load_view_tko" in source
    assert "store_view_tko" in source
