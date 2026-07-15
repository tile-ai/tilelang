"""Structured lowering coverage: accepted, supported, and rejected constructs."""

from __future__ import annotations


import re
import pytest

import tilelang
import tilelang.testing
from tilelang import language as T
from tilelang import tvm as tvm
from tilelang.tileir import checks
from tvm import tirx
from tilelang.jit.adapter.tileir.adapter import TileIRKernelAdapter
from tilelang.tileir.errors import TileIRLoweringNotImplementedError, _UnsupportedTileIRNode
from tilelang.tileir import lowering as tileir_lowering
from tilelang.tileir.semantic import extract_semantic_program
from tilelang.backend.target import determine_target

Range = tvm.ir.Range
structural_equal = tvm.ir.structural_equal
Target = tvm.target.Target


from tileir_test_utils import (
    _build_tileir_module_for_test,
    _prepared_tileir_kernel_for_test,
    _tileir_source_for_test,
    _semantic_kinds,
    _semantic_tile_ops,
)


def test_tileir_structured_lowering_accepts_barrier_scheduling_hints():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

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
            tirx.Evaluate(
                tirx.call_intrin(
                    "handle",
                    tirx.op.Op.get("tl.mbarrier_wait_parity"),
                    tirx.IntImm("int32", 0),
                    tirx.IntImm("int32", 0),
                )
            ),
        ),
    ).with_attr("global_symbol", "mbarrier_wait")
    target = determine_target("tileir -arch=sm_120", return_object=True)
    extract_semantic_program(prim_func)

    module = _build_tileir_module_for_test(prim_func, target)

    assert "tl.mbarrier_wait_parity" not in str(module)


def test_tileir_structured_lowering_accepts_storage_sync_hint():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

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
            tirx.Evaluate(tirx.call_intrin("handle", "tirx.tvm_storage_sync", "shared")),
        ),
    ).with_attr("global_symbol", "storage_sync")
    target = determine_target("tileir -arch=sm_120", return_object=True)
    extract_semantic_program(prim_func)

    module = _build_tileir_module_for_test(prim_func, target)

    assert "tvm_storage_sync" not in str(module)


def test_tileir_structured_lowering_supports_debug_assert_and_print():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    block_x = tirx.IterVar(
        Range(tirx.IntImm("int32", 0), tirx.IntImm("int32", 1)),
        tirx.Var("block_x", "int32"),
        tirx.IterVar.ThreadIndex,
        "blockIdx.x",
    )
    body = tirx.SeqStmt(
        [
            tirx.Evaluate(
                tirx.call_intrin(
                    "void",
                    tirx.op.Op.get("tl.device_assert_with_msg"),
                    tirx.IntImm("bool", 1),
                    tirx.StringImm("ok"),
                )
            ),
            tirx.Evaluate(tirx.call_extern("handle", "debug_print_var", "value", tirx.IntImm("int32", 7))),
        ]
    )
    prim_func = tirx.PrimFunc(
        [],
        tirx.AttrStmt(block_x, "thread_extent", tirx.IntImm("int32", 1), body),
    ).with_attr("global_symbol", "debug_ops")
    target = determine_target("tileir -arch=sm_120", return_object=True)
    extract_semantic_program(prim_func)
    module = _build_tileir_module_for_test(prim_func, target)
    source = str(module)

    assert "assert" in source
    assert "print_tko" in source


def test_tileir_structured_lowering_pads_non_power_local_tiles():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def non_power_tile_kernel(out):
        n = T.dynamic("n")
        out: T.Tensor((n, 24), T.float32)

        with T.Kernel(n) as bx:
            tmp = T.alloc_fragment(24, T.float32)
            for j in T.Parallel(24):
                tmp[j] = T.cast(j, T.float32)
                out[bx, j] = tmp[j]

    target = determine_target("tileir -arch=sm_120", return_object=True)
    prim_func = non_power_tile_kernel.get_tir(None)
    _, prepared = TileIRKernelAdapter._prepare_device_module(prim_func, target)
    tileir_lowering._extract_semantic_program_for_lowering(prepared, target)
    module = _build_tileir_module_for_test(prepared, target)

    source = str(module)

    assert "tile<24x" not in source
    assert "tile<32xf32>" in source


def test_tileir_structured_lowering_broadcasts_local_vector_load():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def vector_broadcast_kernel(out):
        out: T.Tensor((128, 64), T.float32)

        with T.Kernel(1):
            vec = T.alloc_fragment((128,), T.float32)
            mat = T.alloc_fragment((128, 64), T.float32)
            for i in T.Parallel(128):
                vec[i] = T.cast(i, T.float32)
            for i, j in T.Parallel(128, 64):
                mat[i, j] = vec[i]
            T.copy(mat, out)

    source = _tileir_source_for_test(vector_broadcast_kernel, None)

    assert "tile<128x1xf32> -> tile<128x64xf32>" in source
    assert "cat " not in source


def test_tileir_rejects_rank1_simt_scalar_gather():
    """A collective thread-index tile must not be collapsed to lane zero."""
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def simt_scalar_gather(inp, out):
        inp: T.Tensor((128,), T.int8)
        out: T.Tensor((128,), T.int8)

        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((128,), T.int8)
            tx = T.get_thread_binding()
            T.copy(inp, shared)
            out[tx] = shared[tx]

    with pytest.raises(_UnsupportedTileIRNode, match="per-thread SIMT gather"):
        _tileir_source_for_test(simt_scalar_gather, None, None)


def test_tileir_rejects_rank1_simt_scalar_scatter():
    """A rank-1 thread index cannot select one scalar shared-tile element."""
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def simt_scalar_scatter(out):
        out: T.Tensor((128,), T.int8)

        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((128,), T.int8)
            tx = T.get_thread_binding()
            shared[tx] = T.cast(7, T.int8)
            T.copy(shared, out)

    with pytest.raises(_UnsupportedTileIRNode, match="per-thread SIMT scatter"):
        _tileir_source_for_test(simt_scalar_scatter, None)


def test_tileir_structured_lowering_projects_singleton_parallel_store():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def projected_store_kernel(out):
        out: T.Tensor((64,), T.float32)

        with T.Kernel(1):
            vec = T.alloc_fragment((64,), T.float32)
            for _i, j in T.Parallel(1, 64):
                vec[j] = vec[j] + T.cast(j, T.float32)
            T.copy(vec, out)

    source = _tileir_source_for_test(projected_store_kernel, None)

    assert "tile<1x64xf32> -> tile<64xf32>" in source


def test_tileir_structured_lowering_supports_parallel_alloc_var():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def parallel_alloc_var_kernel(out):
        out: T.Tensor((16, 8), T.float32)

        with T.Kernel(1):
            frag = T.alloc_fragment((16, 8), T.float32)
            for i, j in T.Parallel(16, 8):
                value = T.alloc_var(T.float32)
                if i + j > 0:
                    value = 1.0
                else:
                    value = 0.0
                frag[i, j] = value
            T.copy(frag, out)

    source = _tileir_source_for_test(parallel_alloc_var_kernel, None)

    assert "select" in source


def test_tileir_structured_lowering_lowers_i2u_decode_extern():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def i2u_decode_kernel(src, out):
        src: T.Tensor((4,), T.int8)
        out: T.Tensor((16,), T.int8)

        with T.Kernel(1):
            packed = T.alloc_local((4,), T.int8)
            decoded = T.alloc_local((16,), T.int8)
            T.copy(src, packed)
            T.call_extern(
                "handle",
                "decode_i2u_to_i8s",
                T.access_ptr(packed, "r"),
                T.access_ptr(decoded, "w"),
            )
            T.copy(decoded, out)

    source = _tileir_source_for_test(i2u_decode_kernel, None, None)

    assert "andi" in source
    assert "trunci" in source


def test_tileir_structured_lowering_lowers_dp4a_extern():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def dp4a_kernel(a, b, out):
        a: T.Tensor((4,), T.int8)
        b: T.Tensor((4,), T.int8)
        out: T.Tensor((1,), T.int32)

        with T.Kernel(1):
            a_local = T.alloc_local((4,), T.int8)
            b_local = T.alloc_local((4,), T.int8)
            acc = T.alloc_local((1,), T.int32)
            T.copy(a, a_local)
            T.copy(b, b_local)
            T.clear(acc)
            T.dp4a(a_local[0], b_local[0], acc[0])
            T.copy(acc, out)

    source = _tileir_source_for_test(dp4a_kernel, None, None, None)

    assert "exti" in source
    assert "muli" in source
    assert "addi" in source


def test_tileir_structured_lowering_lowers_cumsum_to_scan():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def cumsum_kernel(a, b):
        a: T.Tensor((32,), T.float32)
        b: T.Tensor((32,), T.float32)

        with T.Kernel(1, threads=128):
            tile = T.alloc_shared((32,), T.float32)
            T.copy(a, tile)
            T.cumsum(src=tile, dim=0)
            T.copy(tile, b)

    target, prepared, semantic_program = _prepared_tileir_kernel_for_test(cumsum_kernel, None, None)
    assert "tl.tileop.cumsum" in _semantic_tile_ops(semantic_program.kernels[0].body)

    module = _build_tileir_module_for_test(prepared, target)
    source = str(module)

    assert "scan" in source


def test_tileir_structured_lowering_lowers_while_loop_control():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def while_kernel(out):
        out: T.Tensor((1,), T.int32)

        with T.Kernel(1, threads=1):
            i = T.alloc_var(T.int32, 0)
            total = T.alloc_var(T.int32, 0)
            while i < 4:
                total += i
                i += 1
            out[0] = total

    target, prepared, semantic_program = _prepared_tileir_kernel_for_test(while_kernel, None)
    assert "while" in _semantic_kinds(semantic_program.kernels[0].body)

    module = _build_tileir_module_for_test(prepared, target)
    source = str(module)

    assert "loop" in source
    assert "break" in source
    assert "continue" in source


def test_tileir_structured_lowering_supports_while_break_and_continue():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def while_control_kernel(out):
        out: T.Tensor((1,), T.int32)

        with T.Kernel(1, threads=1):
            i = T.alloc_var(T.int32, 0)
            total = T.alloc_var(T.int32, 0)
            while i < 6:
                i += 1
                if i == 2:
                    continue
                if i == 5:
                    break
                total += i
            out[0] = total

    target, prepared, semantic_program = _prepared_tileir_kernel_for_test(while_control_kernel, None)
    assert {"tir.break_loop", "tir.continue_loop"}.issubset(_semantic_tile_ops(semantic_program.kernels[0].body))

    module = _build_tileir_module_for_test(prepared, target)
    source = str(module)

    assert "break" in source
    assert "continue" in source


def test_tileir_structured_lowering_preserves_f32_gemm_as_tf32_mma():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def f32_gemm_kernel(a, b, c):
        a: T.Tensor((32, 256), T.float32)
        b: T.Tensor((32, 256), T.float32)
        c: T.Tensor((32, 32), T.float32)

        with T.Kernel(1):
            a_shared = T.alloc_shared((32, 256), T.float32)
            b_shared = T.alloc_shared((32, 256), T.float32)
            acc = T.alloc_fragment((32, 32), T.float32)
            T.copy(a, a_shared)
            T.copy(b, b_shared)
            T.clear(acc)
            T.gemm(a_shared, b_shared, acc, transpose_B=True)
            T.copy(acc, c)

    target = determine_target("tileir -arch=sm_120", return_object=True)
    prim_func = f32_gemm_kernel.get_tir(None, None, None)
    _, prepared = TileIRKernelAdapter._prepare_device_module(prim_func, target)
    tileir_lowering._extract_semantic_program_for_lowering(prepared, target)
    module = _build_tileir_module_for_test(prepared, target)

    source = str(module)

    assert "tile<32x256xtf32>" in source
    assert "tile<256x32xtf32>" in source
    assert "tile<32x256xf32>, tile<256x32xf32>, tile<32x32xf32>" not in source


def test_tileir_structured_lowering_honors_gemm_clear_accum():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def clear_accum_kernel(a, b, c):
        a: T.Tensor((32, 64), T.float16)
        b: T.Tensor((64, 32), T.float16)
        c: T.Tensor((32, 32), T.float32)

        with T.Kernel(1):
            a_shared = T.alloc_shared((32, 64), T.float16)
            b_shared = T.alloc_shared((64, 32), T.float16)
            acc = T.alloc_fragment((32, 32), T.float32)
            T.copy(a, a_shared)
            T.copy(b, b_shared)
            T.fill(acc, 1)
            T.gemm(a_shared, b_shared, acc, clear_accum=True)
            T.copy(acc, c)

    target = determine_target("tileir -arch=sm_120", return_object=True)
    prim_func = clear_accum_kernel.get_tir(None, None, None)
    _, prepared = TileIRKernelAdapter._prepare_device_module(prim_func, target)
    tileir_lowering._extract_semantic_program_for_lowering(prepared, target)
    module = _build_tileir_module_for_test(prepared, target)

    source = str(module)

    assert " = mmaf " in source
    assert "<f32: 0.000000e+00> : tile<32x32xf32>" in source


def test_tileir_structured_lowering_uses_fma_under_fast_math():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def fast_math_kernel(a, b, c):
        a: T.Tensor((32,), T.float32)
        b: T.Tensor((32,), T.float32)
        c: T.Tensor((32,), T.float32)

        with T.Kernel(1):
            a_frag = T.alloc_fragment((32,), T.float32)
            b_frag = T.alloc_fragment((32,), T.float32)
            c_frag = T.alloc_fragment((32,), T.float32)
            T.copy(a, a_frag)
            T.copy(b, b_frag)
            for i in T.Parallel(32):
                c_frag[i] = a_frag[i] * 2.0 + b_frag[i]
            T.copy(c_frag, c)

    target = determine_target("tileir -arch=sm_120", return_object=True)
    prim_func = fast_math_kernel.get_tir(None, None, None)
    _, prepared = TileIRKernelAdapter._prepare_device_module(prim_func, target)
    tileir_lowering._extract_semantic_program_for_lowering(prepared, target)

    precise_module = _build_tileir_module_for_test(prepared, target)
    fast_module = _build_tileir_module_for_test(
        prepared,
        target,
        options=tileir_lowering._lowering_options(prepared, {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}),
    )

    assert " fma " not in str(precise_module)
    assert " fma " in str(fast_module)


def test_tileir_structured_lowering_supports_more_math_intrinsics():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def math_intrinsics_kernel(a, b):
        a: T.Tensor((32,), T.float32)
        b: T.Tensor((32,), T.float32)

        with T.Kernel(1):
            frag = T.alloc_fragment((32,), T.float32)
            out = T.alloc_fragment((32,), T.float32)
            T.copy(a, frag)
            for i in T.Parallel(32):
                value = T.abs(frag[i]) + T.sqrt(frag[i]) + T.log(frag[i]) + T.tanh(frag[i])
                out[i] = value + T.pow(frag[i], 2.0)
            T.copy(out, b)

    source = _tileir_source_for_test(math_intrinsics_kernel, None, None)

    assert "absf" in source
    assert "sqrt" in source
    assert "log" in source
    assert "tanh" in source
    assert "pow" in source


def test_tileir_structured_lowering_supports_integer_abs_select():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def int_abs_kernel(out):
        out: T.Tensor((32,), T.int32)

        with T.Kernel(1):
            frag = T.alloc_fragment((32,), T.int32)
            for i in T.Parallel(32):
                frag[i] = T.abs(T.cast(i, T.int32) - 16)
            T.copy(frag, out)

    source = _tileir_source_for_test(int_abs_kernel, None)

    assert "select" in source


def test_tileir_structured_lowering_supports_shift_ops():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def shift_kernel(a, b):
        a: T.Tensor((32,), T.uint32)
        b: T.Tensor((32,), T.uint32)

        with T.Kernel(1):
            packed = T.alloc_fragment((32,), T.uint32)
            shifted = T.alloc_fragment((32,), T.uint32)
            T.copy(a, packed)
            for i in T.Parallel(32):
                shifted[i] = (packed[i] >> T.cast(1, T.uint32)) << T.cast(1, T.uint32)
            T.copy(shifted, b)

    source = _tileir_source_for_test(shift_kernel, None, None)

    assert "shri" in source
    assert "unsigned" in source
    assert "shli" in source


def test_tileir_structured_lowering_elides_varlen_negative_offset_zero_div_guard():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def guarded_div_kernel(a, n, o):
        a: T.Tensor((64, 64), T.float32)
        n: T.int32
        o: T.Tensor((64, 64), T.float32)

        with T.Kernel(1):
            frag = T.alloc_fragment((64, 64), T.float32)
            T.copy(a, frag)
            for i, j in T.Parallel(64, 64):
                frag[i, j] = 0 if i + n < 0 else frag[i, j] / 2.0
            T.copy(frag, o)

    target = determine_target("tileir -arch=sm_120", return_object=True)
    prim_func = guarded_div_kernel.get_tir(None, None, None)
    _, prepared = TileIRKernelAdapter._prepare_device_module(prim_func, target)
    tileir_lowering._extract_semantic_program_for_lowering(prepared, target)
    module = _build_tileir_module_for_test(prepared, target)

    source = str(module)

    # The zero-div guard (0 if i+n<0 else frag/2.0) is lowered as a
    # select+divf pair. Guard elision is left to the cuda_tile MLIR optimizer.
    assert "divf" in source
    # The guard remains explicit at the TileIR level.
    assert "select" in source


def test_tileir_structured_lowering_keeps_dynamic_shape_stride_alignment_assumptions():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def dynamic_stride_kernel(a, b):
        n = T.dynamic("n")
        a: T.Tensor((n, 256), T.bfloat16)
        b: T.Tensor((n, 32), T.float32)

        with T.Kernel(T.ceildiv(n, 32)) as bx:
            a_frag = T.alloc_fragment((32, 256), T.bfloat16)
            b_frag = T.alloc_fragment((32, 32), T.float32)
            T.copy(a[bx * 32, 0], a_frag)
            T.copy(b[bx * 32, 0], b_frag)
            T.copy(b_frag, b[bx * 32, 0])

    target = determine_target("tileir -arch=sm_120", return_object=True)
    prim_func = dynamic_stride_kernel.get_tir(None, None)
    _, prepared = TileIRKernelAdapter._prepare_device_module(prim_func, target)
    tileir_lowering._extract_semantic_program_for_lowering(prepared, target)
    module = _build_tileir_module_for_test(prepared, target)

    source = str(module)

    assert "tensor_view<?x?xbf16, strides=[256,1]>" in source
    assert "tensor_view<?x?xf32, strides=[32,1]>" in source
    assert "assume div_by<8>" in source
    assert "assume div_by<4>" in source


def test_tileir_structured_lowering_uses_view_store_for_full_parallel_partition():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def parallel_view_store_kernel(out):
        n = T.dynamic("n")
        out: T.Tensor((n, 24), T.float32)

        with T.Kernel(T.ceildiv(n, 32)) as bx:
            frag = T.alloc_fragment((32, 32), T.float32)
            T.clear(frag)
            for i, j in T.Parallel(32, 32):
                if j < 24:
                    out[bx * 32 + i, j] = frag[i, j]

    target = determine_target("tileir -arch=sm_120", return_object=True)
    prim_func = parallel_view_store_kernel.get_tir(None)
    _, prepared = TileIRKernelAdapter._prepare_device_module(prim_func, target)
    tileir_lowering._extract_semantic_program_for_lowering(prepared, target)
    module = _build_tileir_module_for_test(prepared, target)

    source = str(module)

    assert "tensor_view<?x?xf32, strides=[24,1]>" in source
    # Masked GLOBAL stores use a TRUE predicated store (per-element pointers
    # + store_ptr_tko with a mask operand): masked lanes are not written. The
    # earlier select-zero + store_view_tko form wrote ZERO to masked lanes,
    # which clobbers live data whenever they alias valid elements.
    assert "store_ptr_tko" in source
    assert ", %2 token=" in source or "i1" in source.split("store_ptr_tko", 1)[1].split("\n", 1)[0]


def test_tileir_structured_lowering_does_not_derive_latency_from_num_stages():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def pipelined_copy_kernel(a, b):
        n = T.dynamic("n")
        a: T.Tensor((n, 32), T.bfloat16)
        b: T.Tensor((n, 32), T.bfloat16)

        with T.Kernel(T.ceildiv(n, 32)) as bx:
            frag = T.alloc_fragment((32, 32), T.bfloat16)
            for _ in T.Pipelined(2, num_stages=3):
                T.copy(a[bx * 32, 0], frag)
                T.copy(frag, b[bx * 32, 0])

    target = determine_target("tileir -arch=sm_120", return_object=True)
    prim_func = pipelined_copy_kernel.get_tir(None, None)
    _, prepared = TileIRKernelAdapter._prepare_device_module(prim_func, target)
    tileir_lowering._extract_semantic_program_for_lowering(prepared, target)
    module = _build_tileir_module_for_test(prepared, target)

    source = str(module)

    assert "for %loopIdx" in source
    assert "load_view_tko weak" in source
    # `num_stages` is a buffering-depth knob, not a memory-latency hint: it must not
    # produce a load/store latency hint (the assembler schedules the loop itself).
    assert "optimization_hints" not in source
    assert "latency" not in source
    # `a` is load-only (never stored), so it carries no token and reads the root
    # token; the store to `b` consumes `b`'s own carried token, not the load result.
    load_input = re.search(r"load_view_tko weak \S+\[[^\]]*\] token = (%\w+)", source)
    store_input = re.search(r"store_view_tko weak \S+, \S+\[[^\]]*\] token = (%\w+)", source)
    assert load_input is not None and store_input is not None
    assert load_input.group(1) != store_input.group(1)


def test_tileir_structured_lowering_prunes_redundant_global_bounds_masks():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def dynamic_row_copy_kernel(a, b, offsets):
        n = T.dynamic("n")
        a: T.Tensor((n, 16, 32), T.float16)
        b: T.Tensor((n, 16, 32), T.float16)
        offsets: T.Tensor((1,), T.int32)

        with T.Kernel(1, 16) as (_, by):
            frag = T.alloc_fragment((32, 32), T.float16)
            row = offsets[0]
            T.copy(a[row : row + 32, by, :], frag)
            T.copy(frag, b[row : row + 32, by, :])

    target = determine_target("tileir -arch=sm_120", return_object=True)
    prim_func = dynamic_row_copy_kernel.get_tir(None, None, None)
    _, prepared = TileIRKernelAdapter._prepare_device_module(prim_func, target)
    tileir_lowering._extract_semantic_program_for_lowering(prepared, target)
    module = _build_tileir_module_for_test(prepared, target)

    source = str(module)

    # Dynamic 3D slices use the partition-view path. Redundant bounds-mask
    # optimization is left to the cuda_tile MLIR optimizer.
    assert "load_view_tko weak" in source
    assert "store_view_tko weak" in source


def test_tileir_structured_lowering_carries_tokens_through_if():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def conditional_copy_kernel(src, dst, flag):
        src: T.Tensor((32,), T.float32)
        dst: T.Tensor((32,), T.float32)
        flag: T.Tensor((1,), T.int32)

        with T.Kernel(1):
            frag = T.alloc_fragment((32,), T.float32)
            T.copy(src, frag)
            if flag[0] > 0:
                T.copy(frag, dst)
            T.copy(frag, dst)

    source = _tileir_source_for_test(conditional_copy_kernel, None, None, None)

    assert " = if " in source
    # The conditional branch stores to `dst`, so the if must carry at least one
    # token result for that buffer; the unconditional store after the if then
    # orders against it.  The result list may also include tile values for
    # REGISTER buffers live across the branch.
    assert re.search(r"if .* -> \([^)]*token", source)
    assert source.count("store_view_tko weak") == 2


def _loop_result_token_count(source: str) -> int:
    """Number of `token`-typed results carried out of the (first) for-loop."""
    match = re.search(r"for %loopIdx[^{]*-> \(([^)]*)\)", source)
    return match.group(1).count("token") if match else 0


def test_tileir_structured_lowering_keeps_independent_loop_loads_unserialized():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def two_operand_loop(a, b, c):
        n = T.dynamic("n")
        a: T.Tensor((n, 32), T.float16)
        b: T.Tensor((n, 32), T.float16)
        c: T.Tensor((n, 32), T.float16)

        with T.Kernel(T.ceildiv(n, 32)) as bx:
            fa = T.alloc_fragment((32, 32), T.float16)
            fb = T.alloc_fragment((32, 32), T.float16)
            for _ in T.Pipelined(2, num_stages=2):
                T.copy(a[bx * 32, 0], fa)
                T.copy(b[bx * 32, 0], fb)
                T.copy(fa, c[bx * 32, 0])

    source = _tileir_source_for_test(two_operand_loop, None, None, None)
    # a and b are load-only inputs: each reads the same root token (independent, not
    # chained through one serial token) and carries nothing across the loop. Only the
    # written buffer c carries its ordering tokens (LAST_OP + LAST_STORE).
    load_tokens = re.findall(r"load_view_tko weak \S+\[[^\]]*\] token = (%\w+)", source)
    assert len(load_tokens) >= 2
    assert load_tokens[0] == load_tokens[1]
    assert _loop_result_token_count(source) == 2


def test_tileir_structured_lowering_keeps_distinct_buffer_loads_independent():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def two_load_flat(a, b, c):
        a: T.Tensor((32,), T.float32)
        b: T.Tensor((32,), T.float32)
        c: T.Tensor((32,), T.float32)

        with T.Kernel(1):
            fa = T.alloc_fragment((32,), T.float32)
            fb = T.alloc_fragment((32,), T.float32)
            T.copy(a, fa)
            T.copy(b, fb)
            T.copy(fa, c)

    source = _tileir_source_for_test(two_load_flat, None, None, None)
    # Distinct, never-written buffers -> both loads take the root token; neither is
    # chained behind the other.
    load_inputs = re.findall(r"load_view_tko weak \S+\[[^\]]*\] token = (%\w+)", source)
    assert len(load_inputs) >= 2
    assert load_inputs[0] == load_inputs[1]


def test_tileir_structured_lowering_orders_read_after_write_on_same_buffer():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def raw_kernel(a):
        a: T.Tensor((32,), T.float32)

        with T.Kernel(1):
            fr = T.alloc_fragment((32,), T.float32)
            T.copy(a, fr)
            T.copy(fr, a)
            T.copy(a, fr)

    source = _tileir_source_for_test(raw_kernel, None)
    # The reload of `a` must depend on the store to `a` (RAW): it consumes the
    # store's result token.
    store = re.search(r"(%\w+) = store_(?:ptr|view)_tko weak", source)
    assert store is not None
    tok = store.group(1)
    assert f"token={tok}" in source or f"token = {tok}" in source


def test_tileir_structured_lowering_carries_no_tokens_for_local_only_loop():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def local_only_loop(src, dst):
        src: T.Tensor((32,), T.float32)
        dst: T.Tensor((32,), T.float32)

        with T.Kernel(1):
            fa = T.alloc_fragment((32,), T.float32)
            fb = T.alloc_fragment((32,), T.float32)
            T.copy(src, fa)
            for _ in T.serial(4):
                T.copy(fa, fb)
                T.copy(fb, fa)
            T.copy(fa, dst)

    source = _tileir_source_for_test(local_only_loop, None, None)
    # The loop body only moves data between local fragments (no global memory op),
    # so it carries no tokens across iterations.
    assert _loop_result_token_count(source) == 0


def test_tileir_structured_lowering_rejects_explicit_pipeline_schedule_annotations():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def explicitly_scheduled_pipeline_kernel(a):
        a: T.Tensor((64, 32), T.float32)

        with T.Kernel(1):
            frag = T.alloc_fragment((32, 32), T.float32)
            for _ in T.Pipelined(2, order=[0], stage=[0]):
                T.copy(a[0, 0], frag)

    target = determine_target("tileir -arch=sm_120", return_object=True)
    prim_func = explicitly_scheduled_pipeline_kernel.get_tir(None)
    _, prepared = TileIRKernelAdapter._prepare_device_module(prim_func, target)
    tileir_lowering._extract_semantic_program_for_lowering(prepared, target)

    with pytest.raises(TileIRLoweringNotImplementedError, match="explicit pipeline schedule"):
        _build_tileir_module_for_test(prepared, target)
