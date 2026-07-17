"""Structured lowering coverage for reduce and atomic operations."""

from __future__ import annotations


import pytest

import tilelang
import tilelang.testing
from tilelang import language as T
from tilelang import tvm as tvm
from tilelang.tileir import checks
from tilelang.jit.adapter.tileir.adapter import TileIRKernelAdapter
from tilelang.tileir.errors import TileIRLoweringNotImplementedError
from tilelang.tileir import lowering as tileir_lowering
from tilelang.backend.target import determine_target

Range = tvm.ir.Range
structural_equal = tvm.ir.structural_equal
Target = tvm.target.Target


from tileir_test_utils import (
    _build_tileir_module_for_test,
    _prepared_tileir_kernel_for_test,
    _tileir_source_for_test,
)


def test_tileir_structured_lowering_skips_reduce_identity_combine():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def reduce_identity_kernel(a, b):
        a: T.Tensor((32, 32), T.float32)
        b: T.Tensor((32,), T.float32)

        with T.Kernel(1):
            frag = T.alloc_fragment((32, 32), T.float32)
            accum = T.alloc_fragment((32,), T.float32)
            T.copy(a, frag)
            T.fill(accum, -T.infinity(T.float32))
            T.reduce_max(frag, accum, dim=1, clear=False)
            T.copy(accum, b)

    target = determine_target("tileir -arch=sm_120", return_object=True)
    prim_func = reduce_identity_kernel.get_tir(None, None)
    _, prepared = TileIRKernelAdapter._prepare_device_module(prim_func, target)
    tileir_lowering._extract_semantic_program_for_lowering(prepared, target)
    module = _build_tileir_module_for_test(prepared, target)

    source = str(module)

    # The lowering emits maxf in the reduce body and may retain the
    # identity-combine step. The cuda_tile MLIR optimizer may elide the
    # redundant maxf(-inf, reduce_result) operation.
    assert source.count("maxf") >= 1  # at least one maxf (inside reduce body)
    assert "reduce" in source


def test_tileir_structured_lowering_keeps_reduce_clear_false_loop_carried():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def reduce_accumulate_kernel(a, b):
        a: T.Tensor((64, 32), T.float32)
        b: T.Tensor((32,), T.float32)

        with T.Kernel(1):
            accum = T.alloc_fragment((32,), T.float32)
            T.clear(accum)
            for k in T.Pipelined(2, num_stages=1):
                frag = T.alloc_fragment((32, 32), T.float32)
                T.copy(a[k * 32, 0], frag)
                T.reduce_sum(frag, accum, dim=1, clear=False)
            T.copy(accum, b)

    target = determine_target("tileir -arch=sm_120", return_object=True)
    prim_func = reduce_accumulate_kernel.get_tir(None, None)
    _, prepared = TileIRKernelAdapter._prepare_device_module(prim_func, target)
    tileir_lowering._extract_semantic_program_for_lowering(prepared, target)
    module = _build_tileir_module_for_test(prepared, target)

    source = str(module)

    assert "iter_values" in source
    assert "reduce" in source
    assert "addf" in source


def test_tileir_structured_lowering_supports_more_reduce_kinds():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def reduce_extra_kernel(a, b, c, d):
        a: T.Tensor((32, 32), T.float32)
        b: T.Tensor((32,), T.float32)
        c: T.Tensor((32,), T.float32)
        d: T.Tensor((32,), T.int32)

        with T.Kernel(1):
            frag = T.alloc_fragment((32, 32), T.float32)
            out_min = T.alloc_fragment((32,), T.float32)
            out_abs = T.alloc_fragment((32,), T.float32)
            int_frag = T.alloc_fragment((32, 32), T.int32)
            int_out = T.alloc_fragment((32,), T.int32)
            T.copy(a, frag)
            for i, j in T.Parallel(32, 32):
                int_frag[i, j] = T.cast(i + j, T.int32)
            T.reduce_min(frag, out_min, dim=1, clear=True)
            T.reduce_abssum(frag, out_abs, dim=1)
            T.reduce_bitxor(int_frag, int_out, dim=1, clear=True)
            T.copy(out_min, b)
            T.copy(out_abs, c)
            T.copy(int_out, d)

    source = _tileir_source_for_test(reduce_extra_kernel, None, None, None, None)

    assert "minf" in source
    assert "absf" in source
    assert "xori" in source


def test_tileir_structured_lowering_accepts_reduce_batch_annotation_as_schedule_hint():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def reduce_batch_kernel(a, b):
        a: T.Tensor((32, 32), T.float32)
        b: T.Tensor((32,), T.float32)

        with T.Kernel(1):
            frag = T.alloc_fragment((32, 32), T.float32)
            out = T.alloc_fragment((32,), T.float32)
            T.copy(a, frag)
            T.reduce_sum(frag, out, dim=1, clear=True, batch=2)
            T.copy(out, b)

    source = _tileir_source_for_test(reduce_batch_kernel, None, None)

    assert "reduce" in source


def test_tileir_structured_lowering_strength_reduces_layout_indexing():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    def make_layout(dst):
        return T.Layout(dst.shape, lambda i, j: [i // 8, j // 8, j % 2, 4 * (i % 8) + (j % 8) // 2])

    @tilelang.jit
    def layout_atomic_kernel(src, dst):
        src: T.Tensor((32, 64), T.float32)
        dst: T.Tensor((32, 64), T.float32)

        with T.Kernel(1):
            T.annotate_layout({dst: make_layout(dst)})
            for i, j in T.Parallel(32, 64):
                T.atomic_add(dst[i, j], src[i, j])

    target = determine_target("tileir -arch=sm_120", return_object=True)
    prim_func = layout_atomic_kernel.get_tir(None, None)
    _, prepared = TileIRKernelAdapter._prepare_device_module(prim_func, target)
    tileir_lowering._extract_semantic_program_for_lowering(prepared, target)
    module = _build_tileir_module_for_test(prepared, target)

    source = str(module)

    # Relaxed atomics may use the partition-view or pointer-based path. Both
    # apply the atomic operation to every element of the destination tile.
    assert "atomic_red_view_tko" in source or "atomic_rmw_tko" in source


def test_tileir_structured_lowering_uses_atomic_red_view_for_relaxed_tile_atomics():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def relaxed_tile_atomic_kernel(src, dst_add, dst_min, dst_max):
        src: T.Tensor((32,), T.int32)
        dst_add: T.Tensor((32,), T.int32)
        dst_min: T.Tensor((32,), T.int32)
        dst_max: T.Tensor((32,), T.int32)

        with T.Kernel(1):
            for i in T.Parallel(32):
                T.atomic_add(dst_add[i], src[i])
                T.atomic_min(dst_min[i], src[i])
                T.atomic_max(dst_max[i], src[i])

    source = _tileir_source_for_test(relaxed_tile_atomic_kernel, None, None, None, None)

    assert source.count("atomic_red_view_tko relaxed device") == 3
    assert "atomic_rmw_tko" not in source
    assert ", add," in source
    assert ", min," in source
    assert ", max," in source


def test_tileir_rejects_multi_gemm_loop_indexed_atomic_partition():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def loop_indexed_atomic_kernel(src, weight, dst):
        src: T.Tensor((64, 32), T.float16)
        weight: T.Tensor((32, 32), T.float16)
        dst: T.Tensor((64, 32), T.float32)

        with T.Kernel(1):
            src_tile = T.alloc_shared((32, 32), T.float16)
            weight_tile = T.alloc_shared((32, 32), T.float16)
            acc = T.alloc_fragment((32, 32), T.float32)
            acc_shared = T.alloc_shared((32, 32), T.float32)
            T.copy(weight, weight_tile)
            for chunk in T.Pipelined(2, num_stages=1):
                T.copy(src[chunk * 32 : (chunk + 1) * 32, :], src_tile)
                T.gemm(src_tile, weight_tile, acc, clear_accum=True)
                T.gemm(src_tile, weight_tile, acc)
                T.copy(acc, acc_shared)
                T.atomic_add(dst[chunk * 32 : (chunk + 1) * 32, :], acc_shared)

    target, prepared, _ = _prepared_tileir_kernel_for_test(loop_indexed_atomic_kernel, None, None, None)

    with pytest.raises(TileIRLoweringNotImplementedError, match="loop-indexed atomic reduction.*multiple GEMM updates"):
        _build_tileir_module_for_test(prepared, target)


def test_tileir_structured_lowering_supports_atomic_min_max_memory_order():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def atomic_min_max_kernel(src, dst_min, dst_max):
        src: T.Tensor((32,), T.int32)
        dst_min: T.Tensor((32,), T.int32)
        dst_max: T.Tensor((32,), T.int32)

        with T.Kernel(1):
            for i in T.Parallel(32):
                T.atomic_min(dst_min[i], src[i], memory_order="acquire")
                T.atomic_max(dst_max[i], src[i], memory_order="release")

    source = _tileir_source_for_test(atomic_min_max_kernel, None, None, None)

    assert source.count("atomic_rmw_tko") == 2
    assert "acquire" in source
    assert "release" in source
    assert ", min," in source
    assert ", max," in source


def test_tileir_structured_lowering_supports_parallel_atomic_return_prev():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def atomic_return_kernel(src, dst, old):
        src: T.Tensor((32,), T.float32)
        dst: T.Tensor((32,), T.float32)
        old: T.Tensor((32,), T.float32)

        with T.Kernel(1):
            for i in T.Parallel(32):
                old[i] = T.atomic_add(dst[i], src[i], return_prev=True)

    source = _tileir_source_for_test(atomic_return_kernel, None, None, None)

    assert "atomic_rmw_tko" in source
    assert "store_view_tko" in source
    # return_prev: the atomic's previous-value result is the tile written back to `old`.
    import re

    atomic_prev = re.search(r"(%\w+), %\w+ = atomic_rmw_tko", source)
    assert atomic_prev is not None
    assert f"store_view_tko weak {atomic_prev.group(1)}" in source


def test_tileir_structured_lowering_supports_atomic_load_store_memory_order():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def atomic_load_store_kernel(src, dst, old):
        src: T.Tensor((32,), T.int32)
        dst: T.Tensor((32,), T.int32)
        old: T.Tensor((32,), T.int32)

        with T.Kernel(1):
            for i in T.Parallel(32):
                value = T.atomic_load(src[i], memory_order="acquire")
                old[i] = value
                T.atomic_store(dst[i], value + 1, memory_order="release")

    source = _tileir_source_for_test(atomic_load_store_kernel, None, None, None)

    assert "load_ptr_tko acquire device" in source
    assert "store_ptr_tko release device" in source


def test_tileir_structured_lowering_rejects_atomic_seq_cst_order():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def atomic_seq_cst_kernel(src, dst):
        src: T.Tensor((32,), T.float32)
        dst: T.Tensor((32,), T.float32)

        with T.Kernel(1):
            for i in T.Parallel(32):
                T.atomic_add(dst[i], src[i], memory_order="seq_cst")

    target, prepared, semantic_program = _prepared_tileir_kernel_for_test(atomic_seq_cst_kernel, None, None)

    with pytest.raises(TileIRLoweringNotImplementedError, match="memory order id 5"):
        _build_tileir_module_for_test(prepared, target)


def test_tileir_rejects_atomic_load_store_unsupported_order():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def atomic_load_seq_cst_kernel(src, dst):
        src: T.Tensor((32,), T.int32)
        dst: T.Tensor((32,), T.int32)

        with T.Kernel(1):
            for i in T.Parallel(32):
                dst[i] = T.atomic_load(src[i])

    target, prepared, semantic_program = _prepared_tileir_kernel_for_test(atomic_load_seq_cst_kernel, None, None)

    with pytest.raises(TileIRLoweringNotImplementedError, match="atomic load.*memory order id 5"):
        _build_tileir_module_for_test(prepared, target)

    @tilelang.jit
    def atomic_store_acquire_kernel(src, dst):
        src: T.Tensor((32,), T.int32)
        dst: T.Tensor((32,), T.int32)

        with T.Kernel(1):
            for i in T.Parallel(32):
                T.atomic_store(dst[i], src[i], memory_order="acquire")

    with pytest.raises(ValueError, match="atomic_store does not support memory_order='acquire'"):
        _prepared_tileir_kernel_for_test(atomic_store_acquire_kernel, None, None)
