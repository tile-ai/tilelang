"""Execute TileLang ThreadSync output through TileLang's CUDA emitter."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx as tvm
from tvm_tirx import ir

import tilelang
import tilelang.language as T
from tilelang.language.tir.entry import prim_func
from tilelang.testing.ir import collect_calls, collect_nodes
from testing.python.cuda.test_native_codegen_cuda import build_cuda


def _ordered(function):
    if not tvm.cuda(0).exist:
        pytest.skip("CUDA runtime and device required for synchronization execution")
    arch = os.environ.get("TILELANG_TEST_CUDA_ARCH", "sm_" + tvm.cuda(0).compute_version.replace(".", ""))
    function = function.with_attr("target", tvm.target.Target({"kind": "cuda", "arch": arch}))
    function = tilelang.transform.ThreadSync("shared")(tvm.IRModule({"main": function}))["main"]
    return function.without_attr("target")


def _check(module, values, expected, tmp_path):
    device = tvm.cuda(0)
    source = tvm.runtime.tensor(values, device)
    output = tvm.runtime.tensor(np.full_like(expected, -999), device)
    for _ in range(10):
        module(source, output)
        np.testing.assert_array_equal(output.numpy(), expected)
    maps = Path("/proc/self/maps")
    libraries = (
        sorted({line.split()[-1] for line in maps.read_text().splitlines() if ".so" in line and ("tvm" in line or "tilelang" in line)})
        if maps.exists()
        else []
    )
    (tmp_path / "execution.json").write_text(
        json.dumps(
            {
                "tilelang": tilelang.__file__,
                "tvm_tirx": tvm.__file__,
                "tvm_ffi": tvm_ffi.__file__,
                "loaded_libraries": libraries,
                "input": values.tolist(),
                "expected": expected.tolist(),
                "actual": output.numpy().tolist(),
                "iterations": 10,
                "oracle": "exact integer equality",
            },
            indent=2,
        )
    )


def test_native_threadsync_cross_warp_exchange_executes(tmp_path):
    @prim_func
    def kernel(source: T.Tensor((64,), "int32"), output: T.Tensor((64,), "int32")):
        _bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 64)
        shared = T.alloc_shared((64,), "int32", scope="shared")
        shared[tx] = source[tx]
        output[tx] = shared[(tx + 32) % 64]

    ordered = _ordered(kernel)
    assert len(collect_calls(ordered, op="tirx.gpu_storage_sync")) == 1
    module, source = build_cuda(ordered, tmp_path)
    assert "__syncthreads" in source
    values = np.arange(64, dtype="int32") * 7 - 19
    _check(module, values, np.roll(values, -32), tmp_path)


def test_native_threadsync_partial_warps_execute(tmp_path):
    @prim_func
    def kernel(source: T.Tensor((128,), "int32"), output: T.Tensor((128,), "int32")):
        _bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        shared = T.alloc_shared((64,), "int32", scope="shared")
        if tx < 64:
            shared[tx] = source[tx]
            output[tx] = shared[(tx + 32) % 64]
        else:
            output[tx] = source[tx]

    ordered = _ordered(kernel)
    barriers = collect_calls(ordered, op="tl.storage_sync")
    assert len(barriers) == 1 and int(barriers[0].args[2]) == 64
    module, source = build_cuda(ordered, tmp_path)
    assert f"tl::__sync_thread_partial({int(barriers[0].args[1])}, 64)" in source
    values = np.arange(128, dtype="int32") * 7 - 19
    expected = values.copy()
    expected[:64] = np.roll(values[:64], -32)
    _check(module, values, expected, tmp_path)


def test_native_threadsync_write_only_loop_carry_executes(tmp_path):
    @prim_func
    def kernel(source: T.Tensor((64,), "int32"), output: T.Tensor((64,), "int32")):
        _bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 64)
        shared = T.alloc_shared((64,), "int32", scope="shared")
        for iteration in range(4):
            shared[(tx + iteration * 32) % 64] = source[tx] + iteration
        output[tx] = shared[tx]

    ordered = _ordered(kernel)
    loops = collect_nodes(ordered, ir.For)
    assert len(loops) == 1
    assert len(collect_calls(loops[0], op="tirx.gpu_storage_sync")) == 1
    assert len(collect_calls(ordered, op="tirx.gpu_storage_sync")) == 2
    module, source = build_cuda(ordered, tmp_path)
    assert "__syncthreads" in source
    values = np.arange(64, dtype="int32") * 7 - 19
    _check(module, values, np.roll(values, -32) + np.int32(3), tmp_path)
