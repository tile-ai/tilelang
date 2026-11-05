"""Native TileLang CUDA rules are selected before the generic target rules."""

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx.script import tirx as T

from testing.python.cuda.test_native_codegen_cuda import build_cuda


@pytest.mark.parametrize("dtype", ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"])
def test_popcount_preserves_input_bit_width(dtype, tmp_path):
    info = np.iinfo(dtype)
    values = np.array([info.min, -1 if info.min < 0 else info.max - 1, 0, 1, info.max], dtype=dtype)

    @T.function
    def population(source: T.Tensor((5,), dtype), output: T.Tensor((5,), dtype)):
        with T.launch_thread("blockIdx.x", 1) as _bx, T.launch_thread("threadIdx.x", 32) as tx:
            if tx < 5:
                output[tx] = T.popcount(source[tx])

    target = tvm_tirx.target.Target("cuda", host="c")
    function = population.with_attr("target", target).with_attr("global_symbol", "main")
    lowered = tvm_ffi.get_global_func("tl.transform.LowerIntrin")()(tvm_tirx.IRModule({"main": function}))
    (tmp_path / "tilelang_lowered.py").write_text(lowered.script())
    assert "popcount(" not in lowered.script()
    module, source = build_cuda(lowered["main"], tmp_path)
    assert ("__popcll" if info.bits == 64 else "__popc") in source
    dev = tvm_tirx.cuda(0)
    output = tvm_tirx.runtime.tensor(np.zeros_like(values), dev)
    module(tvm_tirx.runtime.tensor(values, dev), output)
    mask = (1 << info.bits) - 1
    expected = np.array([(int(value) & mask).bit_count() for value in values], dtype=dtype)
    np.testing.assert_array_equal(output.numpy(), expected)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_isfinite_result_remains_boolean(dtype, tmp_path):
    def isfinite(value):
        return tvm_tirx.ir.Call("tl.isfinite", [value])

    @T.function
    def finite(source: T.Tensor((5,), dtype), output: T.Tensor((5,), "bool")):
        with T.launch_thread("blockIdx.x", 1) as _bx, T.launch_thread("threadIdx.x", 32) as tx:
            if tx < 5:
                output[tx] = isfinite(source[tx])

    target = tvm_tirx.target.Target("cuda", host="c")
    function = finite.with_attr("target", target).with_attr("global_symbol", "main")
    lowered = tvm_ffi.get_global_func("tl.transform.LowerIntrin")()(tvm_tirx.IRModule({"main": function}))
    module, source = build_cuda(lowered["main"], tmp_path)
    assert "isfinite(" in source
    values = np.array([-np.inf, np.inf, np.nan, 0, -1.25], dtype=dtype)
    dev = tvm_tirx.cuda(0)
    output = tvm_tirx.runtime.tensor(np.zeros(5, dtype="bool"), dev)
    module(tvm_tirx.runtime.tensor(values, dev), output)
    np.testing.assert_array_equal(output.numpy(), np.isfinite(values))
