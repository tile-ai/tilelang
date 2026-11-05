"""Compile and execute synchronous async-copy control flow on a CUDA device."""

import json
import os

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx

from test_native_async_copy_control_flow import integer, wait
from test_native_cuda_copy_execute import build_cuda_copy
from test_native_reducer_contracts import allocate, tensor


@pytest.mark.parametrize("case", ["conditional_wait", "both_branch_wait", "zero_trip_wait", "partial_wait", "trailing_copy", "backedge"])
def test_async_copy_control_flow_native_gpu(case, tmp_path):
    assert tvm_tirx.cuda(0).exist, "A CUDA device is required for this execution suite"
    size = 32
    source, output = tensor("A", [3 * size], scope="global"), tensor("B", [3 * size], scope="global")
    flags = tensor("flags", [1], "int32", "global")
    shared = tensor("S", [size], scope="shared")
    thread, block, iteration, count = [ir.Var(name, "int32") for name in ("thread", "block", "iteration", "count")]

    def copy(row):
        return ir.TensorStore(shared, [thread], ir.TensorLoad(source, [row * size + thread]))

    def consume(row):
        return ir.TensorStore(output, [row * size + thread], ir.TensorLoad(shared, [thread]))

    body = [allocate(shared), ir.Bind(count, ir.TensorLoad(flags, [integer(0)])), copy(0)]
    if case == "conditional_wait":
        body.extend([ir.If(count > 0, [wait()], None), consume(0)])
    elif case == "both_branch_wait":
        body.extend([ir.If(count > 0, [wait()], [wait()]), consume(0)])
    elif case == "zero_trip_wait":
        body.extend([ir.For(iteration, integer(0), count, integer(1), ir.ForKind.DEFAULT, [wait()]), consume(0)])
    elif case == "partial_wait":
        body.extend([wait(1), consume(0)])
    elif case == "trailing_copy":
        body.extend([ir.If(count > 0, [wait(), copy(1)], [wait(), copy(2)]), consume(0)])
    else:
        body.extend(
            [
                ir.For(iteration, integer(0), count, integer(1), ir.ForKind.DEFAULT, [consume(iteration), copy(iteration + 1)]),
                consume(count),
            ]
        )
    body = ir.RegionStmt(
        "tirx.launch_thread",
        [ir.StringImm("blockIdx.x"), integer(1)],
        [block],
        {},
        [ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), integer(size)], [thread], {}, body)],
    )
    arch = os.environ.get("TILELANG_TEST_CUDA_ARCH", "sm_" + tvm_tirx.cuda(0).compute_version.replace(".", ""))
    original = (
        tirx.Function([source, output, flags], [body], ir.NoneType())
        .with_attr("global_symbol", "main")
        .with_attr("target", tvm_tirx.target.Target({"kind": "cuda", "arch": arch}))
    )
    assert tirx.analysis.check_well_formed(original)
    serialized = ir.save_json(original)
    transformed = tvm_ffi.get_global_func("tl.testing.InjectPTXAsyncCopy")(original.body, False)
    assert transformed["injected"]
    value = tirx.Function(original.params, transformed["statement"], original.ret_type, original.attrs)
    assert ir.save_json(original) == serialized
    assert tirx.analysis.check_well_formed(value)
    (tmp_path / "native-input.json").write_text(serialized)
    (tmp_path / "native-injected.json").write_text(ir.save_json(value))
    module, code = build_cuda_copy(value, tmp_path.resolve())
    assert "cp_async_gs" in code and "cp_async_wait" in code
    device = tvm_tirx.cuda(0)
    inputs = np.arange(3 * size, dtype="float32") * 7 + 3
    source_tensor = tvm_tirx.runtime.tensor(inputs, device)
    values = [0, 1, 2] if case in ("zero_trip_wait", "backedge") else [0, 1]
    results = []
    for flag in values:
        destination = tvm_tirx.runtime.tensor(np.zeros_like(inputs), device)
        flag_tensor = tvm_tirx.runtime.tensor(np.array([flag], dtype="int32"), device)
        module(source_tensor, destination, flag_tensor)
        expected = np.zeros_like(inputs)
        if case == "trailing_copy":
            row = 1 if flag else 2
            expected[:size] = inputs[row * size : (row + 1) * size]
        elif case == "backedge":
            expected[: (flag + 1) * size] = inputs[: (flag + 1) * size]
        else:
            expected[:size] = inputs[:size]
        np.testing.assert_array_equal(destination.numpy(), expected)
        results.append({"flag": flag, "passed": True})
    (tmp_path / "numeric.json").write_text(
        json.dumps({"case": case, "values": results, "oracle": "exact per-iteration float32 values", "device": str(device)}, indent=2)
    )
