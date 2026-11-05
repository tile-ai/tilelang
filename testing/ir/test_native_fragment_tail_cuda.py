"""Execute guarded fragment tails through TileLang CUDA, NVCC and the driver."""

import json

import numpy as np
import pytest
import tvm_tirx
from tvm_tirx import ir

from test_native_ascend_ops import subregion
from test_native_cuda_ops_execute import execute
from test_native_reducer_contracts import allocate, integer, region, tensor, tileop
from test_native_webgpu_ops import _function


@pytest.mark.parametrize("size", [7, 257])
@pytest.mark.parametrize("operation", ["fill", "copy"])
@pytest.mark.parametrize("dtype", ["float32", "int32"])
def test_fragment_tail_executes_on_cuda_with_untouched_sentinels(size, operation, dtype, tmp_path):
    device = tvm_tirx.cuda(0)
    assert device.exist, "This suite requires the assigned CUDA GPU"
    source = tensor("A", [size + 16], dtype, "global")
    output = tensor("B", [size + 16], dtype, "global")
    fragment = tensor("F", [size], dtype, "local.fragment")
    constant = ir.FloatImm(dtype, -0.0) if dtype == "float32" else integer(17, dtype)
    body = [allocate(fragment), tileop("fill", region(fragment), constant)]
    if operation == "copy":
        body.append(tileop("copy", subregion(source, [8], [size]), region(fragment)))
    body.append(tileop("copy", region(fragment), subregion(output, [8], [size])))
    arch = "sm_" + device.compute_version.replace(".", "")
    value = _function([source, output], body).with_attr("target", tvm_tirx.target.Target({"kind": "cuda", "arch": arch}))
    inputs = np.arange(size + 16, dtype=dtype) * 3 - 29
    initial = np.full(size + 16, -333, dtype=dtype)
    expected = initial.copy()
    expected[8 : 8 + size] = inputs[8 : 8 + size] if operation == "copy" else (-0.0 if dtype == "float32" else 17)
    results, source_code = execute(value, [inputs, initial], tmp_path)
    np.testing.assert_array_equal(results[0], inputs)
    np.testing.assert_array_equal(results[1], expected)
    if dtype == "float32":
        np.testing.assert_array_equal(results[1].view("uint32"), expected.view("uint32"))
    assert "__global__" in source_code
    (tmp_path / "numeric.json").write_text(
        json.dumps(
            {
                "target": str(value.attrs["target"]),
                "device": str(device),
                "dtype": dtype,
                "size": size,
                "operation": operation,
                "sentinel_elements": 16,
                "gpu_execution": "passed",
            },
            indent=2,
        )
    )
