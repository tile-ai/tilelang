import numpy as np
import pytest

from tvm_tirx import runtime, tirx
from tilelang._compat import ir
from tilelang._compat import tirx as primitives
from tvm_tirx.ir import prim
from tvm_tirx.target import Target


@pytest.mark.parametrize("backend", ["c", "llvm", "cuda"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("lanes", [1, 4])
@pytest.mark.parametrize("predicate", ["isfinite", "isinf", "isnan"])
def test_finiteness_classification_including_nan_and_infinities(backend, dtype, lanes, predicate, compile_tilelang):
    device = runtime.cuda(0) if backend == "cuda" else runtime.cpu(0)
    if not device.exist:
        pytest.skip("Requested device is unavailable")
    target = Target(backend)
    if backend == "cuda":
        target = Target(
            {
                "kind": "cuda",
                "arch": f"sm_{device.compute_version.replace('.', '')}",
                "backend_config": {"cuda": {"compiler": "nvcc"}},
            },
            host="c",
        )
    source = ir.Var("source", tirx.TensorType([8], dtype))
    output = ir.Var("output", tirx.TensorType([8], "uint8"))
    index = ir.Var("index", "int32")
    indices = [index if lanes == 1 else prim.Ramp(index * lanes, 1, lanes)]
    result_type = ir.prim_type_from_dlpack((1, 8, lanes))
    classified = getattr(primitives, predicate)(ir.TensorLoad(source, indices))
    value = prim.Select(classified, ir.const(1, result_type), ir.const(0, result_type))
    store = ir.TensorStore(output, indices, value)
    if backend == "cuda":
        body = ir.RegionStmt(
            "tirx.launch_thread",
            [ir.StringImm("threadIdx.x"), ir.const(8 // lanes)],
            [index],
            None,
            store,
        )
    else:
        body = ir.For(index, 0, 8 // lanes, 1, ir.ForKind.DEFAULT, store)
    function = tirx.Function([source, output], body).with_attr("global_symbol", "classify")
    executable = compile_tilelang(function, target)
    values = np.array([0, 1, -1, np.inf, -np.inf, np.nan, 1e-20, 1e20], dtype=dtype)
    actual = runtime.empty((8,), "uint8", device=device)
    executable(runtime.tensor(values, device=device), actual)
    np.testing.assert_array_equal(actual.numpy(), getattr(np, predicate)(values).astype("uint8"))
