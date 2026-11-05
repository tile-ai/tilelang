"""Mixed index widths in dynamic strided CPU loops retain typed vector planning."""

import numpy as np
import json
import os
from pathlib import Path
import shutil
import subprocess
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx
from tilelang.testing.packed import compile_packed_function

from test_native_reducer_contracts import integer, objects


def _program(symbolic_stride, index_dtype, *, static_extent=None):
    size, stride = ir.Var("size", "int64"), ir.Var("stride", "int64")
    length = size if static_extent is None else integer(static_extent, "int64")
    row, column = ir.Var("row", index_dtype), ir.Var("column", index_dtype)
    raw = ir.Var("storage", tirx.TensorType([(size + 2) * stride], "float32"))
    output = ir.Var("output", tirx.TensorType([length, integer(2, "int64")], "float32"))
    row_stride = stride if symbolic_stride else integer(4, "int64")
    layout = tirx.TileLayout(tirx.layout.S[(length, 2) : (row_stride, 1)] + 6)
    view = ir.Var("view", tirx.TensorType([length, 2], "float32", layout=layout, align=4))
    declaration = ir.Bind(view, ir.Call("tirx.decl_tensor", [raw.data], ty_args=[view.ty]))
    store = ir.TensorStore(output, [row, column], ir.TensorLoad(view, [row, column]) + ir.FloatImm("float32", 3))
    inner = ir.For(column, integer(0, index_dtype), integer(2, index_dtype), integer(1, index_dtype), ir.ForKind.PARALLEL, [store])
    outer = ir.For(row, integer(0, index_dtype), ir.prim.Cast(index_dtype, length), integer(1, index_dtype), ir.ForKind.PARALLEL, [inner])
    value = tirx.Function([size, stride, raw, output], [declaration, outer], ir.PrimType("void"))
    return value.with_attr("global_symbol", "main").with_attr("target", tvm_tirx.target.Target({"kind": "llvm", "mcpu": "generic"}))


@pytest.mark.parametrize("size", [1, 3, 7])
@pytest.mark.parametrize("stride", [4, 5])
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_dynamic_strided_parallel_layout_lowering_and_cpu_values(size, stride, index_dtype):
    value = _program(True, index_dtype).with_attr("target", tvm_tirx.target.Target("c"))
    before = ir.save_json(value)
    assert tirx.analysis.check_well_formed(value)
    module = ir.IRModule({"main": value})
    inferred = tvm_ffi.get_global_func("tl.transform.LayoutInference")()(module)
    lowered = tvm_ffi.get_global_func("tl.transform.LowerTileOp")()(inferred)
    assert ir.save_json(value) == before
    assert tirx.analysis.check_well_formed(lowered)
    assert not tirx.analysis.undefined_vars(lowered["main"])
    lowered = tvm_ffi.get_global_func("tl.transform.FinalizeAnnotations")("c")(lowered)
    compiled = compile_packed_function(lowered, target="c")
    storage = np.arange((size + 2) * stride, dtype="float32")
    source = tvm_tirx.runtime.tensor(storage)
    output = tvm_tirx.runtime.tensor(np.full((size, 2), -99, dtype="float32"))
    compiled(size, stride, source, output)
    expected = storage[6 + np.arange(size)[:, None] * stride + np.arange(2)] + 3
    np.testing.assert_array_equal(output.numpy(), expected)


@pytest.mark.parametrize("noalias", [False, True])
def test_known_even_stride_still_produces_vector_memory(noalias):
    value = _program(False, "int32", static_extent=4)
    if noalias:
        value = value.with_attr("tirx.noalias", True)
    module = tvm_ffi.get_global_func("tl.transform.LayoutInference")()(ir.IRModule({"main": value}))
    module = tvm_ffi.get_global_func("tl.transform.LowerTileOp")()(module)
    assert tirx.analysis.check_well_formed(module)
    assert any(loop.kind == ir.ForKind.VECTORIZED and int(loop.end - loop.begin) == 2 for loop in objects(module, ir.For))
    module = tvm_ffi.get_global_func("tl.transform.VectorizeLoop")(True)(module)
    assert tirx.analysis.check_well_formed(module)
    if noalias:
        assert any(store.indices[-1].ty.dtype.lanes == 2 for store in objects(module, ir.TensorStore))
    else:
        # The original raw Function has no disjoint-storage contract. Its
        # strided view and output could overlap across scalar iterations.
        assert all(store.indices[-1].ty.dtype.lanes == 1 for store in objects(module, ir.TensorStore))
        assert any(loop.kind == ir.ForKind.DEFAULT for loop in objects(module, ir.For))


def test_dynamic_host_control_cuda_static_compilation(tmp_path):
    import tilelang.language as T
    from tvm_tirx.tirx._compile import _split_host_device_modules
    from tvm_tirx.tirx.compilation_pipeline import tirx_pipeline

    n = T.dynamic("n", "int64")

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32")):
        if n > 0:
            for step in T.serial(3):
                if step != 1:
                    with T.Kernel(T.ceildiv(n, 128), threads=128) as block:
                        for i in T.Parallel(128):
                            index = block * 128 + i
                            if index < n:
                                B[index] = A[index] + T.cast(step, "float32")

    target = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a"}, host="c")
    value = main.with_attr("target", target)
    original = ir.save_json(value)
    module = ir.IRModule({"main": value})
    module = tvm_ffi.get_global_func("tl.transform.MaterializeKernelLaunch")(False, True, [integer(128)], None, None)(module)
    for name in ("LayoutInference", "LowerTileOp", "LowerAccessPtr"):
        module = tvm_ffi.get_global_func("tl.transform." + name)()(module)
        assert tirx.analysis.check_well_formed(module)
    assert ir.save_json(value) == original
    module = tvm_ffi.get_global_func("tl.transform.FinalizeAnnotations")("cuda")(module)
    lowering, _, finalize_device = tirx_pipeline(target=target)
    lowered = lowering(module)
    _, devices = _split_host_device_modules(lowered)
    assert len(devices) == 1
    (tmp_path / "lowered.json").write_text(ir.save_json(lowered))
    for (device_target, _), device in devices.items():
        device = finalize_device()(device)
        source = tvm_ffi.get_global_func("target.build.tilelang_cuda_without_compile")(device, device_target).inspect_source("cu")
        path = tmp_path / "kernel.cu"
        path.write_text(source)
        repository = Path(os.environ.get("TILELANG_OPS_EXAMPLE_REPOSITORY", Path(__file__).resolve().parents[2]))
        nvcc = shutil.which("nvcc")
        assert nvcc is not None
        command = [
            nvcc,
            "-std=c++17",
            "--cubin",
            "-arch=sm_100a",
            "-I" + str(repository / "src"),
            "-I" + str(repository / "3rdparty/cutlass/include"),
            str(path),
            "-o",
            str(tmp_path / "kernel.cubin"),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        (tmp_path / "nvcc.log").write_text(result.stdout + result.stderr)
        (tmp_path / "nvcc.json").write_text(json.dumps({"command": command, "exit_code": result.returncode}))
        assert result.returncode == 0, result.stdout + result.stderr
