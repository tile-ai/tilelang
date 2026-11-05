"""Public callee-output ABI over native typed workers and framework allocation."""

import os
import subprocess
import sys

import pytest
import torch
import tvm_ffi
import tvm_tirx

import tilelang
import tilelang.cpu.language as T


def compile_output(function, outputs, kind="c"):
    if kind == "llvm" and not tvm_tirx.runtime.enabled("llvm"):
        pytest.skip("Optional LLVM code generation is disabled")
    # CPU defaults to caller allocation; the explicit function ABI exercises
    # the same callee path selected by CUDA's execution-backend manifest.
    function = function.with_attr("tilelang_callee_allocated_outputs", 1)
    target = tvm_tirx.target.Target({"kind": kind, **({"mcpu": "generic"} if kind == "llvm" else {})})
    host = target
    return tilelang.compile(function, out_idx=outputs, target=target, target_host=host, execution_backend="tvm_ffi")


def dynamic_pair():
    n = T.dynamic("n", "int64")

    @T.prim_func
    def main(B: T.Tensor((n,), "float32"), A: T.Buffer((n,), "float32", align=4), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            for i in T.Parallel(n):
                B[i] = A[i] - 1.0
                C[i] = A[i] + 2.0

    return main


@pytest.mark.parametrize("kind", ["llvm", "c"])
@pytest.mark.parametrize("extent", [0, 1, 17, 257])
def test_callee_multi_output_order_and_dynamic_shape(kind, extent):
    kernel = compile_output(dynamic_pair(), [2, 0], kind)
    values = torch.arange(extent + 5, dtype=torch.float32)[5:]
    plus, minus = kernel(values)
    assert isinstance(plus, torch.Tensor) and isinstance(minus, torch.Tensor)
    torch.testing.assert_close(plus, values + 2.0, rtol=0, atol=0)
    torch.testing.assert_close(minus, values - 1.0, rtol=0, atol=0)
    assert plus.shape == minus.shape == values.shape


@pytest.mark.parametrize("extent", [0, 1, 17])
def test_callee_scalar_only_late_shape(extent):
    n = T.dynamic("n", "int64")

    @T.prim_func
    def main(B: T.Tensor((n,), "int32")):
        with T.Kernel(1):
            for i in T.Parallel(n):
                B[i] = T.int32(i) + 3

    # This ABI test makes the free shape symbol a late explicit parameter.
    # Frontend spelling compatibility is covered independently.
    from tvm_tirx import tirx

    function = tirx.Function([main.params[0], n], main.body, main.ret_type, main.attrs)
    kernel = compile_output(function, [0])
    output = kernel(extent)
    torch.testing.assert_close(output, torch.arange(extent, dtype=torch.int32) + 3, rtol=0, atol=0)


def test_callee_single_output_and_rank_zero():
    @T.prim_func
    def main(B: T.Tensor((), "float32"), value: T.float32):
        with T.Kernel(1):
            B[()] = value + 0.5

    kernel = compile_output(main, [0])
    result = kernel(2.0)
    assert isinstance(result, torch.Tensor) and result.ndim == 0
    assert result.item() == 2.5


def test_callee_export_reload_and_function_lifetime(tmp_path):
    kernel = compile_output(dynamic_pair(), [2, 0])
    x = torch.arange(7, dtype=torch.float32)
    torch.testing.assert_close(kernel(x)[0], x + 2, rtol=0, atol=0)
    executable = kernel.adapter.get_exportable_executable()
    library = tmp_path / "output.so"
    executable.export_library(library, **kernel.adapter.get_export_kwargs())
    script = tmp_path / "reload.py"
    script.write_text("""import gc, sys
import torch
import tilelang
import tvm_ffi
module = tvm_ffi.load_module(sys.argv[1])
entry = module.main
del module
gc.collect()
for n in (0, 1, 19):
    values = torch.arange(n, dtype=torch.float32)
    plus, minus = entry(values, values)
    torch.testing.assert_close(plus, values + 2, rtol=0, atol=0)
    torch.testing.assert_close(minus, values - 1, rtol=0, atol=0)
assert "tvm" not in sys.modules
print("output reload and lifetime passed")
""")
    result = subprocess.run([sys.executable, str(script), str(library)], env=os.environ.copy(), capture_output=True, text=True)
    (tmp_path / "reload.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr


def test_callee_schema_rejects_corruption():
    kernel = compile_output(dynamic_pair(), [2, 0])
    schema = kernel.adapter.host_mod.attrs["tl.output_module_schema"]
    loader = tvm_ffi.get_global_func("ffi.Module.load_from_bytes.tilelang.output")
    for corrupt in (bytes(schema)[:9], bytes(schema) + b"tail", b"bad schema"):
        with pytest.raises(ValueError):
            loader(corrupt)
