"""Actual native FFI adapter calls; full JIT/pass-pipeline tests are separate."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir

import tilelang.cpu.language as T
from tilelang.engine.param import KernelParam
from tilelang.jit.adapter import TVMFFIKernelAdapter
from tilelang.testing.packed import build_packed_module


def make_adapter(kind):
    if kind == "llvm" and not tvm_tirx.runtime.enabled("llvm"):
        pytest.skip("Optional LLVM code generation is disabled")
    n = T.dynamic("n", "int64")

    @T.prim_func
    def repeat_add(B: T.Tensor((2 * n,), "float32"), A: T.Buffer((n,), "float32", align=4), value: T.float32):
        with T.Kernel(1):
            for i in T.serial(2 * n):
                B[i] = A[i % n] + value

    materialize = tvm_ffi.get_global_func("tl.transform.MaterializeKernelLaunch")(False, False, None, None, None)
    native = materialize(tvm_tirx.IRModule({"main": repeat_add}))["main"].with_attr("global_symbol", "main")
    target = tvm_tirx.target.Target({"kind": kind, **({"mcpu": "generic"} if kind == "llvm" else {})})
    module = build_packed_module(native, target=target)
    params = [KernelParam.from_var(param) for param in repeat_add.params]
    return TVMFFIKernelAdapter(params, [0], target, repeat_add, rt_mod=module)


@pytest.mark.parametrize("kind", ["llvm", "c"])
def test_torch_adapter_dynamic_output_before_input(kind):
    adapter = make_adapter(kind)
    assert adapter.executable is None
    for extent in (0, 1, 17, 257):
        # A nonzero underlying storage offset must not be applied twice.
        source = torch.arange(extent + 5, dtype=torch.float32)[5:]
        output = adapter(source, 1.25)
        assert isinstance(output, torch.Tensor)
        torch.testing.assert_close(output, source.repeat(2) + 1.25, rtol=0, atol=0)
    assert adapter.executable is not None
    with pytest.raises(ValueError, match="expected 2 inputs"):
        adapter(torch.ones(3))
    with pytest.raises(RuntimeError, match="expected rank 1"):
        adapter(torch.ones(3, 2), 1.0)
    with pytest.raises(RuntimeError, match="stride"):
        adapter(torch.arange(18, dtype=torch.float32)[::2], 1.0)


@pytest.mark.parametrize("kind", ["llvm", "c"])
def test_adapter_export_and_independent_process_reload(kind, tmp_path):
    adapter = make_adapter(kind)
    source = torch.arange(7, dtype=torch.float32)
    torch.testing.assert_close(adapter(source, 2.0), source.repeat(2) + 2, rtol=0, atol=0)
    executable = adapter.get_exportable_executable()
    library = tmp_path / "kernel.so"
    from tilelang.env import TILELANG_TEMPLATE_PATH

    executable.export_library(library, options=["-std=c++17", "-I" + TILELANG_TEMPLATE_PATH])
    function_path = tmp_path / "function.json"
    function_path.write_text(ir.save_json(adapter.prim_func))
    script = tmp_path / "reload.py"
    script.write_text("""import json
import sys
from pathlib import Path
import torch
import tilelang
import tvm_tirx
from tilelang.engine.param import KernelParam
from tilelang.jit.adapter import TVMFFIKernelAdapter
from tilelang.jit.adapter.base import CachedTextSource
from tvm_tirx import ir
root = Path(sys.argv[1])
function = ir.load_json((root / "function.json").read_text())
adapter = TVMFFIKernelAdapter.from_database(
    [KernelParam.from_var(p) for p in function.params], [0], sys.argv[2], function,
    CachedTextSource(text=""), CachedTextSource(text=""), str(root / "kernel.so"))
for extent in (1, 17, 31):
    source = torch.arange(extent, dtype=torch.float32)
    torch.testing.assert_close(adapter(source, 3.0), source.repeat(2) + 3, rtol=0, atol=0)
assert "tvm" not in sys.modules
mapped = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                 if "libtvm" in line and line.split()[-1].startswith("/")})
assert not any("/3rdparty/tvm/" in path for path in mapped)
print(json.dumps({"status": "passed", "cases": 3, "mapped_libraries": mapped}))
""")
    result = subprocess.run([sys.executable, str(script), str(tmp_path), kind], capture_output=True, text=True, env=os.environ.copy())
    (tmp_path / "reload.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(result.stdout.strip().splitlines()[-1])
    assert record["status"] == "passed" and record["cases"] == 3


def test_adapter_concurrent_calls_keep_shape_bindings_local():
    from concurrent.futures import ThreadPoolExecutor

    adapter = make_adapter("c")

    def run(extent):
        source = torch.arange(extent, dtype=torch.float32)
        output = adapter(source, 0.5)
        np.testing.assert_array_equal(output.numpy(), np.tile(source.numpy(), 2) + np.float32(0.5))

    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(run, (1, 7, 17, 31, 3, 8, 29, 2)))
