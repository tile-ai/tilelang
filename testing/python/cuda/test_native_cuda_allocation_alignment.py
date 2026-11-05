"""The TileLang CUDA emitter retains alignment promised by native tensor types."""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest
import tvm_ffi
from tvm_tirx import IRModule, ir, target, tirx


def emit(scope, alignment, dtype, count, tmp_path):
    storage = ir.Var("arena", tirx.TensorType((count,), dtype, scope=scope, data_alignment=alignment))
    allocation = ir.Bind(storage, ir.Call("tirx.alloc_tensor", [], ty_args=[storage.ty]))
    function = tirx.Function([], [allocation]).with_attr("global_symbol", "allocation_probe").with_attr("calling_conv", 2)
    module = tvm_ffi.get_global_func("target.build.tilelang_cuda_without_compile")(
        IRModule({"allocation_probe": function}), target.Target({"kind": "cuda", "arch": "sm_100a"})
    )
    source = module.inspect_source("cuda")
    (tmp_path / "allocation.cu").write_text(source)
    return source


@pytest.mark.parametrize(
    "scope,alignment,expected,dtype,count,physical_count",
    [
        # Zero requests the native TensorType default (64 bytes).
        ("shared", 0, 64, "uint8", 257, 257),
        ("shared", 8, 16, "uint8", 257, 257),
        ("shared", 256, 256, "uint8", 257, 257),
        ("shared", 2048, 2048, "float32", 32, 32),
        ("shared", 256, 256, "int4", 257, 129),
        ("shared.dyn", 0, 1024, "uint8", 257, None),
        ("shared.dyn", 256, 1024, "uint8", 257, None),
        ("shared.dyn", 2048, 2048, "uint8", 257, None),
    ],
)
def test_declared_alignment_keeps_storage_floor_and_arena_requirement(scope, alignment, expected, dtype, count, physical_count, tmp_path):
    source = emit(scope, alignment, dtype, count, tmp_path)
    declaration = next(line for line in source.splitlines() if re.search(r"\barena\[", line))
    assert f"__align__({expected})" in declaration
    assert ("extern __shared__" in declaration) == (scope == "shared.dyn")
    assert f"arena[{'' if physical_count is None else physical_count}]" in declaration


def test_non_power_of_two_shared_alignment_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="alignment must be a power of two"):
        emit("shared", 96, "uint8", 128, tmp_path)


@pytest.mark.parametrize("scope", ["shared", "shared.dyn"])
def test_cuda_sdk_accepts_large_native_alignment(scope, tmp_path):
    emit(scope, 2048, "uint8", 256, tmp_path)
    nvcc = shutil.which("nvcc")
    root = Path(__file__).resolve().parents[3]
    cutlass = Path(os.environ.get("TILELANG_TEST_CUTLASS_INCLUDE", root / "3rdparty/cutlass/include"))
    if nvcc is None or not (cutlass / "cute").is_dir():
        pytest.skip("The CUDA SDK and fixed CUTLASS headers are needed for device compilation")
    command = [
        nvcc,
        "-std=c++17",
        "--cubin",
        "-arch=sm_100a",
        f"-I{root / 'src'}",
        f"-I{cutlass}",
        str(tmp_path / "allocation.cu"),
        "-o",
        str(tmp_path / "allocation.cubin"),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    (tmp_path / "nvcc.json").write_text(json.dumps({"command": command, "exit_code": result.returncode}, indent=2))
    (tmp_path / "nvcc.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
