"""TileLang owns directional FP8 storage aliases; native TIRx remains exact."""

from functools import lru_cache
import os
import subprocess
import sys

import pytest
from tvm_tirx import ir, tirx
from tvm_tirx.ir import prim

from tilelang.testing.packed import compile_packed_function

from test_native_packed_subbyte_abi import _descriptor, _invoke


_ALIASES = [
    ("float8_e4m3", "float8_e4m3fn"),
    ("float8_e4m3", "float8_e4m3fnuz"),
    ("float8_e5m2", "float8_e5m2fnuz"),
]


@lru_cache(None)
def _compile(expected, policy):
    size = ir.Var("size", "int64")
    tensor = ir.Var("input", tirx.TensorType([size], expected))
    function = tirx.Function([tensor], ir.Return(size), ir.PrimType("int64"))
    function = function.with_attr("global_symbol", "main")
    return compile_packed_function(function, tilelang_owned=policy)


@pytest.mark.parametrize("expected,storage", _ALIASES)
@pytest.mark.parametrize("lanes", [1, 2, 4])
def test_storage_aliases_preserve_lane_count_and_logical_shape(expected, storage, lanes):
    suffix = "" if lanes == 1 else f"x{lanes}"
    descriptor = _descriptor((7,), storage + suffix)
    assert _invoke(_compile(expected + suffix, True), descriptor) == 7
    with pytest.raises(TypeError, match="dtype"):
        _invoke(_compile(expected + suffix, False), descriptor)


@pytest.mark.parametrize("expected,storage", _ALIASES)
def test_specific_fp8_types_do_not_acquire_reverse_aliases(expected, storage):
    with pytest.raises(RuntimeError, match="dtype"):
        _invoke(_compile(storage, True), _descriptor((7,), expected))


@pytest.mark.parametrize(
    "expected,storage",
    [
        ("float8_e4m3", "float8_e5m2"),
        ("float8_e5m2", "float8_e4m3fn"),
        ("float8_e4m3fn", "float8_e4m3fnuz"),
        ("float8_e4m3", "uint8"),
    ],
)
def test_equal_storage_width_does_not_make_unlisted_aliases(expected, storage):
    with pytest.raises(RuntimeError, match="dtype"):
        _invoke(_compile(expected, True), _descriptor((7,), storage))


@pytest.mark.parametrize("expected,storage", _ALIASES)
@pytest.mark.parametrize("field,value", [("bits", 16), ("bits", 4), ("lanes", 2), ("lanes", 0)])
def test_aliases_do_not_relax_bits_or_lanes(expected, storage, field, value):
    descriptor = _descriptor((7,), storage)
    setattr(descriptor.dtype, field, value)
    with pytest.raises(RuntimeError, match="dtype"):
        _invoke(_compile(expected, True), descriptor)


@pytest.mark.parametrize("dtype", ["float8_e4m3", "float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz"])
@pytest.mark.parametrize("policy", [False, True])
def test_exact_dtype_is_still_accepted(dtype, policy):
    assert _invoke(_compile(dtype, policy), _descriptor((7,), dtype)) == 7


@pytest.mark.parametrize("expected,storage", _ALIASES)
def test_alias_storage_is_not_numerically_converted(expected, storage):
    tensor = ir.Var("input", tirx.TensorType([1], expected))
    bits = tirx.reinterpret(ir.TensorLoad(tensor, [0]), ty=ir.PrimType("uint8"))
    function = tirx.Function([tensor], ir.Return(prim.Cast("int32", bits)), ir.PrimType("int32"))
    function = function.with_attr("global_symbol", "main")
    compiled = compile_packed_function(function)
    descriptor = _descriptor((1,), storage)
    import ctypes

    for byte in (0x00, 0x80, 0x7F, 0xFF, 0x3C):
        ctypes.c_uint8.from_address(descriptor.data).value = byte
        assert _invoke(compiled, descriptor) == byte


def _shape_output_kernel(dtype, callee):
    import tilelang
    import tilelang.cpu.language as T

    size = T.dynamic("size", "int64")

    @T.prim_func
    def fill(A: T.Tensor((size,), dtype), B: T.Tensor((size,), "int32")):
        with T.Kernel(1):
            for index in T.serial(size):
                B[index] = T.int32(index) + 2

    if callee:
        fill = fill.with_attr("tilelang_callee_allocated_outputs", True)
    target = {"kind": "c"}
    return tilelang.compile(fill, out_idx=[1], execution_backend="tvm_ffi", target=target, target_host=target)


@pytest.mark.parametrize("expected,storage", _ALIASES)
@pytest.mark.parametrize("callee", [False, True])
def test_fp8_input_alias_policy_reaches_allocated_output_worker(expected, storage, callee):
    import torch

    kernel = _shape_output_kernel(expected, callee)
    for size in (0, 1, 17):
        source = torch.zeros(size + 3, dtype=getattr(torch, storage))[3:]
        actual = kernel(source)
        torch.testing.assert_close(actual, torch.arange(size, dtype=torch.int32) + 2, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="dtype"):
        kernel(torch.zeros(7, dtype=torch.uint8))


def test_fp8_allocated_output_policy_survives_export_and_new_process_reload(tmp_path):
    kernel = _shape_output_kernel("float8_e4m3", True)
    executable = kernel.adapter.get_exportable_executable()
    library = tmp_path / "fp8_output.so"
    executable.export_library(library, **kernel.adapter.get_export_kwargs())
    script = tmp_path / "reload.py"
    script.write_text("""import gc, json, sys
from pathlib import Path
import torch
import tilelang
import tvm_ffi
module = tvm_ffi.load_module(sys.argv[1])
entry = module["fill"]
del module
gc.collect()
for dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
    for n in (0, 1, 19):
        source = torch.zeros(n + 3, dtype=dtype)[3:]
        actual = entry(source, source)
        torch.testing.assert_close(actual, torch.arange(n, dtype=torch.int32) + 2, rtol=0, atol=0)
try:
    source = torch.zeros(7, dtype=torch.uint8)
    entry(source, source)
except RuntimeError as error:
    assert "dtype" in str(error)
else:
    raise AssertionError("Unlisted uint8 alias was accepted")
assert "tvm" not in sys.modules
maps = Path("/proc/self/maps")
mapped = None
if maps.is_file():
    mapped = sorted({line.split()[-1] for line in maps.read_text().splitlines()
                     if "libtvm" in line and line.split()[-1].startswith("/")})
    assert not any("/3rdparty/tvm/" in path for path in mapped)
print(json.dumps({"status": "passed", "mapped_libraries": mapped,
                  "library_map_available": mapped is not None}))
""")
    result = subprocess.run(
        [sys.executable, str(script), str(library)],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    (tmp_path / "reload.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
