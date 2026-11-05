"""Evaluate emitted MLX index code over finite physical thread-block grids."""

import ctypes
import json
import os
import re
import shlex
import shutil
import subprocess
import sys

import numpy as np
import pytest

import tilelang
import tilelang.language as T


def _mapping_kernel(panel, physical_rows, logical_columns):
    @T.prim_func
    def main(Output: T.Tensor((physical_rows * panel, logical_columns), "int32")):
        with T.Kernel(logical_columns * panel, physical_rows, threads=32) as (bx, by):
            T.use_swizzle(panel_size=panel, order="mlx")
            if T.get_thread_binding() == 0:
                logical_x = bx // panel
                logical_y = by * panel + bx % panel
                Output[logical_y, logical_x] = logical_y * 997 + logical_x

    return main


def _host_source(generated, panel, physical_rows, logical_columns):
    # Retain the emitted scalar function body exactly. Only Metal includes,
    # annotations and storage qualifiers require a host syntax shim.
    portable = re.sub(r"^#include <metal[^>]*>\s*$", "", generated, flags=re.MULTILINE)
    portable = re.sub(r"\[\[.*?\]\]", "", portable, flags=re.DOTALL)
    portable = portable.replace("using namespace metal;", "")
    signature = re.search(r"\bvoid\s+main_kernel\s*\((.*?)\)\s*\{", portable, flags=re.DOTALL)
    assert signature is not None, generated
    arguments = []
    for parameter in signature.group(1).split(","):
        name = re.search(r"([A-Za-z_][A-Za-z_0-9]*)\s*$", parameter).group(1)
        if "*" in parameter:
            assert name.startswith("Output"), parameter
            arguments.append("output")
        elif name == "blockIdx":
            arguments.append("uint3(x, y, 0)")
        elif name == "__gridDim":
            arguments.append(f"uint3({logical_columns * panel}, {physical_rows}, 1)")
        elif name == "threadIdx":
            arguments.append("uint3(0, 0, 0)")
        elif name == "__simd_group_id":
            arguments.append("0")
        else:
            raise AssertionError(f"Unexpected Metal parameter in scalar oracle: {parameter}")
    preamble = """
#include <cstdint>
#define kernel
#define device
#define threadgroup
#define thread
#define constant
using uint = unsigned int;
using ushort = unsigned short;
struct uint3 {
  unsigned int x, y, z;
  uint3(unsigned int x, unsigned int y, unsigned int z) : x(x), y(y), z(z) {}
};
"""
    wrapper = f"""
extern "C" void run_grid(int32_t* output) {{
  for (unsigned int y = 0; y < {physical_rows}; ++y) {{
    for (unsigned int x = 0; x < {logical_columns * panel}; ++x) {{
      main_kernel({", ".join(arguments)});
    }}
  }}
}}
"""
    return preamble + portable + wrapper


@pytest.mark.parametrize("panel", [1, 2, 4, 8])
@pytest.mark.parametrize("physical_rows", [1, 3])
def test_emitted_mlx_mapping_has_one_write_per_logical_coordinate(panel, physical_rows, tmp_path):
    if not sys.platform.startswith("linux"):
        pytest.skip("This host C++ syntax-shim probe currently uses a Linux shared library")
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler or shutil.which(compiler[0]) is None:
        pytest.skip("A host C++ compiler is required to evaluate emitted scalar Metal code")
    columns = 5
    artifact = tilelang.lower(
        _mapping_kernel(panel, physical_rows, columns),
        target={"kind": "metal", "keys": ["metal", "gpu"]},
        target_host={"kind": "llvm", "mcpu": "generic"},
    )
    generated = artifact.kernel_source
    (tmp_path / "kernel.metal").write_text(generated)
    source = tmp_path / "host.cc"
    source.write_text(_host_source(generated, panel, physical_rows, columns))
    library = tmp_path / "host.so"
    command = [*compiler, "-std=c++17", "-O2", "-shared", "-fPIC", str(source), "-o", str(library)]
    (tmp_path / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    compile_result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (tmp_path / "compile.log").write_text(compile_result.stdout)
    assert compile_result.returncode == 0, compile_result.stdout
    loaded = ctypes.CDLL(str(library))
    run = loaded.run_grid
    run.argtypes = [ctypes.POINTER(ctypes.c_int32)]
    run.restype = None
    count = physical_rows * panel * columns
    sentinel = np.int32(-2147483647)
    # The extra storage safely exposes the old double-mapping writes as guard
    # corruption. It does not change the kernel's declared output shape.
    output = np.full(count * 16, sentinel, dtype="int32")
    run(output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
    expected = np.array([row * 997 + col for row in range(physical_rows * panel) for col in range(columns)], dtype="int32")
    np.save(tmp_path / "actual.npy", output)
    np.save(tmp_path / "expected.npy", expected)
    np.testing.assert_array_equal(output[:count], expected)
    np.testing.assert_array_equal(output[count:], sentinel)
