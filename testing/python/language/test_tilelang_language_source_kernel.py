import inspect
import os
import re
import tempfile
from pathlib import Path

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing


CUDA_SOURCE = """
extern "C" __global__ void external_copy(float* A, float* B, int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        B[i] = A[i];
    }
}
"""

CUDA_SOURCE_WITH_EXPLICIT_ENTRY = """
extern "C" __global__ void helper_kernel(float* A) {
    if ((int)(blockIdx.x * blockDim.x + threadIdx.x) == 0) {
        A[0] = 0.0f;
    }
}

extern "C" __global__ void external_copy_entry(float* A, float* B, int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        B[i] = A[i];
    }
}
"""


def make_source_kernel(source_code_or_path: str | os.PathLike[str], *, entry_name: str | None = None):
    N = T.dynamic("N")

    @T.prim_func
    def main(
        A: T.Tensor((N,), T.float32),
        B: T.Tensor((N,), T.float32),
    ):
        T.CUDASourceCodeKernel(
            source_code_or_path,
            T.ceildiv(N, 128),
            entry_name=entry_name,
            threads=128,
        )

    return main


def get_single_device_function(device_mod):
    functions = list(device_mod.functions.items())
    assert len(functions) == 1
    return functions[0]


def get_single_device_function_name(device_mod) -> str:
    g_var, _ = get_single_device_function(device_mod)
    return g_var.name_hint


def get_single_device_function_global_symbol(device_mod) -> str:
    _, func = get_single_device_function(device_mod)
    return str(func.attrs["global_symbol"])


def test_source_kernel_api_signature():
    params = inspect.signature(T.CUDASourceCodeKernel).parameters

    assert "source_code_or_path" in params
    assert "entry_name" in params
    assert "source_code" not in params
    assert "source_path" not in params


@tilelang.testing.requires_cuda
def test_source_kernel_inline_codegen():
    artifact = tilelang.lower(make_source_kernel(CUDA_SOURCE), target="cuda")
    function_name = get_single_device_function_name(artifact.device_mod)
    global_symbol = get_single_device_function_global_symbol(artifact.device_mod)

    assert global_symbol == "external_copy"
    assert re.search(
        rf"__global__\s+void\s+(?:__launch_bounds__\([^\)]*\)\s+)?{re.escape(global_symbol)}\s*\(",
        artifact.kernel_source,
    )
    assert "B[i] = A[i];" in artifact.kernel_source
    if function_name != global_symbol:
        assert not re.search(
            rf"__global__\s+void\s+(?:__launch_bounds__\([^\)]*\)\s+)?{re.escape(function_name)}\s*\(",
            artifact.kernel_source,
        )


@tilelang.testing.requires_cuda
def test_source_kernel_loads_from_file():
    with tempfile.NamedTemporaryFile("w", suffix=".cu", delete=False, encoding="utf-8") as f:
        f.write(CUDA_SOURCE)
        source_path = f.name

    try:
        artifact = tilelang.lower(make_source_kernel(Path(source_path)), target="cuda")
    finally:
        os.unlink(source_path)

    assert "B[i] = A[i];" in artifact.kernel_source


@tilelang.testing.requires_cuda
def test_source_kernel_explicit_entry_name():
    artifact = tilelang.lower(
        make_source_kernel(
            CUDA_SOURCE_WITH_EXPLICIT_ENTRY,
            entry_name="external_copy_entry",
        ),
        target="cuda",
    )
    function_name = get_single_device_function_name(artifact.device_mod)
    global_symbol = get_single_device_function_global_symbol(artifact.device_mod)

    assert global_symbol == "external_copy_entry"
    assert re.search(
        rf"__global__\s+void\s+(?:__launch_bounds__\([^\)]*\)\s+)?{re.escape(global_symbol)}\s*\(",
        artifact.kernel_source,
    )
    assert "helper_kernel(" in artifact.kernel_source
    assert "B[i] = A[i];" in artifact.kernel_source
    if function_name != global_symbol:
        assert not re.search(
            rf"__global__\s+void\s+(?:__launch_bounds__\([^\)]*\)\s+)?{re.escape(function_name)}\s*\(",
            artifact.kernel_source,
        )


@tilelang.testing.requires_cuda
def test_source_kernel_abi_tracks_external_entry_signature():
    N = T.dynamic("N")

    @T.prim_func
    def main(
        A: T.Tensor((N,), T.float32),
        B: T.Tensor((N,), T.float32),
        C: T.Tensor((N,), T.float32),
        alpha: T.int32,
    ):
        T.CUDASourceCodeKernel(CUDA_SOURCE, T.ceildiv(N, 128), threads=128)

    artifact = tilelang.lower(main, target="cuda", enable_host_codegen=False, enable_device_compile=False)
    function_name = get_single_device_function_name(artifact.device_mod)
    device_script = artifact.device_mod.script()
    signature = next(line.strip() for line in device_script.splitlines() if f"def {function_name}(" in line)
    assert 'A: T.handle("float32", "global")' in signature
    assert 'B: T.handle("float32", "global")' in signature
    assert "N: T.int32" in signature
    assert "C:" not in signature
    assert "alpha:" not in signature


@tilelang.testing.requires_cuda
def test_source_kernel_can_mix_with_regular_kernel():
    N = T.dynamic("N")

    @T.prim_func
    def main(
        A: T.Tensor((N,), T.float32),
        B: T.Tensor((N,), T.float32),
    ):
        T.CUDASourceCodeKernel(CUDA_SOURCE, T.ceildiv(N, 128), threads=128)
        with T.Kernel(T.ceildiv(N, 128), threads=128):
            if T.get_thread_binding() == 0 and T.get_block_binding() == 0:
                B[0] = A[0]

    artifact = tilelang.lower(main, target="cuda")

    definitions = re.findall(r"__global__\s+void(?:\s+__launch_bounds__\([^\)]*\))?\s+\w+\s*\([^;]*\)\s*\{", artifact.kernel_source)
    assert len(definitions) == 2
    assert "external_copy(" in artifact.kernel_source
    assert "B[i] = A[i];" in artifact.kernel_source
    assert "B[0] = A[0];" in artifact.kernel_source


@tilelang.testing.requires_cuda
def test_source_kernel_duplicate_entry_name_rejected():
    func_a = make_source_kernel(CUDA_SOURCE).with_attr("global_symbol", "source_kernel_a")
    func_b = make_source_kernel(CUDA_SOURCE).with_attr("global_symbol", "source_kernel_b")
    mod = tilelang.tvm.IRModule(
        {
            func_a.attrs["global_symbol"]: func_a,
            func_b.attrs["global_symbol"]: func_b,
        }
    )

    with pytest.raises(
        tilelang.tvm.TVMError,
        match=r"Duplicate CUDA kernel global_symbol `external_copy`",
    ):
        tilelang.lower(mod, target="cuda")


if __name__ == "__main__":
    tilelang.testing.main()
