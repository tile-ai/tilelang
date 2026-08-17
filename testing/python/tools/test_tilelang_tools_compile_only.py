import subprocess
import sys
from pathlib import Path

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.tools.compile_only import compile_kernel_source, cuda_codegen_available, resolve_target

_COMPILE_ONLY_EXAMPLE = """\
import tilelang
import tilelang.language as T


@tilelang.jit
def add(A, B):
    N = 64
    A: T.Tensor((N,), T.float16)
    B: T.Tensor((N,), T.float16)
    C = T.empty((N,), T.float16)
    with T.Kernel(1, threads=64):
        for i in T.Parallel(N):
            C[i] = A[i] + B[i]
    return C
"""


@tilelang.jit
def _add(A, B):
    N = 64
    A: T.Tensor((N,), T.float16)
    B: T.Tensor((N,), T.float16)
    C = T.empty((N,), T.float16)
    with T.Kernel(1, threads=64):
        for i in T.Parallel(N):
            C[i] = A[i] + B[i]
    return C


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, "-I", "-m", "tilelang.tools.compile_only", *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )


def _looks_like_generated_source(text: str) -> bool:
    source = text.strip()
    if not source:
        return False
    # Real lower() text: CPU C / HIP C / CUDA C / TIR. Do not require __global__.
    markers = (
        "#include",
        'extern "C"',
        "tilelang target",
        "primfn",
        "@T.prim_func",
        "_kernel",
    )
    return any(marker in source for marker in markers)


def test_compile_kernel_source_default_cpu_c():
    source = compile_kernel_source(_add.get_tir())

    assert source.strip()
    assert _looks_like_generated_source(source)
    assert "__global__" not in source
    assert "AnnotateDeviceBoundTmaCopies" not in source


def test_compile_only_cli_writes_kernel_source(tmp_path: Path):
    example = tmp_path / "example.py"
    output = tmp_path / "out.s"
    example.write_text(_COMPILE_ONLY_EXAMPLE)

    result = _run_cli("--output_file", str(output), str(example))

    assert result.returncode == 0, result.stderr
    assert output.is_file()
    text = output.read_text()
    assert text.strip()
    assert _looks_like_generated_source(text)
    assert "__global__" not in text


def test_compile_only_cli_reports_compile_error(tmp_path: Path):
    example = tmp_path / "broken.py"
    output = tmp_path / "out.s"
    example.write_text("x = 1\n")

    result = _run_cli("--output_file", str(output), str(example))

    assert result.returncode != 0
    assert result.stderr.strip()
    assert "no @tilelang.jit kernel or PrimFunc found" in result.stderr
    assert "CUDA" not in result.stderr or "driver" not in result.stderr.lower()
    assert not output.is_file() or not output.read_text().strip()


def test_compile_only_cli_requires_output_file(tmp_path: Path):
    example = tmp_path / "example.py"
    example.write_text(_COMPILE_ONLY_EXAMPLE)

    result = _run_cli(str(example))

    assert result.returncode != 0
    assert "output_file" in result.stderr


def test_compile_only_cli_rejects_auto_target(tmp_path: Path):
    example = tmp_path / "example.py"
    output = tmp_path / "out.s"
    example.write_text(_COMPILE_ONLY_EXAMPLE)

    with pytest.raises(ValueError, match="auto"):
        resolve_target("auto")

    result = _run_cli("--target", "auto", "--output_file", str(output), str(example))

    assert result.returncode != 0
    assert "auto" in result.stderr
    assert not output.is_file() or not output.read_text().strip()


@pytest.mark.skipif(
    cuda_codegen_available(),
    reason="CUDA codegen FFI present; soft-fail path is for Metal-style wheels",
)
def test_compile_only_cli_cuda_soft_fails_without_ffi(tmp_path: Path):
    example = tmp_path / "example.py"
    output = tmp_path / "out.s"
    example.write_text(_COMPILE_ONLY_EXAMPLE)

    with pytest.raises(RuntimeError, match="CUDA codegen FFI missing"):
        compile_kernel_source(_add.get_tir(), "cuda")

    result = _run_cli("--target", "cuda", "--output_file", str(output), str(example))

    assert result.returncode != 0
    assert "CUDA codegen FFI missing" in result.stderr
    assert not output.is_file() or not output.read_text().strip()


@pytest.mark.skipif(
    not cuda_codegen_available(),
    reason="CUDA codegen FFI missing (e.g. macOS Metal wheel)",
)
def test_compile_kernel_source_optional_cuda():
    source = compile_kernel_source(_add.get_tir(), "cuda")

    assert source.strip()
    assert _looks_like_generated_source(source)
    assert "__global__" in source
    assert "add_kernel" in source


@pytest.mark.skipif(
    not cuda_codegen_available(),
    reason="CUDA codegen FFI missing (e.g. macOS Metal wheel)",
)
def test_compile_only_cli_optional_cuda_target(tmp_path: Path):
    example = tmp_path / "example.py"
    output = tmp_path / "out.s"
    example.write_text(_COMPILE_ONLY_EXAMPLE)

    result = _run_cli("--target", "cuda", "--output_file", str(output), str(example))

    assert result.returncode == 0, result.stderr
    text = output.read_text()
    assert text.strip()
    assert _looks_like_generated_source(text)
    assert "__global__" in text
    assert "add_kernel" in text


if __name__ == "__main__":
    tilelang.testing.main()
