import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.transform import PassConfigKey
from tilelang.tools.compile_only import (
    _is_cuda_target,
    _pass_context_config,
    compile_kernel_source,
    cuda_codegen_available,
    discover_prim_func,
    resolve_target,
)

_CAN_EMIT_LINE_DIRECTIVES = hasattr(PassConfigKey, "TL_EMIT_LINE_DIRECTIVES")

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
            C[i] = A[i] + B[i]  # line_marker_store
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
            C[i] = A[i] + B[i]  # line_marker_store
    return C


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, "-I", "-m", "tilelang.tools.compile_only", *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )


def _line_directives(source: str) -> list[tuple[int, str]]:
    return [(int(num), fname) for num, fname in re.findall(r'^#line (\d+) "(.*)"$', source, re.M)]


def _marker_line(path: Path, marker: str) -> int:
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if marker in line:
            return i
    raise ValueError(f"marker not found: {marker}")


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


def test_pass_context_config_enables_line_directives_when_supported():
    config = _pass_context_config()
    if _CAN_EMIT_LINE_DIRECTIVES:
        assert config.get(PassConfigKey.TL_EMIT_LINE_DIRECTIVES) is True
    else:
        assert config == {}


def test_line_directive_parser_reads_file_and_line():
    source = '#line 12 "/tmp/example.py"\nint x;\n'
    assert _line_directives(source) == [(12, "/tmp/example.py")]


@pytest.mark.skipif(
    not _CAN_EMIT_LINE_DIRECTIVES,
    reason="tl.emit_line_directives is not in this wheel; CI merge with main has it",
)
def test_compile_kernel_source_emits_line_directives():
    source = compile_kernel_source(_add.get_tir())
    directives = _line_directives(source)
    assert directives, f"no #line directives:\n{source}"
    store_line = _marker_line(Path(__file__), "line_marker_store")
    files = {fname for _, fname in directives}
    assert any(Path(fname).name == Path(__file__).name for fname in files), files
    assert any(num == store_line and Path(fname).name == Path(__file__).name for num, fname in directives), (
        store_line,
        directives,
        source,
    )


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


@pytest.mark.skipif(
    not _CAN_EMIT_LINE_DIRECTIVES,
    reason="tl.emit_line_directives is not in this wheel; CI merge with main has it",
)
def test_compile_only_cli_emits_line_directives(tmp_path: Path):
    example = tmp_path / "example.py"
    output = tmp_path / "out.s"
    example.write_text(_COMPILE_ONLY_EXAMPLE)

    result = _run_cli("--output_file", str(output), str(example))

    assert result.returncode == 0, result.stderr
    text = output.read_text()
    directives = _line_directives(text)
    assert directives, f"no #line directives:\n{text}"
    store_line = _marker_line(example, "line_marker_store")
    assert any(num == store_line and Path(fname).name == example.name for num, fname in directives), (
        store_line,
        directives,
        text,
    )


def test_compile_only_cli_reports_compile_error(tmp_path: Path):
    example = tmp_path / "broken.py"
    output = tmp_path / "out.s"
    example.write_text("x = 1\n")

    result = _run_cli("--output_file", str(output), str(example))

    assert result.returncode != 0
    assert result.stderr.strip()
    assert "no @tilelang.jit kernel or PrimFunc found" in result.stderr
    # A4: this is a compile diagnostic, not a missing-GPU / missing-driver abort.
    assert "CUDA" not in result.stderr
    assert "driver" not in result.stderr.lower()
    assert not output.is_file() or not output.read_text().strip()


def test_compile_only_cli_unlinks_stale_output_on_failure(tmp_path: Path):
    # CE reuses --output_file. A failed compile must not leave yesterday's assembly.
    example = tmp_path / "broken.py"
    output = tmp_path / "out.s"
    stale = "YESTERDAY_ASSEMBLY_MUST_NOT_SURVIVE\n"
    output.write_text(stale)
    example.write_text("x = 1\n")

    result = _run_cli("--output_file", str(output), str(example))

    assert result.returncode != 0
    assert "no @tilelang.jit kernel or PrimFunc found" in result.stderr
    assert not output.is_file() or output.read_text() != stale


def test_compile_only_cli_keeps_input_when_output_is_same_path(tmp_path: Path):
    example = tmp_path / "example.py"
    example.write_text(_COMPILE_ONLY_EXAMPLE)

    result = _run_cli("--output_file", str(example), str(example))

    assert result.returncode != 0
    assert example.is_file()
    assert example.read_text() == _COMPILE_ONLY_EXAMPLE
    assert "output_file" in result.stderr
    assert "input" in result.stderr.lower()


def test_compile_only_cli_keeps_input_when_output_is_symlink(tmp_path: Path):
    example = tmp_path / "example.py"
    alias = tmp_path / "out.py"
    example.write_text(_COMPILE_ONLY_EXAMPLE)
    alias.symlink_to(example)

    result = _run_cli("--output_file", str(alias), str(example))

    assert result.returncode != 0
    assert example.is_file()
    assert example.read_text() == _COMPILE_ONLY_EXAMPLE
    assert alias.is_symlink()


def test_compile_only_cli_reports_syntax_error(tmp_path: Path):
    example = tmp_path / "bad.py"
    output = tmp_path / "out.s"
    example.write_text("def (\n")

    result = _run_cli("--output_file", str(output), str(example))

    assert result.returncode != 0
    assert result.stderr.strip()
    assert "SyntaxError" in result.stderr or "invalid syntax" in result.stderr.lower()
    assert not output.is_file() or not output.read_text().strip()


def test_compile_only_cli_reports_empty_file(tmp_path: Path):
    example = tmp_path / "empty.py"
    output = tmp_path / "out.s"
    example.write_text("")

    result = _run_cli("--output_file", str(output), str(example))

    assert result.returncode != 0
    assert "no @tilelang.jit kernel or PrimFunc found" in result.stderr
    assert not output.is_file() or not output.read_text().strip()


def test_compile_only_cli_reports_missing_input_file(tmp_path: Path):
    missing = tmp_path / "no_such.py"
    output = tmp_path / "out.s"

    result = _run_cli("--output_file", str(output), str(missing))

    assert result.returncode != 0
    assert result.stderr.strip()
    assert not output.is_file() or not output.read_text().strip()


def test_compile_only_cli_requires_output_file(tmp_path: Path):
    example = tmp_path / "example.py"
    example.write_text(_COMPILE_ONLY_EXAMPLE)

    result = _run_cli(str(example))

    assert result.returncode != 0
    assert "output_file" in result.stderr


def test_discover_prim_func_respects_module_order():
    # A later @tilelang.jit must not steal an earlier PrimFunc (module order).
    prim = _add.get_tir()
    module = ModuleType("tilelang_compile_only_order")
    module.early_prim = prim
    module.later_jit = _add

    found = discover_prim_func(module)

    assert found is prim


def test_is_cuda_target_recognizes_option_strings():
    assert _is_cuda_target("cuda")
    assert _is_cuda_target("cuda -arch=sm_90")
    assert _is_cuda_target("CUDA -arch=sm_80")
    assert _is_cuda_target('{"kind": "cuda", "arch": "sm_90"}')
    assert not _is_cuda_target("c")
    assert not _is_cuda_target("llvm")
    assert not _is_cuda_target('{"kind": "c"}')


def test_resolve_target_parses_cuda_options():
    pinned = resolve_target("cuda")
    assert pinned == {"kind": "cuda", "arch": "sm_80"}

    cli = resolve_target("cuda -arch=sm_90")
    assert cli["kind"] == "cuda"
    assert cli["arch"] == "sm_90"

    json_target = resolve_target('{"kind": "cuda", "arch": "sm_90"}')
    assert json_target["kind"] == "cuda"
    assert json_target["arch"] == "sm_90"


def test_resolve_target_rejects_json_auto():
    with pytest.raises(ValueError, match="auto"):
        resolve_target('{"kind": "auto"}')


def test_resolve_target_pins_json_cuda_without_arch():
    assert resolve_target('{"kind": "cuda"}') == {"kind": "cuda", "arch": "sm_80"}


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


def test_compile_only_cli_rejects_json_auto_target(tmp_path: Path):
    example = tmp_path / "example.py"
    output = tmp_path / "out.s"
    example.write_text(_COMPILE_ONLY_EXAMPLE)

    result = _run_cli("--target", '{"kind": "auto"}', "--output_file", str(output), str(example))

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

    with pytest.raises(RuntimeError, match="CUDA codegen FFI missing"):
        compile_kernel_source(_add.get_tir(), "cuda -arch=sm_90")
    with pytest.raises(RuntimeError, match="CUDA codegen FFI missing"):
        compile_kernel_source(_add.get_tir(), '{"kind": "cuda", "arch": "sm_90"}')

    optioned = tmp_path / "out_sm90.s"
    result = _run_cli("--target", "cuda -arch=sm_90", "--output_file", str(optioned), str(example))

    assert result.returncode != 0
    assert "CUDA codegen FFI missing" in result.stderr
    assert not optioned.is_file() or not optioned.read_text().strip()


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
