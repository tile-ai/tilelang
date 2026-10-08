from tilelang.contrib import cc
import ctypes
import os
import pytest
import psutil
import subprocess
import sys


def test_create_shared_with_explicit_compiler_and_cxx17(tmp_path):
    compiler = cc.get_cc()
    if not compiler:
        pytest.skip("No host compiler is available")
    include = tmp_path / "include with spaces"
    include.mkdir()
    (include / "value.h").write_text("constexpr int value = 40;\n")
    source = tmp_path / "answer.cpp"
    source.write_text(
        '#include "value.h"\n'
        "#ifdef _WIN32\n#define EXPORT __declspec(dllexport)\n#else\n#define EXPORT\n#endif\n"
        'extern "C" EXPORT int answer() { if constexpr (value > 0) { return value + OFFSET; } else { return 0; } }\n'
    )
    library = tmp_path / ("answer." + cc.create_shared.output_format)
    environment_before = dict(os.environ)
    cc.create_shared(str(library), [str(source)], cc=compiler, options=["-std=c++17", "-I" + str(include), "-DOFFSET=2"])
    assert ctypes.CDLL(str(library)).answer() == 42
    assert dict(os.environ) == environment_before


def test_cross_compiler_does_not_persist_per_call_options():
    calls = []

    def compile_func(outputs, objects, options):
        calls.append((outputs, objects, options))

    fcompile = cc.cross_compiler(compile_func, options=["-base"])
    fcompile("first.so", ["first.o"], options=["-first"])
    fcompile("second.so", ["second.o"], options=["-second"])

    assert calls[0][2] == ["-base", "-first"]
    assert calls[1][2] == ["-base", "-second"]


def test_create_executable(tmp_path):
    compiler = cc.get_cc()
    if not compiler:
        pytest.skip("No host compiler is available")
    source = tmp_path / "probe.cpp"
    source.write_text('#include <string>\nint main() { return std::string("hello").size() == 5 ? 0 : 1; }\n')
    executable = tmp_path / ("probe.exe" if sys.platform == "win32" else "probe")
    cc.create_executable(str(executable), str(source), cc=compiler, options=["-std=c++17"])
    subprocess.run([str(executable)], check=True)


def test_compiler_failure_reports_command(tmp_path):
    source = tmp_path / "invalid.cpp"
    source.write_text("this is not valid C++;\n")
    compiler = cc.get_cc()
    if not compiler:
        pytest.skip("No host compiler is available")
    with pytest.raises(RuntimeError, match="Compilation error") as error:
        cc.create_shared(str(tmp_path / ("invalid." + cc.create_shared.output_format)), str(source), cc=compiler)
    assert str(source) in str(error.value)
    assert "Command line:" in str(error.value)


def test_compiler_timeout_terminates_process(tmp_path):
    pid = tmp_path / "compiler.pid"
    script = "import os, pathlib, sys, time; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)"
    with pytest.raises(subprocess.TimeoutExpired):
        cc._run_compiler([sys.executable, "-c", script, str(pid)], timeout=1)
    try:
        process = psutil.Process(int(pid.read_text()))
        process.wait(timeout=5)
    except psutil.NoSuchProcess:
        pass
