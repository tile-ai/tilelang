from tilelang.contrib import cc
import ctypes
import os
import pytest


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
