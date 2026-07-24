from tilelang.contrib import cc


def test_cross_compiler_does_not_persist_per_call_options():
    calls = []

    def compile_func(outputs, objects, options):
        calls.append((outputs, objects, options))

    fcompile = cc.cross_compiler(compile_func, options=["-base"])
    fcompile("first.so", ["first.o"], options=["-first"])
    fcompile("second.so", ["second.o"], options=["-second"])

    assert calls[0][2] == ["-base", "-first"]
    assert calls[1][2] == ["-base", "-second"]
