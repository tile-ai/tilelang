from tilelang.contrib import cc, msvc


def test_cross_compiler_does_not_persist_per_call_options():
    calls = []

    def compile_func(outputs, objects, options):
        calls.append((outputs, objects, options))

    fcompile = cc.cross_compiler(compile_func, options=["-base"])
    fcompile("first.so", ["first.o"], options=["-first"])
    fcompile("second.so", ["second.o"], options=["-second"])

    assert calls[0][2] == ["-base", "-first"]
    assert calls[1][2] == ["-base", "-second"]


def test_windows_arch_prefers_native_architecture(monkeypatch):
    monkeypatch.setenv("PROCESSOR_ARCHITECTURE", "AMD64")
    monkeypatch.setenv("PROCESSOR_ARCHITEW6432", "ARM64")

    assert msvc._windows_arch() == "arm64"


def test_vsdevcmd_uses_native_architecture(monkeypatch):
    command_lines = []

    class CompletedProcess:
        returncode = 0
        stdout = "PATH=C:\\tools\n"

    def run(command_line, **_kwargs):
        command_lines.append(command_line)
        return CompletedProcess()

    monkeypatch.setattr(msvc, "_windows_arch", lambda: "arm64")
    monkeypatch.setattr(msvc.subprocess, "run", run)

    compiler_env = msvc._import_vsdevcmd_environment("C:\\VS\\VsDevCmd.bat")

    assert compiler_env is not None
    assert "-arch=arm64 -host_arch=arm64" in command_lines[0]
