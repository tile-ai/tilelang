"""Compiler and linker identities used by the Ascend caches."""

import subprocess

import pytest

from tilelang.contrib import bisheng
from tilelang.env import env


@pytest.fixture(autouse=True)
def reset_bisheng_version_cache(monkeypatch):
    bisheng.get_bisheng_version.cache_clear()
    bisheng.get_ld_lld_version.cache_clear()
    monkeypatch.setattr(env, "TILELANG_COMPILE_TIMEOUT_SECONDS", "")
    yield
    bisheng.get_bisheng_version.cache_clear()
    bisheng.get_ld_lld_version.cache_clear()


def test_ld_lld_version_preserves_build_identity_and_is_cached(monkeypatch):
    version = "LLD 15.0.5 (build-revision) (compatible with GNU linkers)"
    queries = []
    monkeypatch.setattr(bisheng, "find_ld_lld_path", lambda: "/opt/cann/bin/ld.lld")

    def fake_check_output(command, **kwargs):
        queries.append(command)
        assert kwargs == {"stderr": subprocess.STDOUT, "text": True, "timeout": None}
        return version + "\n"

    monkeypatch.setattr(bisheng.subprocess, "check_output", fake_check_output)
    assert bisheng.get_ld_lld_version() == version
    assert bisheng.get_ld_lld_version() == version
    assert queries == [["/opt/cann/bin/ld.lld", "--version"]]


@pytest.mark.parametrize("output", ["", "\n", "unknown version\n"])
def test_ld_lld_version_rejects_unrecognized_output(monkeypatch, output):
    monkeypatch.setattr(bisheng, "find_ld_lld_path", lambda: "/opt/cann/bin/ld.lld")
    monkeypatch.setattr(bisheng.subprocess, "check_output", lambda *args, **kwargs: output)
    with pytest.raises(RuntimeError, match="Cannot determine ld.lld version"):
        bisheng.get_ld_lld_version()


@pytest.mark.parametrize("tool", ["bisheng", "ld_lld"])
def test_tool_version_query_failure_is_not_cached(monkeypatch, tool):
    monkeypatch.setattr(bisheng, f"find_{tool}_path", lambda: "/opt/tool")
    calls = []

    def fail_query(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command, "version query failed")

    monkeypatch.setattr(bisheng.subprocess, "check_output", fail_query)
    for _ in range(2):
        with pytest.raises(subprocess.CalledProcessError):
            getattr(bisheng, f"get_{tool}_version")()
    assert len(calls) == 2


@pytest.mark.parametrize("tool, name", [("bisheng", "Bisheng"), ("ld_lld", "ld.lld")])
def test_tool_version_query_timeout(monkeypatch, tool, name):
    monkeypatch.setattr(bisheng, f"find_{tool}_path", lambda: "/opt/tool")
    monkeypatch.setattr(env, "TILELANG_COMPILE_TIMEOUT_SECONDS", "2.5")

    def timeout_query(command, **kwargs):
        assert kwargs["timeout"] == 2.5
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(bisheng.subprocess, "check_output", timeout_query)
    with pytest.raises(RuntimeError, match=f"{name} version query timed out after 2.5 seconds"):
        getattr(bisheng, f"get_{tool}_version")()


def test_bisheng_version_preserves_build_identity_and_ignores_installation_path(monkeypatch, tmp_path):
    version = "2026-08-27T22:23:38+08:00 clang version 15.0.5 (clang-5c68a1cb1231 flang-5c68a1cb1231)"
    queries = []

    def fake_check_output(command, **kwargs):
        queries.append(command)
        assert kwargs["text"] is True
        assert kwargs["stderr"] == subprocess.STDOUT
        assert kwargs["timeout"] is None
        return f"{version}\nTarget: aarch64-unknown-linux-gnu\nThread model: posix\nInstalledDir: {command[0]}\n"

    monkeypatch.setattr(bisheng.subprocess, "check_output", fake_check_output)
    for directory in ("original", "relocated"):
        compiler_home = tmp_path / directory
        compiler = compiler_home / "bin" / "bisheng"
        compiler.parent.mkdir(parents=True)
        compiler.touch()
        monkeypatch.setenv("BISHENG_HOME", str(compiler_home))
        # Simulate a fresh process using the same compiler at a different path.
        bisheng.get_bisheng_version.cache_clear()
        assert bisheng.get_bisheng_version() == version
        assert bisheng.get_bisheng_version() == version
        assert queries[-1] == [str(compiler), "--version"]
    assert len(queries) == 2


@pytest.mark.parametrize("output", ["", "\n", "unknown version\nInstalledDir: /opt/bisheng/bin\n"])
def test_bisheng_version_rejects_unrecognized_output(monkeypatch, output):
    monkeypatch.setattr(bisheng, "find_bisheng_path", lambda: "/opt/bisheng/bin/bisheng")
    monkeypatch.setattr(bisheng.subprocess, "check_output", lambda *args, **kwargs: output)
    with pytest.raises(RuntimeError, match="Cannot determine Bisheng version"):
        bisheng.get_bisheng_version()
