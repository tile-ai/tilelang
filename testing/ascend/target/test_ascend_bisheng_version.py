"""Compiler identity used by the Ascend device binary cache."""

import subprocess

import pytest

from tilelang.contrib import bisheng
from tilelang.env import env


@pytest.fixture(autouse=True)
def reset_bisheng_version_cache(monkeypatch):
    bisheng.get_bisheng_version.cache_clear()
    monkeypatch.setattr(env, "TILELANG_COMPILE_TIMEOUT_SECONDS", "")
    yield
    bisheng.get_bisheng_version.cache_clear()


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


def test_bisheng_version_query_failure_is_not_cached(monkeypatch):
    monkeypatch.setattr(bisheng, "find_bisheng_path", lambda: "/opt/bisheng/bin/bisheng")
    calls = []

    def fail_query(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command, "version query failed")

    monkeypatch.setattr(bisheng.subprocess, "check_output", fail_query)
    for _ in range(2):
        with pytest.raises(subprocess.CalledProcessError):
            bisheng.get_bisheng_version()
    assert len(calls) == 2


def test_bisheng_version_query_timeout(monkeypatch):
    monkeypatch.setattr(bisheng, "find_bisheng_path", lambda: "/opt/bisheng/bin/bisheng")
    monkeypatch.setattr(env, "TILELANG_COMPILE_TIMEOUT_SECONDS", "2.5")

    def timeout_query(command, **kwargs):
        assert kwargs["timeout"] == 2.5
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(bisheng.subprocess, "check_output", timeout_query)
    with pytest.raises(RuntimeError, match="Bisheng version query timed out after 2.5 seconds"):
        bisheng.get_bisheng_version()
