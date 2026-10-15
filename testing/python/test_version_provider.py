from __future__ import annotations

import importlib.util
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _clear_version_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "CUDA_VERSION",
        "NO_GIT_VERSION",
        "NO_TOOLCHAIN_VERSION",
        "NO_VERSION_LABEL",
        "TILELANG_BUILD_WHEEL_WITH_DATE",
        "USE_CUDA",
        "USE_ROCM",
    ):
        monkeypatch.delenv(name, raising=False)


def _load_provider(root: Path):
    spec = importlib.util.spec_from_file_location(
        f"_tilelang_test_version_provider_{uuid.uuid4().hex}",
        root / "version_provider.py",
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_provider_tree(root: Path) -> None:
    shutil.copy(REPO_ROOT / "version_provider.py", root / "version_provider.py")
    (root / "VERSION").write_text("1.2.3\n", encoding="utf-8")


def test_extracted_sdist_uses_plain_version_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _clear_version_env(monkeypatch)
    _make_provider_tree(tmp_path)
    (tmp_path / ".git_commit.txt").write_text("abcdef1234567890\n", encoding="utf-8")

    provider = _load_provider(tmp_path)

    assert provider.dynamic_metadata("version") == "1.2.3"


def test_extracted_sdist_can_opt_into_version_label(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _clear_version_env(monkeypatch)
    monkeypatch.setenv("NO_VERSION_LABEL", "OFF")
    monkeypatch.setenv("NO_TOOLCHAIN_VERSION", "ON")
    _make_provider_tree(tmp_path)
    (tmp_path / ".git_commit.txt").write_text("abcdef1234567890\n", encoding="utf-8")

    provider = _load_provider(tmp_path)

    assert provider.dynamic_metadata("version") == "1.2.3+gitabcdef12"


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required for checkout version test")
def test_git_checkout_uses_version_label_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _clear_version_env(monkeypatch)
    monkeypatch.setenv("NO_TOOLCHAIN_VERSION", "ON")
    _make_provider_tree(tmp_path)

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "TileLang Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "tilelang-test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "VERSION", "version_provider.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        encoding="utf-8",
    ).stdout.strip()

    provider = _load_provider(tmp_path)

    assert provider.dynamic_metadata("version") == f"1.2.3+git{commit[:8]}"
    assert (tmp_path / ".git_commit.txt").read_text(encoding="utf-8") == commit
