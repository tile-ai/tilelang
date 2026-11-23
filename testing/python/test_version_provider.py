from __future__ import annotations

import importlib.util
import runpy
import shutil
import uuid
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _clear_version_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "NO_GIT_VERSION",
        "NO_VERSION_LABEL",
        "TILELANG_BUILD_WHEEL_WITH_DATE",
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
    _make_provider_tree(tmp_path)
    (tmp_path / ".git_commit.txt").write_text("abcdef1234567890\n", encoding="utf-8")

    provider = _load_provider(tmp_path)

    assert provider.dynamic_metadata("version") == "1.2.3+gitabcdef12"


def test_build_date_handles_current_and_older_wheels():
    # Packaging tests run without loading the native TileLang library.
    build_date = runpy.run_path(str(REPO_ROOT / "tilelang/utils/version.py"))["build_date"]
    assert build_date("1.2.3+d20260127.gitabcdef12") == 20260127
    assert build_date("1.2.3+d20260127") == 20260127
    assert build_date("1.2.3+cuda.d20260127.gitabcdef12") == 20260127
    assert build_date("1.2.3+gitabcdef12") is None
