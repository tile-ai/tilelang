from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from functools import lru_cache

ROOT = Path(__file__).parent

base_version = (ROOT / "VERSION").read_text().strip()
# When installing a sdist,
# the installed version needs to match the sdist version,
# so pip will complain when we install `tilelang-0.1.6.post2+gitxxxx.tar.gz`.
# To workaround that, when building sdist,
# we do not add version label and use a file to store the git hash instead.
git_pin = ROOT / ".git_commit.txt"


def _read_cmake_bool(i: str | None, default=False):
    if i is None:
        return default
    return i.lower() not in ("0", "false", "off", "no", "n", "")


@lru_cache(maxsize=1)
def get_git_commit_id() -> str | None:
    """Get the current git commit hash by running git in the current file's directory."""

    if (ROOT / ".git").exists():
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, encoding="utf-8")
        if r.returncode == 0:
            _git = r.stdout.strip()
            git_pin.write_text(_git)
            return _git

    if git_pin.exists():
        return git_pin.read_text().strip()

    return None


def dynamic_metadata(field: str, settings: dict[str, object] | None = None) -> str:
    assert field == "version"

    version = base_version

    # Pin the source commit for sdists, even when the public version omits it.
    get_git_commit_id()

    no_version_label = _read_cmake_bool(
        os.environ.get("NO_VERSION_LABEL"),
        default=not (ROOT / ".git").exists(),
    )
    if not no_version_label:
        exts = []

        # Add build date if TILELANG_BUILD_WHEEL_WITH_DATE is set
        if _read_cmake_bool(os.environ.get("TILELANG_BUILD_WHEEL_WITH_DATE")):
            exts.append(f"d{time.strftime('%Y%m%d')}")

        if _read_cmake_bool(os.environ.get("NO_GIT_VERSION")):
            pass
        elif git_hash := get_git_commit_id():
            exts.append(f"git{git_hash[:8]}")
        else:
            exts.append("gitunknown")

        if exts:
            version += "+" + ".".join(exts)

    return version


__all__ = ["dynamic_metadata"]
