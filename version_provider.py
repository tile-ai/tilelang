from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent

base_version = (ROOT / 'VERSION').read_text().strip()


def _read_cmake_bool(i: str | None, default=False):
    if i is None:
        return default
    return i.lower() not in ('0', 'false', 'off', 'no', 'n', '')


def get_git_commit_id() -> str | None:
    """Get the current git commit hash by running git in the current file's directory."""

    r = subprocess.run(['git', 'rev-parse', 'HEAD'],
                       cwd=ROOT,
                       capture_output=True,
                       encoding='utf-8')
    if r.returncode == 0:
        return r.stdout.strip()
    else:
        return 'unknown'


def dynamic_metadata(
    field: str,
    settings: dict[str, object] | None = None,
) -> str:
    assert field == 'version'

    version = base_version

    if not _read_cmake_bool(os.environ.get('NO_VERSION_LABEL')):
        exts = []
        backend = None
        if _read_cmake_bool(os.environ.get('NO_TOOLCHAIN_VERSION')):
            pass
        elif platform.system() == 'Darwin':
            # only on macosx_11_0_arm64, not necessary
            # backend = 'metal'
            pass
        elif _read_cmake_bool(os.environ.get('USE_ROCM', '')):
            backend = 'rocm'
        elif 'USE_CUDA' in os.environ and not _read_cmake_bool(os.environ.get('USE_CUDA')):
            backend = 'cpu'
        else:  # cuda
            # Read nvcc version from env.
            # This is not exactly how it should be,
            # but works for now if building in a nvidia/cuda image.
            if cuda_version := os.environ.get('CUDA_VERSION'):
                major, minor, *_ = cuda_version.split('.')
                backend = f'cu{major}{minor}'
            else:
                backend = 'cuda'
        if backend:
            exts.append(backend)

        if _read_cmake_bool(os.environ.get('NO_GIT_VERSION')):
            pass
        elif git_hash := get_git_commit_id():
            exts.append(f'git{git_hash[:8]}')

        if exts:
            version += '+' + '.'.join(exts)

    return version


__all__ = ["dynamic_metadata"]
