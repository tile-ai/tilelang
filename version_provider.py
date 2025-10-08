from __future__ import annotations

import os
import platform
import subprocess
from typing import Optional
from pathlib import Path

ROOT = Path(__file__).parent

base_version = (ROOT / 'VERSION').read_text().strip()


def _read_cmake_bool(i: str | None):
    return i and i.lower() not in ('0', 'false', 'off', 'no', 'n', '')


def get_git_commit_id() -> Optional[str]:
    """Get the current git commit hash by running git in the current file's directory."""
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       cwd=os.path.dirname(os.path.abspath(__file__)),
                                       stderr=subprocess.DEVNULL,
                                       encoding='utf-8').strip()
    # FileNotFoundError is raised when git is not installed
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def dynamic_metadata(
    field: str,
    settings: dict[str, object] | None = None,
) -> str:
    assert field == 'version'

    exts = []
    backend = None
    if _read_cmake_bool(os.environ.get('NO_TOOLCHAIN_VERSION')):
        pass
    elif platform.system() == 'Darwin':
        backend = 'metal'
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

    if git_hash := get_git_commit_id():
        exts.append(f'git{git_hash[:8]}')

    version = base_version

    if exts:
        version += '+' + '.'.join(exts)

    return version


__all__ = ["dynamic_metadata"]
