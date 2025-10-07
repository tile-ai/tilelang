from __future__ import annotations

import os
import subprocess
from typing import Optional

base_version = '0.1.6'


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
    try:
        import torch

        if torch.cuda.is_available():
            backend = 'cuda' if torch.version.hip is None else 'rocm'
        elif torch.mps.is_availabe():
            backend = 'metal'
        else:
            backend = 'cpu'

        exts.append(backend)
    except Exception:
        pass

    if git_hash := get_git_commit_id():
        exts.append(f'git{git_hash[:8]}')

    version = base_version

    if exts:
        version += '+' + '.'.join(exts)

    return version


__all__ = ["dynamic_metadata"]
