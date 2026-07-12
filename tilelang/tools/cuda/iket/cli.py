"""Host-side output and IKET CLI helpers."""

from __future__ import annotations

import os
import shlex
from collections.abc import Sequence
from pathlib import Path


_output_dir: Path | None = None


def set_output_dir(path: str | os.PathLike[str] | None) -> Path | None:
    """Set the process-local IKET export directory used by helper APIs."""
    global _output_dir
    if path is None:
        _output_dir = None
        os.environ.pop("TL_IKET_OUTPUT_DIR", None)
        return None

    _output_dir = Path(path).expanduser().absolute()
    _output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TL_IKET_OUTPUT_DIR"] = str(_output_dir)
    return _output_dir


def output_dir(default: str | os.PathLike[str] | None = None) -> Path | None:
    """Return the configured IKET export directory, if any."""
    if _output_dir is not None:
        return _output_dir
    env_dir = os.environ.get("TL_IKET_OUTPUT_DIR")
    if env_dir:
        return Path(env_dir).expanduser().absolute()
    if default is None:
        return None
    return Path(default).expanduser().absolute()


def output_path(name: str, *, directory: str | os.PathLike[str] | None = None) -> Path:
    """Return a path under the configured IKET export directory."""
    base = output_dir(directory)
    if base is None:
        raise RuntimeError("IKET output directory is not configured")
    base.mkdir(parents=True, exist_ok=True)
    return base / name


def trace_files(*, directory: str | os.PathLike[str] | None = None) -> list[Path]:
    """Return generated IKET JSON traces in size-descending order."""
    base = output_dir(directory)
    if base is None or not base.exists():
        return []
    return sorted(base.glob("*.trace.json"), key=lambda path: path.stat().st_size, reverse=True)


def profile_command(
    command: str | Sequence[str],
    *,
    directory: str | os.PathLike[str],
    postprocess: str = "all",
    clobber: bool = True,
) -> str:
    """Build an IKET CLI command that exports traces to ``directory``."""
    parts = ["python", "-m", "iket.cli.main", "--output-dir", str(Path(directory).expanduser())]
    if clobber:
        parts.append("--clobber")
    parts.extend(["profile", "--postprocess", postprocess, "--"])
    command_prefix = " ".join(shlex.quote(str(item)) for item in parts)
    if isinstance(command, str):
        return f"{command_prefix} {command}"
    return " ".join([command_prefix, *(shlex.quote(str(item)) for item in command)])


def _snapshot_output_dir() -> tuple[Path | None, str | None]:
    return _output_dir, os.environ.get("TL_IKET_OUTPUT_DIR")


def _restore_output_dir(previous_output_dir: Path | None, previous_env_output_dir: str | None) -> None:
    global _output_dir
    _output_dir = previous_output_dir
    if previous_env_output_dir is None:
        os.environ.pop("TL_IKET_OUTPUT_DIR", None)
    else:
        os.environ["TL_IKET_OUTPUT_DIR"] = previous_env_output_dir
