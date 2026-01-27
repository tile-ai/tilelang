"""Version utilities for tilelang."""

from __future__ import annotations

import re


def build_date(version_str: str | None = None) -> int | None:
    """Extract build date (YYYYMMDD) from version string.

    Args:
        version_str: Version string like "0.1.7.post3+cuda.d20260127.gita17230e4".
                     If None, uses tilelang.__version__.

    Returns:
        Build date as integer (e.g., 20260127), or None if not found.

    Example:
        >>> import tilelang
        >>> if tilelang.build_date() >= 20260127:
        ...     print("Version meets requirement")
    """
    if version_str is None:
        import tilelang

        version_str = tilelang.__version__

    match = re.search(r"\.d(\d{8})\.", version_str)
    if match:
        return int(match.group(1))
    return None
