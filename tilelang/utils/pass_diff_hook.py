"""Framework-level pass diff hook.

When the ``TILELANG_PASS_DIFF`` environment variable is enabled, this module
monkey-patches ``tvm.ir.transform.Pass.__call__`` to automatically capture
IR before/after every pass and generate diff reports.

Usage (no code changes needed)::

    TILELANG_PASS_DIFF=terminal python3 my_script.py   # colored terminal diff
    TILELANG_PASS_DIFF=html python3 my_script.py       # HTML report
    TILELANG_PASS_DIFF=both python3 my_script.py       # both
    python3 my_script.py                               # off (zero overhead)
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

_original_call = None
_pass_counter = 0
_html_steps: list[dict] = []
_html_path: str | None = None
_mode: str | None = None
_atexit_registered = False

# Preloaded at install time for performance (avoid per-pass imports)
_diff_utils = None


def _get_pass_name(p) -> str:
    """Extract a human-readable name from a pass object."""
    try:
        info = p.info
        name = getattr(info, "name", None)
        if name:
            return str(name)
    except Exception:
        pass
    return type(p).__name__


def _patched_call(self, mod):
    """Replacement for Pass.__call__ that captures before/after IR."""
    global _pass_counter, _html_steps

    _pass_counter += 1
    pass_name = _get_pass_name(self)

    # Use preloaded utilities
    compute_diff = _diff_utils["compute_diff"]
    count_changes = _diff_utils["count_changes"]
    print_colored = _diff_utils["print_colored"]
    print_header = _diff_utils["print_header"]
    generate_html = _diff_utils["generate_html"]
    dim = _diff_utils["dim"]
    reset = _diff_utils["reset"]

    mode = _mode or "terminal"

    # Capture before
    try:
        before_script = mod.script()
    except Exception:
        before_script = "<failed to capture>"

    before_lines = before_script.splitlines()

    # Run the original pass
    result = _original_call(self, mod)

    # Capture after
    try:
        after_script = result.script()
    except Exception:
        after_script = "<failed to capture>"

    after_lines = after_script.splitlines()

    # Compute diff
    diff_lines = compute_diff(before_lines, after_lines, context=3)
    insertions, deletions = count_changes(diff_lines)
    changed = insertions > 0 or deletions > 0

    step = {
        "name": f"Pass {_pass_counter}: {pass_name}",
        "before_script": before_script,
        "after_script": after_script,
        "diff_lines": diff_lines,
        "insertions": insertions,
        "deletions": deletions,
        "changed": changed,
    }

    # Terminal output
    if mode in ("terminal", "both"):
        print_header(_pass_counter, "∞", pass_name)
        print_colored(diff_lines)
        if changed:
            print(f"\n  {dim}>>> +{insertions} insertion(s), -{deletions} deletion(s){reset}")
        else:
            print(f"\n  {dim}>>> (no changes){reset}")

    # Accumulate for HTML
    if mode in ("html", "both"):
        _html_steps.append(step)
        # Flush HTML on every pass so partial results are available
        try:
            generate_html(_html_steps, _html_path)
        except Exception as e:
            print(f"\n[pass_diff] HTML flush error: {e}")

    return result


def install_pass_diff_hook() -> None:
    """Install the pass diff hook if TILELANG_PASS_DIFF is enabled.

    Called automatically from ``tilelang/__init__.py``. Safe to call
    multiple times — only the first call has effect.
    """
    global _original_call, _html_path, _diff_utils, _mode, _atexit_registered

    from tilelang.env import env

    mode = env.get_pass_diff_mode()
    if mode is None:
        return  # disabled — zero overhead from here on

    # Already installed?
    if _original_call is not None:
        return

    _mode = mode

    # Preload diff utilities once at install time
    from tilelang.utils.pass_diff import (
        _compute_diff,
        _count_changes,
        _generate_html,
        _print_colored_diff,
        _print_step_header,
        _DIM,
        _RESET,
    )

    _diff_utils = {
        "compute_diff": _compute_diff,
        "count_changes": _count_changes,
        "generate_html": _generate_html,
        "print_colored": _print_colored_diff,
        "print_header": _print_step_header,
        "dim": _DIM,
        "reset": _RESET,
    }

    from tvm.ir.transform import Pass

    _original_call = Pass.__call__
    Pass.__call__ = _patched_call  # type: ignore[assignment]

    # Set up HTML output path if needed
    if mode in ("html", "both"):
        diff_dir = str(env.TILELANG_PASS_DIFF_OUTPUT)
        os.makedirs(diff_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        _html_path = os.path.join(diff_dir, f"pass_diff_{timestamp}.html")

    import atexit

    def _flush_html():
        if _html_path and _html_steps:
            generate_html = _diff_utils["generate_html"]
            try:
                generate_html(_html_steps, _html_path)
                print(f"\n[pass_diff] HTML report: {_html_path}")
            except Exception as e:
                print(f"\n[pass_diff] Error generating HTML: {e}")

    if not _atexit_registered:
        atexit.register(_flush_html)
        _atexit_registered = True


def uninstall_pass_diff_hook() -> None:
    """Remove the pass diff hook and restore original Pass.__call__."""
    global _original_call, _html_steps, _pass_counter, _diff_utils, _html_path, _mode

    if _original_call is None:
        return

    from tvm.ir.transform import Pass

    Pass.__call__ = _original_call  # type: ignore[assignment]
    _original_call = None
    _html_steps.clear()
    _pass_counter = 0
    _diff_utils = None
    _html_path = None
    _mode = None
