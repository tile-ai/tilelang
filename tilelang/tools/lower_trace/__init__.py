"""IR Lower Trace — zero-intrusion debug tool for visualizing compilation passes.

Generates a self-contained HTML page and/or terminal diff showing all passes in the
compilation pipeline with side-by-side IR diff for each pass.

Usage::

    TL_LOWER_TRACE=1 python my_kernel.py        # HTML report
    TL_LOWER_TRACE=terminal python my_kernel.py  # terminal diff only
    TL_LOWER_TRACE=both python my_kernel.py      # both terminal and HTML

Programmatic API::

    from tilelang.tools.lower_trace import lower_trace

    lower_trace(func, my_pass, mode="terminal")
    lower_trace(func, [pass_a, pass_b], mode="both", html_path="diff.html")
"""

from __future__ import annotations

from .core import (
    enable,
    disable,
    reset,
    LowerRecord,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_CODEGEN,
    _get_pass_display_name,
)

__all__ = [
    "STATUS_CODEGEN",
    "STATUS_COMPLETED",
    "STATUS_FAILED",
    "STATUS_SKIPPED",
    "LowerRecord",
    "disable",
    "enable",
    "lower_trace",
    "reset",
]


def lower_trace(
    func_or_mod,
    passes,
    *,
    mode: str = "terminal",
    context: int = 3,
    html_path: str = "lower_trace_report.html",
) -> list[dict]:
    """Compare IR before and after each pass in a chain.

    Parameters
    ----------
    func_or_mod : PrimFunc or IRModule
        The starting IR.
    passes : Pass or list[Pass] or list[tuple[str, Pass]]
        A single pass, a list of passes, or a list of (name, pass) pairs.
    mode : {"terminal", "html", "both"}
        Output mode.
    context : int
        Number of context lines in the unified diff (default 3).
    html_path : str
        Output path for HTML report (default ``lower_trace_report.html``).

    Returns
    -------
    list[dict]
        One entry per pass step, each containing:
        ``name``, ``before_script``, ``after_script``, ``diff_lines``,
        ``insertions``, ``deletions``, ``changed``.
    """
    from tilelang import tvm
    from .diff import unified_diff

    mode = str(mode).strip().lower()
    if mode not in ("terminal", "html", "both"):
        raise ValueError(f"mode must be one of 'terminal', 'html', 'both', got {mode!r}")

    if isinstance(func_or_mod, tvm.IRModule):
        mod = func_or_mod
    else:
        mod = tvm.IRModule({"main": func_or_mod})

    if not isinstance(passes, (list, tuple)):
        passes = [passes]

    named_passes: list[tuple[str, object]] = []
    for p in passes:
        if isinstance(p, (list, tuple)) and len(p) == 2:
            named_passes.append((str(p[0]), p[1]))
        else:
            named_passes.append((_get_pass_display_name(p), p))

    results: list[dict] = []

    for step_idx, (name, p) in enumerate(named_passes, 1):
        before_script = mod.script()
        mod = p(mod)
        after_script = mod.script()

        diff_text = unified_diff(
            before_script,
            after_script,
            before_label=f"step {step_idx} before",
            after_label=f"step {step_idx} after",
            context=context,
            color=False,
        )
        diff_lines = diff_text.splitlines() if diff_text else []

        insertions = sum(1 for d in diff_lines if d.startswith("+") and not d.startswith("+++"))
        deletions = sum(1 for d in diff_lines if d.startswith("-") and not d.startswith("---"))
        changed = insertions > 0 or deletions > 0

        step_result = {
            "name": name,
            "before_script": before_script,
            "after_script": after_script,
            "diff_lines": diff_lines,
            "insertions": insertions,
            "deletions": deletions,
            "changed": changed,
        }
        results.append(step_result)

        if mode in ("terminal", "both"):
            header = f"\n{'=' * 60}\n  Pass {step_idx}: {name}\n{'=' * 60}\n"
            print(header)
            if changed:
                colored = unified_diff(
                    before_script,
                    after_script,
                    before_label=f"step {step_idx} before",
                    after_label=f"step {step_idx} after",
                    context=context,
                    color=True,
                )
                print(colored, end="")
                print(f"\n  >>> +{insertions} insertion(s), -{deletions} deletion(s)")
            else:
                print("  (no changes)")

    if mode in ("html", "both"):
        from .html import generate_html

        records = []
        for i, r in enumerate(results):
            changed = r["changed"]
            records.append(
                LowerRecord(
                    phase="lower_trace",
                    name=r["name"],
                    index=i,
                    before_text=r["before_script"],
                    after_text=r["after_script"],
                    changed=changed,
                    add_lines=r["insertions"],
                    del_lines=r["deletions"],
                )
            )
        generate_html(records, html_path)
        print(f"\nHTML report written to: {html_path}")

    return results
