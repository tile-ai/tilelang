"""IR pass diff tool — compare TIR before/after each pass in a chain.

Usage
-----
::

    from tilelang.utils.pass_diff import pass_diff

    # Single pass, terminal colored diff
    pass_diff(func, tilelang.transform.ThreadSync("shared"))

    # Pass chain with named steps
    pass_diff(func, [
        ("AnnotateDeviceRegions", tvm.tirx.transform.AnnotateDeviceRegions()),
        ("SplitHostDevice",       tvm.tirx.transform.SplitHostDevice()),
        ("ThreadSync",            tilelang.transform.ThreadSync("shared")),
    ])

    # Generate HTML report
    pass_diff(func, passes, mode="html")

    # Both terminal + HTML
    pass_diff(func, passes, mode="both")
"""

from __future__ import annotations

import difflib
import html as html_module
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tilelang.tvm import IRModule

# ANSI color codes
_RED = "\033[91m"
_GREEN = "\033[92m"
_CYAN = "\033[96m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Core diff logic
# ---------------------------------------------------------------------------


def _to_mod(func_or_mod) -> IRModule:
    """Convert a PrimFunc or IRModule to IRModule."""
    from tilelang import tvm

    if isinstance(func_or_mod, tvm.IRModule):
        return func_or_mod
    return tvm.IRModule({"main": func_or_mod})


def _get_script(mod: IRModule) -> str:
    """Get the TIR script text from an IRModule."""
    return mod.script()


def _compute_diff(
    before_lines: list[str],
    after_lines: list[str],
    context: int,
    before_label: str = "before",
    after_label: str = "after",
) -> list[str]:
    """Compute unified diff between two line lists."""
    return list(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=before_label,
            tofile=after_label,
            lineterm="",
            n=context,
        )
    )


def _count_changes(diff_lines: list[str]) -> tuple[int, int]:
    """Count insertions and deletions from a unified diff."""
    insertions = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
    deletions = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))
    return insertions, deletions


# ---------------------------------------------------------------------------
# Terminal colored output
# ---------------------------------------------------------------------------


def _print_colored_diff(diff_lines: list[str]) -> None:
    """Print unified diff with ANSI colors to stdout."""
    if not diff_lines:
        print(f"  {_DIM}(no changes){_RESET}")
        return

    for line in diff_lines:
        if line.startswith("+++") or line.startswith("---"):
            print(f"{_BOLD}{line}{_RESET}")
        elif line.startswith("@@"):
            print(f"{_CYAN}{line}{_RESET}")
        elif line.startswith("+"):
            print(f"{_GREEN}{line}{_RESET}")
        elif line.startswith("-"):
            print(f"{_RED}{line}{_RESET}")
        else:
            print(line)


def _print_step_header(step_idx: int, total: int | str, name: str) -> None:
    """Print a pass step header."""
    width = 60
    print()
    print(f"{_BOLD}{'=' * width}{_RESET}")
    print(f"{_BOLD}  Pass {step_idx}/{total}: {name}{_RESET}")
    print(f"{_BOLD}{'=' * width}{_RESET}")


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pass Diff Report</title>
<style>
  :root {{
    --bg-primary: #1e1e2e;
    --bg-secondary: #181825;
    --bg-surface: #11111b;
    --bg-hover: #313244;
    --border: #45475a;
    --border-light: #585b70;
    --text-primary: #cdd6f4;
    --text-secondary: #a6adc8;
    --text-muted: #6c7086;
    --accent: #89b4fa;
    --accent-dim: #1e3a5f;
    --green: #a6e3a1;
    --green-bg: #1a2e1a;
    --green-inline: #2d4a2d;
    --red: #f38ba8;
    --red-bg: #2e1a1a;
    --red-inline: #4a2d2d;
    --yellow: #f9e2af;
    --peach: #fab387;
    --mauve: #cba6f7;
  }}
  [data-theme="light"] {{
    --bg-primary: #eff1f5;
    --bg-secondary: #e6e9ef;
    --bg-surface: #dce0e8;
    --bg-hover: #ccd0da;
    --border: #bcc0cc;
    --border-light: #acb0be;
    --text-primary: #4c4f69;
    --text-secondary: #5c5f77;
    --text-muted: #8c8fa1;
    --accent: #1e66f5;
    --accent-dim: #ccd7f5;
    --green: #40a02b;
    --green-bg: #d4edda;
    --green-inline: #b8dbbf;
    --red: #d20f39;
    --red-bg: #f8d7da;
    --red-inline: #f0b8be;
    --yellow: #df8e1d;
    --peach: #fe640b;
    --mauve: #8839ef;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'SF Mono', Consolas, monospace;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 13px;
    line-height: 1.6;
    tab-size: 4;
  }}

  /* ── Toolbar ── */
  .toolbar {{
    position: sticky; top: 0; z-index: 100;
    display: flex; align-items: center; gap: 16px;
    padding: 10px 20px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
    backdrop-filter: blur(8px);
  }}
  .toolbar-title {{
    font-size: 14px; font-weight: 700;
    color: var(--accent);
    letter-spacing: 0.5px;
  }}
  .toolbar-stats {{
    display: flex; gap: 12px; font-size: 12px;
    color: var(--text-secondary);
  }}
  .toolbar-stats .stat {{
    display: flex; align-items: center; gap: 4px;
    padding: 2px 8px;
    border-radius: 4px;
    background: var(--bg-surface);
    border: 1px solid var(--border);
  }}
  .stat-num {{ font-weight: 700; }}
  .stat-add {{ color: var(--green); }}
  .stat-del {{ color: var(--red); }}
  .stat-pass {{ color: var(--accent); }}
  .toolbar-actions {{
    margin-left: auto;
    display: flex; gap: 8px;
  }}
  .toolbar-btn {{
    padding: 4px 12px;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--bg-surface);
    color: var(--text-secondary);
    cursor: pointer;
    font-family: inherit;
    font-size: 11px;
    transition: all 0.15s;
  }}
  .toolbar-btn:hover {{
    background: var(--bg-hover);
    color: var(--text-primary);
    border-color: var(--border-light);
  }}

  /* ── Pass Section ── */
  .pass-section {{
    margin: 2px 0;
    border-bottom: 1px solid var(--border);
  }}
  .pass-header {{
    display: flex; align-items: center; gap: 10px;
    padding: 8px 20px;
    background: var(--bg-secondary);
    cursor: pointer;
    user-select: none;
    transition: background 0.15s;
  }}
  .pass-header:hover {{ background: var(--bg-hover); }}
  .pass-chevron {{
    color: var(--text-muted);
    font-size: 10px;
    transition: transform 0.2s;
    width: 16px; text-align: center;
  }}
  .pass-section.open .pass-chevron {{ transform: rotate(90deg); }}
  .pass-name {{
    font-weight: 600; font-size: 13px;
    color: var(--text-primary);
  }}
  .pass-badge {{
    font-size: 10px; font-weight: 700;
    padding: 1px 8px;
    border-radius: 3px;
    letter-spacing: 0.3px;
  }}
  .pass-badge.changed {{
    background: var(--peach);
    color: var(--bg-surface);
  }}
  .pass-badge.unchanged {{
    background: var(--border);
    color: var(--text-muted);
  }}
  .pass-stats {{
    font-size: 11px; color: var(--text-muted);
    margin-left: auto;
  }}

  /* ── Diff Body ── */
  .pass-body {{
    display: none;
    border-top: 1px solid var(--border);
  }}
  .pass-section.open .pass-body {{ display: block; }}

  .diff-wrapper {{
    overflow-x: auto;
    background: var(--bg-surface);
  }}
  .diff-table {{
    width: 100%;
    border-collapse: collapse;
  }}
  .col-ln {{ width: 48px; }}
  .col-code {{ width: auto; }}
  .col-gutter {{ width: 1px; }}
  .diff-table td {{
    vertical-align: top;
    padding: 0;
    white-space: pre;
    font-family: inherit;
    font-size: 12.5px;
    line-height: 1.65;
  }}

  /* Line number gutter */
  .ln {{
    width: 48px; min-width: 48px;
    text-align: right;
    padding: 0 8px 0 4px;
    color: var(--text-muted);
    user-select: none;
    border-right: 1px solid var(--border);
    background: var(--bg-secondary);
    font-size: 11px;
  }}

  /* Code content */
  .code {{
    padding: 0 12px;
    width: 50%;
    overflow-x: auto;
  }}

  /* Gutter between panels */
  .gutter {{
    width: 1px !important;
    min-width: 0;
    max-width: 1px;
    padding: 0;
    background: var(--border);
  }}

  /* Row colors */
  tr.ctx .code {{ color: var(--text-secondary); }}
  tr.add .code {{ background: var(--green-bg); color: var(--green); }}
  tr.add .ln {{ background: var(--green-bg); color: var(--green); }}
  tr.del .code {{ background: var(--red-bg); color: var(--red); }}
  tr.del .ln {{ background: var(--red-bg); color: var(--red); }}

  /* Pair row: left=del, right=add */
  tr.pair .del-ln {{ background: var(--red-bg); color: var(--red); }}
  tr.pair .del-code {{ background: var(--red-bg); color: var(--red); }}
  tr.pair .add-ln {{ background: var(--green-bg); color: var(--green); }}
  tr.pair .add-code {{ background: var(--green-bg); color: var(--green); }}

  /* Empty cells for unpaired lines */
  .empty-ln {{ background: var(--bg-surface); }}
  .empty-code {{ background: var(--bg-surface); }}
  tr.hunk td {{
    background: var(--accent-dim);
    color: var(--accent);
    padding: 2px 12px;
    font-size: 11px;
    font-style: italic;
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
  }}
  /* Inline word highlight */
  .hl {{ font-weight: 700; text-decoration: underline; text-underline-offset: 2px; }}
  tr.add .hl {{ background: var(--green-inline); }}
  tr.del .hl {{ background: var(--red-inline); }}

  /* No change message */
  .no-change {{
    padding: 24px;
    text-align: center;
    color: var(--text-muted);
    font-style: italic;
    font-size: 12px;
    background: var(--bg-surface);
  }}

  /* ── Panel toolbar (copy buttons) ── */
  .panel-bar {{
    display: flex; align-items: center;
    padding: 4px 12px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
    font-size: 11px;
    color: var(--text-muted);
  }}
  .panel-bar .panel-label {{
    font-weight: 600;
    color: var(--text-secondary);
  }}
  .copy-btn {{
    margin-left: auto;
    padding: 2px 10px;
    border: 1px solid var(--border);
    border-radius: 3px;
    background: var(--bg-surface);
    color: var(--text-muted);
    cursor: pointer;
    font-family: inherit;
    font-size: 10px;
    transition: all 0.15s;
  }}
  .copy-btn:hover {{
    background: var(--bg-hover);
    color: var(--text-primary);
    border-color: var(--border-light);
  }}
  .copy-btn.copied {{
    color: var(--green);
    border-color: var(--green);
  }}

  /* ── Full content view for unchanged passes ── */
  .full-content {{
    overflow-x: auto;
    background: var(--bg-surface);
  }}
  .full-content pre {{
    margin: 0;
    padding: 12px 16px;
    font-family: inherit;
    font-size: 12.5px;
    line-height: 1.65;
    color: var(--text-secondary);
    white-space: pre;
  }}

  /* ── Expand context (native <details>) ── */
  .expand-details {{
    border: 1px solid var(--border);
    border-radius: 4px;
    margin: 4px 0;
    background: var(--bg-secondary);
  }}
  .expand-summary {{
    padding: 4px 16px;
    cursor: pointer;
    color: var(--accent);
    font-size: 11px;
    font-weight: 600;
    user-select: none;
    list-style: none;
    text-align: center;
  }}
  .expand-summary::-webkit-details-marker {{ display: none; }}
  .expand-summary::before {{
    content: "▶ ";
    font-size: 9px;
    display: inline-block;
    transition: transform 0.15s;
    margin-right: 4px;
  }}
  details[open] > .expand-summary::before {{
    transform: rotate(90deg);
  }}
  .expand-summary:hover {{
    color: var(--text-primary);
    background: var(--bg-hover);
  }}
  .expand-details .diff-table {{
    border-top: 1px solid var(--border);
  }}

  /* Timestamp footer */
  .footer {{
    padding: 8px 20px;
    text-align: right;
    font-size: 11px;
    color: var(--text-muted);
    background: var(--bg-secondary);
    border-top: 1px solid var(--border);
  }}
</style>
</head>
<body>

<div class="toolbar">
  <span class="toolbar-title">⚡ Pass Diff Report</span>
  <div class="toolbar-stats">
    <span class="stat"><span class="stat-num stat-pass">{total_passes}</span> passes</span>
    <span class="stat"><span class="stat-num stat-pass">{changed_passes}</span> changed</span>
    <span class="stat"><span class="stat-num stat-add">+{total_add}</span></span>
    <span class="stat"><span class="stat-num stat-del">-{total_del}</span></span>
  </div>
  <div class="toolbar-actions">
    <button class="toolbar-btn" id="theme-btn">☀ Light</button>
    <button class="toolbar-btn" id="expand-all-btn">▼ Expand All</button>
    <button class="toolbar-btn" id="collapse-all-btn">▲ Collapse All</button>
  </div>
</div>

{steps}

<div class="footer">Generated at {timestamp}</div>

<script>
// ── Pass section toggle (collapse/expand) ──
document.querySelectorAll('.pass-header').forEach(h => {{
  h.addEventListener('click', () => h.parentElement.classList.toggle('open'));
}});

// ── Toolbar: Expand All / Collapse All ──
document.getElementById('expand-all-btn').addEventListener('click', () => {{
  document.querySelectorAll('.pass-section').forEach(s => s.classList.add('open'));
}});
document.getElementById('collapse-all-btn').addEventListener('click', () => {{
  document.querySelectorAll('.pass-section').forEach(s => s.classList.remove('open'));
}});

// ── Toolbar: Theme toggle ──
document.getElementById('theme-btn').addEventListener('click', () => {{
  const html = document.documentElement;
  const btn = document.getElementById('theme-btn');
  if (html.getAttribute('data-theme') === 'light') {{
    html.removeAttribute('data-theme');
    btn.textContent = '☀ Light';
    localStorage.setItem('pass-diff-theme', 'dark');
  }} else {{
    html.setAttribute('data-theme', 'light');
    btn.textContent = '🌙 Dark';
    localStorage.setItem('pass-diff-theme', 'light');
  }}
}});

// ── Copy buttons ──
document.querySelectorAll('.copy-btn[data-copy-target]').forEach(btn => {{
  btn.addEventListener('click', () => {{
    const el = document.getElementById(btn.getAttribute('data-copy-target'));
    if (!el) return;
    navigator.clipboard.writeText(el.textContent).then(() => {{
      const orig = btn.textContent;
      btn.textContent = '✓ Copied';
      btn.classList.add('copied');
      setTimeout(() => {{ btn.textContent = orig; btn.classList.remove('copied'); }}, 1500);
    }});
  }});
}});

// ── Expand details: toggle summary text ──
document.querySelectorAll('.expand-details').forEach(details => {{
  details.addEventListener('toggle', () => {{
    const summary = details.querySelector('.expand-summary');
    if (!summary) return;
    const count = summary.getAttribute('data-count');
    const pos = summary.getAttribute('data-pos');
    const verb = details.open ? 'Hide' : 'Show';
    summary.textContent = `⋯ ${{verb}} ${{count}} lines ${{pos}}`;
  }});
}});

// ── Restore saved theme ──
if (localStorage.getItem('pass-diff-theme') === 'light') {{
  document.documentElement.setAttribute('data-theme', 'light');
  document.getElementById('theme-btn').textContent = '🌙 Dark';
}}
</script>
</body>
</html>
"""


def _parse_diff_into_rows(
    diff_lines: list[str],
    before_lines: list[str] | None = None,
    after_lines: list[str] | None = None,
    max_expand: int = 20,
) -> tuple[list[dict], list[dict]]:
    """Parse unified diff lines into aligned left/right rows.

    When ``before_lines`` / ``after_lines`` are provided, expand sections
    are collected (up to ``max_expand`` lines each direction) for rendering
    as native ``<details>`` elements outside the diff table.

    Returns
    -------
    tuple[list[dict], list[dict]]
        (visible_rows, expand_sections).  Expand sections have keys:
        position ('before'|'after'), hunk_idx, direction, count, lines.
    """
    import re

    # ── Phase 1: parse visible diff rows ──
    visible_rows = []
    i = 0
    left_num = 0
    right_num = 0
    # Track hunk boundaries: list of (before_start, before_end, after_start, after_end)
    hunk_ranges: list[tuple[int, int, int, int]] = []
    current_hunk_start_left = 0
    current_hunk_start_right = 0

    while i < len(diff_lines):
        line = diff_lines[i]

        if line.startswith("---") or line.startswith("+++"):
            i += 1
            continue

        if line.startswith("@@"):
            # Save previous hunk range
            if hunk_ranges or visible_rows:
                hunk_ranges.append(
                    (
                        current_hunk_start_left,
                        left_num,
                        current_hunk_start_right,
                        right_num,
                    )
                )
            m = re.match(r"@@ -(\d+)", line)
            if m:
                left_num = int(m.group(1)) - 1
                current_hunk_start_left = left_num
            m2 = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)", line)
            if m2:
                right_num = int(m2.group(1)) - 1
                current_hunk_start_right = right_num
            visible_rows.append({"type": "hunk", "left": line, "right": "", "left_num": None, "right_num": None})
            i += 1
            continue

        if line.startswith("-"):
            if i + 1 < len(diff_lines) and diff_lines[i + 1].startswith("+"):
                left_num += 1
                right_num += 1
                visible_rows.append(
                    {
                        "type": "pair",
                        "left": line[1:],
                        "right": diff_lines[i + 1][1:],
                        "left_num": left_num,
                        "right_num": right_num,
                    }
                )
                i += 2
                continue
            else:
                left_num += 1
                visible_rows.append({"type": "del", "left": line[1:], "right": "", "left_num": left_num, "right_num": None})
                i += 1
                continue

        if line.startswith("+"):
            right_num += 1
            visible_rows.append({"type": "add", "left": "", "right": line[1:], "left_num": None, "right_num": right_num})
            i += 1
            continue

        # Context line
        left_num += 1
        right_num += 1
        visible_rows.append(
            {
                "type": "ctx",
                "left": line[1:] if len(line) > 0 else "",
                "right": line[1:] if len(line) > 0 else "",
                "left_num": left_num,
                "right_num": right_num,
            }
        )
        i += 1

    # Save last hunk range
    if current_hunk_start_left or visible_rows:
        hunk_ranges.append(
            (
                current_hunk_start_left,
                left_num,
                current_hunk_start_right,
                right_num,
            )
        )

    # ── Phase 2: collect expand sections if full scripts provided ──
    if before_lines is None or after_lines is None or not hunk_ranges:
        return visible_rows, []

    expand_sections: list[dict] = []
    hunk_idx = 0

    for row in visible_rows:
        if row["type"] == "hunk":
            if hunk_idx < len(hunk_ranges):
                bl_start, bl_end, al_start, al_end = hunk_ranges[hunk_idx]

                # ── Expand UP: lines before this hunk ──
                if hunk_idx > 0:
                    prev_bl_end = hunk_ranges[hunk_idx - 1][1]
                    prev_al_end = hunk_ranges[hunk_idx - 1][3]
                else:
                    prev_bl_end = 0
                    prev_al_end = 0

                up_start_left = max(prev_bl_end, bl_start - max_expand)
                up_start_right = max(prev_al_end, al_start - max_expand)
                up_count_left = bl_start - up_start_left
                up_count_right = al_start - up_start_right

                if up_count_left > 0 or up_count_right > 0:
                    max_up = max(up_count_left, up_count_right)
                    lines = []
                    for j in range(max_up):
                        ln_left = up_start_left + j + 1 if j < up_count_left else None
                        ln_right = up_start_right + j + 1 if j < up_count_right else None
                        left_content = before_lines[ln_left - 1] if ln_left and ln_left <= len(before_lines) else ""
                        right_content = after_lines[ln_right - 1] if ln_right and ln_right <= len(after_lines) else ""
                        lines.append({"left": left_content, "right": right_content, "left_num": ln_left, "right_num": ln_right})
                    expand_sections.append(
                        {
                            "position": "before",
                            "hunk_idx": hunk_idx,
                            "direction": "up",
                            "count": max_up,
                            "lines": lines,
                        }
                    )

            hunk_idx += 1

    # ── Expand DOWN after last hunk ──
    if hunk_ranges:
        bl_start, bl_end, al_start, al_end = hunk_ranges[-1]
        down_count_left = min(max_expand, len(before_lines) - bl_end)
        down_count_right = min(max_expand, len(after_lines) - al_end)

        if down_count_left > 0 or down_count_right > 0:
            max_down = max(down_count_left, down_count_right)
            lines = []
            for j in range(max_down):
                ln_left = bl_end + j + 1 if j < down_count_left else None
                ln_right = al_end + j + 1 if j < down_count_right else None
                left_content = before_lines[ln_left - 1] if ln_left and ln_left <= len(before_lines) else ""
                right_content = after_lines[ln_right - 1] if ln_right and ln_right <= len(after_lines) else ""
                lines.append({"left": left_content, "right": right_content, "left_num": ln_left, "right_num": ln_right})
            expand_sections.append(
                {
                    "position": "after",
                    "hunk_idx": len(hunk_ranges) - 1,
                    "direction": "down",
                    "count": max_down,
                    "lines": lines,
                }
            )

    return visible_rows, expand_sections


def _inline_highlight(old_text: str, new_text: str) -> tuple[str, str]:
    """Compute word-level inline highlights between old and new text.

    Returns (old_html, new_html) with <span class='hl'> wrapping changed words.
    """
    import difflib as _dl

    old_words = old_text.split(" ")
    new_words = new_text.split(" ")

    sm = _dl.SequenceMatcher(None, old_words, new_words)

    old_parts = []
    new_parts = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        old_chunk = " ".join(old_words[i1:i2])
        new_chunk = " ".join(new_words[j1:j2])
        if tag == "equal":
            old_parts.append(html_module.escape(old_chunk))
            new_parts.append(html_module.escape(new_chunk))
        else:
            if old_chunk:
                old_parts.append(f'<span class="hl">{html_module.escape(old_chunk)}</span>')
            if new_chunk:
                new_parts.append(f'<span class="hl">{html_module.escape(new_chunk)}</span>')

    return " ".join(old_parts), " ".join(new_parts)


def _rows_to_html_table(rows: list[dict]) -> str:
    """Convert parsed diff rows into a Beyond Compare style side-by-side HTML table."""
    table_rows = []

    for row in rows:
        ln_left = str(row["left_num"]) if row["left_num"] is not None else ""
        ln_right = str(row["right_num"]) if row["right_num"] is not None else ""

        if row["type"] == "hunk":
            table_rows.append(f'<tr class="hunk"><td colspan="5">{html_module.escape(row["left"])}</td></tr>')

        elif row["type"] == "ctx":
            left_esc = html_module.escape(row["left"])
            right_esc = html_module.escape(row["right"])
            table_rows.append(
                f'<tr class="ctx">'
                f'<td class="ln">{ln_left}</td><td class="code">{left_esc}</td>'
                f'<td class="gutter"></td>'
                f'<td class="ln">{ln_right}</td><td class="code">{right_esc}</td>'
                f"</tr>"
            )

        elif row["type"] == "pair":
            old_hl, new_hl = _inline_highlight(row["left"], row["right"])
            table_rows.append(
                f'<tr class="pair">'
                f'<td class="ln del-ln">{ln_left}</td><td class="code del-code">{old_hl}</td>'
                f'<td class="gutter"></td>'
                f'<td class="ln add-ln">{ln_right}</td><td class="code add-code">{new_hl}</td>'
                f"</tr>"
            )

        elif row["type"] == "del":
            left_esc = html_module.escape(row["left"])
            table_rows.append(
                f'<tr class="del">'
                f'<td class="ln">{ln_left}</td><td class="code">{left_esc}</td>'
                f'<td class="gutter"></td>'
                f'<td class="ln empty-ln"></td><td class="code empty-code">&nbsp;</td>'
                f"</tr>"
            )

        elif row["type"] == "add":
            right_esc = html_module.escape(row["right"])
            table_rows.append(
                f'<tr class="add">'
                f'<td class="ln empty-ln"></td><td class="code empty-code">&nbsp;</td>'
                f'<td class="gutter"></td>'
                f'<td class="ln">{ln_right}</td><td class="code">{right_esc}</td>'
                f"</tr>"
            )

    return (
        f'<table class="diff-table">'
        f"<colgroup>"
        f'<col class="col-ln"><col class="col-code">'
        f'<col class="col-gutter">'
        f'<col class="col-ln"><col class="col-code">'
        f"</colgroup>"
        f"{''.join(table_rows)}</table>"
    )


def _split_rows_by_hunk(visible_rows: list[dict]) -> list[list[dict]]:
    """Split visible_rows into per-hunk lists.

    Each hunk starts with a row of type 'hunk' (the @@ line). Returns a list
    of hunk row groups. If there are no hunks, returns an empty list.
    """
    if not visible_rows:
        return []

    hunks: list[list[dict]] = []
    current: list[dict] = []

    for row in visible_rows:
        if row["type"] == "hunk":
            if current:
                hunks.append(current)
            current = [row]
        elif current:
            current.append(row)

    if current:
        hunks.append(current)

    return hunks


def _render_expand_details(expand_sections: list[dict]) -> dict[int, tuple[str, str]]:
    """Render expand sections as native ``<details>`` elements.

    Returns a dict mapping hunk_idx to (before_html, after_html) — HTML to
    place before and after each hunk's diff table respectively.
    """
    result: dict[int, tuple[str, str]] = {}

    for sec in expand_sections:
        hunk_idx = sec["hunk_idx"]
        direction = sec["direction"]
        count = sec["count"]
        pos = "above" if direction == "up" else "below"
        label = f"⋯ Show {count} lines {pos}"

        rows_html = []
        for line in sec["lines"]:
            ln_left = str(line["left_num"]) if line["left_num"] is not None else ""
            ln_right = str(line["right_num"]) if line["right_num"] is not None else ""
            left_esc = html_module.escape(line["left"])
            right_esc = html_module.escape(line["right"])
            rows_html.append(
                f'<tr class="ctx">'
                f'<td class="ln">{ln_left}</td><td class="code">{left_esc}</td>'
                f'<td class="gutter"></td>'
                f'<td class="ln">{ln_right}</td><td class="code">{right_esc}</td>'
                f"</tr>"
            )

        details_html = (
            f'<details class="expand-details">'
            f'<summary class="expand-summary" data-count="{count}" data-pos="{pos}">'
            f"{label}</summary>"
            f'<table class="diff-table">'
            f"<colgroup>"
            f'<col class="col-ln"><col class="col-code">'
            f'<col class="col-gutter">'
            f'<col class="col-ln"><col class="col-code">'
            f"</colgroup>"
            f"{''.join(rows_html)}</table>"
            f"</details>"
        )

        before_html, after_html = result.get(hunk_idx, ("", ""))
        if sec["position"] == "before":
            result[hunk_idx] = (before_html + details_html, after_html)
        else:
            result[hunk_idx] = (before_html, after_html + details_html)

    return result


def _generate_html(steps: list[dict], path: str) -> str:
    """Generate an HTML diff report.

    Parameters
    ----------
    steps : list of dict
        Each dict has keys: name, diff_lines, insertions, deletions, changed
    path : str
        Output file path.

    Returns
    -------
    str
        The output file path.
    """
    import time

    total_add = sum(s["insertions"] for s in steps)
    total_del = sum(s["deletions"] for s in steps)
    changed_passes = sum(1 for s in steps if s["changed"])

    html_steps = []
    for idx, step in enumerate(steps):
        changed = step["changed"]
        badge_class = "changed" if changed else "unchanged"
        badge_text = f"+{step['insertions']}/-{step['deletions']}" if changed else "unchanged"
        stats_text = f"+{step['insertions']} -{step['deletions']}" if changed else ""

        before_id = f"before_{idx}"
        after_id = f"after_{idx}"

        if changed:
            before_split = step["before_script"].splitlines()
            after_split = step["after_script"].splitlines()
            rows, expand_sections = _parse_diff_into_rows(
                step["diff_lines"],
                before_split,
                after_split,
            )
            hunk_rows = _split_rows_by_hunk(rows)
            expand_map = _render_expand_details(expand_sections)

            hunk_blocks = []
            for i, hunk_row_list in enumerate(hunk_rows):
                table_html = _rows_to_html_table(hunk_row_list)
                expand_before, expand_after = expand_map.get(i, ("", ""))
                hunk_blocks.append(f"{expand_before}{table_html}{expand_after}")

            body = (
                f'<div class="diff-wrapper">'
                f'<div id="{before_id}" style="display:none">{html_module.escape(step["before_script"])}</div>'
                f'<div id="{after_id}" style="display:none">{html_module.escape(step["after_script"])}</div>'
                f'<div class="panel-bar">'
                f'<span class="panel-label">Before</span>'
                f'<button class="copy-btn" data-copy-target="{before_id}">📋 Copy</button>'
                f'<span class="panel-label" style="margin-left:24px">After</span>'
                f'<button class="copy-btn" data-copy-target="{after_id}">📋 Copy</button>'
                f"</div>"
                f"{''.join(hunk_blocks)}"
                f"</div>"
            )
        else:
            # Unchanged: show full IR content with copy button
            body = (
                f'<div class="full-content">'
                f'<div class="panel-bar">'
                f'<span class="panel-label">IR (unchanged)</span>'
                f'<button class="copy-btn" data-copy-target="{before_id}">📋 Copy</button>'
                f"</div>"
                f'<div id="{before_id}"><pre>{html_module.escape(step["before_script"])}</pre></div>'
                f"</div>"
            )

        open_attr = ' class="pass-section open"' if changed else ' class="pass-section"'
        html_steps.append(f"""\
<div{open_attr}>
  <div class="pass-header">
    <span class="pass-chevron">▶</span>
    <span class="pass-name">{html_module.escape(step["name"])}</span>
    <span class="pass-badge {badge_class}">{badge_text}</span>
    <span class="pass-stats">{stats_text}</span>
  </div>
  <div class="pass-body">
    {body}
  </div>
</div>""")

    html_content = _HTML_TEMPLATE.format(
        steps="\n".join(html_steps),
        total_passes=len(steps),
        changed_passes=changed_passes,
        total_add=total_add,
        total_del=total_del,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pass_diff(
    func_or_mod,
    passes,
    *,
    mode: str = "terminal",
    context: int = 3,
    html_path: str = "pass_diff_report.html",
) -> list[dict]:
    """Compare IR before and after each pass in a chain.

    Parameters
    ----------
    func_or_mod : PrimFunc or IRModule
        The starting IR.
    passes : Pass or list[Pass] or list[tuple[str, Pass]]
        A single pass, a list of passes, or a list of (name, pass) pairs.
        If passes are not named, a default name is derived from the pass object.
    mode : {"terminal", "html", "both"}
        Output mode. ``"terminal"`` prints colored diff to stdout.
        ``"html"`` generates an HTML file. ``"both"`` does both.
    context : int
        Number of context lines in the unified diff (default 3).
    html_path : str
        Output path for HTML report (default ``pass_diff_report.html``).

    Returns
    -------
    list[dict]
        One entry per pass step, each containing:
        ``name``, ``before_script``, ``after_script``, ``diff_lines``,
        ``insertions``, ``deletions``, ``changed``.
    """
    mode = str(mode).strip().lower()
    if mode not in ("terminal", "html", "both"):
        raise ValueError(f"mode must be one of 'terminal', 'html', 'both', got {mode!r}")

    # Normalize passes to list of (name, pass)
    if not isinstance(passes, (list, tuple)):
        passes = [passes]

    named_passes: list[tuple[str, object]] = []
    for p in passes:
        if isinstance(p, (list, tuple)) and len(p) == 2:
            named_passes.append((str(p[0]), p[1]))
        else:
            name = type(p).__name__
            named_passes.append((name, p))

    total = len(named_passes)
    mod = _to_mod(func_or_mod)
    results: list[dict] = []

    for step_idx, (name, p) in enumerate(named_passes, 1):
        before_script = _get_script(mod)

        # Apply pass
        mod = p(mod)

        after_script = _get_script(mod)

        before_lines = before_script.splitlines()
        after_lines = after_script.splitlines()
        diff_lines = _compute_diff(before_lines, after_lines, context)
        insertions, deletions = _count_changes(diff_lines)
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

        # Terminal output
        if mode in ("terminal", "both"):
            _print_step_header(step_idx, total, name)
            _print_colored_diff(diff_lines)
            if changed:
                print(f"\n  {_DIM}>>> +{insertions} insertion(s), -{deletions} deletion(s){_RESET}")
            else:
                print(f"\n  {_DIM}>>> (no changes){_RESET}")

    # HTML output
    if mode in ("html", "both"):
        _generate_html(results, html_path)
        if mode == "html":
            print(f"HTML report written to: {html_path}")
        else:
            print(f"\nHTML report also written to: {html_path}")

    return results
