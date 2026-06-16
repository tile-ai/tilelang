"""Diff utilities for lower trace."""

from __future__ import annotations

import difflib


_ANSI_RESET = "\033[0m"
_ANSI_RED = "\033[31m"
_ANSI_GREEN = "\033[32m"
_ANSI_YELLOW = "\033[33m"
_ANSI_BLUE = "\033[34m"
_ANSI_CYAN = "\033[36m"
_ANSI_BOLD = "\033[1m"
_ANSI_DIM = "\033[2m"


def _esc(text: str) -> str:
    """Escape HTML special characters."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _inline_diff(line_before: str, line_after: str) -> tuple[str, str, bool]:
    """Compute character-level inline diff between two lines.

    Returns (left_html, right_html, is_ws_only) with highlighted changes.
    """
    sm = difflib.SequenceMatcher(None, line_before, line_after)
    left_parts = []
    right_parts = []
    is_ws_only = True

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            eq = _esc(line_before[i1:i2])
            left_parts.append(eq)
            right_parts.append(eq)
        else:
            left_chunk = line_before[i1:i2] if i2 > i1 else ""
            right_chunk = line_after[j1:j2] if j2 > j1 else ""
            if left_chunk.strip() != "" or right_chunk.strip() != "":
                is_ws_only = False

            if tag == "replace":
                left_parts.append(f'<span class="del-word">{_esc(left_chunk)}</span>')
                right_parts.append(f'<span class="add-word">{_esc(right_chunk)}</span>')
            elif tag == "delete":
                left_parts.append(f'<span class="del-word">{_esc(left_chunk)}</span>')
            elif tag == "insert":
                right_parts.append(f'<span class="add-word">{_esc(right_chunk)}</span>')

    return "".join(left_parts), "".join(right_parts), is_ws_only


def _merge_whitespace_diffs(opcodes: list, before_lines: list, after_lines: list) -> list:
    """Post-process opcodes to merge adjacent delete+insert into replace when whitespace-only."""
    result = []
    i = 0
    while i < len(opcodes):
        tag, i1, i2, j1, j2 = opcodes[i]

        if tag == "delete" and i + 1 < len(opcodes) and opcodes[i + 1][0] == "insert":
            _, _, _, j1_next, j2_next = opcodes[i + 1]
            del_lines = list(range(i1, i2))
            ins_lines = list(range(j1_next, j2_next))

            matched_del = set()
            matched_ins = set()
            pairs = []

            for di, d_idx in enumerate(del_lines):
                for ii, i_idx in enumerate(ins_lines):
                    if ii in matched_ins:
                        continue
                    if before_lines[d_idx].strip() == after_lines[i_idx].strip():
                        pairs.append((d_idx, i_idx))
                        matched_del.add(di)
                        matched_ins.add(ii)
                        break

            if pairs:
                pairs.sort()
                prev_d = i1
                prev_i = j1_next
                for d_idx, i_idx in pairs:
                    while prev_d < d_idx:
                        result.append(("delete", prev_d, prev_d + 1, prev_i, prev_i))
                        prev_d += 1
                    while prev_i < i_idx:
                        result.append(("insert", d_idx, d_idx, prev_i, prev_i + 1))
                        prev_i += 1
                    result.append(("replace", d_idx, d_idx + 1, i_idx, i_idx + 1))
                    prev_d = d_idx + 1
                    prev_i = i_idx + 1
                for d_idx in range(prev_d, i2):
                    result.append(("delete", d_idx, d_idx + 1, prev_i, prev_i))
                for i_idx in range(prev_i, j2_next):
                    result.append(("insert", i2, i2, i_idx, i_idx + 1))

                i += 2
                continue

        elif tag == "insert" and i + 1 < len(opcodes) and opcodes[i + 1][0] == "delete":
            _, i1_next, i2_next, _, _ = opcodes[i + 1]
            ins_lines = list(range(j1, j2))
            del_lines = list(range(i1_next, i2_next))

            matched_ins = set()
            matched_del = set()
            pairs = []

            for ii, i_idx in enumerate(ins_lines):
                for di, d_idx in enumerate(del_lines):
                    if di in matched_del:
                        continue
                    if after_lines[i_idx].strip() == before_lines[d_idx].strip():
                        pairs.append((d_idx, i_idx))
                        matched_ins.add(ii)
                        matched_del.add(di)
                        break

            if pairs:
                pairs.sort()
                prev_d = i1_next
                prev_i = j1
                for d_idx, i_idx in pairs:
                    while prev_i < i_idx:
                        result.append(("insert", prev_d, prev_d, prev_i, prev_i + 1))
                        prev_i += 1
                    while prev_d < d_idx:
                        result.append(("delete", prev_d, prev_d + 1, prev_i, prev_i))
                        prev_d += 1
                    result.append(("replace", d_idx, d_idx + 1, i_idx, i_idx + 1))
                    prev_d = d_idx + 1
                    prev_i = i_idx + 1
                for i_idx in range(prev_i, j2):
                    result.append(("insert", prev_d, prev_d, i_idx, i_idx + 1))
                for d_idx in range(prev_d, i2_next):
                    result.append(("delete", d_idx, d_idx + 1, j2, j2))

                i += 2
                continue

        result.append((tag, i1, i2, j1, j2))
        i += 1

    return result


def _flush_gap_rows(target: list, gap_l: list, gap_r: list) -> None:
    """Append position-paired rows for unmatched left/right lines within a gap.

    Pairs ``gap_l[k]`` with ``gap_r[k]`` for inline diff rendering; leftovers
    become single-side rows. Both lists are ascending, so the appended rows keep
    the left and right columns ascending.
    """
    for k in range(max(len(gap_l), len(gap_r))):
        if k < len(gap_l) and k < len(gap_r):
            target.append((gap_l[k], gap_r[k], False))
        elif k < len(gap_l):
            target.append((gap_l[k], None, False))
        else:
            target.append((None, gap_r[k], False))


def _make_diff_html(before_text: str, after_text: str, context: int = 3) -> str:
    """Generate a GitHub-style side-by-side diff HTML table."""
    before_lines = before_text.splitlines()
    after_lines = after_text.splitlines()

    sm = difflib.SequenceMatcher(None, before_lines, after_lines)
    opcodes = sm.get_opcodes()

    opcodes = _merge_whitespace_diffs(opcodes, before_lines, after_lines)

    if all(tag == "equal" for tag, *_ in opcodes):
        return '<p class="noop-msg">No differences.</p>'

    before_collapse = [True] * len(before_lines)
    after_collapse = [True] * len(after_lines)
    for tag, i1, i2, j1, j2 in opcodes:
        if tag != "equal":
            for i in range(i1, i2):
                before_collapse[i] = False
            for j in range(j1, j2):
                after_collapse[j] = False
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for k in range(min(context, i2 - i1)):
                before_collapse[i1 + k] = False
                after_collapse[j1 + k] = False
            for k in range(min(context, i2 - i1)):
                before_collapse[i2 - 1 - k] = False
                after_collapse[j2 - 1 - k] = False

    rows = []

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                collapsed = before_collapse[i]
                hidden_attr = ' class="row-hidden" data-collapse="1"' if collapsed else ""
                ln_l = f'<td class="ln ln-eq" data-side="l" data-idx="{i}">{i + 1}</td>'
                ln_r = f'<td class="ln ln-eq" data-side="r" data-idx="{j}">{j + 1}</td>'
                rows.append(
                    f"<tr{hidden_attr}>{ln_l}"
                    f'<td class="sg"></td><td class="eq">{_esc(before_lines[i])}</td>'
                    f"{ln_r}"
                    f'<td class="sg"></td><td class="eq">{_esc(after_lines[j])}</td></tr>'
                )
        elif tag == "replace":
            # Monotone matching: LCS on stripped content guarantees both the
            # left and right line-number columns render in ascending order.
            # Greedy bipartite matching + sort could pair a later left line to
            # an earlier right line, making the right column jump backwards.
            before_stripped = [before_lines[i].strip() for i in range(i1, i2)]
            after_stripped = [after_lines[j].strip() for j in range(j1, j2)]
            inner = difflib.SequenceMatcher(None, before_stripped, after_stripped)
            matched: list[tuple[int, int]] = []
            for t2, a1, a2, b1, _b2 in inner.get_opcodes():
                if t2 == "equal":
                    for k in range(a2 - a1):
                        matched.append((i1 + a1 + k, j1 + b1 + k))

            matched_left = {li for li, _ in matched}
            matched_right = {ri for _, ri in matched}
            unmatched_left = [i for i in range(i1, i2) if i not in matched_left]
            unmatched_right = [j for j in range(j1, j2) if j not in matched_right]

            all_rows: list = []
            up = vp = 0

            for li, ri in matched:
                gap_l = []
                while up < len(unmatched_left) and unmatched_left[up] < li:
                    gap_l.append(unmatched_left[up])
                    up += 1
                gap_r = []
                while vp < len(unmatched_right) and unmatched_right[vp] < ri:
                    gap_r.append(unmatched_right[vp])
                    vp += 1
                _flush_gap_rows(all_rows, gap_l, gap_r)
                all_rows.append((li, ri, True))

            _flush_gap_rows(all_rows, unmatched_left[up:], unmatched_right[vp:])

            for li, ri, is_matched in all_rows:
                if li is not None and ri is not None:
                    left_html, right_html, is_ws_only = _inline_diff(before_lines[li], after_lines[ri])
                    if is_ws_only and is_matched:
                        ln_l = f'<td class="ln ln-ws" data-side="l" data-idx="{li}">{li + 1}</td>'
                        ln_r = f'<td class="ln ln-ws" data-side="r" data-idx="{ri}">{ri + 1}</td>'
                        rows.append(
                            f"<tr>{ln_l}"
                            f'<td class="sg sg-ws">~</td><td class="ws">{left_html}</td>'
                            f"{ln_r}"
                            f'<td class="sg sg-ws">~</td><td class="ws">{right_html}</td></tr>'
                        )
                    else:
                        ln_l = f'<td class="ln ln-del" data-side="l" data-idx="{li}">{li + 1}</td>'
                        ln_r = f'<td class="ln ln-add" data-side="r" data-idx="{ri}">{ri + 1}</td>'
                        rows.append(
                            f"<tr>{ln_l}"
                            f'<td class="sg sg-del">\u2212</td><td class="del">{left_html}</td>'
                            f"{ln_r}"
                            f'<td class="sg sg-add">+</td><td class="add">{right_html}</td></tr>'
                        )
                elif li is not None:
                    ln_l = f'<td class="ln ln-del" data-side="l" data-idx="{li}">{li + 1}</td>'
                    rows.append(
                        f"<tr>{ln_l}"
                        f'<td class="sg sg-del">\u2212</td><td class="del">{_esc(before_lines[li])}</td>'
                        f'<td class="ln"></td><td class="sg"></td><td></td></tr>'
                    )
                else:
                    ln_r = f'<td class="ln ln-add" data-side="r" data-idx="{ri}">{ri + 1}</td>'
                    rows.append(
                        f'<tr><td class="ln"></td><td class="sg"></td><td></td>'
                        f"{ln_r}"
                        f'<td class="sg sg-add">+</td><td class="add">{_esc(after_lines[ri])}</td></tr>'
                    )
        elif tag == "delete":
            for i in range(i1, i2):
                ln_l = f'<td class="ln ln-del" data-side="l" data-idx="{i}">{i + 1}</td>'
                rows.append(
                    f"<tr>{ln_l}"
                    f'<td class="sg sg-del">\u2212</td><td class="del">{_esc(before_lines[i])}</td>'
                    f'<td class="ln"></td><td class="sg"></td><td></td></tr>'
                )
        elif tag == "insert":
            for j in range(j1, j2):
                ln_r = f'<td class="ln ln-add" data-side="r" data-idx="{j}">{j + 1}</td>'
                rows.append(
                    f'<tr><td class="ln"></td><td class="sg"></td><td></td>'
                    f"{ln_r}"
                    f'<td class="sg sg-add">+</td><td class="add">{_esc(after_lines[j])}</td></tr>'
                )

    # --- Identify contiguous runs of collapsed (hidden) rows ---
    collapse_flags = ['data-collapse="1"' in r and 'class="row-hidden"' in r for r in rows]

    run_list: list[tuple[int, int]] = []
    in_run = False
    for idx, is_collapsed in enumerate(collapse_flags):
        if is_collapsed and not in_run:
            run_start = idx
            in_run = True
        elif not is_collapsed and in_run:
            run_list.append((run_start, idx))
            in_run = False
    if in_run:
        run_list.append((run_start, len(rows)))

    # Tag hidden rows with their run index (used by JS to scope expansion)
    for run_idx, (r1, r2) in enumerate(run_list):
        for i in range(r1, r2):
            rows[i] = rows[i].replace('class="row-hidden"', f'class="row-hidden" data-run="{run_idx}"')

    # --- Insert expand button rows ---
    # One "↑↓ Expand" button after each run, expanding hidden rows in both
    # directions.  Insertions are applied from bottom to top so indices stay valid.
    insertions: list[tuple[int, str]] = []
    for ri, (_r1, r2) in enumerate(run_list):
        insertions.append(
            (
                r2,
                f'<tr class="btn-row">'
                f'<td colspan="6" data-run="{ri}" '
                f'onclick="expand(this,20,event)">'
                f'<span class="exp-arrow">\u2191\u2193</span>'
                f'<span class="exp-label">Expand</span>'
                f"</td></tr>",
            )
        )
    for pos, html in sorted(insertions, key=lambda x: x[0], reverse=True):
        rows.insert(pos, html)

    return (
        '<div class="diff-table-wrap">'
        "<table><colgroup>"
        '<col style="width:50px"><col style="width:20px"><col>'
        '<col style="width:50px"><col style="width:20px"><col>'
        "</colgroup>" + "\n".join(rows) + "</table></div>"
    )


def unified_diff(
    before_text: str,
    after_text: str,
    before_label: str = "before",
    after_label: str = "after",
    context: int = 3,
    color: bool = True,
) -> str:
    """Generate a unified diff string, optionally with terminal ANSI colors."""
    before_lines = before_text.splitlines(keepends=True)
    after_lines = after_text.splitlines(keepends=True)

    diff = list(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=before_label,
            tofile=after_label,
            n=context,
        )
    )

    if not diff:
        return ""

    if not color:
        return "".join(diff)

    colored = []
    for line in diff:
        if line.startswith("---") or line.startswith("+++"):
            colored.append(f"{_ANSI_BOLD}{line}{_ANSI_RESET}")
        elif line.startswith("@@"):
            colored.append(f"{_ANSI_CYAN}{line}{_ANSI_RESET}")
        elif line.startswith("-"):
            colored.append(f"{_ANSI_RED}{line}{_ANSI_RESET}")
        elif line.startswith("+"):
            colored.append(f"{_ANSI_GREEN}{line}{_ANSI_RESET}")
        else:
            colored.append(line)

    return "".join(colored)


def print_diff(
    before_text: str,
    after_text: str,
    before_label: str = "before",
    after_label: str = "after",
    context: int = 3,
    color: bool = True,
) -> bool:
    """Print a unified diff to stdout. Returns True if there were differences."""
    result = unified_diff(before_text, after_text, before_label, after_label, context, color)
    if result:
        print(result, end="")
        return True
    return False


def _count_changes(diff_lines: list[str]) -> tuple[int, int]:
    """Count insertions and deletions from a unified diff."""
    insertions = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
    deletions = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))
    return insertions, deletions
