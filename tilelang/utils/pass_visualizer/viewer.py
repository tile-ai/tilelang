"""Interactive pass browser: render the SBlock structure tree after every CUDA
lowering pass as a single self-contained HTML file.

Left pane is the ordered pass list (click to select). Right pane shows the
structure tree for the selected pass, with lines added by that pass highlighted
green and lines it removed shown ghosted red — so switching between passes makes
"what this pass changed" obvious.

Reuses the tree-rendering logic from core.py verbatim; this file only adds
per-pass capture, line-level diffing (difflib), and HTML emission.

Run with::

    python -m tilelang.utils.pass_visualizer.viewer \\
        tilelang/utils/pass_visualizer/examples/gemm_relu.py \\
        --set M=1024 --set N=1024 --set K=1024 \\
        --set block_M=128 --set block_N=128 --set block_K=32 \\
        --out gemm_relu_passes.html
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import html
import io
import json
import os
import re

from tilelang.engine.semantic_check import PreLowerSemanticCheck

from . import core as M

# Field names that inspect_structure prints to the LEFT of the "→" arrow.
_FIELDS = {
    "iter_vars",
    "reads",
    "writes",
    "alloc_buffers",
    "match_buffers",
    "init?",
    "annotations",
    "params",
    "ret_type",
    "buffer_map",
    "attrs",
}
_NODES = ("SBlock", "PrimFunc", "AttrStmt", "SeqStmt", "For", "IfThenElse", "Evaluate", "BufferStore", "SBlockRealize")

# Inline styles so highlighting survives even where external CSS classes don't
# (some IDE HTML previews strip <style> or sanitize class names).
_STY_TILEOP = "background:#ff9d3c;color:#1e1e1e;font-weight:700;padding:0 4px;border-radius:3px"  # tile ops
_STY_SYNC = "background:#9d7bff;color:#1e1e1e;font-weight:700;padding:0 4px;border-radius:3px"  # sync primitives
_STY_HW = "background:#5fb3d4;color:#1e1e1e;font-weight:600;padding:0 4px;border-radius:3px"  # lowered hw intrinsics
_STY_FIELD = "color:#9cdcfe"
_STY_NODE = "color:#4ec9b0"
_STY_TY = "color:#c586c0"

# Op classification. Only genuine operators are highlighted; scalar intrinsics
# (max/Cast/bitwise_xor) and DSL constructors (T.Tensor/T.Kernel/T.alloc_*) are
# intentionally left plain.
#
# 1) Tile ops — the authoritative set: every operator registered in C++ via
#    TIR_REGISTER_TL_TILE_OP ("tl.tileop.*") in src/op/*.cc. These are the
#    high-level tile/fragment operators LowerTileOp consumes. ('region' is a
#    TileOperator too but appears as an argument everywhere, so we leave it plain
#    to avoid noise.)
_TILE_OPS = (
    "gemm",
    "gemm_sp",
    "copy",
    "tma_copy",
    "fill",
    "clear",
    "reduce",
    "cumsum",
    "cummax",
    "transpose",
    "im2col",
    "atomicadd",
    "atomic_add",
    "atomicmax",
    "atomicmin",
    "finalize_reducer",
)
# 2) Synchronization primitives — thread/warp sync emitted by warp-specialization
#    and pipelining passes (NOT tile ops).
_SYNC_OPS = ("mbarrier_wait_parity", "ptx_arrive_barrier", "mbarrier_expect_tx", "tl_shuffle_elect", "barrier_wait", "fence_barrier_init")
# 3) Lowered hardware intrinsics — the PTX/TMA-level result of LowerTileOp.
_HW_OPS = ("ptx_mma", "ptx_ldmatrix", "tma_load", "create_tma_descriptor", "tvm_access_ptr", "access_ptr")

_TILEOP_RE = re.compile(r"\bT\.(" + "|".join(_TILE_OPS) + r")(?=\(|\s|$)")
_SYNC_RE = re.compile(r"\bT\.(" + "|".join(_SYNC_OPS) + r")(?=\(|\s|$)")
_HW_RE = re.compile(r"\bT\.(" + "|".join(_HW_OPS) + r")(?=\(|\s|$)")


def _highlight(line: str) -> str:
    """Server-side syntax highlight one text line into safe HTML with inline styles."""
    s = html.escape(line)
    # operator highlighting, by class (tile / sync / lowered-hardware)
    s = _TILEOP_RE.sub(rf'<span style="{_STY_TILEOP}">T.\1</span>', s)
    s = _SYNC_RE.sub(rf'<span style="{_STY_SYNC}">T.\1</span>', s)
    s = _HW_RE.sub(rf'<span style="{_STY_HW}">T.\1</span>', s)
    # node-type labels
    s = re.sub(r"\b(" + "|".join(_NODES) + r")\b", rf'<span style="{_STY_NODE}">\1</span>', s)
    # field names (token immediately before the → arrow)
    s = re.sub(
        r"\b([A-Za-z_?]+)(?=\s*→)", lambda m: f'<span style="{_STY_FIELD}">{m.group(1)}</span>' if m.group(1) in _FIELDS else m.group(0), s
    )
    # python type annotations after a colon: ":IntImm", ": Array", "[value:Call]"
    s = re.sub(r":\s*([A-Z][A-Za-z0-9_]*)\b", rf': <span style="{_STY_TY}">\1</span>', s)
    return s


def _capture_tree(mod) -> list[str]:
    """Render core.inspect_structure(mod) into a list of text lines."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        M.inspect_structure(mod)
    return buf.getvalue().splitlines()


def _diff_rows(prev: list[str], cur: list[str]) -> list[dict]:
    """Merge prev->cur into display rows tagged equal / add / del.

    'add'  = line present in cur but not prev (this pass introduced it)
    'del'  = line present in prev but not cur (this pass removed it; shown ghosted)
    'equal'= unchanged carry-over
    """
    rows: list[dict] = []
    sm = difflib.SequenceMatcher(a=prev, b=cur, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for line in cur[j1:j2]:
                rows.append({"t": "equal", "s": line})
        elif tag == "insert":
            for line in cur[j1:j2]:
                rows.append({"t": "add", "s": line})
        elif tag == "delete":
            for line in prev[i1:i2]:
                rows.append({"t": "del", "s": line})
        elif tag == "replace":
            for line in prev[i1:i2]:
                rows.append({"t": "del", "s": line})
            for line in cur[j1:j2]:
                rows.append({"t": "add", "s": line})
    # Pre-render highlighted HTML + tile-op flag for every row (server-side, so it
    # does not depend on browser-side JS or external CSS).
    for r in rows:
        r["h"] = _highlight(r["s"])
        r["op"] = bool(_TILEOP_RE.search(r["s"]))
    return rows


def build_pass_data(path: str, factory: str | None, target: str, kwargs: dict[str, object], source: str) -> tuple[str, list[dict]]:
    """Run the pipeline pass-by-pass, capturing a tree + diff for each stage.

    Returns (kernel_name, stages) where each stage is:
        {name, flag, rows}
    'flag' is changed/no-op; 'rows' is the diff vs the previous stage's tree.
    Stage 0 is the TileLang source; stage 1 the pipeline input ("(input)").
    """
    module = M.load_user_module(path)

    if factory is not None:
        if not hasattr(module, factory):
            raise SystemExit(f"Name {factory!r} not found in {path}.")
        kernels = {factory: getattr(module, factory)}
    else:
        kernels = M.discover_jit_kernels(module)
        if not kernels:
            raise SystemExit(f"No @tilelang.jit kernel found in {path}. For a bare @T.prim_func factory, pass --factory NAME.")

    # The viewer focuses on a single kernel; take the first discovered one.
    name, kernel = next(iter(kernels.items()))
    func = M.kernel_to_tir(kernel, **kwargs)
    mod, resolved_target = M.build_module(func, target=target)

    PreLowerSemanticCheck(mod)
    stages = M.build_pass_stages(resolved_target)

    captured: list[dict] = []

    # Stage 0: the TileLang source itself, shown in the same right-hand pane.
    # Rows are plain (no diff) but still tile-op highlighted.
    src_rows = [{"t": "equal", "s": ln, "h": _highlight(ln), "op": bool(_TILEOP_RE.search(ln))} for ln in source.split("\n")]
    captured.append({"name": "source code", "flag": "source", "rows": src_rows})

    prev_lines: list[str] = []
    input_lines = _capture_tree(mod)
    captured.append(
        {
            "name": "(input)",
            "flag": "input",
            "rows": _diff_rows([], input_lines),
        }
    )
    prev_lines = input_lines

    for idx, (pname, transform) in enumerate(stages, start=1):
        before = str(mod)
        with resolved_target:  # post-LayoutInference passes need Target.Current()
            mod = transform(mod)
        after = str(mod)
        changed = before != after

        cur_lines = _capture_tree(mod)
        captured.append(
            {
                "name": f"[{idx:02d}] {pname}",
                "flag": "changed" if changed else "no-op",
                "rows": _diff_rows(prev_lines, cur_lines),
            }
        )
        prev_lines = cur_lines

    return name, captured


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Pass browser — __TITLE__</title>
<style>
  :root {
    --bg: #1e1e1e; --fg: #d4d4d4; --panel: #252526; --border: #3c3c3c;
    --sel: #094771; --add-bg: #14361d; --add-fg: #6ccb6c;
    --del-bg: #3a1d1d; --del-fg: #d36b6b; --muted: #858585;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    background: var(--bg); color: var(--fg); font-size: 13px; height: 100vh;
    display: flex; overflow: hidden;
  }
  #left {
    width: 360px; min-width: 260px; flex-shrink: 0; overflow-y: auto;
    background: var(--panel); border-right: 1px solid var(--border);
    transition: width .12s ease, min-width .12s ease;
  }
  #left.collapsed { width: 34px; min-width: 34px; overflow: hidden; }
  #left.collapsed #passlist { display: none; }
  #left.collapsed h2 .lbl { display: none; }
  #left h2 {
    margin: 0; padding: 12px 14px; font-size: 12px; text-transform: uppercase;
    letter-spacing: .08em; color: var(--muted); position: sticky; top: 0;
    background: var(--panel); border-bottom: 1px solid var(--border);
    cursor: pointer; user-select: none; display: flex; align-items: center; gap: 8px;
  }
  #left h2:hover { color: var(--fg); }
  #left h2 .arrow { font-size: 10px; }
  .pass {
    padding: 8px 14px; cursor: pointer; border-bottom: 1px solid #2d2d2d;
    display: flex; align-items: center; gap: 8px; white-space: nowrap;
  }
  .pass:hover { background: #2a2d2e; }
  .pass.sel { background: var(--sel); }
  .pass .nm { flex: 1; overflow: hidden; text-overflow: ellipsis; }
  .badge {
    font-size: 10px; padding: 1px 7px; border-radius: 10px; flex-shrink: 0;
  }
  .badge.changed { background: var(--add-bg); color: var(--add-fg); }
  .badge.no-op   { background: #333; color: var(--muted); }
  .badge.input   { background: #2d3b55; color: #8ab4f8; }
  .badge.source  { background: #4a3a1a; color: #ffba6b; }
  .delta { font-size: 10px; color: var(--muted); flex-shrink: 0; }
  #right { flex: 1; overflow: auto; padding: 0; }
  #head {
    position: sticky; top: 0; background: var(--bg); padding: 12px 18px;
    border-bottom: 1px solid var(--border); z-index: 1;
  }
  #head .title { font-size: 15px; }
  #head .sub { color: var(--muted); font-size: 12px; margin-top: 4px; }
  #head .legend { margin-top: 8px; display: flex; gap: 16px; font-size: 11px; }
  #head .legend span { display: flex; align-items: center; gap: 6px; }
  .sw { width: 11px; height: 11px; border-radius: 2px; display: inline-block; }
  .sw.add { background: var(--add-fg); } .sw.del { background: var(--del-fg); }
  .sw.tileop { background: #ff9d3c; }
  .sw.sync { background: #9d7bff; }
  .sw.hw { background: #5fb3d4; }
  pre#tree { margin: 0; padding: 14px 18px; white-space: pre; tab-size: 2; }
  .row { display: block; padding: 0 6px; border-radius: 2px; }
  .row.add { background: var(--add-bg); color: var(--add-fg); }
  .row.del { background: var(--del-bg); color: var(--del-fg); text-decoration: line-through; opacity: .8; }
  .row .mk { display: inline-block; width: 14px; color: var(--muted); user-select: none; }
  .row.add .mk { color: var(--add-fg); } .row.del .mk { color: var(--del-fg); }
  /* tile-op lines: left accent bar + faint wash so the actual ops stand out */
  .row.tileop-row {
    background: rgba(255, 166, 87, 0.13);
    box-shadow: inset 4px 0 0 #ff8c1a;
  }
  .row.tileop-row.add { box-shadow: inset 4px 0 0 #ff8c1a; }
  .row.tileop-row.del { box-shadow: inset 4px 0 0 #8a5a1a; }
  /* token highlighting, layered on top of the diff row colors */
  .tileop {
    color: #1e1e1e; font-weight: 700; background: #ff9d3c;
    padding: 0 4px; border-radius: 3px;
  }
  .field  { color: #9cdcfe; }                          /* SBlock/PrimFunc field names */
  .node   { color: #4ec9b0; }                          /* Stmt node-type labels */
  .ty     { color: #c586c0; }                          /* Python type annotations (:IntImm, :Map, ...) */
  kbd { background:#333;border-radius:3px;padding:1px 5px;border:1px solid #555;font-size:11px; }
</style>
</head>
<body>
<div id="left"><h2 id="passhdr"><span class="arrow">&#9664;</span><span class="lbl">Passes</span></h2><div id="passlist"></div></div>
<div id="right">
  <div id="head">
    <div class="title" id="h-title"></div>
    <div class="sub" id="h-sub"></div>
    <div class="legend">
      <span><i class="sw add"></i>added by this pass</span>
      <span><i class="sw del"></i>removed by this pass</span>
      <span><i class="sw tileop"></i>tile op</span>
      <span><i class="sw sync"></i>sync primitive</span>
      <span><i class="sw hw"></i>lowered hw intrinsic</span>
      <span>Use <kbd>&uarr;</kbd>/<kbd>&darr;</kbd> to step through passes</span>
    </div>
  </div>
  <pre id="tree"></pre>
</div>
<script>
const DATA = __DATA__;
let cur = 0;

function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function countDelta(stage) {
  let a = 0, d = 0;
  for (const r of stage.rows) { if (r.t === 'add') a++; else if (r.t === 'del') d++; }
  return [a, d];
}

function renderList() {
  const el = document.getElementById('passlist');
  el.innerHTML = '';
  DATA.forEach((st, i) => {
    const [a, d] = countDelta(st);
    const row = document.createElement('div');
    row.className = 'pass' + (i === cur ? ' sel' : '');
    row.onclick = () => select(i);
    const delta = (st.flag === 'changed') ? `<span class="delta">+${a} -${d}</span>` : '';
    row.innerHTML = `<span class="nm">${escapeHtml(st.name)}</span>`
      + `<span class="badge ${st.flag}">${st.flag}</span>` + delta;
    el.appendChild(row);
  });
}

function renderTree() {
  const st = DATA[cur];
  const [a, d] = countDelta(st);
  document.getElementById('h-title').textContent = st.name;
  document.getElementById('h-sub').textContent =
    st.flag === 'source' ? 'original TileLang source'
    : st.flag === 'input' ? 'pipeline input (before any pass)'
    : `${st.flag} — ${a} line(s) added, ${d} line(s) removed vs previous pass`;
  const pre = document.getElementById('tree');
  pre.innerHTML = '';
  for (const r of st.rows) {
    const span = document.createElement('span');
    const cls = r.t === 'add' ? 'add' : r.t === 'del' ? 'del' : '';
    // r.h = server-side highlighted HTML; r.op = line carries a tile-op call.
    span.className = 'row' + (cls ? ' ' + cls : '') + (r.op ? ' tileop-row' : '');
    const mark = r.t === 'add' ? '+' : r.t === 'del' ? '-' : ' ';
    span.innerHTML = `<span class="mk">${mark}</span>${r.h}`;
    pre.appendChild(span);
  }
}

function select(i) {
  cur = Math.max(0, Math.min(DATA.length - 1, i));
  renderList();
  renderTree();
  const sel = document.querySelector('.pass.sel');
  if (sel) sel.scrollIntoView({block: 'nearest'});
}

document.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowDown') { select(cur + 1); e.preventDefault(); }
  else if (e.key === 'ArrowUp') { select(cur - 1); e.preventDefault(); }
});

// Click the "Passes" header to collapse / expand the left panel.
document.getElementById('passhdr').onclick = () => {
  const left = document.getElementById('left');
  const collapsed = left.classList.toggle('collapsed');
  left.querySelector('.arrow').innerHTML = collapsed ? '&#9654;' : '&#9664;';
};

renderList();
renderTree();
</script>
</body>
</html>
"""


def emit_html(title: str, stages: list[dict]) -> str:
    data_json = json.dumps(stages)
    out = HTML_TEMPLATE.replace("__TITLE__", html.escape(title))
    out = out.replace("__DATA__", data_json)
    return out


def emit_txt(title: str, stages: list[dict]) -> str:
    """Plain-text rendering: each stage's tree, one stage after another.

    Uses the raw row text (`s`), with a +/- marker for diff rows so the same
    information shown in the HTML is greppable / diffable as text.
    """
    lines: list[str] = [f"########## kernel: {title} ##########", ""]
    for st in stages:
        lines.append(f"===== {st['name']}  [{st['flag']}] =====")
        for r in st["rows"]:
            mark = "+" if r["t"] == "add" else "-" if r["t"] == "del" else " "
            lines.append(f"{mark} {r['s']}")
        lines.append("")
    return "\n".join(lines)


def _write_outputs(name: str, stages: list[dict], html_path: str) -> str:
    """Write the HTML and a sibling .txt; return the txt path."""
    os.makedirs(os.path.dirname(os.path.abspath(html_path)), exist_ok=True)
    with open(html_path, "w") as f:
        f.write(emit_html(name, stages))
    txt_path = os.path.splitext(html_path)[0] + ".txt"
    with open(txt_path, "w") as f:
        f.write(emit_txt(name, stages))
    return txt_path


def main():
    parser = argparse.ArgumentParser(description="Build an interactive HTML pass browser for a TileLang kernel.")
    parser.add_argument("path", help="Path to a Python file containing a @tilelang.jit kernel.")
    parser.add_argument("--factory", default=None, help="Name of the kernel to analyze (default: first discovered).")
    parser.add_argument("--target", default="auto", help="Compilation target (default: auto).")
    parser.add_argument(
        "--set",
        dest="kwargs",
        action="append",
        default=[],
        metavar="K=V",
        help="key=value arg forwarded to the kernel factory (repeatable).",
    )
    parser.add_argument("--out", default=None, help="Output HTML path (default: <kernel>_passes.html next to the source).")
    args = parser.parse_args()

    with open(args.path) as f:
        source = f.read()

    name, stages = build_pass_data(args.path, args.factory, args.target, M._parse_kv(args.kwargs), source)

    out_path = args.out
    if out_path is None:
        base = os.path.splitext(os.path.basename(args.path))[0]
        out_path = os.path.join(os.path.dirname(os.path.abspath(args.path)), f"{base}_passes.html")

    txt_path = _write_outputs(name, stages, out_path)

    print(f"Wrote interactive pass browser ({len(stages)} stages) -> {out_path}")
    print(f"Wrote text dump -> {txt_path}")


if __name__ == "__main__":
    main()
