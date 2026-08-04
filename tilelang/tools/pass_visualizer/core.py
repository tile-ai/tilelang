"""Core helpers for the pass visualizer: load a user TileLang kernel, capture
real compiler-pass executions, and render a PrimFunc's SBlock structure tree.

``StructureTreePassInstrument`` observes the canonical CUDA lowering prologue
through TVM's ``PassInstrument`` API.  The visualizer therefore follows the
passes that actually execute instead of maintaining a duplicate pass list.

The kernel file is taken as input; any ``@tilelang.jit`` kernel in the file is
auto-discovered. These helpers are consumed by ``viewer.py`` to emit an
interactive HTML pass browser.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import logging
from dataclasses import dataclass

from tilelang import tvm as tvm
from tvm import tirx
from tvm.tirx import PrimFunc, SBlock
from tvm.target import Target

from tilelang.jit import JITImpl

try:
    from tilelang.backend.target import determine_target
except ImportError:  # installed package (0.1.x) exposes it here
    from tilelang.utils.target import determine_target
from tilelang.engine.lower import canon_target_host

logger = logging.getLogger("tilelang.pass_visualizer")


def load_user_module(path: str):
    """Import an arbitrary user TileLang source file as a module."""
    spec = importlib.util.spec_from_file_location("_user_kernel", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot import a Python module from {path!r}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discover_jit_kernels(module) -> dict[str, JITImpl]:
    """Find every `@tilelang.jit` object (JITImpl) defined at module top level."""
    found: dict[str, JITImpl] = {}
    for name in dir(module):
        if name.startswith("__"):
            continue
        obj = getattr(module, name)
        if isinstance(obj, JITImpl):
            found[name] = obj
    return found


def kernel_to_tir(kernel, **kwargs) -> tirx.PrimFunc:
    """Elaborate a kernel into its un-lowered PrimFunc (TIR).

    Accepts three forms:
      * JITImpl  (@tilelang.jit)        -> .get_tir(**kwargs)
      * a factory callable (@T.prim_func wrapper returning a PrimFunc) -> call it
      * a PrimFunc already                -> returned as-is
    """
    if isinstance(kernel, JITImpl):
        return kernel.get_tir(**kwargs)
    if isinstance(kernel, tirx.PrimFunc):
        return kernel
    if callable(kernel):
        func = kernel(**kwargs)
        if not isinstance(func, tirx.PrimFunc):
            raise SystemExit(
                f"Factory returned {type(func).__name__}, expected a PrimFunc. Make sure the function returns the inner @T.prim_func."
            )
        return func
    raise SystemExit(f"Don't know how to turn {type(kernel).__name__} into TIR.")


def build_module(func: tirx.PrimFunc, target: str | Target = "auto"):
    """Wrap a PrimFunc into an IRModule and resolve the (target, target_host) pair."""
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})

    if isinstance(target, str):
        target = determine_target(target)
    target_host = canon_target_host(target, None)
    target_host = tvm.target.Target(target_host)
    target = tvm.target.Target(target, target_host)
    return mod, target


def _fmt_shape(shape) -> list:
    out = []
    for d in shape:
        try:
            out.append(int(d))
        except (TypeError, ValueError):
            out.append(d)
    return out


def _fmt_buffer(buf) -> str:
    return f"Buffer({buf.name}, shape={_fmt_shape(buf.shape)}, dtype={buf.dtype}, scope={buf.scope()})"


def _layout_fields(obj) -> list[tuple[str, object]] | None:
    """If obj is a Layout/Fragment/Target, return its (field, value) pairs, else None."""
    cls = type(obj).__name__
    if cls == "Layout":
        return [("input_size", obj.input_size), ("forward_index", obj.forward_index)]
    if cls == "Fragment":
        return [
            ("input_size", obj.input_size),
            ("forward_index", obj.forward_index),
            ("forward_thread", obj.forward_thread),
            ("replicate_size", obj.replicate_size),
            ("thread_range", obj.thread_range),
        ]
    if cls == "Target":
        return [("kind", obj.kind.name), ("keys", list(obj.keys)), ("arch", obj.attrs.get("arch", "?")), ("host", obj.host)]
    return None


def _print_annotation_value(val, indent: str) -> None:
    """Recursively expand an annotation value: Map -> per-key, Layout/Fragment ->
    per-field, everything else -> a single inline line."""
    # Map-like (has .items): expand each key on its own line.
    if hasattr(val, "items") and not isinstance(val, str):
        items = list(val.items())
        for j, (k, v) in enumerate(items):
            key = getattr(k, "name", k)
            last = j == len(items) - 1
            conn = "└─" if last else "├─"
            fields = _layout_fields(v)
            if fields is not None:
                print(f"{indent}{conn} {key} : {type(v).__name__}")
                pad = indent + ("       " if last else "│      ")
                for fn, fv in fields:
                    print(f"{pad}{fn:<14}= {fv}")
            else:
                print(f"{indent}{conn} {key} = {v}")
        return
    # Bare Layout/Fragment (not inside a Map).
    fields = _layout_fields(val)
    if fields is not None:
        for fn, fv in fields:
            print(f"{indent}{fn:<14}= {fv}")
        return
    print(f"{indent}{val}")


def _print_sblock(blk: SBlock, indent: str) -> None:
    """Print one SBlock's stored fields, then recurse into its body."""
    print(f"{indent}SBlock({blk.name_hint!r})")
    inner = indent + "    "
    print(f"{inner}├─ iter_vars    → {[iv.var.name for iv in blk.iter_vars]}")
    print(f"{inner}├─ reads        → {[r.buffer.name for r in blk.reads]}")
    print(f"{inner}├─ writes       → {[w.buffer.name for w in blk.writes]}")
    if blk.alloc_buffers:
        print(f"{inner}├─ alloc_buffers → ({len(blk.alloc_buffers)})")
        for buf in blk.alloc_buffers:
            print(f"{inner}│      {_fmt_buffer(buf)}")
    else:
        print(f"{inner}├─ alloc_buffers → []")
    print(f"{inner}├─ match_buffers → {[m.buffer.name for m in blk.match_buffers]}")
    print(f"{inner}├─ init?        → {blk.init is not None}")
    if blk.annotations:
        print(f"{inner}├─ annotations  →")
        items = list(blk.annotations.items())
        for j, (k, v) in enumerate(items):
            last = j == len(items) - 1
            conn = "└─" if last else "├─"
            # Map / Layout / Fragment values expand; scalars stay inline.
            if (hasattr(v, "items") and not isinstance(v, str)) or _layout_fields(v):
                print(f"{inner}│   {conn} {k} →")
                _print_annotation_value(v, inner + ("│       " if not last else "│       "))
            else:
                print(f"{inner}│   {conn} {k} = {v}")
    else:
        print(f"{inner}├─ annotations  → {{}}")
    print(f"{inner}└─ body")
    _walk_stmt(blk.body, inner + "       ")


# Positional-arg -> field-name schema for each tile op, taken from the C++
# constructors in src/op/*.cc. Index i names args[i]; a trailing "*rest" entry
# captures any remaining (optional / variadic) args as one line. Args beyond the
# named ones (e.g. gemm's optional mbar/scale-factor tail) fall into "*rest".
_TILEOP_FIELDS = {
    "gemm": [
        "a_region",
        "b_region",
        "c_region",
        "transA",
        "transB",
        "M",
        "N",
        "K",
        "policy",
        "clearAccum",
        "strideA",
        "strideB",
        "offsetA",
        "offsetB",
        "kPack",
        "wgWait",
    ],
    "copy": ["src_region", "dst_region"],
    "tma_copy": ["src_region", "dst_region"],
    "fill": ["dst", "value"],
    "reduce": ["src_region", "dst_region", "reduce_type", "dim", "clear"],
    "atomicadd": ["src_region", "dst_region"],
    "atomicmax": ["src_region", "dst_region"],
    "atomicmin": ["src_region", "dst_region"],
}


def _opname(call) -> str | None:
    """Return the short tile-op name ('gemm') for a Call to tl.tileop.*, else None."""
    op = getattr(call, "op", None)
    name = getattr(op, "name", "") if op is not None else ""
    if name.startswith("tl.tileop."):
        return name[len("tl.tileop.") :]
    return None


def _print_tileop(call, opname: str, indent: str) -> None:
    """Expand a tile-op Call by field name instead of printing one long line."""
    print(f"{indent}Evaluate: T.{opname}")
    fields = _TILEOP_FIELDS.get(opname)
    args = list(call.args)
    inner = indent + "    "
    if fields is None:
        # Unknown op: fall back to positional listing so nothing is hidden.
        for i, a in enumerate(args):
            last = i == len(args) - 1
            print(f"{inner}{'└─' if last else '├─'} arg{i} = {a}")
        return
    named = [(fname, args[i]) for i, fname in enumerate(fields) if i < len(args)]
    rest = args[len(fields) :]
    for j, (fname, val) in enumerate(named):
        last = (j == len(named) - 1) and not rest
        print(f"{inner}{'└─' if last else '├─'} {fname:<11}= {val}")
    if rest:
        print(f"{inner}└─ *rest      = {list(rest)}")


def _walk_stmt(node, indent: str) -> None:
    """Descend the Stmt tree, printing structural nodes and recursing into SBlocks.

    Only nesting-bearing nodes are expanded; leaf statements are summarized on one
    line so the SBlock nesting stays the visible backbone of the tree.
    """
    from tvm import tirx

    if isinstance(node, tirx.SBlockRealize):
        _walk_stmt(node.block, indent)
    elif isinstance(node, SBlock):
        _print_sblock(node, indent)
    elif isinstance(node, tirx.AttrStmt):
        # launch_thread / sblock_attr / kWarpSpecializationScope ... — show key+value, descend.
        if node.attr_key == "thread_extent":
            label = f"launch_thread {node.node.var.name} (extent={node.value})"
            print(f"{indent}AttrStmt[{label}]")
        else:
            print(f"{indent}AttrStmt[{node.attr_key}]")
            print(f"{indent}    └─ value = {node.value}")
        _walk_stmt(node.body, indent + "    ")
    elif isinstance(node, tirx.SeqStmt):
        for s in node.seq:
            _walk_stmt(s, indent)
    elif isinstance(node, tirx.For):
        head = f"For({node.loop_var.name} in {node.min}..{node.min + node.extent}, kind={node.kind})"
        print(f"{indent}{head}")
        if node.annotations:
            for k, v in node.annotations.items():
                # Fragment/Layout/Map annotations expand by field; scalars stay inline.
                if (hasattr(v, "items") and not isinstance(v, str)) or _layout_fields(v):
                    print(f"{indent}    @ann {k} : {type(v).__name__}")
                    _print_annotation_value(v, indent + "        ")
                else:
                    print(f"{indent}    @ann {k} = {v}")
        _walk_stmt(node.body, indent + "    ")
    elif isinstance(node, tirx.IfThenElse):
        print(f"{indent}IfThenElse({node.condition})")
        _walk_stmt(node.then_case, indent + "    ")
        if node.else_case is not None:
            print(f"{indent}else")
            _walk_stmt(node.else_case, indent + "    ")
    elif isinstance(node, tirx.Evaluate):
        # Tile ops (tl.tileop.*) expand by field name; other intrinsic calls expand
        # by positional arg; non-calls print inline.
        call = node.value
        opname = _opname(call) if hasattr(call, "op") else None
        if opname is not None:
            _print_tileop(call, opname, indent)
        elif hasattr(call, "op") and hasattr(call, "args"):
            cname = getattr(call.op, "name", str(call.op))
            short = cname[3:] if cname.startswith("tl.") else cname
            args = list(call.args)
            print(f"{indent}Evaluate: T.{short}")
            for i, a in enumerate(args):
                last = i == len(args) - 1
                print(f"{indent}    {'└─' if last else '├─'} arg{i} = {a}")
        else:
            print(f"{indent}Evaluate: {call}")
    elif isinstance(node, tirx.BufferStore):
        idx = ", ".join(str(i) for i in node.indices)
        print(f"{indent}BufferStore: {node.buffer.name}[{idx}]")
        print(f"{indent}    └─ value = {node.value}")
    else:
        print(f"{indent}{type(node).__name__}")


def inspect_structure(mod: tvm.IRModule) -> None:
    """Print each PrimFunc top-down: params → buffer_map → attrs → body (SBlock tree)."""
    for gv, func in mod.functions.items():
        if not isinstance(func, PrimFunc):
            continue
        print(f"PrimFunc `{gv.name_hint}`")
        print(f"├─ params       → {[p.name for p in func.params]}")
        print(f"├─ ret_type     → {func.ret_type}")
        print("├─ buffer_map   →")
        for var, buf in func.buffer_map.items():
            print(f"│      {var.name:<12} : {_fmt_buffer(buf)}")
        print("├─ attrs        →")
        if func.attrs:
            for k, v in func.attrs.items():
                # Map / Target-valued attrs expand by key/field; scalars stay inline.
                if (hasattr(v, "items") and not isinstance(v, str)) or _layout_fields(v):
                    print(f"│      {k} →")
                    _print_annotation_value(v, "│          ")
                else:
                    print(f"│      {k} = {v}")
        else:
            print("│      {}")
        print("└─ body")
        _walk_stmt(func.body, "       ")
        print()


def capture_structure(mod: tvm.IRModule) -> list[str]:
    """Render ``inspect_structure`` into stable text lines for diffing."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        inspect_structure(mod)
    return buf.getvalue().splitlines()


@dataclass
class PassStructureRecord:
    """Before/after structure snapshots for one top-level compiler pass."""

    name: str
    sequence: int
    before_lines: list[str]
    after_lines: list[str]

    @property
    def changed(self) -> bool:
        return self.before_lines != self.after_lines


@dataclass
class _ActivePass:
    """One active PassInstrument callback frame, including nested passes."""

    name: str
    sequence: int | None
    before_lines: list[str] | None


@tvm.ir.instrument.pass_instrument
class StructureTreePassInstrument:
    """Capture structure trees around the real top-level passes in a pipeline.

    Some top-level TileLang passes invoke nested TVM passes internally.  Those
    callbacks are tracked so before/after events remain correctly paired, but
    only depth-zero passes become browser stages.  This preserves the viewer's
    linear pipeline semantics instead of duplicating a parent's changes in its
    nested implementation details.
    """

    def __init__(self):
        self.records: list[PassStructureRecord] = []
        self.input_lines: list[str] | None = None
        self.incomplete_passes: list[str] = []
        self._stack: list[_ActivePass] = []
        self._next_sequence = 0

    def enter_pass_ctx(self):
        self.records.clear()
        self.input_lines = None
        self.incomplete_passes.clear()
        self._stack.clear()
        self._next_sequence = 0

    def exit_pass_ctx(self):
        # A failing pass has no run_after_pass callback. Preserve its name for
        # diagnostics, then clear the stack so this instrument can be reused.
        if self._stack:
            self.incomplete_passes.extend(frame.name for frame in self._stack)
            self._stack.clear()

    def run_before_pass(self, mod: tvm.IRModule, info):
        name = str(info.name)
        is_top_level = not self._stack
        if is_top_level:
            before_lines = capture_structure(mod)
            if self.input_lines is None:
                self.input_lines = before_lines
            frame = _ActivePass(
                name=name,
                sequence=self._next_sequence,
                before_lines=before_lines,
            )
            self._next_sequence += 1
        else:
            frame = _ActivePass(name=name, sequence=None, before_lines=None)
        self._stack.append(frame)

    def run_after_pass(self, mod: tvm.IRModule, info):
        name = str(info.name)
        if not self._stack:
            logger.warning("Ignoring unmatched after-pass callback for %s", name)
            return

        frame = self._stack.pop()
        if frame.name != name:
            logger.warning("Ignoring mismatched after-pass callback for %s (expected %s)", name, frame.name)
            self._stack.clear()
            return

        if frame.sequence is None:
            return

        assert frame.before_lines is not None
        self.records.append(
            PassStructureRecord(
                name=name,
                sequence=frame.sequence,
                before_lines=frame.before_lines,
                after_lines=capture_structure(mod),
            )
        )

    def ordered_records(self) -> list[PassStructureRecord]:
        """Return completed top-level pass records in execution order."""
        return sorted(self.records, key=lambda record: record.sequence)


def _parse_kv(pairs: list[str]) -> dict[str, object]:
    """Parse `key=value` CLI args, literal-evaluating values when possible."""
    kwargs: dict[str, object] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Expected key=value, got: {pair!r}")
        key, _, raw = pair.partition("=")
        try:
            kwargs[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            kwargs[key] = raw  # fall back to plain string
    return kwargs
