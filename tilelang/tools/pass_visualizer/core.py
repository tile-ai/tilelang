"""Core helpers for the pass visualizer: load a user TileLang kernel, capture
real compiler-pass executions, and render a native function's region structure tree.

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
from tilelang._compat.tvm import tirx
from tvm_tirx import ir
from tvm_tirx.tirx import Function as PrimFunc
from tilelang._compat.tensor import tensor_parameter_map
from tvm_tirx.target import Target

from tilelang.backend import create_backend_context
from tilelang.jit import JITImpl
from tilelang.instrumentation import (
    IncompletePass,
    PassEvent,
    PassEventObserver,
    PassInstrumentationTool,
    StackedPassInstrument,
)

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

    context = create_backend_context(target)
    return mod, context.target


def _fmt_shape(shape) -> list:
    out = []
    for d in shape:
        try:
            out.append(int(d))
        except (TypeError, ValueError):
            out.append(d)
    return out


def _fmt_buffer(buf) -> str:
    return f"Buffer({buf.name}, shape={_fmt_shape(buf.shape)}, dtype={buf.dtype}, scope={buf.scope().name})"


def _layout_fields(obj) -> list[tuple[str, object]] | None:
    """If obj is a Layout/Fragment/Target, return its (field, value) pairs, else None."""
    cls = type(obj).__name__
    if cls == "Layout":
        return [("input_size", obj.get_input_shape()), ("forward_index", obj.index)]
    if cls in ("Fragment", "PartialFragment"):
        rep_field = "replicate_size (partials)" if cls == "PartialFragment" else "replicate_size"
        return [
            ("input_size", obj.get_input_shape()),
            ("forward_index", obj.index),
            ("forward_thread", obj.thread),
            (rep_field, obj.replicate_size),
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


def _print_region(region: ir.RegionStmt, indent: str) -> None:
    """Print a native region's operands, bound variables, attributes, and body."""
    print(f"{indent}RegionStmt({region.op.name!r})")
    inner = indent + "    "
    print(f"{inner}├─ args         → {list(region.args)}")
    print(f"{inner}├─ body_params  → {[parameter.name for parameter in region.body_params]}")
    print(f"{inner}├─ result_vars  → {[variable.name for variable in region.result_vars]}")
    print(f"{inner}├─ attrs")
    _print_annotation_value(region.attrs, inner + "│      ")
    print(f"{inner}└─ body")
    _walk_stmt(region.body, inner + "       ")


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
    """Descend native statement regions and summarize leaves."""
    if isinstance(node, ir.RegionStmt):
        _print_region(node, indent)
    elif isinstance(node, tirx.SeqStmt):
        for s in node.seq:
            _walk_stmt(s, indent)
    elif isinstance(node, tirx.For):
        head = f"For({node.var.name} in {node.begin}..{node.end}, step={node.step}, kind={node.kind})"
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
    elif isinstance(node, ir.Bind):
        print(f"{indent}Bind: {node.var.name}: {node.var.ty} = {node.value}")
    elif isinstance(node, ir.TensorStore):
        idx = ", ".join(str(i) for i in node.indices)
        print(f"{indent}TensorStore: {node.dest.name}[{idx}]")
        print(f"{indent}    └─ value = {node.value}")
    else:
        print(f"{indent}{type(node).__name__}")


def inspect_structure(mod: tvm.IRModule) -> None:
    """Print each PrimFunc top-down: params → tensor contracts → attrs → body (region tree)."""
    for gv, func in mod.functions.items():
        if not isinstance(func, PrimFunc):
            continue
        print(f"PrimFunc `{gv.name_hint}`")
        print(f"├─ params       → {[p.name for p in func.params]}")
        print(f"├─ ret_type     → {func.ret_type}")
        print("├─ tensors      →")
        for var, buf in tensor_parameter_map(func).items():
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


class _StructureTreeObserver(PassEventObserver):
    """Pass-event consumer that snapshots the visualizer's structure tree."""

    def __init__(self):
        self.records: list[PassStructureRecord] = []
        self.input_lines: list[str] | None = None
        self.incomplete_passes: list[str] = []

    def enter_pass_context(self):
        self.records.clear()
        self.input_lines = None
        self.incomplete_passes.clear()

    def pass_started(self, mod: tvm.IRModule, event: PassEvent) -> list[str]:
        before_lines = capture_structure(mod)
        if self.input_lines is None:
            self.input_lines = before_lines
        return before_lines

    def pass_finished(self, mod: tvm.IRModule, event: PassEvent, before_lines: list[str]):
        self.records.append(
            PassStructureRecord(
                name=event.name,
                sequence=event.sequence,
                before_lines=before_lines,
                after_lines=capture_structure(mod),
            )
        )

    def passes_incomplete(self, passes: list[IncompletePass], error: BaseException | None):
        self.incomplete_passes.extend(item.name for item in passes)

    def callback_mismatch(self, actual: str, expected: str | None):
        logger.warning("Ignoring mismatched after-pass callback for %s (expected %s)", actual, expected)


class StructureTreePassInstrument(StackedPassInstrument):
    """Capture structure trees around the real top-level passes in a pipeline.

    Some top-level TileLang passes invoke nested TVM passes internally.  The
    shared stack instrument tracks every callback for pairing, while this
    consumer snapshots only depth-zero passes so the browser remains linear.
    """

    def __init__(self):
        self._structure_observer = _StructureTreeObserver()
        super().__init__(self._structure_observer, capture_nested=False)

    @property
    def records(self) -> list[PassStructureRecord]:
        return self._structure_observer.records

    @property
    def input_lines(self) -> list[str] | None:
        return self._structure_observer.input_lines

    @property
    def incomplete_passes(self) -> list[str]:
        return self._structure_observer.incomplete_passes

    def ordered_records(self) -> list[PassStructureRecord]:
        """Return completed top-level pass records in execution order."""
        return sorted(self.records, key=lambda record: record.sequence)


class StructureTreePassTool(PassInstrumentationTool):
    """Per-viewer tool that creates its PassContext-local capture instrument."""

    def __init__(self) -> None:
        self.instrument: StructureTreePassInstrument | None = None

    def create_pass_instrument(self) -> StructureTreePassInstrument:
        self.instrument = StructureTreePassInstrument()
        return self.instrument


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
