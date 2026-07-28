"""Pattern-based kernel replacement framework for Relax IR.

Matches DAG patterns in the Relax IR using TVM's dataflow pattern
language, then replaces matched subgraphs with optimized TIR PrimFuncs.

Runs **before LegalizeOps** so the IR contains high-level Relax ops
(``R.astype``, ``R.mean``, ``R.rsqrt``, etc.) which are easy to match.

Pattern Registration Protocol
-----------------------------
Each pattern has three components:

1. **pattern_fn** ``() -> (root_pattern, annotations_dict)``
   Builds the DFPattern DAG.  Annotations are named wildcards whose
   matched Relax vars are extracted as ``InputInfo`` for the builder.
   Annotation vars are allowed to be externally used (they're inputs,
   not intermediates).

2. **builder_fn** ``(inputs: dict[str, InputInfo], params: dict) -> PrimFunc``
   Constructs the replacement TIR function.  ``inputs`` maps annotation
   names to ``InputInfo(var, shape, dtype)``.  ``params`` comes from the
   check function.  The function can be built with ``te.compute`` (for
   schedule-rule kernels) or ``@T.prim_func`` (for TileLang DSL kernels
   with ``tirx.is_scheduled=True``).

3. **check_fn** ``(matched_bindings, annotations) -> CheckResult | dict | None``
   Validates the match and extracts op-specific parameters.  Return
   ``None`` to reject.  Return a ``CheckResult`` (or legacy dict) with:
   - ``params``: passed to builder_fn
   - ``extra_outputs``: multi-output vars (see below)
   - ``opaque``: if True, sets ``op_pattern=8`` (kOpaque) on the TIR

Multi-Output Patterns
---------------------
When ``check_fn`` returns ``extra_outputs=[var]``, the fused kernel
produces multiple tensors.  The framework emits a ``call_tir`` with
tuple return and ``TupleGetItem`` extraction.  Extra output vars are
removed from their original bindings and remapped to the TupleGetItem
results.  Use ``bb.emit_output()`` to convert DataflowVars to regular
Vars before storing in env.

TIR Attributes
--------------
- ``tirx.is_tilelang_kernel``: TileLang DSL kernel with explicit thread bindings.
- ``tirx.is_scheduled``: Function has explicit schedule - skip
  schedule-rule application by ApplyDefaultSchedule.
- ``op_pattern=8``: kOpaque — prevents FuseOps from merging this
  call_tir with neighboring ops (avoids primitive-function lowering conflicts).

Usage::

    my_pattern = make_pattern("my_pattern", pattern_fn, builder_fn, check_fn)
    # In pipeline: PatternRewritePass([my_pattern, ...]) before LegalizeOps
"""

import logging
from dataclasses import dataclass

from tilelang import tvm as tvm
from tvm import relax, tirx as tir, ir
from tvm.relax.transform import FusionPattern

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pattern data types and factory
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Return type for pattern check functions.

    Attributes
    ----------
    params : dict
        Op-specific parameters (e.g. ``head_dim``, ``N``).
        Passed to the builder function.
    extra_outputs : list[relax.Var]
        Intermediate vars that become extra outputs of the fused kernel
        (multi-output patterns). Their original bindings are removed and
        replaced by TupleGetItem on the call_tir result.
    opaque : bool
        If True, the TIR function is marked with ``op_pattern=8``
        (kOpaque) to prevent FuseOps from merging it with neighbors.
    """

    params: dict
    extra_outputs: list = None
    opaque: bool = False

    def __post_init__(self):
        if self.extra_outputs is None:
            self.extra_outputs = []


@dataclass
class _RegisteredPattern:
    name: str
    pattern_fn: object  # () -> (DFPattern, dict[str, DFPattern])
    builder_fn: object  # (dict[str, InputInfo]) -> tir.PrimFunc
    check_fn: object = None  # optional extra validation


@dataclass
class InputInfo:
    """Info about a matched input extracted by the framework."""

    var: relax.Var
    shape: list[int]
    dtype: str


def make_pattern(name, pattern_fn, builder_fn, check_fn=None):
    """Build a reusable :class:`_RegisteredPattern` describing one rewrite.

    Parameters
    ----------
    name : str
        Human-readable name for logging.
    pattern_fn : callable
        ``() -> (root_pattern, annotations_dict)``
        Build the DFPattern graph. Return the root pattern and a dict
        mapping names to wildcard patterns for input extraction.
    builder_fn : callable
        ``(inputs: dict[str, InputInfo], params: dict) -> tir.PrimFunc``
        Given the matched inputs + extracted params, return a TIR PrimFunc.
        ``params`` is the dict returned by ``check_fn`` (empty if no check_fn).
    check_fn : callable, optional
        ``(matched_bindings: dict[str, VarBinding], annotations: dict[str, Var]) -> dict | None``
        Validate semantics of the matched subgraph and extract op-specific
        parameters (e.g. half_dim, axis). Return a dict of params on success,
        None to reject the match.

    The returned object is plain data — pass it (usually several at once) to
    :class:`PatternRewritePass` to activate.
    """
    return _RegisteredPattern(name, pattern_fn, builder_fn, check_fn)


# ---------------------------------------------------------------------------
# Recursive expression remapper
# ---------------------------------------------------------------------------


def _expr_uses_var(expr, var):
    """Check if a Relax expression references a specific Var."""
    if isinstance(expr, relax.Var):
        return expr.same_as(var)
    if isinstance(expr, relax.Call):
        if isinstance(expr.op, relax.Var) and expr.op.same_as(var):
            return True
        return any(_expr_uses_var(a, var) for a in expr.args)
    if isinstance(expr, relax.Tuple):
        return any(_expr_uses_var(f, var) for f in expr.fields)
    if isinstance(expr, relax.TupleGetItem):
        return _expr_uses_var(expr.tuple_value, var)
    return False


from tilelang.graph.utils import remap_expr as _remap_expr


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------


@tvm.transform.module_pass(opt_level=0, name="PatternRewritePass")
class PatternRewritePass:
    """Relax pass: match an explicit list of patterns and replace with call_tir.

    Place before ``LegalizeOps`` in the pipeline.  ``patterns`` is a list of
    objects returned by :func:`make_pattern`.
    """

    def __init__(self, patterns):
        self._patterns = list(patterns)

    def transform_module(self, mod: tvm.IRModule, _ctx) -> tvm.IRModule:
        if not self._patterns:
            return mod

        main_func = None
        for gvar, func in mod.functions.items():
            if isinstance(func, relax.Function) and gvar.name_hint == "main":
                main_func = func
                break
        if main_func is None:
            return mod

        body = main_func.body
        if not isinstance(body, relax.SeqExpr) or not body.blocks:
            return mod

        # Collect bindings
        bindings = []
        for block in body.blocks:
            for b in block.bindings:
                if isinstance(b, relax.VarBinding):
                    bindings.append(b)

        # For each registered pattern, find all matches
        all_replacements = []  # (matched_var, input_infos, builder_fn, name)

        for reg in self._patterns:
            root_pat, annotations = reg.pattern_fn()

            # Build FusionPattern with optional check
            fusion_pat = FusionPattern(
                reg.name,
                root_pat,
                annotation_patterns=annotations,
                check=reg.check_fn,
            )

            # Use TVM's pattern matching
            matches = _find_pattern_matches(main_func, fusion_pat, annotations, bindings)

            for matched_root_var, matched_annotations, matched_binding_vars in matches:
                # Validate: intermediates must be single-use (only within pattern)
                # Annotation vars (external inputs like q, cos, sin) are ALLOWED
                # to be used outside the pattern — they're inputs, not intermediates.
                # Use same_as for identity comparison since Var objects may differ.
                annotation_var_set = set()
                for av in matched_annotations.values():
                    if isinstance(av, relax.Var):
                        annotation_var_set.add(av)
                        # Also find the binding var that produces this value
                        for b in bindings:
                            if b.var.same_as(av):
                                annotation_var_set.add(b.var)
                intermediates = set()
                for bv in matched_binding_vars:
                    if bv.same_as(matched_root_var):
                        continue
                    is_annotation = any(bv.same_as(av) for av in annotation_var_set)
                    if not is_annotation:
                        intermediates.add(bv)
                externally_used = False
                for b in bindings:
                    if b.var in matched_binding_vars:
                        continue  # skip bindings within the pattern
                    for ivar in intermediates:
                        if _expr_uses_var(b.value, ivar):
                            externally_used = True
                            break
                    if externally_used:
                        break
                # Also check the output expression
                if not externally_used:
                    for ivar in intermediates:
                        if _expr_uses_var(body.body, ivar):
                            externally_used = True
                            break
                if externally_used:
                    continue

                # Extract input info from annotations
                input_infos = {}
                for ann_name, ann_var in matched_annotations.items():
                    sinfo = ann_var.struct_info_ if hasattr(ann_var, "struct_info_") else None
                    if sinfo and isinstance(sinfo, relax.TensorStructInfo) and sinfo.shape:
                        shape_vals = sinfo.shape.values
                        if all(isinstance(s, tir.IntImm) for s in shape_vals):
                            shape = [int(s) for s in shape_vals]
                        else:
                            shape = None
                        dtype = str(sinfo.dtype)
                    else:
                        shape = None
                        dtype = None
                    input_infos[ann_name] = InputInfo(var=ann_var, shape=shape, dtype=dtype)

                # Skip if any input has unknown shape (dynamic — not supported yet)
                if any(info.shape is None for info in input_infos.values()):
                    continue

                # Semantic check: validate and extract op-specific params
                check_result = CheckResult(params={})
                if reg.check_fn is not None:
                    matched_bindings_dict = {}
                    for b in bindings:
                        if b.var in matched_binding_vars:
                            matched_bindings_dict[b.var.name_hint] = b
                    raw = reg.check_fn(matched_bindings_dict, matched_annotations)
                    if raw is None:
                        continue  # check rejected this match
                    # Accept both dict (legacy) and CheckResult
                    if isinstance(raw, CheckResult):
                        check_result = raw
                    else:
                        check_result = CheckResult(
                            params={k: v for k, v in raw.items() if not k.startswith("_")},
                            extra_outputs=raw.get("_extra_output_vars", []),
                            opaque=raw.get("_opaque", False),
                        )

                # Extra output vars are allowed to be externally used
                if check_result.extra_outputs:
                    intermediates -= set(check_result.extra_outputs)

                all_replacements.append(
                    (
                        matched_root_var,
                        matched_binding_vars,
                        intermediates,
                        input_infos,
                        check_result,
                        reg.builder_fn,
                        reg.name,
                    )
                )

        if not all_replacements:
            return mod

        logger.info("PatternRewritePass: %d replacement(s) to apply", len(all_replacements))

        # Build replacement map: root_var → (tir_gvar, input_infos)
        bb = relax.BlockBuilder()
        remove_vars = set()  # vars to skip during rebuild
        rewrite_map = {}  # root_var → (tir_gvar, input_infos, name)

        for root_var, matched_bvars, intermediates_set, input_infos, check_result, builder_fn, name in all_replacements:
            # Skip if root already claimed by a higher-priority pattern
            if root_var in rewrite_map:
                continue
            # Skip if any matched binding overlaps with already-claimed vars
            if matched_bvars & remove_vars:
                continue

            try:
                tir_func = builder_fn(input_infos, check_result.params)
                if check_result.opaque:
                    tir_func = tir_func.with_attr("op_pattern", 8)  # kOpaque
            except Exception as e:
                logger.debug("Pattern %s builder failed: %s", name, e)
                continue

            tir_gvar = bb.add_func(tir_func, f"fused_{name}")
            remove_vars |= intermediates_set
            for ev in check_result.extra_outputs:
                remove_vars.add(ev)
            rewrite_map[root_var] = (tir_gvar, input_infos, check_result, name)

        if not rewrite_map:
            return mod

        # Rebuild main function
        call_tir_op = ir.Op.get("relax.call_tir")

        with bb.function("main", main_func.params):
            with bb.dataflow():
                env = {p: p for p in main_func.params}

                for binding in bindings:
                    if binding.var in remove_vars:
                        continue

                    if binding.var in rewrite_map:
                        tir_gvar, input_infos, check_result, name = rewrite_map[binding.var]
                        extra_outputs = check_result.extra_outputs
                        if extra_outputs:
                            results = _emit_multi_output(bb, env, tir_gvar, input_infos, check_result, binding, extra_outputs, call_tir_op)
                            # Map the primary output
                            var = results[0]
                            # Map extra outputs
                            for extra_var, result_var in zip(extra_outputs, results[1:]):
                                env[extra_var] = result_var
                        else:
                            var = _emit_replacement(bb, env, tir_gvar, input_infos, binding, call_tir_op)
                        logger.debug("Replaced %s with fused_%s", binding.var.name_hint, name)
                    else:
                        value = _remap_expr(binding.value, env)
                        var = bb.emit(value, name_hint=binding.var.name_hint)

                    env[binding.var] = var

                output = _remap_expr(body.body, env)
                bb.emit_output(output)
            bb.emit_func_output(output)

        new_mod = bb.get()
        for gvar, func in mod.functions.items():
            if gvar.name_hint != "main" and gvar.name_hint not in {g.name_hint for g in new_mod.functions}:
                new_mod[gvar] = func
        return new_mod


def _emit_multi_output(bb, env, tir_gvar, input_infos, params, binding, extra_output_vars, call_tir_op):
    """Emit call_tir with tuple output for a multi-output pattern.

    The TIR function has N inputs + K outputs.  ``CallTIRRewrite``
    (downstream) allocates output buffers.  We emit a single
    ``call_tir`` that returns a Tuple of K tensors, then extract
    each with ``TupleGetItem``.

    TIR buffer order: [inputs..., extra_outputs..., primary_output]
    Return order:     [primary_output, extra_output_0, ...]
    """
    tir_func = bb.get()[tir_gvar]

    # Count TIR input vs output buffers
    n_tir_params = len([p for p in tir_func.params if p in tir_func.buffer_map])
    n_extra = len(extra_output_vars)
    # Layout: [inputs..., extra_outputs..., primary_output]
    n_tir_inputs = n_tir_params - n_extra - 1

    all_bufs = []
    for p in tir_func.params:
        if p in tir_func.buffer_map:
            all_bufs.append(tir_func.buffer_map[p])
    output_bufs = all_bufs[n_tir_inputs:]

    # Match inputs by TIR buffer name (skip extra annotations like "x")
    tir_input_names = [all_bufs[i].name for i in range(n_tir_inputs)]
    tir_input_shapes = [[int(s) for s in all_bufs[i].shape] for i in range(n_tir_inputs)]

    tir_inputs = []
    for idx, buf_name in enumerate(tir_input_names):
        if buf_name in input_infos:
            info = input_infos[buf_name]
            mapped = env.get(info.var, info.var)
            if info.shape and tir_input_shapes[idx] != info.shape:
                mapped = bb.emit(relax.op.reshape(mapped, tir_input_shapes[idx]), name_hint=f"{buf_name}_flat")
            tir_inputs.append(mapped)
        else:
            # Fallback: skip
            pass
    if len(tir_inputs) != n_tir_inputs:
        # Name matching failed, use positional order
        tir_inputs = [env.get(info.var, info.var) for info in list(input_infos.values())[:n_tir_inputs]]

    # Build output sinfos from TIR output buffers
    out_sinfos = []
    for buf in output_bufs:
        shape = [int(s) for s in buf.shape]
        out_sinfos.append(relax.TensorStructInfo(shape, str(buf.dtype)))

    # Emit call_tir with tuple output
    if len(out_sinfos) == 1:
        call = relax.Call(call_tir_op, [tir_gvar, relax.Tuple(tir_inputs)], sinfo_args=out_sinfos)
        result = bb.emit(call, name_hint=binding.var.name_hint)
        return [result]

    call = relax.call_tir(tir_gvar, relax.Tuple(tir_inputs), out_sinfo=out_sinfos)
    tuple_out = bb.emit(call, name_hint="fused_multi_out")

    # Extract outputs: TIR order is [extra_out_0, ..., primary_out]
    # Return order is [primary_out, extra_out_0, ...]
    results = []
    all_output_vars = list(extra_output_vars) + [binding.var]
    for i, ov in enumerate(all_output_vars):
        hint = ov.name_hint if hasattr(ov, "name_hint") else f"out_{i}"
        item = bb.emit(relax.TupleGetItem(tuple_out, i), name_hint=hint)
        # Promote to non-DataflowVar so downstream _remap_expr
        # doesn't embed DataflowVars into call_packed args.
        item = bb.emit_output(item)
        results.append(item)

    # Reorder: primary last in TIR → first in return
    return [results[-1]] + results[:-1]


def _emit_replacement(bb, env, tir_gvar, input_infos, binding, call_tir_op):
    """Emit call_tir for a matched pattern.

    Passes inputs directly (no reshape) — the TIR function must accept
    the original tensor shapes.  Output shape/dtype are inferred from
    the TIR function's last (output) buffer.
    """
    tir_func = bb.get()[tir_gvar]

    # Count TIR input vs output buffers from the function signature
    n_tir_params = len([p for p in tir_func.params if p in tir_func.buffer_map])
    # Heuristic: last buffer is output, rest are inputs
    n_tir_inputs = n_tir_params - 1

    # Infer output shape from the TIR function's last buffer
    out_buf = None
    for i, p in enumerate(tir_func.params):
        if p in tir_func.buffer_map and i == n_tir_inputs:
            out_buf = tir_func.buffer_map[p]
    if out_buf is not None:
        out_shape = [int(s) for s in out_buf.shape]
        out_dtype = str(out_buf.dtype)
    else:
        first_info = next(iter(input_infos.values()))
        out_shape = first_info.shape
        out_dtype = first_info.dtype

    # Match annotations to TIR input buffers by name.
    # Extra annotations (e.g. "q" for externally-used check) are skipped.
    tir_input_names = []
    for i, p in enumerate(tir_func.params):
        if p in tir_func.buffer_map and i < n_tir_inputs:
            tir_input_names.append(tir_func.buffer_map[p].name)

    # Collect TIR input buffer shapes for potential reshape
    tir_input_shapes = []
    for i, p in enumerate(tir_func.params):
        if p in tir_func.buffer_map and i < n_tir_inputs:
            tir_input_shapes.append([int(s) for s in tir_func.buffer_map[p].shape])
        else:
            tir_input_shapes.append(None)

    tir_inputs = []
    for idx, buf_name in enumerate(tir_input_names):
        if buf_name in input_infos:
            info = input_infos[buf_name]
            mapped = env.get(info.var, info.var)
            # Reshape if TIR expects different shape (e.g. flattened 2D vs 3D)
            if tir_input_shapes[idx] is not None and info.shape is not None and tir_input_shapes[idx] != info.shape:
                mapped = bb.emit(relax.op.reshape(mapped, tir_input_shapes[idx]), name_hint=f"{buf_name}_flat")
            tir_inputs.append(mapped)
        else:
            break
    if len(tir_inputs) != n_tir_inputs:
        tir_inputs = [env.get(info.var, info.var) for info in list(input_infos.values())[:n_tir_inputs]]

    out_sinfo = relax.TensorStructInfo(out_shape, out_dtype)
    call = relax.Call(call_tir_op, [tir_gvar, relax.Tuple(tir_inputs)], sinfo_args=[out_sinfo])
    result = bb.emit(call, name_hint=binding.var.name_hint)

    # Reshape if TIR output shape differs from binding's expected shape
    # (e.g. TIR uses flattened (M,N) but binding expects (B,S,N))
    orig_si = binding.var.struct_info_
    if isinstance(orig_si, relax.TensorStructInfo) and orig_si.shape and all(isinstance(s, tir.IntImm) for s in orig_si.shape.values):
        orig_shape = [int(s) for s in orig_si.shape.values]
        if orig_shape != out_shape:
            result = bb.emit(relax.op.reshape(result, orig_shape), name_hint=binding.var.name_hint)
    return result


# ---------------------------------------------------------------------------
# Pattern matching using TVM's infrastructure
# ---------------------------------------------------------------------------


def _find_pattern_matches(main_func, fusion_pattern, annotations, bindings):
    """Find all non-overlapping matches of a FusionPattern in the function.

    Returns list of (root_var, matched_annotation_vars, matched_binding_vars).
    """
    body = main_func.body
    if not isinstance(body, relax.SeqExpr):
        return []

    # Build var→binding and var→value maps
    var_to_value = {}
    for block in body.blocks:
        for b in block.bindings:
            if isinstance(b, relax.VarBinding):
                var_to_value[b.var] = b.value

    root_pattern = fusion_pattern.pattern
    ann_patterns = fusion_pattern.annotation_patterns or {}

    results = []
    used_vars = set()

    # Try matching from each binding (reverse order — outputs first)
    for binding in reversed(bindings):
        if binding.var in used_vars:
            continue

        match_result = _try_match(binding.var, binding.value, root_pattern, ann_patterns, var_to_value, main_func.params)

        if match_result is None:
            continue

        matched_anns, matched_bvars = match_result

        # Annotation vars (external inputs) can be shared across matches.
        # Only claim intermediates + root for overlap detection.
        ann_var_set = set()
        for av in matched_anns.values():
            if isinstance(av, relax.Var):
                ann_var_set.add(av)
        exclusive_vars = {v for v in matched_bvars if not any(v.same_as(a) for a in ann_var_set)}

        if exclusive_vars & used_vars:
            continue

        used_vars |= exclusive_vars
        results.append((binding.var, matched_anns, matched_bvars))

    return results


def _try_match(root_var, root_value, root_pattern, ann_patterns, var_to_value, func_params):
    """Try to match root_pattern starting from root_value.

    Returns (annotation_var_map, matched_binding_vars) or None.
    """
    from tvm.relax.dpl.pattern import WildcardPattern, CallPattern, TuplePattern

    # Simple recursive pattern matcher
    env = {}  # DFPattern → matched Relax Expr
    matched_binding_vars = set()

    def resolve(expr):
        """Follow a Var binding to its defining expression, recording the
        intermediate binding var along the way so it gets added to the
        matched set."""
        if isinstance(expr, relax.Var) and expr in var_to_value:
            matched_binding_vars.add(expr)
            return var_to_value[expr]
        return expr

    def match(pattern, expr):
        # Check if this pattern was already matched (diamond handling)
        if pattern in env:
            return _same_expr(env[pattern], expr)

        if isinstance(pattern, WildcardPattern):
            env[pattern] = expr
            return True

        if isinstance(pattern, CallPattern):
            if not isinstance(expr, relax.Call):
                return False
            # Match the op — pattern.op may be ExprPattern(Op) or Op
            if pattern.op is not None:
                from tvm.relax.dpl.pattern import ExprPattern

                pat_op = pattern.op
                if isinstance(pat_op, ExprPattern):
                    pat_op = pat_op.expr
                if isinstance(pat_op, ir.Op) and isinstance(expr.op, ir.Op):
                    if pat_op.name != expr.op.name:
                        return False
                elif isinstance(pat_op, ir.Op):
                    return False  # pattern expects Op but expr has something else
            # Match args
            if len(pattern.args) != len(expr.args):
                return False
            for p_arg, e_arg in zip(pattern.args, expr.args):
                if not match(p_arg, resolve(e_arg)):
                    return False
            env[pattern] = expr
            return True

        if isinstance(pattern, TuplePattern):
            if not isinstance(expr, relax.Tuple):
                return False
            if len(pattern.fields) != len(expr.fields):
                return False
            for p_field, e_field in zip(pattern.fields, expr.fields):
                if not match(p_field, resolve(e_field)):
                    return False
            env[pattern] = expr
            return True

        # For other pattern types, just record as wildcard
        env[pattern] = expr
        return True

    if not match(root_pattern, root_value):
        return None

    matched_binding_vars.add(root_var)

    # Extract annotation matches
    matched_anns = {}
    for name, ann_pat in ann_patterns.items():
        if ann_pat in env:
            expr = env[ann_pat]
            # Resolve to the original Var if possible
            if isinstance(expr, relax.Var):
                matched_anns[name] = expr
            elif isinstance(expr, relax.Call):
                # The matched expr is a call — find which binding var produces it
                for bvar, bval in var_to_value.items():
                    if bval is expr:
                        matched_anns[name] = bvar
                        break
            else:
                matched_anns[name] = expr
        else:
            return None  # annotation not matched

    # Annotations for external inputs should be function params or
    # bindings NOT in the matched set
    for _name, var in matched_anns.items():
        if isinstance(var, relax.Var) and var in matched_binding_vars:
            # This "input" is actually an intermediate — check if it's
            # also a func param (which is fine)
            param_set = set(func_params)
            if var not in param_set:
                # It's an intermediate that was matched — could be ok
                # if the pattern uses it as both input and intermediate
                pass

    return matched_anns, matched_binding_vars


def _same_expr(a, b):
    """Check if two Relax expressions refer to the same value."""
    if isinstance(a, relax.Var) and isinstance(b, relax.Var):
        return a.same_as(b)
    return a is b
