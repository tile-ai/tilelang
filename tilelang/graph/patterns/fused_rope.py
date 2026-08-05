"""Fused RoPE pattern: reshape + permute + rotary position embedding.

Matches (before LegalizeOps):
    x (B,S,H*D) → reshape (B,S,H,D) → permute (B,H,S,D) → RoPE → out (B,H,S,D)

Replaces with a single te.compute that reads (B,S,H*D) and writes
(B,H,S,D) with RoPE applied — reshape+permute+RoPE in one pass.

The permuted tensor q feeds multiply + 2 strided_slices (diamond).
We mark q as an annotation so the externally-used check allows the
strided_slices (which are dead code after rewrite).

In addition to the per-rope pattern, this file provides
:func:`fuse_qk_rope_pass` — a Relax module pass that runs *after* the
standard pattern rewrites and pairs sibling ``fused_fused_rope`` calls
sharing cos/sin into a single dual-output kernel.  This halves the
RoPE launch count from 64 to 32 per LLaMA-2 decode step.
"""

import logging

from tvm.relax.dpl.pattern import wildcard, is_op
from tvm import tirx as tir, te, relax

import tilelang.language as T
from tilelang import tvm
from tilelang.graph.pattern_rewrite import make_pattern, InputInfo
from tilelang.graph.utils import get_static_shape

logger = logging.getLogger(__name__)


def _fused_rope_pattern():
    """Match reshape → permute → RoPE chain."""
    x = wildcard()  # pre-reshape: (B, S, H*D) from linear
    cos = wildcard()  # expanded cos
    sin = wildcard()  # expanded sin

    # reshape + permute → q
    x_reshaped = is_op("relax.reshape")(x, wildcard())
    q = is_op("relax.permute_dims")(x_reshaped)

    # RoPE on q
    q_cos = is_op("relax.multiply")(q, cos)
    q2 = is_op("relax.strided_slice")(q, wildcard(), wildcard(), wildcard(), wildcard())
    q2r = is_op("relax.reshape")(q2, wildcard())
    _neg = is_op("relax.negative")(q2r)
    rotated = is_op("relax.concat")(wildcard())
    rot_sin = is_op("relax.multiply")(rotated, sin)
    out = is_op("relax.add")(q_cos, rot_sin)

    # q is annotation → the strided_slices on q (for rotate_half) are
    # allowed to be "external" uses. They become dead code after rewrite.
    return out, {"x": x, "q": q, "cos": cos, "sin": sin}


def _fused_rope_check(matched_bindings, annotations):
    """Validate shapes and extract H, D from x and cos."""

    x_var = annotations.get("x")
    cos_var = annotations.get("cos")
    if x_var is None or cos_var is None:
        return None

    x_shape = get_static_shape(x_var)
    cos_shape = get_static_shape(cos_var)
    if x_shape is None or cos_shape is None:
        return None
    if len(x_shape) < 2:
        return None

    head_dim = cos_shape[-1]
    hidden = x_shape[-1]
    if hidden % head_dim != 0:
        return None
    num_heads = hidden // head_dim
    half_dim = head_dim // 2
    if half_dim == 0 or head_dim % 2 != 0:
        return None

    return {
        "num_heads": num_heads,
        "head_dim": head_dim,
        "half_dim": half_dim,
        "x_shape": x_shape,
        "cos_shape": cos_shape,
        "_opaque": True,  # prevent a FuseOps group that primitive-function lowering cannot handle
    }


def _fused_rope_builder(inputs: dict[str, InputInfo], params: dict):
    """Build te.compute: (B,S,H*D) → reshape+permute+RoPE → (B,H,S,D)."""
    x_info = inputs["x"]
    cos_info = inputs["cos"]
    sin_info = inputs["sin"]

    x_shape = tuple(x_info.shape)  # (B, S, H*D)
    cos_shape = tuple(cos_info.shape)
    H = params["num_heads"]
    D = params["head_dim"]
    half = params["half_dim"]
    dtype = x_info.dtype

    B = x_shape[0]
    S = x_shape[1]
    out_shape = (B, H, S, D)

    x = te.placeholder(x_shape, name="x", dtype=dtype)
    cos_t = te.placeholder(cos_shape, name="cos", dtype=cos_info.dtype)
    sin_t = te.placeholder(cos_shape, name="sin", dtype=sin_info.dtype)

    def _rope(b, h, s, d):
        # Fused reshape+permute: x[b, s, h*D + d]
        x_val = x[b, s, h * D + d].astype("float32")
        paired_d = (d + half) % D
        x_paired = x[b, s, h * D + paired_d].astype("float32")
        sign = tir.const(1, "float32") - tir.const(2, "float32") * (d < tir.const(half, d.dtype)).astype("float32")

        if len(cos_shape) == 4:
            c = cos_t[0, 0, s, d].astype("float32")
            sn = sin_t[0, 0, s, d].astype("float32")
        elif len(cos_shape) == 3:
            c = cos_t[0, s, d].astype("float32")
            sn = sin_t[0, s, d].astype("float32")
        else:
            c = cos_t[s, d].astype("float32")
            sn = sin_t[s, d].astype("float32")

        return (x_val * c + sign * x_paired * sn).astype(dtype)

    out = te.compute(out_shape, _rope, name="rope_out")
    return te.create_prim_func([x, cos_t, sin_t, out])


PATTERN = make_pattern("fused_rope", _fused_rope_pattern, _fused_rope_builder, check_fn=_fused_rope_check)


# ---------------------------------------------------------------------------
# Q+K RoPE fusion: pair sibling fused_rope call_tir sites
# ---------------------------------------------------------------------------

# kOpaque: prevents FuseOps from grouping with neighbours, so
# LowerPrimitiveFunctionsToTIR
# leaves the buffer names alone.
_K_OPAQUE = 8


def _build_qk_rope_kernel(q_shape, k_shape, cos_shape, dtype):
    """TileLang DSL kernel that does Q RoPE and K RoPE in one launch.

    Inputs : ``x_q`` (B, S, H_q*D), ``x_k`` (B, S, H_kv*D),
             ``cos``/``sin`` (broadcast shape).
    Outputs: ``q_rot`` (B, H_q, S, D), ``k_rot`` (B, H_kv, S, D).

    Grid is ``(H_q, S)``; each block uses ``half = D/2`` threads.
    Every block applies RoPE to its Q head.  The first ``H_kv`` blocks
    also apply RoPE to the corresponding K head (GQA: ``H_kv ≤ H_q``).
    cos/sin are loaded once per thread and reused for both Q and K.
    """
    B = int(q_shape[0])
    S = int(q_shape[1])
    q_hidden = int(q_shape[-1])
    k_hidden = int(k_shape[-1])
    D = int(cos_shape[-1])
    H_q = q_hidden // D
    H_kv = k_hidden // D
    half = D // 2
    q_out_shape = (B, H_q, S, D)
    k_out_shape = (B, H_kv, S, D)
    cos_ndim = len(cos_shape)

    @T.prim_func
    def kernel(
        x_q: T.Tensor(q_shape, dtype),
        x_k: T.Tensor(k_shape, dtype),
        cos: T.Tensor(cos_shape, dtype),
        sin: T.Tensor(cos_shape, dtype),
        q_rot: T.Tensor(q_out_shape, dtype),
        k_rot: T.Tensor(k_out_shape, dtype),
    ):
        with T.Kernel(H_q, S, threads=half) as (bh, bs):
            for d in T.Parallel(half):
                if cos_ndim == 4:
                    c_lo = T.cast(cos[0, 0, bs, d], "float32")
                    c_hi = T.cast(cos[0, 0, bs, d + half], "float32")
                    s_lo = T.cast(sin[0, 0, bs, d], "float32")
                    s_hi = T.cast(sin[0, 0, bs, d + half], "float32")
                elif cos_ndim == 3:
                    c_lo = T.cast(cos[0, bs, d], "float32")
                    c_hi = T.cast(cos[0, bs, d + half], "float32")
                    s_lo = T.cast(sin[0, bs, d], "float32")
                    s_hi = T.cast(sin[0, bs, d + half], "float32")
                else:
                    c_lo = T.cast(cos[bs, d], "float32")
                    c_hi = T.cast(cos[bs, d + half], "float32")
                    s_lo = T.cast(sin[bs, d], "float32")
                    s_hi = T.cast(sin[bs, d + half], "float32")

                # Q rope — every block
                q_lo = T.cast(x_q[0, bs, bh * D + d], "float32")
                q_hi = T.cast(x_q[0, bs, bh * D + d + half], "float32")
                q_rot[0, bh, bs, d] = T.cast(q_lo * c_lo - q_hi * s_lo, dtype)
                q_rot[0, bh, bs, d + half] = T.cast(q_hi * c_hi + q_lo * s_hi, dtype)

                # K rope — first H_kv blocks only (GQA: H_kv ≤ H_q)
                if bh < H_kv:
                    k_lo = T.cast(x_k[0, bs, bh * D + d], "float32")
                    k_hi = T.cast(x_k[0, bs, bh * D + d + half], "float32")
                    k_rot[0, bh, bs, d] = T.cast(k_lo * c_lo - k_hi * s_lo, dtype)
                    k_rot[0, bh, bs, d + half] = T.cast(k_hi * c_hi + k_lo * s_hi, dtype)

    func = kernel.with_attr("tirx.is_scheduled", True)
    func = func.with_attr("tirx.is_tilelang_kernel", True)
    func = func.with_attr("op_pattern", _K_OPAQUE)
    return func


def _rope_call_args(call):
    """Return (x, cos, sin) for a call_tir to a fused_rope PrimFunc, else None."""
    if not isinstance(call, relax.Call):
        return None
    op_name = call.op.name if hasattr(call.op, "name") else None
    if op_name != "relax.call_tir" or len(call.args) < 2:
        return None
    gv = call.args[0]
    if not isinstance(gv, relax.GlobalVar):
        return None
    name = gv.name_hint
    if not (name.startswith("fused_fused_rope") or name == "fused_rope"):
        return None
    arg_tup = call.args[1]
    if not isinstance(arg_tup, relax.Tuple) or len(arg_tup.fields) != 3:
        return None
    return arg_tup.fields  # (x, cos, sin)


def _static_int_list(sinfo):
    """Convert a TensorStructInfo's shape to a list[int], or None if dynamic."""
    if not (isinstance(sinfo, relax.TensorStructInfo) and sinfo.shape is not None):
        return None
    try:
        return [int(s) for s in sinfo.shape]
    except (TypeError, ValueError):
        return None


def _find_rope_pairs(bindings):
    """Linear pass over bindings → list of (i, j) pairs of rope calls
    whose ``cos``/``sin`` Relax vars match by identity.  Each binding
    appears in at most one pair, paired with the *next* unmatched
    sibling.
    """
    pairs = []
    pending: dict = {}  # (cos_id, sin_id) → first-occurrence index
    for i, bnd in enumerate(bindings):
        if not isinstance(bnd, relax.VarBinding):
            continue
        rargs = _rope_call_args(bnd.value)
        if rargs is None:
            continue
        _, cos_v, sin_v = rargs
        if not (isinstance(cos_v, relax.Var) and isinstance(sin_v, relax.Var)):
            continue
        key = (cos_v, sin_v)
        prev = pending.pop(key, None)
        if prev is None:
            pending[key] = i
        else:
            pairs.append((prev, i))
    return pairs


def fuse_qk_rope_pass(mod):
    """Module pass: pair adjacent ``fused_fused_rope`` call_tir sites
    sharing cos/sin into a single dual-output Q+K kernel.  Halves the
    rope launch count on LLaMA-style attention from 64 to 32 per
    decode step.
    """
    qk_kernel_cache: dict = {}
    call_tir_op = tvm.ir.Op.get("relax.call_tir")
    pairs_fused_total = 0

    def get_or_build(q_shape, k_shape, cos_shape, dtype):
        key = (tuple(q_shape), tuple(k_shape), tuple(cos_shape), dtype)
        if key in qk_kernel_cache:
            return qk_kernel_cache[key]
        head_dim = int(cos_shape[-1])
        q_hidden = int(q_shape[-1])
        k_hidden = int(k_shape[-1])
        if q_hidden % head_dim != 0 or k_hidden % head_dim != 0 or head_dim % 2 != 0:
            qk_kernel_cache[key] = None
            return None
        try:
            kernel = _build_qk_rope_kernel(q_shape, k_shape, cos_shape, dtype)
        except Exception as e:
            logger.warning("Failed to build qk_rope_kernel for %s: %s", key, e)
            qk_kernel_cache[key] = None
            return None
        qk_kernel_cache[key] = kernel
        return kernel

    for gv in list(mod.get_global_vars()):
        fn = mod[gv]
        if not isinstance(fn, relax.Function) or not isinstance(fn.body, relax.SeqExpr):
            continue

        # Pre-scan: skip the rebuild entirely if there are no pairs to fuse.
        all_pairs = []
        for block in fn.body.blocks:
            all_pairs.extend(_find_rope_pairs(list(block.bindings)))
        if not all_pairs:
            continue

        new_inner, local_pairs = _rebuild_with_qk_fusion(fn, gv.name_hint, get_or_build, call_tir_op)
        if new_inner is None or local_pairs == 0:
            continue

        pairs_fused_total += local_pairs

        # Helper kernels added by bb.add_func come first — main's body
        # references their GlobalVars by identity, so they must already be
        # in mod when we install the rebuilt main.
        existing_names = {g.name_hint for g in mod.get_global_vars()}
        new_main = None
        for new_gv in new_inner.get_global_vars():
            if new_gv.name_hint == gv.name_hint:
                new_main = new_inner[new_gv]
            elif new_gv.name_hint not in existing_names:
                mod[new_gv] = new_inner[new_gv]
                existing_names.add(new_gv.name_hint)
        if new_main is not None:
            mod[gv] = new_main

    if pairs_fused_total > 0:
        logger.info("fuse_qk_rope_pass: fused %d Q+K pairs", pairs_fused_total)
    return relax.transform.DeadCodeElimination()(mod)


def _rebuild_with_qk_fusion(fn, name, get_or_build, call_tir_op):
    """Rebuild ``fn`` with fused Q+K rope calls.  Returns (new_module, n_pairs)."""
    bb = relax.BlockBuilder()
    attrs_dict = dict(fn.attrs) if fn.attrs is not None else None
    local_pairs = 0

    with bb.function(name, list(fn.params), attrs=attrs_dict):
        env: dict = {p: p for p in fn.params}
        with bb.dataflow():
            for block in fn.body.blocks:
                bindings = list(block.bindings)
                pair_partner: dict = {}
                for i, j in _find_rope_pairs(bindings):
                    pair_partner[i] = j
                    pair_partner[j] = i

                handled: set = set()
                for i, bnd in enumerate(bindings):
                    if i in handled or not isinstance(bnd, relax.VarBinding):
                        continue

                    j = pair_partner.get(i)
                    if j is not None and j > i:
                        bnd_k = bindings[j]
                        if _emit_fused_pair(bb, env, bnd, bnd_k, get_or_build, call_tir_op):
                            handled.add(j)
                            local_pairs += 1
                            continue

                    new_value = _remap_relax_expr(bnd.value, env)
                    env[bnd.var] = bb.emit(new_value, name_hint=bnd.var.name_hint)

        bb.emit_func_output(_remap_relax_expr(fn.body.body, env))

    return bb.get(), local_pairs


def _emit_fused_pair(bb, env, bnd_q, bnd_k, get_or_build, call_tir_op) -> bool:
    """Emit a fused Q+K rope call_tir for a sibling pair.  Returns True on success."""
    rq = _rope_call_args(bnd_q.value)
    rk = _rope_call_args(bnd_k.value)
    if rq is None or rk is None:
        return False
    x_q, cos_v, sin_v = rq
    x_k = rk[0]

    q_shape = _static_int_list(x_q.struct_info_)
    k_shape = _static_int_list(x_k.struct_info_)
    cos_shape = _static_int_list(cos_v.struct_info_)
    if q_shape is None or k_shape is None or cos_shape is None:
        return False

    kernel = get_or_build(q_shape, k_shape, cos_shape, x_q.struct_info_.dtype)
    if kernel is None:
        return False

    gv_qk = bb.add_func(kernel, "fused_qk_rope")
    tuple_call = relax.Call(
        call_tir_op,
        [gv_qk, relax.Tuple([env.get(x_q, x_q), env.get(x_k, x_k), env.get(cos_v, cos_v), env.get(sin_v, sin_v)])],
        sinfo_args=[relax.TupleStructInfo([bnd_q.value.struct_info_, bnd_k.value.struct_info_])],
    )
    tuple_var = bb.emit(tuple_call, name_hint="qk_rope_pair")
    env[bnd_q.var] = bb.emit(relax.TupleGetItem(tuple_var, 0), name_hint=bnd_q.var.name_hint)
    env[bnd_k.var] = bb.emit(relax.TupleGetItem(tuple_var, 1), name_hint=bnd_k.var.name_hint)
    return True


def _remap_relax_expr(expr, env):
    """Substitute Relax variables in ``expr`` using ``env``.

    Handles the subset of expression types that appear inside Relax
    dataflow blocks (Var, Call, Tuple, TupleGetItem, Constant).  Other
    types are returned unchanged.
    """
    if isinstance(expr, relax.Var):
        return env.get(expr, expr)
    if isinstance(expr, relax.Call):
        new_args = [_remap_relax_expr(a, env) for a in expr.args]
        return relax.Call(expr.op, new_args, expr.attrs, expr.sinfo_args)
    if isinstance(expr, relax.Tuple):
        return relax.Tuple([_remap_relax_expr(f, env) for f in expr.fields])
    if isinstance(expr, relax.TupleGetItem):
        return relax.TupleGetItem(_remap_relax_expr(expr.tuple_value, env), expr.index)
    return expr
