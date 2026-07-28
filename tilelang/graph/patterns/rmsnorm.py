"""Standalone RMSNorm pattern (single-output).

Matches:  astype → power → mean → add(eps) → rsqrt → multiply → astype → multiply(w)
Produces: single call_tir to a te.compute-based kernel.

This handles RMSNorm instances that are NOT preceded by a residual add
(e.g. the first layer in LLaMA). Residual+RMSNorm is handled by
residual_rmsnorm which has higher priority.
"""

from tvm.relax.dpl.pattern import wildcard, is_op
from tvm import tirx as tir, te, relax

from tilelang.graph.pattern_rewrite import make_pattern, InputInfo
from tilelang.graph.utils import get_static_shape


def _rmsnorm_pattern():
    """Match standalone RMSNorm chain (HF LLaMA IR without residual add)."""
    x = wildcard()
    w = wildcard()

    x_cast = is_op("relax.astype")(x)
    x_pow = is_op("relax.power")(x_cast, wildcard())
    x_mean = is_op("relax.mean")(x_pow)
    x_add = is_op("relax.add")(x_mean, wildcard())
    x_rsqrt = is_op("relax.rsqrt")(x_add)
    mul1 = is_op("relax.multiply")(x_cast, x_rsqrt)
    normed_cast = is_op("relax.astype")(mul1)
    out = is_op("relax.multiply")(w, normed_cast)

    return out, {"x": x, "w": w}


def _rmsnorm_check(matched_bindings, annotations):
    """Validate shapes, extract norm dimension and eps."""
    x_var = annotations.get("x")
    w_var = annotations.get("w")
    if x_var is None or w_var is None:
        return None

    x_shape = get_static_shape(x_var)
    w_shape = get_static_shape(w_var)
    if x_shape is None or w_shape is None:
        return None

    N = x_shape[-1]

    # Extract eps from the add(mean, eps) binding
    eps = 1e-6  # fallback
    for _name, b in matched_bindings.items():
        val = b.value
        if not (isinstance(val, relax.Call) and hasattr(val.op, "name") and val.op.name == "relax.add"):
            continue
        # The add(mean_result, eps_const) — eps is the second arg
        if len(val.args) == 2:
            arg1 = val.args[1]
            if isinstance(arg1, relax.Constant):
                eps = float(arg1.data.numpy())
                break

    return {
        "N": N,
        "x_shape": x_shape,
        "w_shape": w_shape,
        "eps": eps,
        "_opaque": True,
    }


def _rmsnorm_builder(inputs: dict[str, InputInfo], params: dict):
    """Build RMSNorm via te.compute: cast → reduce(x²) → rsqrt → normalize."""
    x_info = inputs["x"]
    w_info = inputs["w"]

    x_shape = tuple(x_info.shape)
    w_shape = tuple(w_info.shape)
    N = params["N"]
    in_dtype = x_info.dtype

    # Flatten all dims except the last into M
    M = 1
    for s in x_shape[:-1]:
        M *= s

    x = te.placeholder((M, N), name="x", dtype=in_dtype)
    w = te.placeholder(w_shape, name="w", dtype=w_info.dtype)

    # Cast to float32
    x_fp32 = te.compute((M, N), lambda i, j: x[i, j].astype("float32"), name="x_fp32")

    # Sum of squares (reduction)
    rk = te.reduce_axis((0, N), name="rk")
    sq_sum = te.compute((M,), lambda i: te.sum(x_fp32[i, rk] * x_fp32[i, rk], axis=rk), name="sq_sum")

    # rsqrt(mean + eps) — elementwise on the reduced result
    eps = tir.const(params["eps"], "float32")
    rrms = te.compute((M,), lambda i: tir.rsqrt(sq_sum[i] / N + eps), name="rrms")

    # Normalize: x_fp32 * rrms → cast back → multiply weight
    out = te.compute((M, N), lambda i, j: (x_fp32[i, j] * rrms[i]).astype(in_dtype) * w[j % w_shape[-1]], name="rmsnorm_out")

    return te.create_prim_func([x, w, out])


PATTERN = make_pattern("rmsnorm", _rmsnorm_pattern, _rmsnorm_builder, check_fn=_rmsnorm_check)
