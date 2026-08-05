"""Fused residual add + RMSNorm pattern (multi-output).

Matches:  x = add(hidden, residual) → RMSNorm(x, w) → normed
Produces: (res_sum, normed) via call_tir with TupleGetItem extraction.

The fused TileLang kernel computes both outputs in one pass,
eliminating the separate add kernel and one memory round-trip.
"""

from tvm.relax.dpl.pattern import wildcard, is_op
from tvm import relax

import tilelang.language as T
from tilelang.graph.pattern_rewrite import make_pattern, InputInfo
from tilelang.graph.utils import get_static_shape


def _residual_rmsnorm_pattern():
    """Match add → RMSNorm chain (HF LLaMA IR)."""
    hidden = wildcard()
    residual = wildcard()
    w = wildcard()

    x = is_op("relax.add")(hidden, residual)
    x_cast = is_op("relax.astype")(x)
    x_pow = is_op("relax.power")(x_cast, wildcard())
    x_mean = is_op("relax.mean")(x_pow)
    x_add = is_op("relax.add")(x_mean, wildcard())
    x_rsqrt = is_op("relax.rsqrt")(x_add)
    mul1 = is_op("relax.multiply")(x_cast, x_rsqrt)
    normed_cast = is_op("relax.astype")(mul1)
    out = is_op("relax.multiply")(w, normed_cast)

    # x is annotation: the add result is allowed to be externally used
    return out, {"hidden": hidden, "residual": residual, "x": x, "w": w}


def _residual_rmsnorm_check(matched_bindings, annotations):
    """Validate shapes, find the add var for multi-output."""
    hidden_var = annotations.get("hidden")
    w_var = annotations.get("w")
    if hidden_var is None or w_var is None:
        return None

    hidden_shape = get_static_shape(hidden_var)
    w_shape = get_static_shape(w_var)
    if hidden_shape is None or w_shape is None:
        return None

    N = hidden_shape[-1]

    # Find the add binding var for multi-output
    hidden_var = annotations["hidden"]
    residual_var = annotations["residual"]
    add_var = None
    for _name, b in matched_bindings.items():
        val = b.value
        if not (isinstance(val, relax.Call) and hasattr(val.op, "name") and val.op.name == "relax.add"):
            continue
        if len(val.args) != 2:
            continue
        a0, a1 = val.args
        if (isinstance(a0, relax.Var) and a0.same_as(hidden_var) and isinstance(a1, relax.Var) and a1.same_as(residual_var)) or (
            isinstance(a0, relax.Var) and a0.same_as(residual_var) and isinstance(a1, relax.Var) and a1.same_as(hidden_var)
        ):
            add_var = b.var
            break

    if add_var is None:
        return None

    return {
        "N": N,
        "hidden_shape": hidden_shape,
        "w_shape": w_shape,
        "_extra_output_vars": [add_var],
        "_opaque": True,  # TileLang DSL kernel, don't fuse with neighbors
    }


def _residual_rmsnorm_builder(inputs: dict[str, InputInfo], params: dict):
    """Build dual-output TileLang DSL kernel: (res_sum, normed).

    TIR buffer order: [hidden, residual, w, res_sum, normed]
    """
    hidden_info = inputs["hidden"]
    w_info = inputs["w"]

    x_shape = tuple(hidden_info.shape)
    w_shape = tuple(w_info.shape)
    N = params["N"]
    M = 1
    for s in x_shape[:-1]:
        M *= s
    in_dtype = hidden_info.dtype

    # Use enough threads for register budget
    threads = min(1024, max(128, N // 32))

    @T.prim_func
    def kernel(
        hidden: T.Tensor(x_shape, in_dtype),
        residual: T.Tensor(x_shape, in_dtype),
        w: T.Tensor(w_shape, w_info.dtype),
        res_sum: T.Tensor(x_shape, in_dtype),
        normed: T.Tensor(x_shape, in_dtype),
    ):
        with T.Kernel(M, threads=threads) as bx:
            h_frag = T.alloc_fragment((N,), in_dtype)
            r_frag = T.alloc_fragment((N,), in_dtype)
            x_sq = T.alloc_fragment((N,), "float32")
            sq_sum = T.alloc_fragment((1,), "float32")

            if len(x_shape) == 3:
                T.copy(hidden[0, bx, 0:N], h_frag)
                T.copy(residual[0, bx, 0:N], r_frag)
            else:
                T.copy(hidden[bx, 0:N], h_frag)
                T.copy(residual[bx, 0:N], r_frag)

            # Add + square
            for j in T.Parallel(N):
                val = T.cast(h_frag[j], "float32") + T.cast(r_frag[j], "float32")
                h_frag[j] = T.cast(val, in_dtype)
                x_sq[j] = val * val

            # Write res_sum (first output)
            if len(x_shape) == 3:
                T.copy(h_frag, res_sum[0, bx, 0:N])
            else:
                T.copy(h_frag, res_sum[bx, 0:N])

            # Reduce + normalize
            T.reduce_sum(x_sq, sq_sum, dim=0)
            for i in T.Parallel(1):
                sq_sum[i] = T.rsqrt(sq_sum[i] / N + 1e-6)
            for j in T.Parallel(N):
                h_frag[j] = T.cast(T.cast(h_frag[j], "float32") * sq_sum[0], in_dtype) * w[j % w_shape[-1]]

            # Write normed (second output)
            if len(x_shape) == 3:
                T.copy(h_frag, normed[0, bx, 0:N])
            else:
                T.copy(h_frag, normed[bx, 0:N])

    func = kernel.with_attr("tirx.is_scheduled", True)
    func = func.with_attr("tirx.is_tilelang_kernel", True)
    return func


PATTERN = make_pattern("residual_rmsnorm", _residual_rmsnorm_pattern, _residual_rmsnorm_builder, check_fn=_residual_rmsnorm_check)
