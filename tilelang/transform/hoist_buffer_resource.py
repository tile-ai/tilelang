"""Hoist make_wave_buffer_resource descriptors for gfx950 buffer_load...lds.

On gfx950, the `cp_async_gs_lds_with_rsrc<16>` device template takes a
pre-computed buffer resource descriptor and a pre-computed wave-uniform
base address. Computing those per call would emit 4x readfirstlane plus
the resource bit-cast on every call site. In an unrolled tile-copy loop
the same global buffer is touched many times, so we lift the descriptor
to the kernel prologue once per source buffer and rewrite the calls to
the variant that takes the pre-hoisted pair.

Pipeline order: this pass runs in the OptimizeForTarget phase after
ThreadSync/MergeIfStmt and before MakePackedAPI, which means the
tl::access_ptr calls have already been lowered by `LowerAccessPtr` to
`tir.tvm_access_ptr(ptype, data, offset, extent, rw_mask)`, so the
buffer Var is at args[1] of each access_ptr term.

This pass is gfx950-only: on every other target it returns the PrimFunc
unchanged.

NOTE: AMD vmcnt wait-count scaling (the second half of the reference
implementation on the zty_opt_can_run_1120flops branch) is deliberately
omitted in this commit. It will land as a separate milestone (M6.5)
only if the M6 bench shows a correctness failure attributable to async
wait counts.
"""

from tvm import tir
from tvm.tir import AttrStmt, Call, Evaluate, Var, PrimFunc, stmt_functor
from tvm.tir.transform import prim_func_pass

from tilelang.utils.target import target_is_gfx950

_op_ptx_cp_async_lds = tir.op.Op.get("tl.ptx_cp_async_lds")
_op_ptx_cp_async_lds_rsrc = tir.op.Op.get("tl.ptx_cp_async_lds_rsrc")
_op_tvm_access_ptr = tir.op.Op.get("tir.tvm_access_ptr")


def _extract_buffer_var(access_ptr_expr):
    """Pull the buffer-data Var out of a lowered tvm_access_ptr call.

    After tl.LowerAccessPtr the access pointer is encoded as
    ``tvm_access_ptr(ptype, data, offset, extent, rw_mask)`` so args[1]
    is the Var of interest. Anything else (e.g. an unlowered tl.access_ptr
    or a plain pointer expression) returns None and the call is skipped.
    """
    if not isinstance(access_ptr_expr, Call):
        return None
    if access_ptr_expr.op != _op_tvm_access_ptr:
        return None
    if len(access_ptr_expr.args) < 2:
        return None
    data_arg = access_ptr_expr.args[1]
    if isinstance(data_arg, Var):
        return data_arg
    return None


def _collect_buffer_vars(body):
    """Discover unique source buffer Vars referenced by ptx_cp_async_lds calls.

    Returns an ordered dict {buf_var: (rsrc_var, base_var)} so the prologue
    AttrStmts emit in a stable order.
    """
    buffer_vars = {}

    def _visit(stmt):
        if isinstance(stmt, Evaluate) and isinstance(stmt.value, Call):
            if stmt.value.op == _op_ptx_cp_async_lds:
                # ptx_cp_async_lds args: (dst_access_ptr, src_access_ptr, bytes)
                buf_var = _extract_buffer_var(stmt.value.args[1])
                if buf_var is not None and buf_var not in buffer_vars:
                    rsrc_var = Var("__rsrc_" + buf_var.name, dtype="handle")
                    base_var = Var("__base_" + buf_var.name, dtype="uint32")
                    buffer_vars[buf_var] = (rsrc_var, base_var)

    stmt_functor.post_order_visit(body, _visit)
    return buffer_vars


def _rewrite_calls(body, buffer_vars):
    """Rewrite ptx_cp_async_lds -> ptx_cp_async_lds_rsrc with hoisted vars."""

    def _postorder(op):
        if isinstance(op, Evaluate) and isinstance(op.value, Call):
            if op.value.op == _op_ptx_cp_async_lds:
                buf_var = _extract_buffer_var(op.value.args[1])
                if buf_var is not None and buf_var in buffer_vars:
                    rsrc_var, base_var = buffer_vars[buf_var]
                    new_call = Call(
                        op.value.dtype,
                        _op_ptx_cp_async_lds_rsrc,
                        [
                            op.value.args[0],
                            op.value.args[1],
                            op.value.args[2],
                            rsrc_var,
                            base_var,
                        ],
                    )
                    return Evaluate(new_call)
        return None

    return stmt_functor.ir_transform(body, None, _postorder, ["tir.Evaluate"])


def HoistBufferResource():
    """gfx950: hoist buffer resource descriptors out of the inner copy loop."""

    def pass_fn(func: PrimFunc, _mod, _ctx):
        target = func.attrs.get("target", None)
        if target is None or not target_is_gfx950(target):
            return func

        buffer_vars = _collect_buffer_vars(func.body)
        if not buffer_vars:
            return func

        new_body = _rewrite_calls(func.body, buffer_vars)

        for buf_var, (rsrc_var, base_var) in reversed(list(buffer_vars.items())):
            new_body = AttrStmt(base_var, "buffer_base_var", buf_var, new_body)
            new_body = AttrStmt(rsrc_var, "buffer_resource_var", buf_var, new_body)

        return func.with_body(new_body)

    return prim_func_pass(pass_fn, opt_level=0)
