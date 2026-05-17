"""Hoist make_wave_buffer_resource descriptors + scale AMD async wait counts.

On gfx950, the `cp_async_gs_lds_with_rsrc<16>` device template takes a
pre-computed buffer resource descriptor and a pre-computed wave-uniform
base address. Computing those per call would emit 4x readfirstlane plus
the resource bit-cast on every call site. In an unrolled tile-copy loop
the same global buffer is touched many times, so we lift the descriptor
to the kernel prologue once per source buffer and rewrite the calls to
the variant that takes the pre-hoisted pair.

Second half: AMD vmcnt tracks individual `buffer_load` issues, not the
NVIDIA-style commit groups. `tl::cp_async_wait<N>` lowers to
`s_waitcnt vmcnt(N)`, so a wait-for-N-groups must become
wait-for-(N * loads_per_group) on AMD. NVIDIA's cp.async commits group
every async load issued since the last commit; on AMD we have to scale
the wait count manually. We do that by finding the for-loop that
contains the `ptx_commit_group` call, counting async loads in one
iteration of that loop (multiplied by loop extents for nested unrolls),
and rewriting every positive `ptx_wait_group(n)` to `ptx_wait_group(n *
loads_per_group)`. `ptx_wait_group(0)` (wait-all) stays as `vmcnt(0)`,
which is already correct.

Pipeline order: this pass runs in the OptimizeForTarget phase after
ThreadSync/MergeIfStmt and before MakePackedAPI, which means the
tl::access_ptr calls have already been lowered by `LowerAccessPtr` to
`tir.tvm_access_ptr(ptype, data, offset, extent, rw_mask)`, so the
buffer Var is at args[1] of each access_ptr term.

This pass is gfx950-only: on every other target it returns the PrimFunc
unchanged.
"""

from tvm import tir
from tvm.tir import AttrStmt, Call, Evaluate, Var, PrimFunc, stmt_functor
from tvm.tir.transform import prim_func_pass

from tilelang.utils.target import target_is_gfx950

_op_ptx_cp_async_lds = tir.op.Op.get("tl.ptx_cp_async_lds")
_op_ptx_cp_async_lds_rsrc = tir.op.Op.get("tl.ptx_cp_async_lds_rsrc")
_op_tvm_access_ptr = tir.op.Op.get("tir.tvm_access_ptr")
_op_ptx_commit_group = tir.op.Op.get("tir.ptx_commit_group")
_op_ptx_wait_group = tir.op.Op.get("tir.ptx_wait_group")


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


def _is_async_load_call(stmt):
    if not isinstance(stmt, Evaluate) or not isinstance(stmt.value, Call):
        return False
    op = stmt.value.op
    return op == _op_ptx_cp_async_lds or op == _op_ptx_cp_async_lds_rsrc


def _is_commit_call(stmt):
    if not isinstance(stmt, Evaluate) or not isinstance(stmt.value, Call):
        return False
    return stmt.value.op == _op_ptx_commit_group


def _contains_commit_call(stmt):
    found = [False]

    def _v(s):
        if _is_commit_call(s):
            found[0] = True

    stmt_functor.post_order_visit(stmt, _v)
    return found[0]


def _find_for_with_commit(stmt):
    """Find the innermost For loop whose body contains a commit call."""
    if isinstance(stmt, tir.For):
        inner = _find_for_with_commit(stmt.body)
        if inner is not None:
            return inner
        if _contains_commit_call(stmt.body):
            return stmt
    elif isinstance(stmt, tir.SeqStmt):
        for s in stmt.seq:
            r = _find_for_with_commit(s)
            if r is not None:
                return r
    elif hasattr(stmt, "body"):
        return _find_for_with_commit(stmt.body)
    return None


def _count_async_loads(stmt, multiplier=1):
    if _is_async_load_call(stmt):
        return multiplier
    if isinstance(stmt, tir.For):
        ext = multiplier
        if isinstance(stmt.extent, tir.IntImm):
            ext = multiplier * stmt.extent.value
        return _count_async_loads(stmt.body, ext)
    if isinstance(stmt, tir.SeqStmt):
        return sum(_count_async_loads(s, multiplier) for s in stmt.seq)
    if isinstance(stmt, tir.AttrStmt):
        return _count_async_loads(stmt.body, multiplier)
    if isinstance(stmt, tir.IfThenElse):
        c = _count_async_loads(stmt.then_case, multiplier)
        if stmt.else_case is not None:
            c = max(c, _count_async_loads(stmt.else_case, multiplier))
        return c
    if isinstance(stmt, tir.LetStmt):
        return _count_async_loads(stmt.body, multiplier)
    return 0


def _get_loads_per_group(body):
    for_node = _find_for_with_commit(body)
    if for_node is not None:
        return _count_async_loads(for_node.body)
    return 0


def _fix_amd_wait_counts(body, loads_per_group):
    """Multiply positive ptx_wait_group(n) arguments by loads_per_group.

    Each `tl::cp_async_wait<N>` on AMD lowers to `s_waitcnt vmcnt(N)`,
    which counts individual buffer_loads rather than NVIDIA-style commit
    groups. wait_group(0) (wait-all) stays unchanged because vmcnt(0)
    is already the correct "wait for everything" sentinel.
    """

    def _postorder(op):
        if not isinstance(op, Evaluate):
            return None
        if not isinstance(op.value, Call):
            return None
        if op.value.op != _op_ptx_wait_group:
            return None
        if len(op.value.args) != 1:
            return None
        n_arg = op.value.args[0]
        if not isinstance(n_arg, tir.IntImm):
            return None
        if n_arg.value <= 0:
            return None
        new_call = Call(
            op.value.dtype,
            _op_ptx_wait_group,
            [tir.IntImm(n_arg.dtype, n_arg.value * loads_per_group)],
        )
        return Evaluate(new_call)

    return stmt_functor.ir_transform(body, None, _postorder, ["tir.Evaluate"])


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
    """gfx950: hoist buffer resource descriptors + scale AMD vmcnt waits."""

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

        # AMD wait-count scaling. Only meaningful when there's at least one
        # commit group; otherwise loads_per_group is 0 and we skip.
        loads_per_group = _get_loads_per_group(new_body)
        if loads_per_group > 1:
            new_body = _fix_amd_wait_counts(new_body, loads_per_group)

        return func.with_body(new_body)

    return prim_func_pass(pass_fn, opt_level=0)
