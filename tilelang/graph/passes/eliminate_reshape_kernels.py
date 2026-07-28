"""Replace ``call_tir`` to reshape-pattern TIR with zero-copy ``R.reshape``.

Unlike TVM's ``RewriteDataflowReshape``, this works on ALL bindings (not just
``DataflowVar``), eliminating reshape GPU kernels that would otherwise waste a
kernel launch + memory copy.
"""

from tilelang import tvm as tvm
from tvm import relax, tirx as tir
from tvm.relax.analysis import has_reshape_pattern

_CALL_TIR_OP = tvm.ir.Op.get("relax.call_tir")


@relax.expr_functor.mutator
class _ReshapeEliminator(relax.PyExprMutator):
    def __init__(self, mod):
        super().__init__(mod)
        self.mod = mod

    def visit_call_(self, call):
        call = self.visit_expr_post_order(call)
        if call.op != _CALL_TIR_OP:
            return call

        gv = call.args[0]
        if not isinstance(gv, relax.GlobalVar):
            return call
        fn = self.mod.functions.get(gv)
        if not isinstance(fn, tir.PrimFunc):
            return call

        arg_tuple = call.args[1]
        if not isinstance(arg_tuple, relax.Tuple):
            return call
        args = arg_tuple.fields
        if len(args) == 0:
            return call

        if not has_reshape_pattern(fn):
            return call

        # Verify element count matches (exclude strided_slice etc.)
        inp = args[0]
        inp_sinfo = inp.struct_info_
        res_sinfo = call.struct_info_
        if not isinstance(inp_sinfo, relax.TensorStructInfo) or not isinstance(res_sinfo, relax.TensorStructInfo):
            return call
        if inp_sinfo.dtype != res_sinfo.dtype:
            return call
        if inp_sinfo.ndim < 0 or res_sinfo.ndim < 0:
            return call
        inp_shape = inp_sinfo.shape
        res_shape = res_sinfo.shape
        if inp_shape is None or res_shape is None:
            return call

        return relax.op.reshape(inp, res_shape)


def eliminate_reshape_kernels(mod):
    """Module pass: replace reshape TIR kernels with zero-copy reshapes."""
    eliminator = _ReshapeEliminator(mod)
    for gv, fn in list(mod.functions.items()):
        if isinstance(fn, relax.Function):
            mod[gv] = eliminator.visit_expr(fn)
    # Dead-code elimination removes now-unused TIR reshape functions.
    return relax.transform.DeadCodeElimination()(mod)
