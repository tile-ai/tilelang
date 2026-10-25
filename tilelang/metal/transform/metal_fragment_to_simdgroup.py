"""Give legacy Metal GEMM accumulators a native simdgroup tensor type."""

import tvm_ffi
from tvm_tirx import ir
from tvm_tirx.tirx.transform import function_pass

from tilelang._compat.stmt_functor import post_order_visit
from tilelang._compat.tensor import tensor_type


def _metal_fragment_to_simdgroup(function, module, context):
    target = function.attrs.get("target")
    if target is None or target.kind.name != "metal":
        return function
    replacements = {}

    def collect(node):
        if not isinstance(node, ir.Call) or not isinstance(node.op, ir.Op):
            return
        if node.op.name != "tl.tileop.gemm":
            return
        region = node.args[2]
        if isinstance(region, ir.Call) and isinstance(region.op, ir.Op) and region.op.name == "tl.region":
            region = region.args[0]
        if isinstance(region, ir.TensorRegion):
            tensor = region.source
        elif isinstance(region, ir.TensorLoad):
            tensor = region.source
        else:
            raise TypeError(f"Metal GEMM accumulator requires a tensor region, got {type(region).__name__}")
        if tensor.scope().name == "local.fragment":
            replacements[tensor] = ir.Var(
                tensor.name,
                tensor_type(tensor.shape, tensor.dtype, scope="metal.simdgroup"),
                loc=tensor.loc,
            )

    post_order_visit(function.body, collect)
    if not replacements:
        return function

    def mutate_var(variable, engine):
        return replacements.get(variable, variable)

    def mutate_bind(binding, engine):
        if binding.var not in replacements:
            return engine.default_mutate(binding)
        variable = replacements[binding.var]
        value = binding.value
        if not isinstance(value, ir.Call) or value.op.name != "tirx.alloc_tensor":
            raise ValueError("A Metal simdgroup accumulator must have a native tensor allocation")
        allocation = ir.Call(value.op, list(value.args), attrs=value.attrs, ty_args=[variable.ty], loc=value.loc)
        return ir.Bind(variable, allocation, loc=binding.loc)

    return tvm_ffi.structural_mutate(function, [(ir.Var, mutate_var), (ir.Bind, mutate_bind)])


MetalFragmentToSimdgroup = function_pass(_metal_fragment_to_simdgroup, opt_level=0, name="tl.MetalFragmentToSimdgroup")

__all__ = ["MetalFragmentToSimdgroup"]
