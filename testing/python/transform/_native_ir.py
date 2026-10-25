import tvm_ffi
from tvm_tirx import ir, sym, tirx

from tilelang._compat.stmt_functor import ir_transform, post_order_visit, substitute


def make_function(params, body, ret_type=None, buffer_map=None, attrs=None, span=None, *, loc=None):
    tensors = dict(buffer_map or {})
    parameters = []
    for parameter in params:
        matches = [tensor for pointer, tensor in tensors.items() if tvm_ffi.structural_equal(parameter, pointer)]
        parameters.append(matches[0] if matches else parameter)
    replacements = {
        pointer: tensor.data for pointer, tensor in tensors.items() if isinstance(pointer, ir.Var) and not pointer.same_as(tensor)
    }
    if replacements:
        body = substitute(body, replacements)
    attributes = ir.DictAttrs(attrs) if isinstance(attrs, dict) else attrs
    options = {} if loc is None and span is None else {"loc": loc if loc is not None else span}
    return tirx.Function(parameters, body, ret_type=ret_type or ir.TupleType([]), attrs=attributes, **options)


def allocate_tensor(tensor, annotations=None):
    return ir.Bind(tensor, ir.Call("tirx.alloc_tensor", [], attrs=annotations or {}, ty_args=[tensor.ty]))


def allocations(root):
    tensors = []

    def collect(node):
        if isinstance(node, ir.Bind) and isinstance(node.value, ir.Call) and node.value.op.name == "tirx.alloc_tensor":
            tensors.append(node.var)

    post_order_visit(root, collect)
    return tensors


def flat_access_ptr(tensor, offset, extent, mask):
    indices = []
    for axis in range(len(tensor.shape)):
        stride = 1
        for dimension in tensor.shape[axis + 1 :]:
            stride = stride * dimension
        if axis + 1 == len(tensor.shape):
            indices.append(offset)
        else:
            indices.append(offset // stride)
            offset = offset % stride
    return ir.Call("tl.access_ptr", [ir.TensorLoad(tensor, indices), extent, mask])


def thread_sum(value, destination, thread):
    reducer = ir.LambdaExpr([value.ty, value.ty], lambda left, right: left + right)
    return tirx.gpu_thread_allreduce(reducer, ir.const(0, value.ty), value, True, destination, thread)


def ordinal_loops(root):
    analyzer = sym.Analyzer()

    def rewrite(node):
        if not isinstance(node, ir.For):
            return None
        assert analyzer.can_prove(node.step > 0)
        ordinal = ir.Var(node.var.name + "_ordinal", node.var.ty)
        extent = analyzer.simplify((node.end - node.begin + node.step - 1) // node.step)
        body = substitute(node.body, {node.var: node.begin + ordinal * node.step})
        return ir.For(
            ordinal,
            ir.const(0, ordinal.ty),
            extent,
            ir.const(1, ordinal.ty),
            node.kind,
            body,
            node.annotations,
            node.loc,
        )

    return ir_transform(root, None, rewrite)


def expand_bindings(root, expression):
    definitions = {}

    def collect(node):
        if isinstance(node, ir.Bind) and isinstance(node.var.ty, ir.PrimType):
            definitions[node.var] = substitute(node.value, definitions)

    post_order_visit(root, collect)
    return substitute(expression, definitions)


def storage_root(root, value):
    definitions = {}
    post_order_visit(root, lambda node: definitions.__setitem__(node.var, node.value) if isinstance(node, ir.Bind) else None)
    visited = set()
    while value not in visited:
        visited.add(value)
        if isinstance(value, ir.Var) and value in definitions:
            binding = definitions[value]
            if isinstance(binding, ir.Call) and binding.op.name in ("tirx.decl_tensor", "tirx.reinterpret"):
                value = binding.args[0]
                continue
        if isinstance(value, (ir.TensorLoad, ir.TensorRegion)):
            value = value.source
            continue
        if isinstance(value, ir.Call) and value.op.name in (
            "tirx.tensor_data_ptr",
            "tirx.ptr_byte_offset",
            "tirx.reinterpret",
            "tirx.address_of",
        ):
            value = value.args[0]
            continue
        break
    return value
