from tilelang import tvm
from tvm import tirx
from tvm.tirx import AllocBuffer, AttrStmt, Buffer, BufferLoad, BufferStore, PyStmtExprMutator, Var
from tvm.tirx.transform import prim_func_pass


def _with_local_scope(buffer: Buffer) -> Buffer:
    data = Var(
        buffer.data.name,
        tvm.ir.PointerType(tvm.ir.PrimType(buffer.dtype), "local"),
        buffer.data.span,
    )
    return Buffer(
        data,
        buffer.dtype,
        buffer.shape,
        buffer.strides,
        buffer.axis_separators,
        buffer.elem_offset,
        buffer.name,
        buffer.data_alignment,
        buffer.offset_factor,
        buffer.buffer_type,
        getattr(buffer, "span", None),
    )


@tirx.functor.mutator
class NormalizeMetalLocalVarMutator(PyStmtExprMutator):
    def __init__(self):
        super().__init__()
        self.buffer_remap: dict[Buffer, Buffer] = {}
        self.data_buffer_remap: dict[Var, Buffer] = {}
        self.var_remap: dict[Var, Var] = {}

    def _remap_buffer(self, buffer: Buffer) -> Buffer:
        if buffer in self.buffer_remap:
            return self.buffer_remap[buffer]

        if buffer.data in self.data_buffer_remap:
            new_buffer = self.data_buffer_remap[buffer.data]
            self.buffer_remap[buffer] = new_buffer
            return new_buffer

        if buffer.scope() != "local.var":
            return buffer

        new_buffer = _with_local_scope(buffer)
        self.buffer_remap[buffer] = new_buffer
        self.data_buffer_remap[buffer.data] = new_buffer
        self.var_remap[buffer.data] = new_buffer.data
        return new_buffer

    def visit_var_(self, op: Var):
        return self.var_remap.get(op, op)

    def visit_alloc_buffer_(self, op: AllocBuffer):
        buffer = self._remap_buffer(op.buffer)
        return AllocBuffer(buffer, op.annotations, getattr(op, "span", None))

    def visit_buffer_load_(self, op: BufferLoad):
        buffer = self._remap_buffer(op.buffer)
        indices = [self.visit_expr(index) for index in op.indices]
        predicate = self.visit_expr(op.predicate) if op.predicate is not None else None
        return BufferLoad(buffer, indices, predicate, getattr(op, "span", None))

    def visit_buffer_store_(self, op: BufferStore):
        buffer = self._remap_buffer(op.buffer)
        value = self.visit_expr(op.value)
        indices = [self.visit_expr(index) for index in op.indices]
        predicate = self.visit_expr(op.predicate) if op.predicate is not None else None
        return BufferStore(buffer, value, indices, predicate, getattr(op, "span", None))

    def visit_attr_stmt_(self, op: AttrStmt):
        node = op.node
        if isinstance(node, Buffer):
            node = self._remap_buffer(node)
        elif isinstance(node, Var):
            node = self.var_remap.get(node, node)
        return AttrStmt(node, op.attr_key, self.visit_expr(op.value), self.visit_stmt(op.body), getattr(op, "span", None))


def NormalizeMetalLocalVar():
    def pass_fn(func, mod, ctx):
        mutator = NormalizeMetalLocalVarMutator()
        return func.with_body(mutator.visit_stmt(func.body))

    return prim_func_pass(pass_fn, opt_level=0)
