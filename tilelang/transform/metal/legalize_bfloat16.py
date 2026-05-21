from tilelang import tvm
from tvm import tirx
from tvm.tirx import (
    AllocBuffer,
    AttrStmt,
    Buffer,
    BufferLoad,
    BufferStore,
    Cast,
    DeclBuffer,
    PrimFunc,
    PyStmtExprMutator,
    Var,
)
from tvm.tirx.transform import prim_func_pass


def _is_bfloat16(dtype) -> bool:
    return str(dtype).startswith("bfloat16")


def _dtype(code: str, lanes: int) -> str:
    return code if lanes == 1 else f"{code}x{lanes}"


def _lane_broadcast(value: int, dtype: str, lanes: int):
    imm = tirx.IntImm(dtype, value)
    return imm if lanes == 1 else tirx.Broadcast(imm, lanes)


def _bf16_storage_dtype(dtype) -> str:
    return _dtype("uint16", tvm.DataType(dtype).lanes)


def _float32_dtype(dtype) -> str:
    return _dtype("float32", tvm.DataType(dtype).lanes)


def _uint32_dtype(dtype) -> str:
    return _dtype("uint32", tvm.DataType(dtype).lanes)


def _cast_to_float32(value):
    if str(value.dtype).startswith("float32"):
        return value
    return tirx.Cast(_float32_dtype(value.dtype), value)


def _bf16_bits_to_float32(value):
    lanes = tvm.DataType(value.dtype).lanes
    uint32_dtype = _uint32_dtype(value.dtype)
    shifted = tirx.Cast(uint32_dtype, value) << _lane_broadcast(16, "uint32", lanes)
    return tirx.reinterpret(_float32_dtype(value.dtype), shifted)


def _float32_to_bf16_bits(value):
    value = _cast_to_float32(value)
    lanes = tvm.DataType(value.dtype).lanes
    uint32_dtype = _uint32_dtype(value.dtype)
    bits = tirx.reinterpret(uint32_dtype, value)
    rounding_bias = ((bits >> _lane_broadcast(16, "uint32", lanes)) & _lane_broadcast(1, "uint32", lanes)) + _lane_broadcast(
        0x7FFF, "uint32", lanes
    )
    shifted = (bits + rounding_bias) >> _lane_broadcast(16, "uint32", lanes)
    return tirx.Cast(_dtype("uint16", lanes), shifted)


def _with_storage_dtype(buffer: Buffer, data: Var | None = None) -> Buffer:
    dtype = _bf16_storage_dtype(buffer.dtype)
    if data is None:
        data = Var(
            buffer.data.name,
            tvm.ir.PointerType(tvm.ir.PrimType(dtype), buffer.scope()),
            buffer.data.span,
        )
    return Buffer(
        data,
        dtype,
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
class LegalizeMetalBFloat16Mutator(PyStmtExprMutator):
    """Legalize Metal bf16 storage using TVM's bf16 conversion semantics.

    TVM's generic bf16 legalization has two phases: promote bf16 computation to
    fp32, then rewrite remaining bf16 storage to uint16. TileLang's Metal final
    device IR has already allocated shared/local buffers, so running the generic
    compute phase here can introduce promoted buffers without matching
    allocations. This pass keeps allocated bf16 storage as uint16 and applies
    TVM's u16<->fp32 conversion formulas at load/store and cast boundaries.
    """

    def __init__(self):
        super().__init__()
        self.buffer_remap: dict[Buffer, Buffer] = {}
        self.data_buffer_remap: dict[Var, Buffer] = {}
        self.var_remap: dict[Var, Var] = {}

    def remap_var_def(self, var: Var) -> Var:
        if var in self.var_remap:
            return self.var_remap[var]

        annotation = var.type_annotation
        if not isinstance(annotation, tvm.ir.PointerType):
            return var
        element_type = annotation.element_type
        if not isinstance(element_type, tvm.ir.PrimType):
            return var
        if not _is_bfloat16(element_type.dtype):
            return var

        new_var = Var(
            var.name,
            tvm.ir.PointerType(
                tvm.ir.PrimType(_bf16_storage_dtype(element_type.dtype)),
                annotation.storage_scope,
            ),
            var.span,
        )
        self.var_remap[var] = new_var
        return new_var

    def _remap_buffer(self, buffer: Buffer) -> Buffer:
        if buffer in self.buffer_remap:
            return self.buffer_remap[buffer]

        if buffer.data in self.data_buffer_remap:
            new_buffer = self.data_buffer_remap[buffer.data]
            self.buffer_remap[buffer] = new_buffer
            return new_buffer

        if _is_bfloat16(buffer.dtype):
            data = self.remap_var_def(buffer.data)
            if data.same_as(buffer.data):
                data = None
            new_buffer = _with_storage_dtype(buffer, data)
            self.buffer_remap[buffer] = new_buffer
            self.data_buffer_remap[buffer.data] = new_buffer
            self.var_remap[buffer.data] = new_buffer.data
            return new_buffer

        data = self.remap_var_def(buffer.data)
        if data.same_as(buffer.data):
            return buffer

        new_buffer = Buffer(
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
        self.buffer_remap[buffer] = new_buffer
        self.data_buffer_remap[buffer.data] = new_buffer
        return new_buffer

    def visit_var_(self, op: Var):
        return self.var_remap.get(op, op)

    def visit_alloc_buffer_(self, op: AllocBuffer):
        buffer = self._remap_buffer(op.buffer)
        return AllocBuffer(buffer, op.annotations, getattr(op, "span", None))

    def visit_decl_buffer_(self, op: DeclBuffer):
        buffer = self._remap_buffer(op.buffer)
        return DeclBuffer(buffer, getattr(op, "span", None))

    def visit_buffer_load_(self, op: BufferLoad):
        original_dtype = op.buffer.dtype
        buffer = self._remap_buffer(op.buffer)
        indices = [self.visit_expr(index) for index in op.indices]
        predicate = self.visit_expr(op.predicate) if op.predicate is not None else None
        load = BufferLoad(buffer, indices, predicate, getattr(op, "span", None))
        return _bf16_bits_to_float32(load) if _is_bfloat16(original_dtype) else load

    def visit_buffer_store_(self, op: BufferStore):
        original_dtype = op.buffer.dtype
        buffer = self._remap_buffer(op.buffer)
        value = self.visit_expr(op.value)
        if _is_bfloat16(original_dtype) and not str(value.dtype).startswith("uint16"):
            value = _float32_to_bf16_bits(value)
        indices = [self.visit_expr(index) for index in op.indices]
        predicate = self.visit_expr(op.predicate) if op.predicate is not None else None
        return BufferStore(buffer, value, indices, predicate, getattr(op, "span", None))

    def visit_cast_(self, op: Cast):
        value = self.visit_expr(op.value)
        if _is_bfloat16(op.dtype):
            return _float32_to_bf16_bits(value)
        if str(op.dtype) == str(value.dtype):
            return value
        if _is_bfloat16(value.dtype):
            value = _bf16_bits_to_float32(value)
        return Cast(op.dtype, value, getattr(op, "span", None))

    def visit_attr_stmt_(self, op: AttrStmt):
        node = op.node
        if isinstance(node, Buffer):
            node = self._remap_buffer(node)
        elif isinstance(node, Var):
            node = self.var_remap.get(node, node)
        return AttrStmt(node, op.attr_key, self.visit_expr(op.value), self.visit_stmt(op.body), getattr(op, "span", None))

    def visit_prim_func(self, func: PrimFunc) -> PrimFunc:
        params = [self.remap_var_def(param) for param in func.params]
        body = self.visit_stmt(func.body)
        buffer_map = {
            self.var_remap.get(var, var): self._remap_buffer(buffer)
            for var, buffer in func.buffer_map.items()
        }
        return PrimFunc(params, body, func.ret_type, buffer_map, func.attrs, getattr(func, "span", None))


def LegalizeMetalBFloat16():
    def pass_fn(func, mod, ctx):
        mutator = LegalizeMetalBFloat16Mutator()
        return mutator.visit_prim_func(func)

    return prim_func_pass(pass_fn, opt_level=0)
