from __future__ import annotations

from tvm_tirx import ir, tirx
from tvm_tirx.tirx.transform import function_pass
from tilelang._compat.functor import PyStmtExprVisitor

_VF_BLOCK_NAMES = ("SIMD_VF", "SIMT_VF")
_REGION_WRITE_ACCESS_MASK = 2


class _VFCheckVisitor(PyStmtExprVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.inside_vf = False
        self.inside_simd_vf = False
        self.vf_scope_stack: list[ir.RegionStmt] = []
        self.vf_alloc_owners: dict[ir.Var, tuple[ir.RegionStmt, str]] = {}
        self.vf_local_vars: set[ir.Var] = set()
        self.definitions: dict[ir.Var, ir.Expr] = {}

    def _storage(self, tensor):
        storage = tensor
        expression = tensor
        visited = set()
        while True:
            if isinstance(expression, ir.Var):
                if expression in visited:
                    return storage
                visited.add(expression)
                storage = expression
                expression = self.definitions.get(expression)
            elif isinstance(expression, (ir.TensorLoad, ir.TensorRegion)):
                expression = expression.source
            elif isinstance(expression, ir.Call) and expression.op.name in (
                "tirx.decl_tensor",
                "tirx.tensor_data_ptr",
                "tirx.ptr_byte_offset",
                "tirx.reinterpret",
                "tirx.address_of",
                "tl.access_ptr",
            ):
                expression = expression.args[0]
            else:
                return storage

    def visit_region_stmt_(self, op: ir.RegionStmt) -> None:
        name = op.attrs.get("name_hint")
        if name not in _VF_BLOCK_NAMES:
            super().visit_region_stmt_(op)
            return
        was_inside, was_simd = self.inside_vf, self.inside_simd_vf
        saved_local_vars = self.vf_local_vars
        self.vf_local_vars = set()
        self.inside_vf, self.inside_simd_vf = True, name == "SIMD_VF"
        self.vf_scope_stack.append(op)
        try:
            self.visit_stmt(op.body)
        finally:
            self.vf_scope_stack.pop()
            self.inside_vf, self.inside_simd_vf = was_inside, was_simd
            self.vf_local_vars = saved_local_vars

    def visit_bind_(self, op: ir.Bind) -> None:
        self.definitions[op.var] = op.value
        if self.inside_vf and isinstance(op.value, ir.Call) and op.value.op.name == "tirx.alloc_tensor":
            self.vf_alloc_owners[op.var] = (self.vf_scope_stack[-1], op.var.name)
            if op.var.ty.scope.name == "local.var":
                self.vf_local_vars.add(op.var)
        super().visit_bind_(op)

    def _check_vf_allocation_scope(self, tensor: ir.Var) -> None:
        owner_info = self.vf_alloc_owners.get(self._storage(tensor))
        if owner_info is None:
            return
        owner, name = owner_info
        if any(owner.same_as(scope) for scope in self.vf_scope_stack):
            return
        raise ValueError(
            "[TileLang Semantic Check] "
            f"Buffer '{name}' is allocated inside VF block '{owner.attrs['name_hint']}' but accessed outside its allocation scope. "
            "Move the allocation before entering the VF block."
        )

    def _check_vf_local_var_write(self, tensor: ir.Var) -> None:
        if self.inside_vf and tensor.ty.scope.name == "local.var" and self._storage(tensor) not in self.vf_local_vars:
            raise ValueError(
                "[TileLang Semantic Check] "
                f"local.var buffer '{tensor}' is modified inside a SIMD_VF/SIMT_VF block but is defined outside the VF block. "
                "This is not allowed because the variable would need pointer-type capture."
            )

    def _check_global_access(self, tensor: ir.Var) -> None:
        if self.inside_simd_vf and tensor.ty.scope.name == "global":
            raise ValueError(
                "[TileLang Semantic Check] "
                f"SIMD_VF blocks cannot access global memory: buffer '{tensor.name}'. Move the access outside the SIMD_VF block."
            )

    def visit_for_(self, op: ir.For) -> None:
        if op.kind == ir.ForKind.PARALLEL and not self.inside_vf:
            raise ValueError(
                "[TileLang Semantic Check] "
                f"T.Parallel loop '{op.var.name}' must be inside a VF block (T.SimdVF() or T.SimtVF()). "
                "Parallel loops outside VF blocks are not supported on Ascend NPU."
            )
        super().visit_for_(op)

    @staticmethod
    def _region_tensor(expression):
        while isinstance(expression, ir.Call) and expression.op.name == "tl.region":
            expression = expression.args[0]
        return expression.source if isinstance(expression, (ir.TensorRegion, ir.TensorLoad)) else None

    def _check_copy(self, call: ir.Call) -> None:
        source, destination = [self._region_tensor(argument) for argument in call.args[:2]]
        if source is None or destination is None:
            return
        for tensor in (source, destination):
            self._check_vf_allocation_scope(tensor)
            self._check_global_access(tensor)
        self._check_vf_local_var_write(destination)
        if not self.inside_vf and source.ty.dtype != destination.ty.dtype and source.ty.scope.name != "shared.l0c":
            raise ValueError(
                "[TileLang Semantic Check] "
                f"T.copy() from dtype '{source.ty.dtype}' to dtype '{destination.ty.dtype}' outside a VF block is not allowed. "
                "DMA copies cannot perform type casting — perform the cast inside a VF block (T.SimdVF() or T.SimtVF()) instead."
            )

    def visit_call_(self, op: ir.Call) -> None:
        if isinstance(op.op, ir.Op):
            if op.op.name in ("tl.tileop.ascend_copy", "tl.tileop.copy"):
                self._check_copy(op)
            elif op.op.name == "tl.region" and len(op.args) >= 2:
                tensor = self._region_tensor(op)
                mask = op.args[1]
                if tensor is not None and isinstance(mask, ir.IntImm) and int(mask) & _REGION_WRITE_ACCESS_MASK:
                    self._check_vf_local_var_write(tensor)
            elif op.op.name == "tl.tileop.fill":
                tensor = self._region_tensor(op.args[0])
                if tensor is not None:
                    self._check_vf_local_var_write(tensor)
        super().visit_call_(op)

    def visit_tensor_store_(self, op: ir.TensorStore) -> None:
        self._check_vf_allocation_scope(op.dest)
        self._check_vf_local_var_write(op.dest)
        self._check_global_access(op.dest)
        if not self.inside_vf and isinstance(op.value, ir.TensorLoad) and op.value.source.ty.dtype != op.dest.ty.dtype:
            raise ValueError(
                "[TileLang Semantic Check] "
                f"Buffer store from dtype '{op.value.source.ty.dtype}' to dtype '{op.dest.ty.dtype}' outside a VF block implies a type cast. "
                "DMA copies cannot perform type casting — perform the cast inside a VF block (T.SimdVF() or T.SimtVF()) instead."
            )
        super().visit_tensor_store_(op)

    def visit_tensor_load_(self, op: ir.TensorLoad) -> None:
        self._check_vf_allocation_scope(op.source)
        self._check_global_access(op.source)
        super().visit_tensor_load_(op)

    def visit_tensor_region_(self, op: ir.TensorRegion) -> None:
        self._check_vf_allocation_scope(op.source)
        self._check_global_access(op.source)
        super().visit_tensor_region_(op)


def VFChecker():
    """Check native VF scopes, memory accesses, allocation lifetimes, and scalar captures."""

    def pass_fn(func: tirx.Function, mod, ctx):
        if func.body is not None:
            _VFCheckVisitor().visit_stmt(func.body)
        return func

    return function_pass(pass_fn, opt_level=0)
