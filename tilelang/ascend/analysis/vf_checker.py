from __future__ import annotations

from tvm import tirx
from tvm.tirx import BufferLoad, BufferStore, For, PrimFunc, PyStmtExprVisitor
from tvm.tirx.transform import prim_func_pass

_VF_BLOCK_NAMES = ("SIMD_VF", "SIMT_VF")
_REGION_WRITE_ACCESS_MASK = 2


@tirx.functor.visitor
class _VFCheckVisitor(PyStmtExprVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.inside_vf = False
        self.inside_simd_vf = False
        self.vf_scope_stack: list[tirx.SBlock] = []
        self.vf_alloc_owners: dict[tirx.Var, tuple[tirx.SBlock, str]] = {}
        self.vf_local_vars: set[tirx.Var] = set()

    def visit_sblock_(self, op: tirx.SBlock) -> None:
        if op.name_hint in _VF_BLOCK_NAMES:
            was_inside = self.inside_vf
            was_simd = self.inside_simd_vf
            saved_local_vars = self.vf_local_vars
            self.vf_local_vars = set()
            for buffer in op.alloc_buffers:
                self.vf_alloc_owners[buffer.data] = (op, buffer.name)
                if buffer.scope() == "local.var":
                    self.vf_local_vars.add(buffer.data)
            self.inside_vf = True
            self.inside_simd_vf = op.name_hint == "SIMD_VF"
            self.vf_scope_stack.append(op)
            try:
                self.visit_stmt(op.body)
            finally:
                self.vf_scope_stack.pop()
                self.inside_vf = was_inside
                self.inside_simd_vf = was_simd
                self.vf_local_vars = saved_local_vars
        else:
            if self.inside_vf:
                owner = self.vf_scope_stack[-1]
                for buffer in op.alloc_buffers:
                    self.vf_alloc_owners[buffer.data] = (owner, buffer.name)
                    if buffer.scope() == "local.var":
                        self.vf_local_vars.add(buffer.data)
            super().visit_sblock_(op)

    def visit_alloc_buffer_(self, op: tirx.AllocBuffer) -> None:
        if self.inside_vf:
            owner = self.vf_scope_stack[-1]
            self.vf_alloc_owners[op.buffer.data] = (owner, op.buffer.name)
            if op.buffer.scope() == "local.var":
                self.vf_local_vars.add(op.buffer.data)

    def _check_vf_allocation_scope(self, buffer: tirx.Buffer) -> None:
        owner_info = self.vf_alloc_owners.get(buffer.data)
        if owner_info is None:
            return
        owner, name = owner_info
        if any(owner.same_as(scope) for scope in self.vf_scope_stack):
            return
        raise ValueError(
            "[TileLang Semantic Check] "
            f"Buffer '{name}' is allocated inside VF block "
            f"'{owner.name_hint}' but accessed outside its allocation scope. "
            "Move the allocation before entering the VF block."
        )

    def _check_vf_local_var_write(self, buffer: tirx.Buffer) -> None:
        if self.inside_vf and buffer.scope() == "local.var" and buffer.data not in self.vf_local_vars:
            raise ValueError(
                "[TileLang Semantic Check] "
                f"local.var buffer '{buffer}' is modified "
                "inside a SIMD_VF/SIMT_VF block but is defined outside the VF block. "
                "This is not allowed because the variable would need pointer-type capture."
            )

    def visit_for_(self, op: For) -> None:
        if op.kind == tirx.ForKind.PARALLEL and not self.inside_vf:
            raise ValueError(
                "[TileLang Semantic Check] "
                f"T.Parallel loop '{op.loop_var.name}' must be inside a VF block "
                "(T.SimdVF() or T.SimtVF()). "
                "Parallel loops outside VF blocks are not supported on Ascend NPU."
            )
        super().visit_for_(op)

    def _check_copy(self, call: tirx.Call) -> None:
        src_region = call.args[0]
        dst_region = call.args[1]
        if not isinstance(src_region, tirx.Call) or not isinstance(dst_region, tirx.Call):
            return
        src_buf = src_region.args[0].buffer
        dst_buf = dst_region.args[0].buffer
        self._check_vf_allocation_scope(src_buf)
        self._check_vf_allocation_scope(dst_buf)

        if not self.inside_vf and src_buf.dtype != dst_buf.dtype and src_buf.scope() != "shared.l0c":
            raise ValueError(
                "[TileLang Semantic Check] "
                f"T.copy() from dtype '{src_buf.dtype}' to dtype '{dst_buf.dtype}' "
                "outside a VF block is not allowed. "
                "DMA copies cannot perform type casting — perform the cast "
                "inside a VF block (T.SimdVF() or T.SimtVF()) instead."
            )

        if self.inside_simd_vf and (src_buf.scope() == "global" or dst_buf.scope() == "global"):
            raise ValueError(
                "[TileLang Semantic Check] "
                f"T.copy() involving global buffer '{src_buf.name}' -> '{dst_buf.name}' "
                "inside a SIMD_VF block is not allowed. "
                "SIMD_VF blocks cannot access global memory — move the copy "
                "outside the SIMD_VF block."
            )

    def visit_evaluate_(self, op: tirx.Evaluate) -> None:
        call = op.value
        if isinstance(call, tirx.Call) and call.op == tirx.op.Op.get("tl.tileop.copy"):
            self._check_copy(call)
        super().visit_evaluate_(op)

    def visit_call_(self, op: tirx.Call) -> None:
        if op.op.same_as(tirx.op.Op.get("tl.region")) and len(op.args) >= 2:
            buffer_load = op.args[0]
            access_mask = op.args[1]
            if (
                isinstance(buffer_load, BufferLoad)
                and isinstance(access_mask, tirx.IntImm)
                and access_mask.value & _REGION_WRITE_ACCESS_MASK
            ):
                self._check_vf_local_var_write(buffer_load.buffer)
        super().visit_call_(op)

    def visit_buffer_store_(self, op: BufferStore) -> None:
        self._check_vf_allocation_scope(op.buffer)
        self._check_vf_local_var_write(op.buffer)
        if not self.inside_vf:
            value = op.value
            if isinstance(value, BufferLoad) and value.buffer.dtype != op.buffer.dtype:
                raise ValueError(
                    "[TileLang Semantic Check] "
                    f"Buffer store from dtype '{value.buffer.dtype}' to dtype '{op.buffer.dtype}' "
                    "outside a VF block implies a type cast. "
                    "DMA copies cannot perform type casting — perform the cast "
                    "inside a VF block (T.SimdVF() or T.SimtVF()) instead."
                )

        if self.inside_simd_vf and op.buffer.scope() == "global":
            raise ValueError(
                "[TileLang Semantic Check] "
                f"BufferStore to global buffer '{op.buffer.name}' inside a SIMD_VF block "
                "is not allowed. SIMD_VF blocks cannot access global memory — move the "
                "store outside the SIMD_VF block."
            )
        super().visit_buffer_store_(op)

    def visit_buffer_load_(self, op: BufferLoad) -> None:
        self._check_vf_allocation_scope(op.buffer)
        if self.inside_simd_vf and op.buffer.scope() == "global":
            raise ValueError(
                "[TileLang Semantic Check] "
                f"BufferLoad from global buffer '{op.buffer.name}' inside a SIMD_VF block "
                "is not allowed. SIMD_VF blocks cannot access global memory — move the "
                "load outside the SIMD_VF block."
            )
        super().visit_buffer_load_(op)


def VFChecker():
    """Check VF block usage rules:
    - All T.Parallel loops must be inside SIMD_VF or SIMT_VF blocks.
    - Copy operations outside VF blocks must have matching source and destination dtypes.
    - SIMD_VF blocks must not access global memory.
    - Buffers allocated inside a VF block must not escape that VF scope.
    - local.var buffers allocated outside a VF block must not be modified inside it.
    """

    def pass_fn(func: PrimFunc, mod, ctx):
        _VFCheckVisitor().visit_stmt(func.body)
        return func

    return prim_func_pass(pass_fn, opt_level=0)
