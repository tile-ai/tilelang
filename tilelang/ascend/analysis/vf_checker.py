"""Check Ascend VF memory and capture rules on the native source IR."""

from __future__ import annotations

from contextlib import contextmanager

import tvm_ffi
from tvm_tirx import ir, tirx

from tilelang.analysis._visitor import NativeStmtVisitor, tensor_scope, tilelang_metadata

_VF_BLOCK_NAMES = ("SIMD_VF", "SIMT_VF")


def _vf_kind(node):
    if not node.op.same_as(ir.Op.get("tl.ascend_block")):
        return None
    name = str(node.attrs["name"])
    return name if name in _VF_BLOCK_NAMES else None


class _VFCheckVisitor(NativeStmtVisitor):
    def __init__(self):
        super().__init__()
        self.vf_scope_stack = []
        self.allocation_owners = {}
        self.storage_roots = {}
        self.block_annotations = {}

    @property
    def inside_vf(self):
        return bool(self.vf_scope_stack)

    @property
    def inside_simd_vf(self):
        return any(kind == "SIMD_VF" for _, kind in self.vf_scope_stack)

    @contextmanager
    def scope(self):
        annotations = self.block_annotations
        self.block_annotations = annotations.copy()
        try:
            with super().scope():
                yield
        finally:
            self.block_annotations = annotations

    def _roots(self, expression):
        """Follow tensor/pointer Bind aliases without confusing equal names."""
        if isinstance(expression, ir.Var):
            if isinstance(expression.ty, (tirx.TensorType, ir.PtrType)):
                return self.storage_roots.get(expression, {expression})
            return set()
        roots = set()

        def variable(value, _visitor):
            roots.update(self._roots(value))

        tvm_ffi.structural_visit(expression, (ir.Var, variable))
        return roots

    def _scope_for(self, tensor):
        scopes = {tensor_scope(tensor)}
        for root in self._roots(tensor):
            if isinstance(root.ty, tirx.TensorType):
                scopes.add(tensor_scope(root))
            elif isinstance(root.ty, ir.PtrType):
                scopes.add(str(root.ty.scope))
        return scopes

    def visit_region(self, node, visitor):
        kind = _vf_kind(node)
        # Region arguments are evaluated outside the region's lexical body.
        visitor.visit(node.args)
        visitor.visit(node.attrs)
        pushed = kind is not None
        if pushed:
            self.vf_scope_stack.append((node, kind))
        try:
            with self.scope():
                if node.op.same_as(ir.Op.get("tl.ascend_block")) or node.op.same_as(ir.Op.get("tl.lexical_scope")):
                    self.block_annotations.update(tvm_ffi.get_global_func("tl.CollectSourceBlockAnnotations")(node.body))
                for parameter in node.body_params:
                    self.bindings.pop(parameter, None)
                visitor.visit(node.body)
        finally:
            if pushed:
                self.vf_scope_stack.pop()

    def visit_bind(self, node, visitor):
        super().visit_bind(node, visitor)
        if not isinstance(node.var.ty, (tirx.TensorType, ir.PtrType)):
            return
        allocation = isinstance(node.value, ir.Call) and isinstance(node.value.op, ir.Op) and node.value.op.name == "tirx.alloc_tensor"
        roots = {node.var} if allocation else self._roots(node.value)
        self.storage_roots[node.var] = roots or {node.var}
        if allocation and self.inside_vf:
            self.allocation_owners[node.var] = self.vf_scope_stack[-1]

    def _check_allocation_scope(self, tensor):
        for root in self._roots(tensor):
            owner_info = self.allocation_owners.get(root)
            if owner_info is None:
                continue
            owner, kind = owner_info
            if any(owner.same_as(scope) for scope, _ in self.vf_scope_stack):
                continue
            raise ValueError(
                "[TileLang Semantic Check] "
                f"Buffer '{root.name}' is allocated inside VF block "
                f"'{kind}' but accessed outside its allocation scope. "
                "Move the allocation before entering the VF block."
            )

    def _check_local_var_write(self, tensor):
        if not self.inside_vf or "local.var" not in self._scope_for(tensor):
            return
        current = self.vf_scope_stack[-1][0]
        for root in self._roots(tensor):
            owner = self.allocation_owners.get(root)
            if owner is None or not owner[0].same_as(current):
                raise ValueError(
                    "[TileLang Semantic Check] "
                    f"local.var buffer '{tensor}' is modified "
                    "inside a SIMD_VF/SIMT_VF block but is defined outside the VF block. "
                    "This is not allowed because the variable would need pointer-type capture."
                )

    def _check_access(self, tensor, *, write=False):
        self._check_allocation_scope(tensor)
        if write:
            self._check_local_var_write(tensor)
        scopes = self._scope_for(tensor)
        if self.inside_vf and scopes.intersection({"shared.ss", "shared.ss.dyn"}):
            # SS is scalar-only in the Ascend codegen contract. Diagnose the
            # invalid VF access before scheduling can hide it behind a generic
            # cross-subcore conflict; aliases retain the original SS owner.
            raise RuntimeError("SS supports scalar uint32/uint64 accesses outside VF blocks only")
        if self.inside_simd_vf and "global" in scopes:
            action = "TensorStore to" if write else "TensorLoad from"
            raise ValueError(
                "[TileLang Semantic Check] "
                f"{action} global buffer '{getattr(tensor, 'name', tensor)}' inside a SIMD_VF block "
                "is not allowed. SIMD_VF blocks cannot access global memory — move the "
                "access outside the SIMD_VF block."
            )

    def visit_for(self, node, visitor):
        if node.kind == ir.ForKind.PARALLEL and not self.inside_vf:
            raise ValueError(
                "[TileLang Semantic Check] "
                f"T.Parallel loop '{node.var.name}' must be inside a VF block "
                "(T.SimdVF() or T.SimtVF()). "
                "Parallel loops outside VF blocks are not supported on Ascend NPU."
            )
        super().visit_for(node, visitor)

    def _check_copy(self, call):
        source, destination = call.args[:2]
        if not isinstance(source, ir.TensorRegion) or not isinstance(destination, ir.TensorRegion):
            raise TypeError("Ascend copy operands must be native TensorRegion objects")
        source, destination = source.source, destination.source
        if not self.inside_vf and source.ty.dtype != destination.ty.dtype and tensor_scope(source) != "shared.l0c":
            raise ValueError(
                "[TileLang Semantic Check] "
                f"T.copy() from dtype '{source.ty.dtype}' to dtype '{destination.ty.dtype}' "
                "outside a VF block is not allowed. "
                "DMA copies cannot perform type casting — perform the cast "
                "inside a VF block (T.SimdVF() or T.SimtVF()) instead."
            )

    def visit_call(self, node, visitor):
        name = node.op.name if isinstance(node.op, ir.Op) else None
        if name is not None and node.op.get_attr("TLOpBuilder") is not None:
            if name in {"tl.tileop.copy", "tl.tileop.ascend_copy"}:
                self._check_copy(node)
            # Read/write roles belong to each operator implementation. Native
            # TensorRegion contains coordinates, not the old tl.region rw_mask.
            regions = tvm_ffi.get_global_func("tl.GetTileOpAccessRegions")(node, self.block_annotations)
            for region in regions["reads"]:
                self._check_access(region.source)
            for region in regions["writes"]:
                self._check_access(region.source, write=True)
        elif name == "tl.access_ptr":
            load, _, access_mask = node.args
            self._check_access(load.source, write=bool(int(access_mask) & 2))
        elif (metadata := tilelang_metadata(node)) is not None:
            self.block_annotations.update(metadata)
        super().visit_call(node, visitor)

    def visit_tensor_store(self, node, visitor):
        self._check_access(node.dest, write=True)
        if not self.inside_vf and isinstance(node.value, ir.TensorLoad) and node.value.source.ty.dtype != node.dest.ty.dtype:
            raise ValueError(
                "[TileLang Semantic Check] "
                f"Buffer store from dtype '{node.value.source.ty.dtype}' to dtype '{node.dest.ty.dtype}' "
                "outside a VF block implies a type cast. "
                "DMA copies cannot perform type casting — perform the cast "
                "inside a VF block (T.SimdVF() or T.SimtVF()) instead."
            )
        super().visit_tensor_store(node, visitor)

    def visit_tensor_load(self, node, visitor):
        self._check_access(node.source)
        super().visit_tensor_load(node, visitor)


def VFChecker():
    """Check VF loops, casts, global/SS access, lifetime and scalar capture.

    The pass reads native lexical regions and allocation Bind statements and
    returns the input Function unchanged. Tensor/pointer aliases retain the
    owning allocation's identity and lifetime.
    """

    def pass_fn(func, _mod, _ctx):
        _VFCheckVisitor().visit_stmt(func.body)
        return func

    return tirx.transform.function_pass(pass_fn, opt_level=0, name="tl.VFChecker")
