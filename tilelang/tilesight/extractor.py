"""Extract TileSight's structured graph from TileLang TIR.

The extractor intentionally walks the TIR object model instead of parsing
``mod.script()`` text. This keeps TileSight resilient to TVMScript printer
changes and lets it recognize rewritten high-level tile ops such as
``tl.tileop.tma_copy`` after warp specialization.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from tvm import DataType, IRModule, ir, tirx
from tvm.tirx import PyStmtExprVisitor

from .graph import BufferInfo, DataEdge, ExprInfo, KernelGraph, KernelNode, LoopInfo, RegionInfo, TileOpNode


_COPY_OP_NAMES = {
    "tl.tileop.copy",
    "tl.tileop.async_copy",
    "tl.tileop.tma_copy",
    "tl.tileop.copy_cluster",
}
_LAYOUT_COPY_OP_NAMES = {
    "tl.tileop.transpose",
    "tl.tileop.im2col",
}
_SPARSE_GEMM_OP_NAMES = {
    "tl.tileop.gemm_sp",
    "tl.tileop.wgmma_gemm_sp",
    "tl.tileop.tcgen05_gemm_sp",
}
_GEMM_OP_NAMES = {
    "tl.tileop.gemm",
    "tl.tileop.wgmma_gemm",
    "tl.tileop.tcgen05_gemm",
} | _SPARSE_GEMM_OP_NAMES
_FILL_OP_NAMES = {"tl.tileop.fill"}
_REDUCE_OP_NAMES = {
    "tl.tileop.reduce",
    "tl.tileop.finalize_reducer",
}
_SCAN_OP_NAMES = {
    "tl.tileop.cumsum",
    "tl.tileop.cummax",
}
_ATOMIC_OP_NAMES = {
    "tl.tileop.atomicadd",
    "tl.tileop.atomicmax",
    "tl.tileop.atomicmin",
}
_TILE_OP_NAMES = (
    _COPY_OP_NAMES
    | _LAYOUT_COPY_OP_NAMES
    | _GEMM_OP_NAMES
    | _FILL_OP_NAMES
    | _REDUCE_OP_NAMES
    | _SCAN_OP_NAMES
    | _ATOMIC_OP_NAMES
)
_REGION_OP_NAMES = {"tl.tileop.region", "tl.region"}
_SFU_CALL_TOKENS = (
    "sqrt",
    "rsqrt",
    "exp",
    "tanh",
    "log",
    "sin",
    "cos",
    "tan",
)


def extract_kernel_graph(mod: IRModule) -> KernelGraph:
    extractor = _TIRGraphExtractor()
    return extractor.extract(mod)


@tirx.functor.visitor
class _TIRGraphExtractor(PyStmtExprVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.graph = KernelGraph()
        self.kernel: KernelNode | None = None
        self._buffer_ids: dict[Any, str] = {}
        self._buffer_count = 0
        self._op_count = 0
        self._loop_count = 0
        self._loop_stack: list[str] = []
        self._last_writer: dict[str, str] = {}
        self._bound_exprs: dict[str, tuple[str, list[RegionInfo]]] = {}
        self._extra_input_edges: list[tuple[str, str, str]] = []

    def extract(self, mod: IRModule) -> KernelGraph:
        for global_var, func in mod.functions.items():
            if not isinstance(func, tirx.PrimFunc):
                continue
            self._start_kernel(_func_name(global_var, func))
            self._record_primfunc_buffers(func)
            self._record_primfunc_attrs(func)
            self.visit_stmt(func.body)
            self._finish_kernel()
        return self.graph

    def _start_kernel(self, name: str) -> None:
        self.kernel = KernelNode(name=name)
        self._buffer_ids = {}
        self._buffer_count = 0
        self._op_count = 0
        self._loop_count = 0
        self._loop_stack = []
        self._last_writer = {}
        self._bound_exprs = {}
        self._extra_input_edges = []

    def _finish_kernel(self) -> None:
        if self.kernel is not None and (self.kernel.ops or self.kernel.grid or self.kernel.threads):
            self.graph.kernels.append(self.kernel)
        self.kernel = None

    def _record_primfunc_buffers(self, func: tirx.PrimFunc) -> None:
        for buffer in getattr(func, "buffer_map", {}).values():
            self._record_buffer(buffer, is_param=True)

    def _record_primfunc_attrs(self, func: tirx.PrimFunc) -> None:
        assert self.kernel is not None
        attrs = _mapping_to_python(getattr(func, "attrs", None))
        self.kernel.attrs.update(attrs)
        thread_extent = attrs.get("thread_extent")
        if isinstance(thread_extent, dict):
            for tag, extent in thread_extent.items():
                self._record_launch(str(tag), extent)

    def visit_attr_stmt_(self, op) -> None:
        if str(op.attr_key) == "thread_extent":
            tag = _thread_tag(op.node)
            if tag:
                self._record_launch(tag, op.value)
        super().visit_attr_stmt_(op)

    def visit_decl_buffer_(self, op) -> None:
        buffer = getattr(op, "buffer", None)
        if buffer is not None:
            self._record_buffer(buffer, is_param=False)
        super().visit_decl_buffer_(op)

    def visit_alloc_buffer_(self, op) -> None:
        buffer = getattr(op, "buffer", None)
        if buffer is not None:
            self._record_buffer(buffer, is_param=False)
        super().visit_alloc_buffer_(op)

    def visit_for_(self, op) -> None:
        loop_id = self._enter_loop(op)
        self._loop_stack.append(loop_id)
        saved_bound_exprs = dict(self._bound_exprs)
        super().visit_for_(op)
        self._bound_exprs = saved_bound_exprs
        self._loop_stack.pop()

    def visit_if_then_else_(self, op) -> None:
        condition = getattr(op, "condition", None)
        if condition is not None:
            self.visit_expr(condition)
        saved_bound_exprs = dict(self._bound_exprs)
        then_case = getattr(op, "then_case", None)
        if then_case is not None:
            self._bound_exprs = dict(saved_bound_exprs)
            self.visit_stmt(then_case)
        else_case = getattr(op, "else_case", None)
        if else_case is not None:
            self._bound_exprs = dict(saved_bound_exprs)
            self.visit_stmt(else_case)
        self._bound_exprs = saved_bound_exprs

    def visit_let_stmt_(self, op) -> None:
        self._record_bound_expr(getattr(op, "var", None), getattr(op, "value", None), "tir.let")
        body = getattr(op, "body", None)
        if body is not None:
            self.visit_stmt(body)

    def visit_bind_(self, op) -> None:
        self._record_bound_expr(getattr(op, "var", None), getattr(op, "value", None), "tir.bind")

    def visit_evaluate_(self, op) -> None:
        value = getattr(op, "value", None)
        if isinstance(value, tirx.Call) and self._maybe_record_tile_op(value):
            return
        super().visit_evaluate_(op)

    def visit_buffer_store_(self, op) -> None:
        self._record_buffer_store_op(op)
        super().visit_buffer_store_(op)

    def visit_call_(self, op) -> None:
        self._maybe_record_tile_op(op)
        super().visit_call_(op)

    def _record_launch(self, tag: str, extent: Any) -> None:
        assert self.kernel is not None
        info = _expr_info(extent)
        if tag.startswith("blockIdx."):
            self.kernel.grid[tag] = info
        elif tag.startswith("threadIdx."):
            self.kernel.threads[tag] = info

    def _record_buffer(self, buffer, is_param: bool) -> str:
        assert self.kernel is not None
        key = _buffer_key(buffer)
        if key in self._buffer_ids:
            buffer_id = self._buffer_ids[key]
            if is_param:
                self.kernel.buffers[buffer_id].is_param = True
            return buffer_id

        buffer_id = f"b{self._buffer_count}"
        self._buffer_count += 1
        self._buffer_ids[key] = buffer_id

        dtype = str(getattr(buffer, "dtype", "float32"))
        bytes_per_element = _dtype_bytes(dtype)
        shape = [_expr_info(dim) for dim in (getattr(buffer, "shape", None) or [])]
        strides = [_expr_info(stride) for stride in (getattr(buffer, "strides", None) or [])]
        static_elems = _product_static(shape)
        static_bytes = static_elems * bytes_per_element if static_elems is not None and bytes_per_element is not None else None
        self.kernel.buffers[buffer_id] = BufferInfo(
            id=buffer_id,
            name=_buffer_name(buffer),
            scope=_buffer_scope(buffer),
            dtype=dtype,
            shape=shape,
            strides=strides,
            bytes_per_element=bytes_per_element,
            static_bytes=static_bytes,
            is_param=is_param,
        )
        return buffer_id

    def _enter_loop(self, op) -> str:
        assert self.kernel is not None
        loop_id = f"l{self._loop_count}"
        self._loop_count += 1
        var = getattr(getattr(op, "loop_var", None), "name", None) or _expr_text(getattr(op, "loop_var", ""))
        annotations = _mapping_to_python(getattr(op, "annotations", None))
        self.kernel.loops[loop_id] = LoopInfo(
            id=loop_id,
            var=str(var),
            extent=_expr_info(getattr(op, "extent", None)),
            kind=_loop_kind(op),
            annotations=annotations,
            parent=self._loop_stack[-1] if self._loop_stack else None,
        )
        return loop_id

    def _maybe_record_tile_op(self, call: tirx.Call) -> bool:
        op_name = _op_name(call)
        if op_name in _COPY_OP_NAMES or op_name in _LAYOUT_COPY_OP_NAMES:
            self._record_tile_op("copy", op_name, call)
            return True
        if op_name in _GEMM_OP_NAMES:
            self._record_tile_op("gemm", op_name, call)
            return True
        if op_name in _FILL_OP_NAMES:
            self._record_tile_op("fill", op_name, call)
            return True
        if op_name in _REDUCE_OP_NAMES:
            self._record_tile_op("reduce", op_name, call)
            return True
        if op_name in _SCAN_OP_NAMES:
            self._record_tile_op("scan", op_name, call)
            return True
        if op_name in _ATOMIC_OP_NAMES:
            self._record_tile_op("atomic", op_name, call)
            return True
        return False

    def _record_tile_op(self, kind: str, op_name: str, call: tirx.Call) -> None:
        regions = [region for region in (_region_info(arg, self) for arg in call.args) if region is not None]

        math_shape: dict[str, ExprInfo] = {}
        static_flops = None
        if kind == "gemm":
            shape_offset = 7 if op_name in _SPARSE_GEMM_OP_NAMES else 5
            if len(call.args) >= shape_offset + 3:
                math_shape = {
                    "M": _expr_info(call.args[shape_offset]),
                    "N": _expr_info(call.args[shape_offset + 1]),
                    "K": _expr_info(call.args[shape_offset + 2]),
                }
                m = _static_number(math_shape["M"])
                n = _static_number(math_shape["N"])
                k = _static_number(math_shape["K"])
                if m is not None and n is not None and k is not None:
                    dense_factor = 1.0 if op_name in _SPARSE_GEMM_OP_NAMES else 2.0
                    static_flops = dense_factor * m * n * k
        elif kind in ("reduce", "scan", "atomic"):
            static_flops = _static_elements_from_regions(regions)
        if kind == "atomic":
            for region in regions:
                if region.access == "w":
                    region.access = "rw"

        annotations = _mapping_to_python(getattr(call, "annotations", None))
        annotations.update(_tile_op_annotations(kind, call))
        self._append_op(
            kind=kind,
            op_name=op_name,
            regions=regions,
            annotations=annotations,
            static_flops=static_flops,
            math_shape=math_shape,
        )

    def _record_buffer_store_op(self, op) -> None:
        value = getattr(op, "value", None)
        expr_info = _StoreExprAnalyzer(self).analyze(value)
        self._extra_input_edges.extend(expr_info.bound_sources)
        output_region = _buffer_store_region_info(op, "w", self)
        regions = expr_info.regions + [output_region]
        kind = _classify_store_op(output_region, expr_info)
        static_flops = _store_static_flops(output_region, expr_info, kind)
        op_name = f"tir.store.{kind}"
        annotations = {
            "tir_store_value": _expr_text(value),
            "simt_ops_per_element": expr_info.simt_ops_per_element,
            "sfu_ops_per_element": expr_info.sfu_ops_per_element,
            "explicit_reduction_ops_per_element": expr_info.explicit_reduction_ops_per_element,
        }
        if expr_info.call_names:
            annotations["calls"] = sorted(expr_info.call_names)
        if expr_info.has_self_reduction(output_region):
            annotations["self_reduction"] = True

        self._append_op(
            kind=kind,
            op_name=op_name,
            regions=regions,
            annotations=annotations,
            static_flops=static_flops,
        )

    def _record_bound_expr(self, var, expr, prefix: str) -> None:
        op_id, regions = self._record_pure_expr_op(prefix, expr)
        if var is not None and op_id is not None:
            self._bound_exprs[_var_key(var)] = (op_id, regions)

    def _record_pure_expr_op(self, prefix: str, expr) -> tuple[str | None, list[RegionInfo]]:
        if expr is None:
            return None, []
        expr_info = _StoreExprAnalyzer(self).analyze(expr)
        if not _expr_info_has_work(expr_info):
            return None, []
        self._extra_input_edges.extend(expr_info.bound_sources)
        kind = _classify_expr_op(expr_info)
        static_flops = _expr_static_flops(expr_info, kind)
        annotations = {
            "tir_expr_value": _expr_text(expr),
            "simt_ops_per_element": expr_info.simt_ops_per_element,
            "sfu_ops_per_element": expr_info.sfu_ops_per_element,
            "explicit_reduction_ops_per_element": expr_info.explicit_reduction_ops_per_element,
        }
        if expr_info.call_names:
            annotations["calls"] = sorted(expr_info.call_names)
        op_id = self._append_op(
            kind=kind,
            op_name=f"{prefix}.{kind}",
            regions=expr_info.regions,
            annotations=annotations,
            static_flops=static_flops,
        )
        return op_id, list(expr_info.regions)

    def _append_op(
        self,
        kind: str,
        op_name: str,
        regions: list[RegionInfo],
        annotations: dict[str, Any] | None = None,
        static_bytes: float | None = None,
        static_flops: float | None = None,
        math_shape: dict[str, ExprInfo] | None = None,
    ) -> str:
        assert self.kernel is not None
        if static_bytes is None:
            static_bytes = max((region.static_bytes or 0.0) for region in regions) if regions else None
            if static_bytes == 0:
                static_bytes = None
        input_buffers: list[str] = []
        output_buffers: list[str] = []
        for region in regions:
            if not region.buffer_id:
                continue
            if region.access in ("r", "rw", None) and region.buffer_id not in input_buffers:
                input_buffers.append(region.buffer_id)
            if region.access in ("w", "rw") and region.buffer_id not in output_buffers:
                output_buffers.append(region.buffer_id)

        op_id = f"op{self._op_count}"
        self._op_count += 1
        self.kernel.ops.append(
            TileOpNode(
                id=op_id,
                kind=kind,
                op_name=op_name,
                loop_ids=list(self._loop_stack),
                regions=regions,
                input_buffers=input_buffers,
                output_buffers=output_buffers,
                annotations=annotations or {},
                static_bytes=static_bytes,
                static_flops=static_flops,
                math_shape=math_shape or {},
            )
        )

        for buffer_id in input_buffers:
            writer = self._last_writer.get(buffer_id)
            if writer and writer != op_id:
                self.kernel.edges.append(DataEdge(src_op=writer, dst_op=op_id, buffer_id=buffer_id))
        for buffer_id in output_buffers:
            self._last_writer[buffer_id] = op_id
        for src_op, buffer_id, reason in self._extra_input_edges:
            if src_op != op_id:
                self.kernel.edges.append(DataEdge(src_op=src_op, dst_op=op_id, buffer_id=buffer_id, reason=reason))
        self._extra_input_edges = []
        return op_id

    def buffer_id_for_buffer(self, buffer) -> str:
        return self._record_buffer(buffer, is_param=False)

    def bound_expr_for_var(self, var) -> tuple[str, list[RegionInfo]] | None:
        return self._bound_exprs.get(_var_key(var))


class _StoreExprInfo:
    def __init__(self) -> None:
        self.regions: list[RegionInfo] = []
        self._region_keys: set[str] = set()
        self.call_names: set[str] = set()
        self.simt_ops_per_element = 0
        self.sfu_ops_per_element = 0
        self.reduction_ops_per_element = 0
        self.explicit_reduction_ops_per_element = 0
        self.bound_sources: list[tuple[str, str, str]] = []
        self._bound_source_keys: set[tuple[str, str, str]] = set()

    def add_region(self, region: RegionInfo) -> None:
        key = region.signature or f"{region.buffer_id}:{len(self.regions)}"
        if key in self._region_keys:
            return
        self._region_keys.add(key)
        self.regions.append(region)

    def has_self_reduction(self, output_region: RegionInfo) -> bool:
        if self.reduction_ops_per_element <= 0:
            return False
        return any(region.buffer_id == output_region.buffer_id for region in self.regions)

    def add_bound_source(self, op_id: str, regions: list[RegionInfo]) -> None:
        source_buffers = [region.buffer_id for region in regions if region.buffer_id]
        buffer_ids = source_buffers or [""]
        for buffer_id in buffer_ids:
            key = (op_id, buffer_id, "bound_expr")
            if key in self._bound_source_keys:
                continue
            self._bound_source_keys.add(key)
            self.bound_sources.append(key)


@tirx.functor.visitor
class _StoreExprAnalyzer(PyStmtExprVisitor):
    def __init__(self, extractor: _TIRGraphExtractor) -> None:
        super().__init__()
        self.extractor = extractor
        self.info = _StoreExprInfo()

    def analyze(self, expr) -> _StoreExprInfo:
        if expr is not None:
            self.visit_expr(expr)
        return self.info

    def visit_buffer_load_(self, op) -> None:
        self.info.add_region(_buffer_load_region_info(op, "r", self.extractor))

    def visit_var_(self, op) -> None:
        bound = self.extractor.bound_expr_for_var(op)
        if bound is not None:
            source_op, regions = bound
            self.info.add_bound_source(source_op, regions)

    def visit_call_(self, op) -> None:
        op_name = _op_name(op)
        if op_name not in _REGION_OP_NAMES and op_name not in _TILE_OP_NAMES:
            self.info.call_names.add(op_name)
            category = _call_compute_category(op_name)
            if category == "sfu":
                self.info.sfu_ops_per_element += 1
            elif category == "simt":
                self.info.simt_ops_per_element += 1
            if _call_is_explicit_reduction(op_name):
                self.info.explicit_reduction_ops_per_element += 1
            if _call_is_reduction(op_name):
                self.info.reduction_ops_per_element += 1
        super().visit_call_(op)

    def visit_cast_(self, op) -> None:
        self.info.simt_ops_per_element += 1
        super().visit_cast_(op)

    def visit_select_(self, op) -> None:
        self.info.simt_ops_per_element += 1
        super().visit_select_(op)

    def visit_add_(self, op) -> None:
        self._record_binary_simt(op, reduction=True)

    def visit_sub_(self, op) -> None:
        self._record_binary_simt(op)

    def visit_mul_(self, op) -> None:
        self._record_binary_simt(op)

    def visit_div_(self, op) -> None:
        self._record_binary_simt(op)

    def visit_floor_div_(self, op) -> None:
        self._record_binary_simt(op)

    def visit_floor_mod_(self, op) -> None:
        self._record_binary_simt(op)

    def visit_mod_(self, op) -> None:
        self._record_binary_simt(op)

    def visit_min_(self, op) -> None:
        self._record_binary_simt(op, reduction=True)

    def visit_max_(self, op) -> None:
        self._record_binary_simt(op, reduction=True)

    def visit_lt_(self, op) -> None:
        self._record_binary_simt(op)

    def visit_le_(self, op) -> None:
        self._record_binary_simt(op)

    def visit_gt_(self, op) -> None:
        self._record_binary_simt(op)

    def visit_ge_(self, op) -> None:
        self._record_binary_simt(op)

    def visit_eq_(self, op) -> None:
        self._record_binary_simt(op)

    def visit_ne_(self, op) -> None:
        self._record_binary_simt(op)

    def visit_and_(self, op) -> None:
        self._record_binary_simt(op)

    def visit_or_(self, op) -> None:
        self._record_binary_simt(op)

    def visit_not_(self, op) -> None:
        self.info.simt_ops_per_element += 1
        super().visit_not_(op)

    def _record_binary_simt(self, op, reduction: bool = False) -> None:
        self.info.simt_ops_per_element += 1
        if reduction:
            self.info.reduction_ops_per_element += 1
        self._visit_binary_operands(op)

    def _visit_binary_operands(self, op) -> None:
        for attr_name in ("a", "b"):
            child = getattr(op, attr_name, None)
            if child is not None:
                self.visit_expr(child)


def _func_name(global_var, func: tirx.PrimFunc) -> str:
    if getattr(func, "attrs", None) and func.attrs.get("global_symbol", None):
        return str(func.attrs["global_symbol"])
    name_hint = getattr(global_var, "name_hint", None)
    return str(name_hint or global_var)


def _thread_tag(node: Any) -> str | None:
    tag = getattr(node, "thread_tag", None)
    if tag:
        return str(tag)
    var = getattr(node, "var", None)
    if var is not None:
        name = getattr(var, "name", None)
        if name:
            return str(name).replace("_", ".")
    return None


def _region_info(arg: Any, extractor: _TIRGraphExtractor) -> RegionInfo | None:
    if isinstance(arg, tirx.Call) and _op_name(arg) in _REGION_OP_NAMES:
        return _tile_region_call_info(arg, extractor)
    if isinstance(arg, tirx.BufferRegion):
        return _buffer_region_info(arg, None, extractor)
    if isinstance(arg, tirx.BufferLoad):
        return _buffer_load_region_info(arg, "r", extractor)
    return None


def _tile_op_annotations(kind: str, call: tirx.Call) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if kind == "reduce" and len(call.args) >= 5:
        result["reduce_type"] = _expr_value_or_text(call.args[2])
        result["reduce_dim"] = _expr_value_or_text(call.args[3])
        result["clear"] = _expr_value_or_text(call.args[4])
    elif kind == "scan" and len(call.args) >= 4:
        result["scan_dim"] = _expr_value_or_text(call.args[2])
        result["reverse"] = _expr_value_or_text(call.args[3])
    return result


def _tile_region_call_info(call: tirx.Call, extractor: _TIRGraphExtractor) -> RegionInfo | None:
    if len(call.args) < 2:
        return None
    load = call.args[0]
    if not isinstance(load, tirx.BufferLoad):
        return None
    access = _access_name(call.args[1])
    extents = list(call.args[2:])
    if len(load.indices) > len(extents):
        pad = [tirx.IntImm("int32", 1) for _ in range(len(load.indices) - len(extents))]
        extents = pad + extents
    ranges = [ir.Range.from_min_extent(index, extent) for index, extent in zip(load.indices, extents)]
    return _buffer_region_info(tirx.BufferRegion(load.buffer, ranges), access, extractor)


def _buffer_load_region_info(load: tirx.BufferLoad, access: str | None, extractor: _TIRGraphExtractor) -> RegionInfo:
    ranges = _scalar_access_ranges(load.buffer, load.indices, getattr(load, "dtype", None))
    return _buffer_region_info(tirx.BufferRegion(load.buffer, ranges), access, extractor)


def _buffer_store_region_info(store: tirx.BufferStore, access: str | None, extractor: _TIRGraphExtractor) -> RegionInfo:
    ranges = _scalar_access_ranges(store.buffer, store.indices, getattr(getattr(store, "value", None), "dtype", None))
    return _buffer_region_info(tirx.BufferRegion(store.buffer, ranges), access, extractor)


def _scalar_access_ranges(buffer, indices, dtype: Any = None) -> list[ir.Range]:
    lanes = _dtype_lanes(str(dtype)) if dtype is not None else 1
    buffer_lanes = _dtype_lanes(str(getattr(buffer, "dtype", "")))
    lanes = lanes if buffer_lanes <= 1 else 1
    ranges = []
    saw_ramp = False
    for index in indices:
        if isinstance(index, tirx.Ramp):
            saw_ramp = True
            base = getattr(index, "base", index)
            stride = _static_scalar(getattr(index, "stride", None))
            ramp_lanes = _static_scalar(getattr(index, "lanes", None)) or lanes
            extent = ramp_lanes if stride in (None, 1) else int(ramp_lanes) * int(stride)
            ranges.append(ir.Range.from_min_extent(base, tirx.IntImm("int32", int(extent))))
        else:
            ranges.append(ir.Range.from_min_extent(index, tirx.IntImm("int32", 1)))
    if not saw_ramp and lanes > 1 and ranges:
        last = ranges[-1]
        ranges[-1] = ir.Range.from_min_extent(last.min, tirx.IntImm("int32", lanes))
    if not ranges:
        shape = getattr(buffer, "shape", None) or []
        ranges = [ir.Range.from_min_extent(tirx.IntImm("int32", 0), extent) for extent in shape]
    return ranges


def _buffer_region_info(region, access: str | None, extractor: _TIRGraphExtractor) -> RegionInfo:
    buffer = region.buffer
    buffer_id = extractor.buffer_id_for_buffer(buffer)
    extents = [_expr_info(item.extent) for item in region.region]
    indices = [_expr_info(item.min) for item in region.region]
    buffer_info = extractor.kernel.buffers[buffer_id] if extractor.kernel else None
    static_elems = _product_static(extents)
    static_bytes = (
        static_elems * buffer_info.bytes_per_element
        if buffer_info is not None and static_elems is not None and buffer_info.bytes_per_element is not None
        else None
    )
    name = _buffer_name(buffer)
    signature = f"{name}[{','.join(index.text for index in indices)}]:{','.join(extent.text for extent in extents)}"
    return RegionInfo(
        buffer_id=buffer_id,
        buffer_name=name,
        scope=_buffer_scope(buffer),
        dtype=str(getattr(buffer, "dtype", "")) or None,
        indices=indices,
        extents=extents,
        access=access,
        static_bytes=static_bytes,
        signature=signature,
    )


def _classify_store_op(output_region: RegionInfo, expr_info: _StoreExprInfo) -> str:
    if expr_info.has_self_reduction(output_region) or expr_info.explicit_reduction_ops_per_element > 0:
        return "reduce"
    if expr_info.sfu_ops_per_element > 0:
        return "sfu"
    if expr_info.simt_ops_per_element > 0:
        return "elementwise"
    if not expr_info.regions:
        return "fill"
    return "copy"


def _classify_expr_op(expr_info: _StoreExprInfo) -> str:
    if expr_info.explicit_reduction_ops_per_element > 0:
        return "reduce"
    if expr_info.sfu_ops_per_element > 0:
        return "sfu"
    if expr_info.simt_ops_per_element > 0:
        return "elementwise"
    return "copy"


def _expr_info_has_work(expr_info: _StoreExprInfo) -> bool:
    return bool(
        expr_info.regions
        or expr_info.simt_ops_per_element
        or expr_info.sfu_ops_per_element
        or expr_info.explicit_reduction_ops_per_element
    )


def _store_static_flops(output_region: RegionInfo, expr_info: _StoreExprInfo, kind: str) -> float | None:
    if kind in ("copy", "fill"):
        return None
    elements = _region_static_elements(output_region)
    if elements is None:
        return None
    if kind == "sfu":
        ops = max(expr_info.sfu_ops_per_element + expr_info.simt_ops_per_element, 1)
    elif kind == "reduce":
        ops = max(
            expr_info.reduction_ops_per_element
            + expr_info.explicit_reduction_ops_per_element
            + expr_info.simt_ops_per_element,
            1,
        )
    else:
        ops = max(expr_info.simt_ops_per_element, 1)
    return float(elements * ops)


def _expr_static_flops(expr_info: _StoreExprInfo, kind: str) -> float | None:
    if kind == "copy":
        return None
    elements = _static_elements_from_regions(expr_info.regions) or 1
    if kind == "sfu":
        ops = max(expr_info.sfu_ops_per_element + expr_info.simt_ops_per_element, 1)
    elif kind == "reduce":
        ops = max(
            expr_info.reduction_ops_per_element
            + expr_info.explicit_reduction_ops_per_element
            + expr_info.simt_ops_per_element,
            1,
        )
    else:
        ops = max(expr_info.simt_ops_per_element, 1)
    return float(elements * ops)


def _static_elements_from_regions(regions: list[RegionInfo]) -> float | None:
    elements = [_region_static_elements(region) for region in regions]
    elements = [element for element in elements if element is not None]
    return max(elements) if elements else None


def _region_static_elements(region: RegionInfo) -> float | None:
    bytes_per_element = _dtype_bytes(region.dtype or "")
    if region.static_bytes is None or not bytes_per_element:
        return None
    return region.static_bytes / bytes_per_element


def _call_compute_category(op_name: str) -> str | None:
    lowered = op_name.lower()
    if any(token in lowered for token in _SFU_CALL_TOKENS):
        return "sfu"
    if lowered.startswith(("tir.", "tl.ieee_", "tl.add", "tl.sub", "tl.mul", "tl.fma", "tl.max", "tl.min", "tl.abs")):
        return "simt"
    return None


def _call_is_reduction(op_name: str) -> bool:
    lowered = op_name.lower()
    return any(token in lowered for token in ("reduce", "atomic", "add", "max", "min", "sum"))


def _call_is_explicit_reduction(op_name: str) -> bool:
    lowered = op_name.lower()
    return "reduce" in lowered or "atomic" in lowered or "sum" in lowered


def _op_name(call: tirx.Call) -> str:
    op = getattr(call, "op", None)
    name = getattr(op, "name", None)
    if name:
        return str(name)
    return str(op)


def _buffer_key(buffer) -> Any:
    data = getattr(buffer, "data", None)
    return data if data is not None else buffer


def _var_key(var) -> str:
    name = getattr(var, "name", None)
    if name:
        return str(name)
    return _expr_text(var)


def _buffer_name(buffer) -> str:
    name = getattr(buffer, "name", None)
    if name:
        return str(name)
    data = getattr(buffer, "data", None)
    data_name = getattr(data, "name", None)
    return str(data_name or buffer)


def _buffer_scope(buffer) -> str:
    scope = getattr(buffer, "scope", None)
    if callable(scope):
        return str(scope() or "global")
    return str(scope or "global")


def _loop_kind(op) -> str:
    kind = getattr(op, "kind", None)
    name = getattr(kind, "name", None)
    if name:
        return str(name)
    return str(kind or "serial")


def _access_name(value: Any) -> str | None:
    scalar = _static_scalar(value)
    if scalar == 1:
        return "r"
    if scalar == 2:
        return "w"
    if scalar == 3:
        return "rw"
    if isinstance(scalar, str):
        return scalar
    return None


def _expr_info(expr: Any) -> ExprInfo:
    scalar = _static_scalar(expr)
    return ExprInfo(text=_expr_text(expr), value=scalar if isinstance(scalar, (int, float, str, bool)) else None)


def _expr_value_or_text(expr: Any) -> Any:
    info = _expr_info(expr)
    return info.value if info.value is not None else info.text


def _expr_text(expr: Any) -> str:
    if expr is None:
        return "None"
    value = _static_scalar(expr)
    if value is not None and not isinstance(value, str):
        return str(value)
    text = str(expr)
    return text.strip()


def _static_scalar(expr: Any) -> int | float | str | bool | None:
    if isinstance(expr, (int, float, str, bool)):
        return expr
    for attr_name in ("value",):
        value = getattr(expr, attr_name, None)
        if isinstance(value, (int, float, str, bool)):
            return value
    return None


def _static_number(value: ExprInfo) -> int | float | None:
    return value.value if isinstance(value.value, (int, float)) else None


def _product_static(values: Iterable[ExprInfo]) -> int | None:
    product = 1
    saw_value = False
    for value in values:
        if not isinstance(value.value, int):
            return None
        product *= value.value
        saw_value = True
    return product if saw_value else 1


def _dtype_bytes(dtype: str) -> float | None:
    try:
        data_type = DataType(dtype)
        return data_type.bits * data_type.lanes / 8.0
    except Exception:
        return None


def _dtype_lanes(dtype: str) -> int:
    try:
        return int(DataType(dtype).lanes)
    except Exception:
        return 1


def _mapping_to_python(mapping: Any) -> dict[str, Any]:
    if mapping is None:
        return {}
    items = mapping.items() if hasattr(mapping, "items") else []
    result: dict[str, Any] = {}
    for key, value in items:
        scalar = _static_scalar(value)
        result[str(key)] = scalar if scalar is not None else _expr_text(value)
    return result
