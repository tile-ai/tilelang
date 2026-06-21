"""Rewrite local.fragment to metal.simdgroup for legacy Metal GEMM accumulators.

M5 cooperative-tensor GEMM keeps fragment/local buffers in regular scopes until
Metal codegen sees explicit tl.cooperative_tensor_* builtins.  Only legacy
simdgroup GEMM requires changing the accumulator scope before layout inference.
"""

from __future__ import annotations

from functools import lru_cache

from tvm import IRModule
from tvm import tirx as tir
from tvm.ir import Op, PointerType
from tvm.tirx import SBlock
from tvm.tirx.transform import prim_func_pass


@lru_cache(maxsize=1)
def _get_gemm_ops():
    return frozenset({Op.get("tl.tileop.gemm")})


def _extract_buffer_var_from_region(region_call):
    if not isinstance(region_call, tir.Call) or len(region_call.args) < 1:
        return None
    buf_load = region_call.args[0]
    if isinstance(buf_load, tir.BufferLoad):
        return buf_load.buffer.data
    return None


def _get_num_warps_from_body(body: tir.Stmt) -> int:
    warp_size = 32
    num_threads = None

    def _visitor(stmt):
        nonlocal num_threads
        if (
            isinstance(stmt, tir.AttrStmt)
            and stmt.attr_key == "thread_extent"
            and hasattr(stmt.node, "thread_tag")
            and "threadIdx.x" in str(stmt.node.thread_tag)
        ):
            val = stmt.value
            if isinstance(val, tir.IntImm):
                num_threads = val.value

    tir.stmt_functor.post_order_visit(body, _visitor)
    return num_threads // warp_size if num_threads is not None else 1


def _collect_fragment_gemm_accum_vars(body: tir.Stmt) -> set:
    accum_vars: set = set()
    gemm_ops = _get_gemm_ops()

    def _visitor(stmt):
        if isinstance(stmt, tir.Evaluate) and isinstance(stmt.value, tir.Call):
            call = stmt.value
            if call.op in gemm_ops and len(call.args) >= 3:
                var = _extract_buffer_var_from_region(call.args[2])
                if var is not None and hasattr(var, "type_annotation"):
                    ta = var.type_annotation
                    if ta is not None and hasattr(ta, "storage_scope") and ta.storage_scope == "local.fragment":
                        accum_vars.add(var)

    tir.stmt_functor.post_order_visit(body, _visitor)
    return accum_vars


def _remap_buffer(buf, var_map, num_warps=1):
    old_data = buf.data
    new_data = var_map.get(old_data, None)
    if new_data is None:
        return buf
    total = 1
    for s in buf.shape:
        total *= s.value if isinstance(s, tir.IntImm) else s
    new_total = total // num_warps if num_warps > 1 else total
    return tir.decl_buffer(
        [tir.IntImm("int32", new_total)],
        buf.dtype,
        buf.name,
        data=new_data,
        scope="metal.simdgroup",
        data_alignment=buf.data_alignment,
        offset_factor=buf.offset_factor,
    )


def _remap_buffer_region(region, buf_map):
    if region is None:
        return region
    new_buf = buf_map.get(region.buffer, None)
    if new_buf is None:
        return region
    return tir.BufferRegion(new_buf, region.region)


def _remap_match_buffer(match, buf_map):
    if match is None:
        return match
    new_buf = buf_map.get(match.buffer, None)
    new_src = _remap_buffer_region(match.source, buf_map)
    if new_buf is None and new_src is match.source:
        return match
    return tir.MatchBufferRegion(new_buf if new_buf is not None else match.buffer, new_src)


def _rewrite_scope(body, var_map, num_warps=1):
    body = tir.stmt_functor.substitute(body, var_map)
    buf_map = {}

    def _pre_order(stmt):
        if isinstance(stmt, SBlock):
            new_alloc_bufs = []
            changed = False
            for buf in stmt.alloc_buffers:
                new_buf = _remap_buffer(buf, var_map, num_warps)
                new_alloc_bufs.append(new_buf)
                if not new_buf.same_as(buf):
                    buf_map[buf] = new_buf
                    changed = True
            if changed:
                new_reads = [_remap_buffer_region(r, buf_map) for r in (stmt.reads or [])]
                new_writes = [_remap_buffer_region(w, buf_map) for w in (stmt.writes or [])]
                new_match_bufs = [_remap_match_buffer(m, buf_map) for m in (stmt.match_buffers or [])]
                return SBlock(
                    stmt.iter_vars,
                    new_reads,
                    new_writes,
                    stmt.name_hint,
                    stmt.body,
                    stmt.init,
                    new_alloc_bufs,
                    new_match_bufs,
                    stmt.annotations,
                )
        elif isinstance(stmt, tir.AllocBuffer):
            new_buf = _remap_buffer(stmt.buffer, var_map, num_warps)
            if not new_buf.same_as(stmt.buffer):
                buf_map[stmt.buffer] = new_buf
                return tir.AllocBuffer(new_buf, stmt.annotations, stmt.span)
        return None

    return tir.stmt_functor.ir_transform(body, _pre_order, None, ["tirx.SBlock", "tirx.AllocBuffer"])


def _metal_fragment_to_simdgroup(func: tir.PrimFunc, mod: IRModule, ctx) -> tir.PrimFunc:
    target = func.attrs.get("target", None)
    if target is None or target.kind.name != "metal":
        return func

    accum_vars = _collect_fragment_gemm_accum_vars(func.body)
    if not accum_vars:
        return func

    num_warps = _get_num_warps_from_body(func.body)
    var_map: dict = {}
    for var in accum_vars:
        ptr_type = var.type_annotation
        new_ptr = PointerType(ptr_type.element_type, "metal.simdgroup")
        var_map[var] = tir.Var(var.name, new_ptr)

    return func.with_body(_rewrite_scope(func.body, var_map, num_warps))


MetalFragmentToSimdgroup = prim_func_pass(
    _metal_fragment_to_simdgroup,
    opt_level=0,
    name="tl.MetalFragmentToSimdgroup",
)

__all__ = ["MetalFragmentToSimdgroup"]
