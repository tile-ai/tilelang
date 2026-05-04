"""Rewrite local.fragment → metal.simdgroup for GEMM accumulators on Metal."""

from __future__ import annotations

from tvm import tir, IRModule
from tvm.ir import Op, PointerType
from tvm.tir.transform import prim_func_pass

_GEMM_OPS = None


def _get_gemm_ops():
    global _GEMM_OPS
    if _GEMM_OPS is None:
        _GEMM_OPS = {
            Op.get("tl.tileop.gemm"),
            Op.get("tl.tileop.wgmma_gemm"),
            Op.get("tl.tileop.tcgen05_gemm"),
        }
    return _GEMM_OPS


def _extract_buffer_var_from_region(region_call):
    if not isinstance(region_call, tir.Call):
        return None
    if len(region_call.args) < 1:
        return None
    buf_load = region_call.args[0]
    if isinstance(buf_load, tir.BufferLoad):
        return buf_load.buffer.data
    return None


def _extract_buffer_from_region(region_call):
    """Return the Buffer referenced by a tl.region(BufferLoad, ...) call.

    Used to inspect operand dtypes so we can detect mixed-precision GEMMs
    (which route to the Metal scalar fallback rather than simdgroup MMA
    and therefore must NOT have their accumulator scope rewritten to
    metal.simdgroup).
    """
    if not isinstance(region_call, tir.Call):
        return None
    if len(region_call.args) < 1:
        return None
    buf_load = region_call.args[0]
    if isinstance(buf_load, tir.BufferLoad):
        return buf_load.buffer
    return None


def _collect_fragment_gemm_accum_vars(body: tir.Stmt):
    """Walk the body and return fragment vars safe to rewrite to simdgroup.

    Returns the set of fragment vars used as the C operand of a
    same-dtype GEMM (the ones that will be lowered through
    GemmMetal / simdgroup_matrix_multiply). Vars that appear as the C
    operand of a mixed-dtype GEMM (which will route to the scalar
    fallback GemmMetalScalar) — or as the A operand of such a GEMM
    when A is itself a fragment — are excluded so that the scalar
    fallback can perform its per-element T.cast(..., accum_dtype)
    arithmetic on them without tripping the Metal codegen's check that
    metal.simdgroup allocations are scalar 8x8 blocks.
    """
    simd_accum_vars: set = set()
    scalar_accum_vars: set = set()
    gemm_ops = _get_gemm_ops()

    def _visitor(stmt):
        if isinstance(stmt, tir.Evaluate) and isinstance(stmt.value, tir.Call):
            call = stmt.value
            if call.op in gemm_ops and len(call.args) >= 3:
                a_buf = _extract_buffer_from_region(call.args[0])
                b_buf = _extract_buffer_from_region(call.args[1])
                c_var = _extract_buffer_var_from_region(call.args[2])
                if c_var is None or not hasattr(c_var, "type_annotation"):
                    return
                ta = c_var.type_annotation
                if ta is None or not hasattr(ta, "storage_scope") or ta.storage_scope != "local.fragment":
                    return
                # Mixed-dtype A/B routes to the scalar fallback on Metal
                # (see Gemm._select_gemm_instruction in
                # tilelang/tileop/gemm/__init__.py).
                mixed = (a_buf is not None and b_buf is not None
                         and str(a_buf.dtype) != str(b_buf.dtype))
                if mixed:
                    scalar_accum_vars.add(c_var)
                    # If A is itself a fragment in this scalar gemm, the
                    # scalar fallback reads it scalar — never via
                    # simdgroup_load. Don't promote it.
                    if a_buf is not None and hasattr(a_buf.data, "type_annotation"):
                        a_ta = a_buf.data.type_annotation
                        if (a_ta is not None and hasattr(a_ta, "storage_scope")
                                and a_ta.storage_scope == "local.fragment"):
                            scalar_accum_vars.add(a_buf.data)
                else:
                    simd_accum_vars.add(c_var)

    tir.stmt_functor.post_order_visit(body, _visitor)
    # Conservative: if a var appears in BOTH modes (e.g. fp16 Q@Kt
    # produces S into simdgroup, then mixed-dtype S@V reads S as scalar),
    # keep the scalar scope so the scalar reads typecheck.
    return simd_accum_vars - scalar_accum_vars


def _remap_buffer(buf, var_map):
    old_data = buf.data
    new_data = var_map.get(old_data, None)
    if new_data is None:
        return buf
    return tir.decl_buffer(
        buf.shape,
        buf.dtype,
        buf.name,
        data=new_data,
        scope="metal.simdgroup",
        data_alignment=buf.data_alignment,
        offset_factor=buf.offset_factor,
    )


def _rewrite_scope(body, var_map):
    buf_map = {}

    def _pre_order(stmt):
        if isinstance(stmt, tir.Block):
            new_alloc_bufs = []
            changed = False
            for buf in stmt.alloc_buffers:
                new_buf = _remap_buffer(buf, var_map)
                new_alloc_bufs.append(new_buf)
                if not new_buf.same_as(buf):
                    buf_map[buf] = new_buf
                    changed = True
            if changed:
                new_body = tir.stmt_functor.substitute(stmt.body, var_map)
                new_block = tir.Block(
                    stmt.iter_vars,
                    stmt.reads,
                    stmt.writes,
                    stmt.name_hint,
                    new_body,
                    stmt.init,
                    new_alloc_bufs,
                    stmt.match_buffers,
                    stmt.annotations,
                )
                return (
                    tir.BlockRealize(
                        stmt.iter_vars,
                        tir.const(True, "bool"),
                        new_block,
                    )
                    if False
                    else new_block
                )
        elif isinstance(stmt, tir.Allocate):
            new_var = var_map.get(stmt.buffer_var, None)
            if new_var is not None:
                new_body = tir.stmt_functor.substitute(stmt.body, var_map)
                return tir.Allocate(new_var, stmt.dtype, stmt.extents, stmt.condition, new_body, stmt.annotations)
        return None

    return tir.stmt_functor.ir_transform(body, _pre_order, None, ["tir.Block", "tir.Allocate"])


def _metal_fragment_to_simdgroup(func: tir.PrimFunc, mod: IRModule, ctx) -> tir.PrimFunc:
    target = func.attrs.get("target", None)
    if target is None or target.kind.name != "metal":
        return func

    accum_vars = _collect_fragment_gemm_accum_vars(func.body)
    if not accum_vars:
        return func

    var_map: dict = {}
    for var in accum_vars:
        ptr_type = var.type_annotation
        new_ptr = PointerType(ptr_type.element_type, "metal.simdgroup")
        new_var = tir.Var(var.name, new_ptr)
        var_map[var] = new_var

    new_body = _rewrite_scope(func.body, var_map)
    return func.with_body(new_body)


MetalFragmentToSimdgroup = prim_func_pass(_metal_fragment_to_simdgroup, opt_level=0, name="tl.MetalFragmentToSimdgroup")
