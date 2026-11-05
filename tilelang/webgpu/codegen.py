from __future__ import annotations

import tvm_ffi
from tvm_tirx import IRModule, ir
from tvm_tirx.target import Target

from tilelang.backend.device_codegen import global_func_device_codegen

_build_webgpu = global_func_device_codegen("target.build.webgpu")


def _lower_tilelang_sync(mod: IRModule) -> IRModule:
    """Consume TileLang's legacy barrier operands only at the WebGPU boundary."""

    def rewrite(call):
        if not isinstance(call.op, ir.Op):
            return call
        if call.op.name == "tirx.gpu_storage_sync":
            # Do not turn a malformed native multi-argument call into a valid
            # one. The native operation has always had a scope-only contract.
            call.validate()
            return call
        if call.op.name != "tl.storage_sync":
            return call
        call.validate()
        if call.ty != ir.reinfer_type(call):
            raise TypeError("TileLang storage synchronization must have a void result")
        # The old WebGPU emitter used only the storage scope of sync_threads;
        # IDs/counts still nested inside that call were never evaluated. Keep
        # that whole-workgroup fallback after all synchronization planning.
        # Independent statements already emitted for an operand remain in IR.
        scope = ir.StringImm("shared", loc=call.args[0].loc)
        return ir.Call(
            "tirx.gpu_storage_sync",
            [scope],
            attrs=call.attrs,
            ty_args=call.ty_args,
            ty=call.ty,
            loc=call.loc,
        )

    return tvm_ffi.structural_map(mod, (ir.Call, rewrite))


def build_webgpu(mod: IRModule, target: Target) -> tvm_ffi.Module:
    return _build_webgpu(_lower_tilelang_scopes(_lower_tilelang_sync(mod)), target)


def _lower_tilelang_scopes(mod: IRModule) -> IRModule:
    """Keep lexical blocks at the final boundary to native WebGPU codegen.

    Upstream's workgroup collector and emitter accept ordinary conditionals.
    An always-taken branch preserves the scope of each definition without
    changing allocation placement or copying either codegen implementation.
    Run this after allocation and synchronization planning: subsequent generic
    simplification could flatten the branch and erase its lifetime boundary.
    """

    def rewrite(region):
        if region.op.name != "tl.lexical_scope":
            return region
        if region.args or region.body_params or region.result_vars:
            raise ValueError("A TileLang lexical scope cannot bind or return values")
        for name, value in region.attrs.items():
            if name != "explicit_alloc_scope":
                raise ValueError(f"Unknown tl.lexical_scope attribute: {name}")
            if not isinstance(value, bool):
                raise TypeError("tl.lexical_scope explicit_alloc_scope must be a boolean")
        return ir.If(ir.const(True, "bool"), region.body, None, loc=region.loc)

    return tvm_ffi.structural_map(mod, (ir.RegionStmt, rewrite))
