from __future__ import annotations

import tilelang
from tilelang.backend.base import BackendPassHooks
from tilelang.contrib.nvcc import have_pdl, have_tma
from tilelang.transform import PassContext
from tvm import IRModule
from tvm.target import Target


def allow_warp_specialized(pass_ctx: PassContext | None = None, target: Target | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    if target is None or (not have_tma(target)):
        return False
    disable_warp_specialized = pass_ctx.config.get("tl.disable_warp_specialized", False)
    return not disable_warp_specialized


class NvidiaPassHooks(BackendPassHooks):
    def adjust_aggressive_shared_memory_merge(self, enabled: bool, target: Target) -> bool:
        if allow_warp_specialized(target=target):
            return False
        return enabled

    def pre_layout(self, mod: IRModule, target: Target) -> IRModule:
        if allow_warp_specialized(target=target):
            mod = tilelang.transform.ProducerConsumerWarpSpecialized()(mod)
        return tilelang.transform.LowerBlackwell2SM()(mod)

    def post_tile_lowering(self, mod: IRModule, target: Target) -> IRModule:
        return tilelang.transform.LowerL2Persistent()(mod)

    def optimize_entry(self, mod: IRModule, target: Target) -> IRModule:
        return tilelang.transform.LowerSharedTmem()(mod)

    def lower_shared_barrier(self, mod: IRModule, target: Target) -> IRModule:
        return tilelang.transform.LowerSharedBarrier()(mod)

    def after_shared_barrier_lowering(self, mod: IRModule, target: Target, *, has_tma: bool) -> IRModule:
        if has_tma:
            mod = tilelang.transform.FuseMBarrierArriveExpectTx()(mod)
        return mod

    def before_split_host_device(self, mod: IRModule, target: Target) -> IRModule:
        mod = tilelang.transform.LowerLDGSTG()(mod)
        return tilelang.transform.LowerHopperIntrin()(mod)

    def after_split_host_device(self, mod: IRModule, target: Target) -> IRModule:
        return tilelang.transform.MarkCudaSyncCalls(have_pdl(target))(mod)

    def after_shared_memory_planning(self, mod: IRModule, target: Target) -> IRModule:
        return tilelang.transform.InjectFenceProxy()(mod)

    def after_shared_sync(self, mod: IRModule, target: Target) -> IRModule:
        mod = tilelang.transform.InjectTcgen05Fence()(mod)
        if allow_warp_specialized(target=target):
            mod = tilelang.transform.AnnotateWarpGroupRegAlloc()(mod)
        return mod

    def before_device_codegen(self, mod: IRModule, target: Target) -> IRModule:
        return tilelang.transform.PersistThreadblock()(mod)
