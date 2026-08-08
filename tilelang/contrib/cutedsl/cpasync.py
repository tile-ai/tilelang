from __future__ import annotations

__all__ = [
    # cp.async operations
    "cp_async_commit",
    "cp_async_wait",
    "cp_async_gs",
    "cp_async_gs_conditional",
    "cp_async_shared_global",
    # TMA operations
    "extract_tensormap_ptr",
    "tma_load",
    "tma_store",
    "tma_reduce",
    "tma_store_arrive",
    "tma_store_wait",
    "prefetch_tma_descriptor",
    # Mbarrier operations (merged from mbar.py)
    "mbarrier_init",
    "mbarrier_expect_tx",
    "mbarrier_arrive",
    "arrive_and_expect_tx",
    "mbarrier_cp_async_arrive_noinc",
    "mbarrier_wait",
    "mbarrier_cp_async_arrive",
    "fence_proxy_async",
    "fence_barrier_init",
]

from cutlass.cutlass_dsl import Constexpr, CuTeDSL, T, if_generate, dsl_user_op  # noqa: F401

from cutlass.experimental import primitives as prims
import cutlass
from cutlass._mlir import ir

import cutlass._mlir.dialects.cute as _cute_ir
import cutlass._mlir.dialects.cute_nvgpu as _cute_nvgpu_ir

import cutlass.cute as cute
from cutlass.cute.typing import Int, Int32, Int16, Uint64, Pointer, Union  # noqa: F401
from cutlass.impl_utils import check_value_in

_TMA_ARCHES = ["sm_90", "sm_90a", "sm_100a", "sm_110", "sm_120", "sm_120a"]


class _MbarrierPointerAdapter:
    """Expose a CuTe pointer as the Array-like mbarrier input shape."""

    def __init__(self, ptr: cute.Pointer):
        self._ptr = ptr

    def data_ptr(self, idx=0, *, loc=None, ip=None):
        ptr = cutlass.Pointer(
            self._ptr,
            dtype=cutlass.Uint64,
            space=cutlass.AddressSpace.smem,
            loc=loc,
            ip=ip,
        )
        return ptr + idx if idx != 0 else ptr

    def ir_value(self, *, loc=None, ip=None):
        return self._ptr.ir_value(loc=loc, ip=ip)

    def to_llvm_ptr(self, *, loc=None, ip=None):
        return self._ptr.to_llvm_ptr(loc=loc, ip=ip)


@dsl_user_op
def cp_async_commit(*, loc=None, ip=None) -> None:
    prims.cp_async_commit_group(loc=loc, ip=ip)


@dsl_user_op
def cp_async_wait(n: Int, *, loc=None, ip=None) -> None:
    prims.cp_async_wait_group(n, loc=loc, ip=ip)


@dsl_user_op
def mbarrier_init(mbar_ptr: Pointer, arrive_count: Int, *, loc=None, ip=None) -> None:
    prims.mbarrier_init(mbar_ptr, Int32(arrive_count), loc=loc, ip=ip)


@dsl_user_op
def mbarrier_expect_tx(mbar_ptr: Pointer, tx_count: Int, *, loc=None, ip=None) -> None:
    prims.mbarrier_expect_tx(mbar_ptr, Int32(tx_count), loc=loc, ip=ip)


@dsl_user_op
def arrive_and_expect_tx(mbar_ptr: Pointer, tx_count: Int, *, loc=None, ip=None) -> None:
    prims.mbarrier_arrive_expect_tx(mbar_ptr, Int32(tx_count), loc=loc, ip=ip)


@dsl_user_op
def mbarrier_arrive(mbar_ptr: Pointer, cta_id: Int | None = None, *, loc=None, ip=None) -> None:
    if cta_id is not None:
        mbar_ptr = prims.mapa(mbar_ptr, Int32(cta_id), loc=loc, ip=ip)
    prims.mbarrier_arrive(mbar_ptr, loc=loc, ip=ip)


@dsl_user_op
def mbarrier_cp_async_arrive_noinc(mbar_ptr: Pointer, *, loc=None, ip=None) -> None:
    prims.cp_async_mbarrier_arrive(mbar_ptr, noinc=True, loc=loc, ip=ip)


def cp_async_gs(size, dst, src):
    assert size in [16, 8, 4]
    # use CG (cache global) to by pass L1 when loading contiguous 128B.
    mode = prims.LoadCacheModifier.CG if size == 16 else prims.LoadCacheModifier.CA
    if isinstance(src, cute.Tensor):
        src_ptr = src.iterator
    elif isinstance(src, cute.Pointer):
        src_ptr = src
    else:
        raise ValueError(f"Invalid source type: {type(src)}")
    if isinstance(dst, cute.Tensor):
        dst_ptr = dst.iterator
    elif isinstance(dst, cute.Pointer):
        dst_ptr = dst
    else:
        raise ValueError(f"Invalid destination type: {type(dst)}")
    cp_async_shared_global(dst_ptr, src_ptr, size, mode)


@cute.jit
def cp_async_gs_conditional(size, dst, src, cond):
    if cond:
        cp_async_gs(size, dst, src)


@dsl_user_op
def extract_tensormap_ptr(tma_atom: cute.CopyAtom, *, loc=None, ip=None) -> cute.Pointer:
    """
    extract the tensormap pointer from a TMA Copy Atom.
    :param tma_atom:      The TMA Copy Atom
    :type tma_atom:       CopyAtom
    """
    exec_value = _cute_nvgpu_ir.atom_make_exec_tma(tma_atom._trait.value, loc=loc, ip=ip)
    ptr_type = _cute_ir.PtrType.get(Uint64.mlir_type, _cute_ir.AddressSpace.generic, 64)
    tensormap_ptr = _cute_nvgpu_ir.get_tma_desc_addr(ptr_type, exec_value, loc=loc, ip=ip)
    return tensormap_ptr


def _as_tensormap_ptr(tma_desc, *, loc=None, ip=None):
    """Return a pointer suitable for TensorMap NVVM primitives."""
    if isinstance(tma_desc, cute.CopyAtom):
        return extract_tensormap_ptr(tma_desc, loc=loc, ip=ip)
    if isinstance(tma_desc, cute.Tensor):
        return tma_desc.iterator
    if hasattr(tma_desc, "get_ptr"):
        return tma_desc.get_ptr(loc=loc, ip=ip)
    return tma_desc


@dsl_user_op
def tma_load(
    tma_desc,
    mbar: cute.Pointer,
    smem_ptr: cute.Pointer,
    crd: Int | tuple[Int, ...],
    use_2cta: Constexpr[bool] = False,
    im2col_offsets=None,
    *,
    loc=None,
    ip=None,
) -> None:
    """
    Load data from global memory to shared memory using TMA (Tensor Memory Access).

    :param tma_desc:                 TMA descriptor for the tensor
    :type tma_desc:                  CopyAtom or tensormap_ptr or Tensor of tensormap_ptr
    :param mbar:                     Mbarrier pointer in shared memory
    :type mbar:                      Pointer
    :param smem_ptr:                 Destination pointer in shared memory
    :type smem_ptr:                  Pointer
    :param crd:                      Coordinates tuple for the tensor access
    :type crd:                       tuple[Int, ...]
    """
    arch = CuTeDSL._get_dsl().envar.arch
    check_value_in(arch, _TMA_ARCHES, "arch")

    if not isinstance(crd, tuple) and isinstance(tma_desc, cute.Pointer):
        # Legacy signature: tma_load(smem_ptr, gmem_ptr, mbar, size)
        _smem_ptr = tma_desc
        _gmem_ptr = mbar
        _mbar = smem_ptr
        prims.cp_async_bulk_shared_cluster_global(
            dst_mem=_smem_ptr,
            src_mem=_gmem_ptr,
            mbar=_mbar,
            size=Int32(crd),
            loc=loc,
            ip=ip,
        )
    else:
        tma_desc_ptr = _as_tensormap_ptr(tma_desc, loc=loc, ip=ip)
        # Ensure crd is a tuple (handle single coordinate case)
        if not isinstance(crd, tuple):
            crd = (crd,)
        coordinates = [Int32(i) for i in crd]
        im2col_offsets = [] if im2col_offsets is None else [Int16(i) for i in im2col_offsets]
        mode = prims.TMALoadMode.IM2COL if im2col_offsets else None
        if use_2cta:
            prims.cp_async_bulk_tensor_shared_cluster_global(
                dst_mem=smem_ptr,
                tma_descriptor=tma_desc_ptr,
                coordinates=coordinates,
                mbar=_MbarrierPointerAdapter(mbar),
                im2col_offsets=im2col_offsets,
                l2_cache_hint=Uint64(0x1000000000000000),
                mode=mode,
                group=prims.CTAGroup.CTA_2,
                loc=loc,
                ip=ip,
            )
        else:
            prims.cp_async_bulk_tensor_shared_cta_global(
                dst_mem=smem_ptr,
                tma_descriptor=tma_desc_ptr,
                coordinates=coordinates,
                mbar=mbar,
                im2col_offsets=im2col_offsets,
                l2_cache_hint=Uint64(0x1000000000000000),
                mode=mode,
                loc=loc,
                ip=ip,
            )


@dsl_user_op
def tma_store(tma_desc, smem_ptr: cute.Pointer, crd: Int | tuple[Int, ...], *, loc=None, ip=None) -> None:
    """
    Store data from shared memory to global memory using TMA (Tensor Memory Access).

    :param tma_desc:                 TMA descriptor for the tensor
    :type tma_desc:                  TMA descriptor
    :param smem_ptr:                 Source pointer in shared memory
    :type smem_ptr:                  Pointer
    :param crd:                      Coordinates tuple for the tensor access
    :type crd:                       tuple[Int, ...]
    """
    arch = CuTeDSL._get_dsl().envar.arch
    check_value_in(arch, _TMA_ARCHES, "arch")
    if not isinstance(crd, tuple):
        if arch not in ("sm_90", "sm_90a"):
            raise NotImplementedError("tma_store(size) path is only implemented for sm_90/sm_90a")
        gmem_ptr = tma_desc.align(smem_ptr.alignment)
        _cute_nvgpu_ir.arch_copy_SM90_bulk_copy_s2g(
            dsmem_data_addr=smem_ptr.value,
            gmem_data_addr=gmem_ptr.value,
            size=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), crd),
            loc=loc,
            ip=ip,
        )
    else:
        tma_desc_ptr = _as_tensormap_ptr(tma_desc, loc=loc, ip=ip)
        prims.cp_async_bulk_tensor_global_shared_cta(
            tma_descriptor=tma_desc_ptr,
            src_mem=smem_ptr,
            coordinates=[Int32(i) for i in crd],
            mode=prims.TMAStoreMode.TILE,
            loc=loc,
            ip=ip,
        )


@dsl_user_op
def tma_reduce(tma_desc, smem_ptr: cute.Pointer, crd: Int | tuple[Int, ...], *, loc=None, ip=None) -> None:
    """
    Reduce data from shared memory to global memory using TMA with atomic ADD reduction.

    This performs an atomic add of shared memory data to global memory using
    the TMA unit's reduce capability.

    :param tma_desc:                 TMA descriptor for the tensor
    :type tma_desc:                  TMA descriptor
    :param smem_ptr:                 Source pointer in shared memory
    :type smem_ptr:                  Pointer
    :param crd:                      Coordinates tuple for the tensor access
    :type crd:                       tuple[Int, ...]
    """
    arch = CuTeDSL._get_dsl().envar.arch
    check_value_in(arch, _TMA_ARCHES, "arch")

    tma_desc_ptr = _as_tensormap_ptr(tma_desc, loc=loc, ip=ip)

    # Ensure crd is a tuple
    if not isinstance(crd, tuple):
        crd = (crd,)

    prims.cp_async_bulk_tensor_reduce(
        tma_descriptor=tma_desc_ptr,
        src_mem=smem_ptr,
        red_kind=prims.TMARedux.ADD,
        coordinates=[Int32(i) for i in crd],
        mode=prims.TMAStoreMode.TILE,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tma_store_arrive(*, loc=None, ip=None) -> None:
    """
    Indicate arrival of warp issuing TMA_STORE.
    Corresponds to PTX instruction: cp.async.bulk.commit_group;
    """
    prims.cp_async_bulk_commit_group(loc=loc, ip=ip)


@dsl_user_op
def tma_store_wait(count: int, *, read=None, loc=None, ip=None) -> None:
    """
    Wait for TMA_STORE operations to complete.
    Corresponds to PTX instruction: cp.async.bulk.wait_group{.read} <count>;

    :param count: The number of outstanding bulk async groups to wait for
    :type count: Int
    :param read: Whether to use the PTX .read modifier
    :type read: Optional[bool]
    """
    prims.cp_async_bulk_wait_group(group=count, read=read, loc=loc, ip=ip)


@dsl_user_op
def cp_async_shared_global(
    dst: cute.Pointer, src: cute.Pointer, cp_size: Int, modifier: prims.LoadCacheModifier, *, src_size: Int = None, loc=None, ip=None
) -> None:
    """
    Asynchronously copy data from global memory to shared memory.

    :param dst: Destination pointer in shared memory
    :type dst: Pointer
    :param src: Source pointer in global memory
    :type src: Pointer
    :param size: Size of the copy in bytes
    :type size: Int
    :param modifier: Cache modifier
    :type modifier: Int
    :param cp_size: Optional copy size override
    :type cp_size: Int
    """
    prims.cp_async_shared_global(
        dst=dst,
        src=src,
        size=cp_size,
        modifier=modifier,
        cp_size=src_size,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def prefetch_tma_descriptor(tma_desc, *, loc=None, ip=None) -> None:
    """
    Prefetch a TMA descriptor.
    Corresponds to PTX instruction: prefetch.tensormap;
    """
    tma_desc_ptr = _as_tensormap_ptr(tma_desc, loc=loc, ip=ip)
    prims.prefetch_tensormap(tma_desc_ptr, loc=loc, ip=ip)


@cute.jit
def mbarrier_wait(mbar_ptr: Pointer, phase: Int, timeout_ns: int = 10000000) -> None:
    """Waits on a mbarrier with a specified phase (blocking loop).

    Uses the primitive try-wait wrapper in a spin loop.
    """
    while not prims.mbarrier_try_wait_parity(mbar_ptr, Int32(phase), time_limit=Int32(timeout_ns)):
        pass


@dsl_user_op
def mbarrier_cp_async_arrive(mbar_ptr: Pointer, *, loc=None, ip=None) -> None:
    prims.cp_async_mbarrier_arrive(
        mbar_ptr,
        noinc=False,
        loc=loc,
        ip=ip,
    )


def fence_proxy_async():
    prims.fence_proxy("async_shared", space=prims.SharedSpace.shared_cta)


def fence_barrier_init():
    prims.fence_mbarrier_init()
