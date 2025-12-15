from __future__ import annotations
from cutlass.cutlass_dsl import CuTeDSL, T, if_generate, dsl_user_op  # noqa: F401

from cutlass._mlir.dialects import nvvm, cute_nvgpu  # noqa: F401
from cutlass._mlir import ir

import cutlass._mlir.dialects.cute as _cute_ir
import cutlass._mlir.dialects.cute_nvgpu as _cute_nvgpu_ir

import cutlass.cute as cute
from cutlass.cute.typing import Int, Boolean, Int32, Int16, Uint64, Union  # noqa: F401
from cutlass.impl_utils import check_value_in

from cutlass.cute.arch import cp_async_commit_group as cp_async_commit  # noqa: F401
from cutlass.cute.arch import cp_async_wait_group as cp_async_wait  # noqa: F401

BYTES_PER_TENSORMAP = 128
BYTES_PER_POINTER = 8


def cp_async_gs(size, dst, dst_offset, src, src_offset):
    assert size in [16, 8, 4]
    # use CG (cache global) to by pass L1 when loading contiguous 128B.
    mode = nvvm.LoadCacheModifierKind.CG if size == 16 else nvvm.LoadCacheModifierKind.CA
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
    cp_async_shared_global(dst_ptr + dst_offset, src_ptr + src_offset, size, mode)


@cute.jit
def cp_async_gs_conditional(size, dst, dst_offset, src, src_offset, cond):
    if cond:
        cp_async_gs(size, dst, dst_offset, src, src_offset)


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


@dsl_user_op
def tma_load(tma_desc, mbar: cute.Pointer, smem_ptr: cute.Pointer, crd: Int | tuple[Int, ...], *, loc=None, ip=None) -> None:
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
    check_value_in(arch, ["sm_90", "sm_90a", "sm_100a"], "arch")

    if not isinstance(crd, tuple) and isinstance(tma_desc, cute.Pointer):
        # Legacy signature: tma_load(smem_ptr, gmem_ptr, mbar, size)
        _smem_ptr = tma_desc
        _gmem_ptr = mbar
        _mbar = smem_ptr
        nvvm.cp_async_bulk_shared_cluster_global(
            dst_mem=_smem_ptr.llvm_ptr,
            src_mem=_gmem_ptr.llvm_ptr,
            mbar=_mbar.llvm_ptr,
            size=Int32(crd).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    else:
        if isinstance(tma_desc, cute.CopyAtom):
            tma_desc_ptr = extract_tensormap_ptr(tma_desc)
        elif isinstance(tma_desc, cute.Tensor):
            tma_desc_ptr = tma_desc.iterator
        else:
            tma_desc_ptr = tma_desc
        nvvm.cp_async_bulk_tensor_shared_cluster_global(
            dst_mem=smem_ptr.llvm_ptr,
            tma_descriptor=tma_desc_ptr.llvm_ptr,
            coordinates=[Int32(i).ir_value(loc=loc, ip=ip) for i in crd],
            mbar=mbar.llvm_ptr,
            im2col_offsets=[],
            load_mode=nvvm.CpAsyncBulkTensorLoadMode.TILE,
            group=nvvm.Tcgen05GroupKind.CTA_1,
            use_intrinsic=False,  # set to True would lead to compile error
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
    check_value_in(arch, ["sm_90", "sm_90a", "sm_100a"], "arch")
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
        if isinstance(tma_desc, cute.CopyAtom):
            tma_desc_ptr = extract_tensormap_ptr(tma_desc)
        elif isinstance(tma_desc, cute.Tensor):
            tma_desc_ptr = tma_desc.iterator
        else:
            tma_desc_ptr = tma_desc
        nvvm.cp_async_bulk_tensor_global_shared_cta(
            tma_descriptor=tma_desc_ptr.llvm_ptr,
            src_mem=smem_ptr.llvm_ptr,
            coordinates=[Int32(i).ir_value(loc=loc, ip=ip) for i in crd],
            predicate=None,
            loc=loc,
            ip=ip,
        )


@dsl_user_op
def tma_store_arrive(*, loc=None, ip=None) -> None:
    """
    Indicate arrival of warp issuing TMA_STORE.
    Corresponds to PTX instruction: cp.async.bulk.commit_group;
    """
    nvvm.cp_async_bulk_commit_group(loc=loc, ip=ip)


@dsl_user_op
def tma_store_wait(count: int, *, read=None, loc=None, ip=None) -> None:
    """
    Wait for TMA_STORE operations to complete.
    Corresponds to PTX instruction: cp.async.bulk.wait_group.read <count>;

    :param count: The number of outstanding bulk async groups to wait for
    :type count: Int
    """
    nvvm.cp_async_bulk_wait_group(group=count, read=read, loc=loc, ip=ip)


@dsl_user_op
def cp_async_shared_global(
    dst: cute.Pointer, src: cute.Pointer, cp_size: Int, modifier: nvvm.LoadCacheModifierKind, *, src_size: Int = None, loc=None, ip=None
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
    size = src_size if src_size else cp_size
    nvvm.cp_async_shared_global(
        dst=dst.llvm_ptr,
        src=src.llvm_ptr,
        size=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), size),
        modifier=modifier,
        cp_size=Int32(cp_size).ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def prefetch_tma_descriptor(tma_desc, *, loc=None, ip=None) -> None:
    """
    Prefetch a TMA descriptor.
    Corresponds to PTX instruction: prefetch.tensormap;
    """
    if isinstance(tma_desc, cute.CopyAtom):
        tma_desc_ptr = extract_tensormap_ptr(tma_desc)
    elif isinstance(tma_desc, cute.Tensor):
        tma_desc_ptr = tma_desc.iterator
    else:
        tma_desc_ptr = tma_desc
    nvvm.prefetch_tensormap(tma_desc_ptr.llvm_ptr, loc=loc, ip=ip)
