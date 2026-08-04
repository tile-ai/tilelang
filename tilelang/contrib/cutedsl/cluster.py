import cutlass
from cutlass._mlir.dialects import arith
from cutlass.cute.typing import Int32, Pointer
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims

__all__ = [
    "cluster_arrive_relaxed",
    "cluster_arrive",
    "cluster_wait",
    "cluster_sync",
    "block_rank_in_cluster",
    "clc_try_cancel",
    "clc_try_cancel_multicast",
    "clc_is_canceled",
    "clc_get_first_ctaid_x",
    "clc_get_first_ctaid_y",
    "clc_get_first_ctaid_z",
]


def _smem_ptr(ptr: Pointer, dtype, *, loc=None, ip=None):
    return cutlass.Pointer(
        ptr,
        dtype=dtype,
        space=cutlass.AddressSpace.smem,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cluster_arrive_relaxed(*, loc=None, ip=None) -> None:
    prims.barrier_cluster_arrive_relaxed_aligned(loc=loc, ip=ip)


@dsl_user_op
def cluster_arrive(*, loc=None, ip=None) -> None:
    prims.barrier_cluster_arrive_aligned(loc=loc, ip=ip)


@dsl_user_op
def cluster_wait(*, loc=None, ip=None) -> None:
    prims.barrier_cluster_wait_aligned(loc=loc, ip=ip)


@dsl_user_op
def cluster_sync(*, loc=None, ip=None) -> None:
    prims.barrier_cluster_arrive_aligned(loc=loc, ip=ip)
    prims.barrier_cluster_wait_aligned(loc=loc, ip=ip)


@dsl_user_op
def block_rank_in_cluster(*, loc=None, ip=None) -> Int32:
    return prims.cluster_ctarank(loc=loc, ip=ip)


@dsl_user_op
def clc_try_cancel(result_ptr: Pointer, mbar_ptr: Pointer, *, loc=None, ip=None) -> None:
    result = _smem_ptr(result_ptr, cutlass.Uint32, loc=loc, ip=ip)
    mbar = _smem_ptr(mbar_ptr, cutlass.Uint64, loc=loc, ip=ip)
    prims.clusterlaunchcontrol_try_cancel(result, mbar, loc=loc, ip=ip)


@dsl_user_op
def clc_try_cancel_multicast(
    result_ptr: Pointer, mbar_ptr: Pointer, *, loc=None, ip=None
) -> None:
    result = _smem_ptr(result_ptr, cutlass.Uint32, loc=loc, ip=ip)
    mbar = _smem_ptr(mbar_ptr, cutlass.Uint64, loc=loc, ip=ip)
    prims.clusterlaunchcontrol_try_cancel(result, mbar, multicast=0xFFFFFFFF, loc=loc, ip=ip)


def _clc_response(result_ptr: Pointer, *, loc=None, ip=None):
    result = _smem_ptr(result_ptr, cutlass.Int128, loc=loc, ip=ip)
    response = result.load(alignment=16, loc=loc, ip=ip)
    return response


def _clc_query(query_type: prims.ClusterLaunchControlQueryType, result_ptr: Pointer, *, loc=None, ip=None):
    response = _clc_response(result_ptr, loc=loc, ip=ip)
    query = prims.clusterlaunchcontrol_query_cancel(query_type, response, loc=loc, ip=ip)
    prims.fence_proxy("async_shared", space=prims.SharedSpace.shared_cta, loc=loc, ip=ip)
    return query


@dsl_user_op
def clc_is_canceled(result_ptr: Pointer, *, loc=None, ip=None) -> Int32:
    canceled = _clc_query(
        prims.ClusterLaunchControlQueryType.IS_CANCELED, result_ptr, loc=loc, ip=ip
    )
    return Int32(
        arith.extui(
            Int32.mlir_type,
            canceled.ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def clc_get_first_ctaid_x(result_ptr: Pointer, *, loc=None, ip=None) -> Int32:
    return _clc_query(
        prims.ClusterLaunchControlQueryType.GET_FIRST_CTA_ID_X, result_ptr, loc=loc, ip=ip
    )


@dsl_user_op
def clc_get_first_ctaid_y(result_ptr: Pointer, *, loc=None, ip=None) -> Int32:
    return _clc_query(
        prims.ClusterLaunchControlQueryType.GET_FIRST_CTA_ID_Y, result_ptr, loc=loc, ip=ip
    )


@dsl_user_op
def clc_get_first_ctaid_z(result_ptr: Pointer, *, loc=None, ip=None) -> Int32:
    return _clc_query(
        prims.ClusterLaunchControlQueryType.GET_FIRST_CTA_ID_Z, result_ptr, loc=loc, ip=ip
    )
