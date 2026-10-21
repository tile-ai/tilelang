import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.layout import make_row_major, make_zn_layout, make_zz_layout
from tilelang.utils.target import SUNMMIO_TARGET_DESC


DTYPE = "float16"


def _tile_region(buffer_load, access_type, *extents):
    access_code = {"r": 1, "w": 2, "rw": 3}[access_type]
    return tvm.tir.call_intrin(
        "handle",
        tvm.tir.op.Op.get("tl.tileop.region"),
        buffer_load,
        access_code,
        *extents,
    )


@pytest.fixture(autouse=True)
def disable_tilelang_cache():
    cache_was_enabled = tilelang.env.is_cache_enabled()
    tilelang.env.disable_cache()
    try:
        yield
    finally:
        if cache_was_enabled:
            tilelang.env.enable_cache()
        else:
            tilelang.env.disable_cache()


@pytest.fixture
def _strict_region_validation():
    import tilelang.utils.target as target_utils

    previous = target_utils.ENABLE_SUNMMIO_REGION_VALIDATION
    target_utils.set_sunmmio_region_validation(True)
    try:
        yield
    finally:
        target_utils.set_sunmmio_region_validation(previous)


def _make_copy_kernel(copy_case):
    if copy_case in {
        "dynamic_extent_warns_and_passes",
        "dynamic_row_major_inner_mode_warns_and_passes",
    }:
        shape = (T.dynamic("m"), 96)
        layout = make_row_major(shape)
    elif copy_case in {
        "row_major_3d_outer_slab_valid",
        "row_major_3d_outer_slab_unclipped",
        "row_major_3d_outer_slab_oob",
    }:
        shape = (2, 128, 128)
        layout = make_row_major(shape)
    elif copy_case == "zz_3d_major_dim_split":
        shape = (2, 96, 96)
        layout = make_zz_layout(shape, axes=[1, 2], block_shape=(32, 32))
    elif copy_case.startswith("dynamic_outer_"):
        shape = (T.dynamic("m"), 96)
        layout = make_zz_layout(shape, axes=[0, 1], block_shape=(32, 32))
    elif copy_case == "zz_block_non_major_dim_multi_block":
        shape = (128, 128)
        layout = make_zz_layout(shape, axes=[0, 1], block_shape=(32, 32))
    elif copy_case.startswith("zz_") or copy_case.startswith("zn_"):
        shape = (96, 96)
        if copy_case.startswith("zn_"):
            layout = make_zn_layout(shape, axes=[0, 1], block_shape=(32, 32))
        else:
            layout = make_zz_layout(shape, axes=[0, 1], block_shape=(32, 32))
    else:
        shape = (128, 128)
        layout = make_row_major(shape)

    if copy_case == "row_major_aligned_grid_tile":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[0:64, 0:64], A_shared[64:128, 64:128])

    elif copy_case == "row_major_min_not_region_extent_aligned":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[32:96, 0:64], A_shared[0:64, 0:64])

    elif copy_case == "row_major_extent_not_buffer_shape_factor":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[0:96, 0:96], A_shared[0:96, 0:96])

    elif copy_case == "row_major_1d_tile_view":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[0, 64:128], A_shared[1, 64:128])

    elif copy_case == "row_major_dynamic_min_bounds_unknown":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel() as cid:
                bx = cid % 4
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[bx * 32 : (bx + 1) * 32, 0:64], A_shared[0:32, 0:64])

    elif copy_case == "row_major_dynamic_min_alignment_unknown":
        shape = (512, 256)
        shared_shape = (32, 32)
        layout = make_row_major(shape)
        shared_layout = make_row_major(shared_shape)

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel() as cid:
                bx = cid % 8
                A_shared = T.alloc_shared(shared_shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: shared_layout})
                ko = (bx + bx) % 8
                m_offset = bx * 32 + T.min(bx, 1)
                T.copy(A[m_offset, ko * 32], A_shared)

    elif copy_case == "row_major_dynamic_select_alignment_unknown":
        shape = (512, 512)
        shared_shape = (32, 32)
        layout = make_row_major(shape)
        shared_layout = make_row_major(shared_shape)

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel() as cid:
                bx = cid % 4
                by = cid // 4
                A_shared = T.alloc_shared(shared_shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: shared_layout})
                sum_idx = bx + by
                diff_idx = bx - by
                scaled_idx = sum_idx * 3 + 1
                div_idx = scaled_idx // 2
                mod_idx = div_idx % 7
                lt_cmp = bx < by + 2
                le_cmp = by <= bx + 1
                ne_cmp = bx != by
                eq_cmp = bx == by
                not_lt_le = T.Not(lt_cmp and le_cmp)
                row_cond = (lt_cmp and le_cmp) or (ne_cmp and not_lt_le)
                row_delta = T.Select(
                    row_cond,
                    T.Select(eq_cmp, T.min(mod_idx, 7), T.min(mod_idx + 1, 7)),
                    T.max(diff_idx, 0),
                )
                src_row = by * 64 + 8 + row_delta
                T.copy(A[src_row, bx * 64], A_shared)

    elif copy_case == "row_major_3d_outer_slab_valid":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[0:2, 0:64, 0:64], A_shared[0:2, 64:128, 64:128])

    elif copy_case == "row_major_3d_outer_slab_unclipped":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[1:3, 0:64, 0:64], A_shared[0:2, 0:64, 0:64])

    elif copy_case == "row_major_3d_outer_slab_oob":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.evaluate(
                    tvm.tir.call_intrin(
                        "handle",
                        tvm.tir.op.Op.get("tl.tileop.copy"),
                        _tile_region(A[1, 0, 0], "r", 2, 64, 64),
                        _tile_region(A_shared[0, 0, 0], "w", 2, 64, 64),
                    )
                )

    elif copy_case == "row_major_zero_extent":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.evaluate(
                    tvm.tir.call_intrin(
                        "handle",
                        tvm.tir.op.Op.get("tl.tileop.copy"),
                        _tile_region(A[0, 0], "r", 0, 64),
                        _tile_region(A_shared[0, 0], "w", 0, 64),
                    )
                )

    elif copy_case == "row_major_negative_extent":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.evaluate(
                    tvm.tir.call_intrin(
                        "handle",
                        tvm.tir.op.Op.get("tl.tileop.copy"),
                        _tile_region(A[0, 0], "r", -16, 64),
                        _tile_region(A_shared[0, 0], "w", -16, 64),
                    )
                )

    elif copy_case == "zz_3d_major_dim_split":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[0:2, 0:16, 0:32], A_shared[0:2, 16:32, 32:64])

    elif copy_case == "zz_block_equal":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[0:32, 0:32], A_shared[32:64, 32:64])

    elif copy_case == "zz_block_non_major_dim_split":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[0:32, 0:16], A_shared[32:64, 16:32])

    elif copy_case == "zz_block_non_major_dim_multi_block":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[0:32, 0:64], A_shared[32:64, 0:64])

    elif copy_case == "zn_block_non_major_dim_split":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[0:16, 0:96], A_shared[16:32, 0:96])

    elif copy_case == "zz_block_both_dims_split":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[0:16, 0:16], A_shared[16:32, 16:32])

    elif copy_case == "zz_whole_dim":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[0:96, 0:96], A_shared[0:96, 0:96])

    elif copy_case == "zz_block_major_dim_split":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[0:16, 0:32], A_shared[16:32, 32:64])

    elif copy_case == "zz_block_offset_inside_tile":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[16:48, 0:32], A_shared[0:32, 32:64])

    elif copy_case == "zz_extent_not_coalesced_compatible":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[0:48, 0:48], A_shared[0:48, 0:48])

    elif copy_case == "dynamic_extent_warns_and_passes":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[0 : shape[0], 0:32], A_shared[0 : shape[0], 32:64])

    elif copy_case == "dynamic_outer_static_extent_equal_block":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[0:32, 0:32], A_shared[32:64, 32:64])

    elif copy_case == "dynamic_outer_static_extent_split_block":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[0:16, 0:32], A_shared[16:32, 32:64])

    elif copy_case == "dynamic_outer_static_extent_incompatible":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[0:24, 0:32], A_shared[24:48, 32:64])

    elif copy_case == "dynamic_row_major_inner_mode_warns_and_passes":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[0:32, 0:32], A_shared[0:32, 32:64])

    else:
        raise ValueError(f"unknown copy case: {copy_case}")

    return tvm.IRModule({"main": kernel})


def _run_validate_tile_view_regions(copy_case):
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = _make_copy_kernel(copy_case)
        mod = tvm.tir.transform.BindTarget(target)(mod)
        return tilelang.transform.ValidateTileViewRegions()(mod)


def _make_inferred_layout_copy_kernel(copy_case):
    shape = (96, 96) if copy_case.startswith("zz_") else (128, 128)

    @T.prim_func
    def kernel(A: T.Tensor(shape, DTYPE)):
        with T.Kernel():
            A_shared = T.alloc_shared(shape, DTYPE)

            if copy_case == "row_major_aligned_grid_tile":
                T.copy(A[0:64, 0:64], A_shared[64:128, 64:128])
            elif copy_case == "row_major_min_not_region_extent_aligned":
                T.copy(A[32:96, 0:64], A_shared[0:64, 0:64])
            elif copy_case == "zz_extent_not_coalesced_compatible":
                T.copy(A[0:48, 0:48], A_shared[0:48, 0:48])
            elif copy_case == "zz_block_equal":
                T.copy(A[0:32, 0:32], A_shared[32:64, 32:64])
            else:
                raise ValueError(f"unknown copy case: {copy_case}")

    return tvm.IRModule({"main": kernel})


def _run_kernel_pipeline_to_validate_tile_view_regions(copy_case):
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = _make_inferred_layout_copy_kernel(copy_case)
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tilelang.transform.InferSramScope()(mod)
        mod = tilelang.transform.LegalizeSunmmioDataPath()(mod)
        mod = tilelang.transform.SunmmioLayoutInference()(mod)
        return tilelang.transform.ValidateTileViewRegions()(mod)


def _make_comm_kernel(comm_case):
    shape = (96, 96) if comm_case.endswith("_illegal_non_major_split") else (128, 128)
    layout = make_zz_layout(shape, axes=[0, 1], block_shape=(32, 32))

    if comm_case == "broadcast_valid":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                B_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A_shared: layout, B_shared: layout})
                T.comm.broadcast(
                    A_shared[0:32, 0:32],
                    B_shared[32:64, 32:64],
                    (0, 0),
                    direction="h",
                )

    elif comm_case == "broadcast_illegal_non_major_split":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                B_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A_shared: layout, B_shared: layout})
                T.comm.broadcast(
                    A_shared[0:32, 0:16],
                    B_shared[32:64, 16:32],
                    (0, 0),
                    direction="h",
                )

    elif comm_case == "broadcast_scalar_src_oob":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                B_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A_shared: layout, B_shared: layout})
                T.comm.broadcast(
                    A_shared[999, 0],
                    B_shared[0:32, 0:32],
                    (0, 0),
                    direction="h",
                )

    elif comm_case == "put_valid":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                B_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A_shared: layout, B_shared: layout})
                T.comm.put(
                    A_shared[0:32, 0:32],
                    B_shared[32:64, 32:64],
                    (0, 0),
                    (0, 1),
                )

    elif comm_case == "allgather_legacy_valid":
        recv_shape = (16, 32, 32)
        recv_layout = make_row_major(recv_shape)

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel() as _bx:
                A_shared = T.alloc_shared((32, 32), DTYPE)
                R_shared = T.alloc_shared(recv_shape, DTYPE)
                T.annotate_layout(
                    {
                        A_shared: make_row_major((32, 32)),
                        R_shared: recv_layout,
                    }
                )
                T.comm.all_gather(A_shared, R_shared, direction="all")

    elif comm_case == "allgather_axis_last_valid":
        send_shape = (32, 32)
        recv_shape = (32, 512)

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel() as _bx:
                A_shared = T.alloc_shared(send_shape, DTYPE)
                R_shared = T.alloc_shared(recv_shape, DTYPE)
                T.annotate_layout(
                    {
                        A_shared: make_row_major(send_shape),
                        R_shared: make_row_major(recv_shape),
                    }
                )
                T.comm.all_gather(A_shared, R_shared, direction="all", axis=-1)

    elif comm_case == "allgather_axis0_valid":
        send_shape = (32, 32)
        recv_shape = (512, 32)
        recv_layout = make_zz_layout(recv_shape, axes=[0, 1], block_shape=(32, 32))

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel() as _bx:
                A_shared = T.alloc_shared(send_shape, DTYPE)
                R_shared = T.alloc_shared(recv_shape, DTYPE)
                T.annotate_layout(
                    {
                        A_shared: make_row_major(send_shape),
                        R_shared: recv_layout,
                    }
                )
                T.comm.all_gather(A_shared, R_shared, direction="all", axis=0)

    elif comm_case == "allgather_axis0_slot_extent_mismatch":
        send_shape = (32, 32)
        recv_shape = (544, 32)

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel() as bx:
                A_shared = T.alloc_shared(send_shape, DTYPE)
                R_shared = T.alloc_shared(recv_shape, DTYPE)
                T.annotate_layout(
                    {
                        A_shared: make_row_major(send_shape),
                        R_shared: make_row_major(recv_shape),
                    }
                )
                T.evaluate(
                    tvm.tir.call_intrin(
                        "handle",
                        tvm.tir.op.Op.get("tl.tileop.comm_allgather"),
                        A_shared[0:32, 0:32],
                        R_shared[0:544, 0:32],
                        2,
                        -1,
                        0,
                        bx,
                    )
                )

    elif comm_case == "allgather_axis0_nonzero_recv_min_oob":
        send_shape = (32, 32)
        recv_shape = (512, 32)

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel() as _bx:
                A_shared = T.alloc_shared(send_shape, DTYPE)
                R_shared = T.alloc_shared(recv_shape, DTYPE)
                T.annotate_layout(
                    {
                        A_shared: make_row_major(send_shape),
                        R_shared: make_row_major(recv_shape),
                    }
                )
                T.comm.all_gather(
                    A_shared,
                    R_shared[32:544, 0:32],
                    direction="all",
                    axis=0,
                )

    elif comm_case == "allreduce_low_level_valid":
        send_shape = (32, 32)
        out_shape = send_shape
        gather_shape = (4, *send_shape)

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel() as bx:
                A_shared = T.alloc_shared(send_shape, DTYPE)
                Out_shared = T.alloc_shared(out_shape, DTYPE)
                Row_gather = T.alloc_shared(gather_shape, DTYPE)
                Col_gather = T.alloc_shared(gather_shape, DTYPE)
                T.annotate_layout(
                    {
                        A_shared: make_row_major(send_shape),
                        Out_shared: make_row_major(out_shape),
                        Row_gather: make_row_major(gather_shape),
                        Col_gather: make_row_major(gather_shape),
                    }
                )
                T.evaluate(
                    tvm.tir.call_intrin(
                        "handle",
                        tvm.tir.op.Op.get("tl.tileop.comm_allreduce"),
                        A_shared[0:32, 0:32],
                        Out_shared[0:32, 0:32],
                        Row_gather[0:4, 0:32, 0:32],
                        Col_gather[0:4, 0:32, 0:32],
                        "sum",
                        2,
                        T.bool(True),
                        bx,
                    )
                )

    elif comm_case in {
        "allreduce_low_level_bitwise",
        "allreduce_low_level_bitwise_int32",
        "allreduce_low_level_bitwise_uint32",
        "allreduce_low_level_missing_dst_copy",
    }:
        send_shape = (32, 32)
        gather_shape = (4, *send_shape)
        reduce_type = "bitand" if "bitwise" in comm_case else "sum"
        if comm_case == "allreduce_low_level_bitwise_int32":
            reduce_dtype = "int32"
        elif comm_case == "allreduce_low_level_bitwise_uint32":
            reduce_dtype = "uint32"
        else:
            reduce_dtype = DTYPE
        clear = comm_case != "allreduce_low_level_missing_dst_copy"

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel() as bx:
                A_shared = T.alloc_shared(send_shape, reduce_dtype)
                Out_shared = T.alloc_shared(send_shape, reduce_dtype)
                Row_gather = T.alloc_shared(gather_shape, reduce_dtype)
                Col_gather = T.alloc_shared(gather_shape, reduce_dtype)
                T.annotate_layout(
                    {
                        A_shared: make_row_major(send_shape),
                        Out_shared: make_row_major(send_shape),
                        Row_gather: make_row_major(gather_shape),
                        Col_gather: make_row_major(gather_shape),
                    }
                )
                T.evaluate(
                    tvm.tir.call_intrin(
                        "handle",
                        tvm.tir.op.Op.get("tl.tileop.comm_allreduce"),
                        A_shared[0:32, 0:32],
                        Out_shared[0:32, 0:32],
                        Row_gather[0:4, 0:32, 0:32],
                        Col_gather[0:4, 0:32, 0:32],
                        reduce_type,
                        2,
                        T.bool(clear),
                        bx,
                    )
                )

    else:
        raise ValueError(f"unknown comm case: {comm_case}")

    return tvm.IRModule({"main": kernel})


def _run_comm_validate_tile_view_regions(comm_case):
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = _make_comm_kernel(comm_case)
        mod = tvm.tir.transform.BindTarget(target)(mod)
        return tilelang.transform.ValidateTileViewRegions()(mod)


@pytest.mark.parametrize(
    "copy_case",
    [
        "row_major_aligned_grid_tile",
        "row_major_1d_tile_view",
        "row_major_dynamic_min_bounds_unknown",
        "row_major_dynamic_min_alignment_unknown",
        "row_major_dynamic_select_alignment_unknown",
        "row_major_3d_outer_slab_valid",
        "zz_3d_major_dim_split",
        "zz_block_equal",
        "zz_block_non_major_dim_multi_block",
        "zz_block_major_dim_split",
        "zz_whole_dim",
        "dynamic_extent_warns_and_passes",
        "dynamic_row_major_inner_mode_warns_and_passes",
        "dynamic_outer_static_extent_equal_block",
        "dynamic_outer_static_extent_split_block",
    ],
)
def test_sunmmio_validate_tile_view_regions_accepts_legal_regions(copy_case):
    _run_validate_tile_view_regions(copy_case)


@pytest.mark.parametrize(
    "copy_case, error_msg",
    [
        (
            "row_major_min_not_region_extent_aligned",
            "must align to region extent",
        ),
        (
            "row_major_extent_not_buffer_shape_factor",
            "must divide buffer shape",
        ),
        (
            "row_major_3d_outer_slab_oob",
            "must stay within buffer shape",
        ),
        (
            "row_major_3d_outer_slab_unclipped",
            "must stay within buffer shape",
        ),
        (
            "row_major_zero_extent",
            "must be positive",
        ),
        (
            "row_major_negative_extent",
            "must be positive",
        ),
        (
            "zz_block_both_dims_split",
            "must not split the non-major dimension",
        ),
        (
            "zz_block_non_major_dim_split",
            "must not split the non-major dimension",
        ),
        (
            "zn_block_non_major_dim_split",
            "must not split the non-major dimension",
        ),
        (
            "zz_block_offset_inside_tile",
            "must align to region extent",
        ),
        (
            "zz_extent_not_coalesced_compatible",
            "must be compatible with coalesced extent",
        ),
        (
            "dynamic_outer_static_extent_incompatible",
            "must be compatible with dynamic layout inner static mode shape",
        ),
    ],
)
def test_sunmmio_validate_tile_view_regions_rejects_illegal_regions(copy_case, error_msg):
    with pytest.raises(tvm.error.InternalError, match=error_msg):
        _run_validate_tile_view_regions(copy_case)


def test_sunmmio_validate_tile_view_regions_accepts_kernel_after_layout_inference():
    _run_kernel_pipeline_to_validate_tile_view_regions("zz_block_equal")


@pytest.mark.parametrize(
    "copy_case, error_msg",
    [
        (
            "row_major_min_not_region_extent_aligned",
            "must align to region extent",
        ),
        (
            "zz_extent_not_coalesced_compatible",
            "must be compatible with coalesced extent",
        ),
    ],
)
def test_sunmmio_validate_tile_view_regions_rejects_kernel_after_layout_inference(copy_case, error_msg):
    with pytest.raises(tvm.error.InternalError, match=error_msg):
        _run_kernel_pipeline_to_validate_tile_view_regions(copy_case)


@pytest.mark.parametrize(
    "comm_case",
    [
        "broadcast_valid",
        "put_valid",
        "allgather_legacy_valid",
        "allgather_axis0_valid",
        "allgather_axis_last_valid",
        "allreduce_low_level_valid",
        "allreduce_low_level_bitwise_int32",
        "allreduce_low_level_bitwise_uint32",
    ],
)
def test_sunmmio_validate_tile_view_regions_accepts_comm_regions(comm_case):
    _run_comm_validate_tile_view_regions(comm_case)


@pytest.mark.parametrize(
    "comm_case, error_msg",
    [
        (
            "broadcast_illegal_non_major_split",
            "must not split the non-major dimension",
        ),
        (
            "allgather_axis0_slot_extent_mismatch",
            "must equal send extent",
        ),
    ],
)
def test_sunmmio_validate_tile_view_regions_rejects_comm_regions(comm_case, error_msg):
    with pytest.raises(tvm.error.InternalError, match=error_msg):
        _run_comm_validate_tile_view_regions(comm_case)


@pytest.mark.parametrize(
    "comm_case,error_msg",
    [
        (
            "allreduce_low_level_bitwise",
            "Sunmmio bitwise all_reduce requires input and output dtypes to be both int32 or both uint32",
        ),
        ("allreduce_low_level_missing_dst_copy", "dst_copy must be provided only"),
    ],
)
def test_sunmmio_validate_tile_view_regions_rejects_invalid_allreduce_contract(comm_case, error_msg):
    with pytest.raises(tvm.error.InternalError, match=error_msg):
        _run_comm_validate_tile_view_regions(comm_case)


def test_sunmmio_comm_frontend_rejects_broadcast_source_oob(_strict_region_validation):
    with pytest.raises(ValueError, match="region starts outside buffer shape"):
        _make_comm_kernel("broadcast_scalar_src_oob")


def test_sunmmio_comm_frontend_clips_then_rejects_allgather_recv_oob(_strict_region_validation):
    with (
        pytest.warns(UserWarning, match="explicit BufferRegion exceeds buffer shape and will be clipped"),
        pytest.raises(ValueError, match="src extent is larger than dst extent"),
    ):
        _make_comm_kernel("allgather_axis0_nonzero_recv_min_oob")


if __name__ == "__main__":
    tilelang.testing.main()
