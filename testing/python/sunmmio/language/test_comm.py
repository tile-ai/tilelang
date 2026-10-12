import warnings

import pytest

import tilelang
import tilelang.utils.target as _target_utils
import tilelang.language as T

from tilelang import tvm as tvm
from tilelang.utils.target import determine_target


def _broadcast_lines(script):
    return [line.strip() for line in script.splitlines() if "T.broadcast_(" in line]


def _collect_calls(func, op_name):
    calls = []
    op = tvm.tir.op.Op.get(op_name)

    def visit(node):
        if isinstance(node, tvm.tir.Call) and node.op.same_as(op):
            calls.append(node)

    tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    return calls


def _broadcast_line(src, dst, direction, mask, core, src_offset=0):
    return f"T.broadcast_({src}, {dst}, {direction}, T.int64({mask}), {src_offset}, {core})"


def _broadcast_line_no_core(src, dst, direction, mask, src_offset=0):
    return f"T.broadcast_({src}, {dst}, {direction}, {mask}, {src_offset})"


@pytest.fixture
def _strict_region_validation():
    previous = _target_utils.ENABLE_SUNMMIO_REGION_VALIDATION
    _target_utils.set_sunmmio_region_validation(True)
    try:
        yield
    finally:
        _target_utils.set_sunmmio_region_validation(previous)


_ROW_MASK_BX = "T.int64(15)"

_COL_MASK_BX = "T.int64(15)"


def _expected_axis0_all_lines(buffer):
    src = "T.region(A_shared[0, 0], 1, 128, 128)"
    lines = [_broadcast_line_no_core(src, f"T.region({buffer}[bx * 128, 0], 2, 128, 128)", 1, _COL_MASK_BX)]
    for row in range(4):
        offset = "bx % 4 * 128" if row == 0 else f"bx % 4 * 128 + {row * 512}"
        region = f"T.region({buffer}[{offset}, 0], {{access}}, 128, 128)"
        lines.append(_broadcast_line_no_core(region.format(access=1), region.format(access=2), 0, _ROW_MASK_BX))
    return lines


def _expected_axis_last_horizontal_lines(buffer):
    src = "T.region(A_shared[0, 0], 1, 128, 128)"
    return [
        _broadcast_line_no_core(
            src,
            f"T.region({buffer}[0, bx % 4 * 128], 2, 128, 128)",
            0,
            _ROW_MASK_BX,
        )
    ]


def _expected_axis_last_all_lines(buffer):
    src = "T.region(A_shared[0, 0], 1, 128, 128)"
    lines = [_broadcast_line_no_core(src, f"T.region({buffer}[0, bx * 128], 2, 128, 128)", 1, _COL_MASK_BX)]
    for row in range(4):
        offset = "bx % 4 * 128" if row == 0 else f"bx % 4 * 128 + {row * 512}"
        region = f"T.region({buffer}[0, {offset}], {{access}}, 128, 128)"
        lines.append(_broadcast_line_no_core(region.format(access=1), region.format(access=2), 0, _ROW_MASK_BX))
    return lines


def _allreduce_frontend_kernel(direction="all", clear=True, dtype="float32"):
    shape = (32, 32)
    out_shape = (32,)

    @T.prim_func
    def main(A: T.Tensor(shape, dtype), Out: T.Tensor(out_shape, dtype)):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared(shape, dtype, scope="shared.rsram")
            Out_shared = T.alloc_shared(out_shape, dtype, scope="shared.rsram")

            T.copy(A, A_shared)
            if not clear:
                T.copy(Out, Out_shared)
            T.comm.all_reduce(A_shared, Out_shared, "sum", direction, dim=1, clear=clear)
            T.copy(Out_shared, Out)

    return main


@pytest.mark.parametrize(
    "M, N, block_M, block_N, dtype, accum_dtype",
    [
        (1024, 1024, 128, 128, "float16", "float"),
    ],
)
def test_comm_python_api(M, N, block_M, block_N, dtype, accum_dtype):
    func_str = """
        T.comm_broadcast(A_shared[0:128, 0:128], B_shared[0:128, 0:128], -1, 6, 2)
        T.comm_put(A_shared[0:128, 0:128], B_shared[0:128, 0:128], -1, 6, 11)
        T.comm_allgather(A_shared[0:128, 0:128], C_shared[0:16, 0:128, 0:128], 2, -1, -1, bx)
        T.comm_allgather(A_shared[0:128, 0:128], R0_shared[0:2048, 0:128], 2, -1, 0, bx)
        T.comm_allgather(A_shared[0:128, 0:128], R1_shared[0:128, 0:2048], 2, -1, 1, bx)""".strip()

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared([block_M, block_N], dtype, scope="shared.rsram")
            B_shared = T.alloc_shared([block_M, block_N], dtype, scope="shared.rsram")
            C_shared = T.alloc_shared([16, block_M, block_N], dtype, scope="shared.rsram")
            R0_shared = T.alloc_shared([16 * block_M, block_N], dtype, scope="shared.rsram")
            R1_shared = T.alloc_shared([block_M, 16 * block_N], dtype, scope="shared.rsram")
            T.copy(A[by * block_M, bx * block_N], A_shared)

            T.comm.broadcast(A_shared, B_shared, (1, 2), direction="all")
            T.comm.put(A_shared, B_shared, (1, 2), (2, 3))
            T.comm.all_gather(A_shared, C_shared, direction="all")
            T.comm.all_gather(A_shared, R0_shared, direction="all", axis=0)
            T.comm.all_gather(A_shared, R1_shared, direction="all", axis=-1)

    assert main.script()[-len(func_str) :] == func_str, "The generated script does not match the expected output."


def test_comm_allreduce_frontend_allocates_rsram_temporaries():
    script = _allreduce_frontend_kernel(direction="all", clear=False).script()

    assert 'buffer = T.alloc_buffer((4, 32), scope="shared.rsram")' in script
    assert 'buffer_1 = T.alloc_buffer((4, 32), scope="shared.rsram")' in script
    assert 'buffer_2 = T.alloc_buffer((32,), scope="shared.rsram")' in script
    assert (
        "T.comm_allreduce(A_shared[0:32, 0:32], Out_shared[0:32], "
        'buffer[0:4, 0:32], buffer_1[0:4, 0:32], "sum", 2, 1, '
        "T.bool(False), buffer_2[0:32], bx)"
    ) in script


@pytest.mark.parametrize("direction", ["h", "horizontal", "v", "vertical", "all"])
@pytest.mark.parametrize("clear", [True, False])
@pytest.mark.parametrize("mesh", [(4, 4), (2, 3)])
def test_allreduce_allocates_only_active_carriers(direction, clear, mesh, monkeypatch):
    nrow, ncol = mesh
    monkeypatch.setattr(T.comm, "get_target_mesh_shape", lambda: {"nrow": nrow, "ncol": ncol})
    func = _allreduce_frontend_kernel(direction=direction, clear=clear)
    (call,) = _collect_calls(func, "tl.tileop.comm_allreduce")
    carriers = []

    def collect(node):
        if isinstance(node, tvm.tir.Block):
            carriers.extend(buffer for buffer in node.alloc_buffers if len(buffer.shape) == 2 and not buffer.same_as(call.args[0].buffer))

    tvm.tir.stmt_functor.post_order_visit(func.body, collect)
    assert len(carriers) == (2 if direction == "all" else 1)
    assert call.args[2].buffer.same_as(call.args[3].buffer) == (direction != "all")
    if direction in ("h", "horizontal", "all"):
        assert int(call.args[2].buffer.shape[0]) == ncol
    if direction in ("v", "vertical", "all"):
        assert int(call.args[3].buffer.shape[0]) == nrow


def test_comm_buffer_like_region_python_api():
    @T.prim_func
    def main(A: T.Tensor((128, 128), "float32")):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared([128, 128], "float32", scope="shared.rsram")
            B_shared = T.alloc_shared([128, 128], "float32", scope="shared.rsram")
            C_shared = T.alloc_shared([4, 64, 64], "float32", scope="shared.rsram")
            Out_shared = T.alloc_shared([128], "float32", scope="shared.rsram")
            T.copy(A, A_shared)

            T.comm.broadcast(A_shared[8:72, 16:80], B_shared[24:88, 32:96], (0, 0), direction="h")
            T.comm.put(A_shared[0:64, 0:64], B_shared[32:96, 32:96], (0, 0), (0, 1))
            T.comm.all_gather(A_shared[8:72, 16:80], C_shared[0:4, 0:64, 0:64], direction="h")
            T.comm.all_reduce(A_shared[8:72, 16:80], Out_shared[32:96], "sum", "h", dim=1)

    script = main.script()
    assert "T.comm_broadcast(A_shared[8:72, 16:80], B_shared[24:88, 32:96], -1, 0, 0)" in script
    assert "T.comm_put(A_shared[0:64, 0:64], B_shared[32:96, 32:96], -1, 0, 1)" in script
    assert "T.comm_allgather(A_shared[8:72, 16:80], C_shared[0:4, 0:64, 0:64], 0, -1, -1, bx)" in script
    assert (
        'T.comm_allreduce(A_shared[8:72, 16:80], Out_shared[32:96], buffer[0:4, 0:64], buffer[0:4, 0:64], "sum", 0, 1, T.bool(True), bx)'
    ) in script


def test_comm_one_to_one_regions_follow_copy_normalization_rules(_strict_region_validation):
    @T.prim_func
    def main():
        with T.Kernel():
            src = T.alloc_shared((1, 64, 64), "float32", scope="shared.rsram")
            broadcast_dst = T.alloc_shared((64, 64), "float32", scope="shared.rsram")
            put_dst = T.alloc_shared((32, 32), "float32", scope="shared.rsram")
            shrink_dst = T.alloc_shared((64, 64), "float32", scope="shared.rsram")

            T.comm.broadcast(src, broadcast_dst, (0, 0), direction="h")
            T.comm.put(src[0, 16, 16], put_dst, (0, 0), (0, 1))
            T.comm.broadcast(src[0:1, 0:32, 0:32], shrink_dst, (0, 0), direction="h")

    script = main.script()
    assert "T.comm_broadcast(src[0, 0:64, 0:64], broadcast_dst[0:64, 0:64], -1, 0, 0)" in script
    assert "T.comm_put(src[0, 16:48, 16:48], put_dst[0:32, 0:32], -1, 0, 1)" in script
    assert "T.comm_broadcast(src[0, 0:32, 0:32], shrink_dst[0:32, 0:32], -1, 0, 0)" in script


def test_comm_compact_path_preserves_original_regions():
    previous = _target_utils.ENABLE_SUNMMIO_REGION_VALIDATION
    _target_utils.set_sunmmio_region_validation(False)
    try:

        @T.prim_func
        def main():
            with T.Kernel():
                src = T.alloc_shared((4, 32, 32), "float32", scope="shared.rsram")
                broadcast_dst = T.alloc_shared((4, 32, 32), "float32", scope="shared.rsram")
                put_dst = T.alloc_shared((4, 32, 32), "float32", scope="shared.rsram")
                gather_send = T.alloc_shared((32, 32), "float32", scope="shared.rsram")
                gather_recv = T.alloc_shared((4, 32, 32), "float32", scope="shared.rsram")
                reduce_src = T.alloc_shared((32, 64), "float32", scope="shared.rsram")
                reduce_out = T.alloc_shared((32,), "float32", scope="shared.rsram")

                T.comm.broadcast(src[0:1, 0:32, 0:32], broadcast_dst[0:4, 0:32, 0:32], (0, 0), direction="h")
                T.comm.put(src[0:1, 0:32, 0:32], put_dst[0:4, 0:32, 0:32], (0, 0), (0, 1))
                T.comm.all_gather(gather_send, gather_recv, direction="h")
                T.comm.all_reduce(reduce_src, reduce_out, "sum", "h", dim=1)

        script = main.script()
        assert "T.comm_broadcast(src[0, 0:32, 0:32], broadcast_dst[0:4, 0:32, 0:32]" in script
        assert "T.comm_put(src[0, 0:32, 0:32], put_dst[0:4, 0:32, 0:32]" in script
        assert "T.comm_allgather(gather_send[0:32, 0:32], gather_recv[0:4, 0:32, 0:32]" in script
        assert "T.comm_allreduce(reduce_src[0:32, 0:64], reduce_out[0:32]" in script
    finally:
        _target_utils.set_sunmmio_region_validation(previous)


def test_comm_compact_path_keeps_original_dynamic_extent_behavior():
    previous = _target_utils.ENABLE_SUNMMIO_REGION_VALIDATION
    _target_utils.set_sunmmio_region_validation(False)
    try:
        dynamic_extent = tvm.tir.Var("dynamic_extent", "int32")
        src = tvm.tir.decl_buffer((dynamic_extent, 32), "float32", name="src", scope="shared.rsram")
        same_shape_dst = tvm.tir.decl_buffer((dynamic_extent, 32), "float32", name="same_shape_dst", scope="shared.rsram")
        static_dst = tvm.tir.decl_buffer((64, 32), "float32", name="static_dst", scope="shared.rsram")

        with pytest.raises(ValueError, match="Cannot convert to BufferLoad"):
            T.comm.broadcast(src, same_shape_dst, 0, direction="h")

        with pytest.raises(ValueError, match="same number of dimensions for broadcast"):
            T.comm.broadcast(src, static_dst, 0, direction="h")

        gather_recv = tvm.tir.decl_buffer((4, 64, 32), "float32", name="gather_recv", scope="shared.rsram")
        with pytest.raises(AssertionError, match="Receive buffer shape"):
            T.comm.all_gather(src, gather_recv, direction="h")

        reduce_out = tvm.tir.decl_buffer((64,), "float32", name="reduce_out", scope="shared.rsram")
        with pytest.raises(ValueError, match="Invalid reduce output shape"):
            T.comm.all_reduce(src, reduce_out, "sum", "h", dim=1)
    finally:
        _target_utils.set_sunmmio_region_validation(previous)


def test_comm_compact_path_accepts_unresolved_mesh_shape_without_warning():
    previous = _target_utils.ENABLE_SUNMMIO_REGION_VALIDATION
    _target_utils.set_sunmmio_region_validation(False)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*cannot resolve mesh-symbol extents.*")

            @T.prim_func
            def main():
                with T.Kernel():
                    send = T.alloc_shared((32, 32), "float32", scope="shared.rsram")
                    recv = T.alloc_shared((32, 32 * T.ncols()), "float32", scope="shared.rsram")
                    reduce_src = T.alloc_shared((32 * T.ncols(), 32), "float32", scope="shared.rsram")
                    reduce_out = T.alloc_shared((128,), "float32", scope="shared.rsram")
                    T.comm.all_gather(send, recv, direction="h", axis=-1)
                    T.comm.all_reduce(reduce_src, reduce_out, "sum", "h", dim=1)

        assert "mesh_ncols" in main.script()
    finally:
        _target_utils.set_sunmmio_region_validation(previous)


def test_comm_compact_path_keeps_static_checks_with_unresolved_mesh_shape():
    previous = _target_utils.ENABLE_SUNMMIO_REGION_VALIDATION
    _target_utils.set_sunmmio_region_validation(False)
    try:
        with pytest.raises(AssertionError, match="Receive buffer shape"):

            @T.prim_func
            def main():
                with T.Kernel():
                    send = T.alloc_shared((32, 32), "float32", scope="shared.rsram")
                    recv = T.alloc_shared((64, 32 * T.ncols()), "float32", scope="shared.rsram")
                    T.comm.all_gather(send, recv, direction="h", axis=-1)
    finally:
        _target_utils.set_sunmmio_region_validation(previous)


def test_comm_one_to_one_full_buffer_extent_mismatch_is_rejected(_strict_region_validation):
    with pytest.raises(ValueError, match="exact match is required for Buffer-to-Buffer operation"):

        @T.prim_func
        def main():
            with T.Kernel():
                src = T.alloc_shared((32, 32), "float32", scope="shared.rsram")
                dst = T.alloc_shared((32, 64), "float32", scope="shared.rsram")
                T.comm.broadcast(src, dst, (0, 0), direction="h")


def test_comm_explicit_region_oob_is_warned_and_clipped(_strict_region_validation):
    with pytest.warns(UserWarning, match="T.comm.put explicit BufferRegion exceeds buffer shape"):

        @T.prim_func
        def main():
            with T.Kernel():
                src = T.alloc_shared((32, 32), "float32", scope="shared.rsram")
                dst = T.alloc_shared((32, 32), "float32", scope="shared.rsram")
                T.comm.put(src[0:32, 0:32], dst[0:64, 0:64], (0, 0), (0, 1))

    assert "T.comm_put(src[0:32, 0:32], dst[0:32, 0:32], -1, 0, 1)" in main.script()


def test_comm_all_gather_regions_follow_copy_normalization_rules(_strict_region_validation):
    @T.prim_func
    def main():
        with T.Kernel():
            send = T.alloc_shared((64, 64), "float32", scope="shared.rsram")
            recv_from_load = T.alloc_shared((4, 32, 32), "float32", scope="shared.rsram")
            recv_load = T.alloc_shared((4, 64, 64), "float32", scope="shared.rsram")
            recv_shrink = T.alloc_shared((4, 64, 64), "float32", scope="shared.rsram")
            recv_axis_shrink = T.alloc_shared((130, 64), "float32", scope="shared.rsram")

            T.comm.all_gather(send[16, 16], recv_from_load, direction="h")
            T.comm.all_gather(send[0:32, 0:32], recv_load[0, 0, 0], direction="h")
            T.comm.all_gather(send[0:32, 0:32], recv_shrink, direction="h")
            T.comm.all_gather(send[0:32, 0:32], recv_axis_shrink, direction="h", axis=0)

    script = main.script()
    assert "T.comm_allgather(send[16:48, 16:48], recv_from_load[0:4, 0:32, 0:32], 0, -1, -1," in script
    assert "T.comm_allgather(send[0:32, 0:32], recv_load[0:4, 0:32, 0:32], 0, -1, -1," in script
    assert "T.comm_allgather(send[0:32, 0:32], recv_shrink[0:4, 0:32, 0:32], 0, -1, -1," in script
    assert "T.comm_allgather(send[0:32, 0:32], recv_axis_shrink[0:128, 0:32], 0, -1, 0," in script


def test_comm_all_gather_buffer_load_pair_is_rejected(_strict_region_validation):
    with pytest.raises(ValueError, match="both operands are BufferLoad"):

        @T.prim_func
        def main():
            with T.Kernel():
                send = T.alloc_shared((32, 32), "float32", scope="shared.rsram")
                recv = T.alloc_shared((4, 32, 32), "float32", scope="shared.rsram")
                T.comm.all_gather(send[0, 0], recv[0, 0, 0], direction="h")


def test_comm_all_gather_full_buffer_extent_mismatch_is_rejected(_strict_region_validation):
    with pytest.raises(ValueError, match="exact match is required for Buffer-to-Buffer operation"):

        @T.prim_func
        def main():
            with T.Kernel():
                send = T.alloc_shared((32, 32), "float32", scope="shared.rsram")
                recv = T.alloc_shared((4, 64, 32), "float32", scope="shared.rsram")
                T.comm.all_gather(send, recv, direction="h")


def test_comm_all_reduce_regions_follow_copy_normalization_rules(_strict_region_validation):
    @T.prim_func
    def main():
        with T.Kernel():
            src = T.alloc_shared((64, 64), "float32", scope="shared.rsram")
            out_from_load = T.alloc_shared((32,), "float32", scope="shared.rsram")
            out_load = T.alloc_shared((64,), "float32", scope="shared.rsram")
            out_shrink = T.alloc_shared((64,), "float32", scope="shared.rsram")

            T.comm.all_reduce(src[16, 0], out_from_load, "sum", "h", dim=1)
            T.comm.all_reduce(src[0:32, 0:64], out_load[16], "sum", "h", dim=1)
            T.comm.all_reduce(src[0:32, 0:64], out_shrink, "sum", "h", dim=1)

    script = main.script()
    assert "T.comm_allreduce(src[16:48, 0:64], out_from_load[0:32]" in script
    assert "T.comm_allreduce(src[0:32, 0:64], out_load[16:48]" in script
    assert "T.comm_allreduce(src[0:32, 0:64], out_shrink[0:32]" in script


def test_comm_all_reduce_full_buffer_extent_mismatch_is_rejected(_strict_region_validation):
    with pytest.raises(ValueError, match="exact match is required for Buffer-to-Buffer operation"):

        @T.prim_func
        def main():
            with T.Kernel():
                src = T.alloc_shared((32, 64), "float32", scope="shared.rsram")
                out = T.alloc_shared((64,), "float32", scope="shared.rsram")
                T.comm.all_reduce(src, out, "sum", "h", dim=1)


def test_comm_all_reduce_buffer_load_pair_is_rejected(_strict_region_validation):
    with pytest.raises(ValueError, match="both operands are BufferLoad"):

        @T.prim_func
        def main():
            with T.Kernel():
                src = T.alloc_shared((32, 32), "float32", scope="shared.rsram")
                out = T.alloc_shared((32,), "float32", scope="shared.rsram")
                T.comm.all_reduce(src[0, 0], out[0], "sum", "h", dim=1)


def test_comm_dynamic_core_python_api():
    @T.prim_func
    def main(A: T.Tensor((128, 128), "float32")):
        with T.Kernel(16, threads=128) as bx:
            A_shared = T.alloc_shared([128, 128], "float32", scope="shared.rsram")
            B_shared = T.alloc_shared([128, 128], "float32", scope="shared.rsram")
            T.copy(A, A_shared)

            T.comm.broadcast(A_shared, B_shared, bx, direction="h")
            T.comm.put(A_shared, B_shared, bx, (bx + 1) % 16)

    broadcast_calls = _collect_calls(main, "tl.tileop.comm_broadcast")
    put_calls = _collect_calls(main, "tl.tileop.comm_put")
    assert len(broadcast_calls) == 1
    assert len(put_calls) == 1
    assert isinstance(broadcast_calls[0].args[3], tvm.tir.Var)
    assert tvm.ir.structural_equal(broadcast_calls[0].args[3], put_calls[0].args[3])
    assert tvm.ir.structural_equal(put_calls[0].args[4], (put_calls[0].args[3] + 1) % 16)


def test_comm_dynamic_core_lower():
    @T.prim_func
    def main(A: T.Tensor((128, 128), "float32")):
        with T.Kernel(16, threads=128) as bx:
            A_shared = T.alloc_shared([128, 128], "float32", scope="shared.rsram")
            B_shared = T.alloc_shared([128, 128], "float32", scope="shared.rsram")
            T.copy(A, A_shared)

            T.comm.broadcast(A_shared, B_shared, bx, direction="all")
            T.comm.put(A_shared, B_shared, bx, (bx + 1) % 16)

    mod = tvm.IRModule({"main": main})
    target = determine_target("Sunmmio", return_object=True)
    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tilelang.transform.LowerTileOp()(mod)

    broadcast_calls = _collect_calls(mod["main"], "tl.broadcast_")
    if_nodes = []

    def visit(node):
        if isinstance(node, tvm.tir.IfThenElse):
            if_nodes.append(node)

    tvm.tir.stmt_functor.post_order_visit(mod["main"].body, visit)
    assert len(broadcast_calls) == 9
    assert all(len(call.args) == 6 for call in broadcast_calls)
    assert if_nodes, "Dynamic put lowering should emit runtime routing branches."


@pytest.mark.parametrize(
    "M, N, block_M, block_N, dtype, accum_dtype",
    [
        (1024, 1024, 128, 128, "float16", "float"),
    ],
)
def test_comm_broadcast_lower(M, N, block_M, block_N, dtype, accum_dtype):
    expected = [
        "T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(B_shared[0, 0], 2, 128, 128), 1, T.int64(15), 0, 6)",
        "T.broadcast_(T.region(B_shared[0, 0], 1, 128, 128), T.region(B_shared[0, 0], 2, 128, 128), 0, T.int64(15), 0, 2)",
        "T.broadcast_(T.region(B_shared[0, 0], 1, 128, 128), T.region(B_shared[0, 0], 2, 128, 128), 0, T.int64(15), 0, 6)",
        "T.broadcast_(T.region(B_shared[0, 0], 1, 128, 128), T.region(B_shared[0, 0], 2, 128, 128), 0, T.int64(15), 0, 10)",
        "T.broadcast_(T.region(B_shared[0, 0], 1, 128, 128), T.region(B_shared[0, 0], 2, 128, 128), 0, T.int64(15), 0, 14)",
    ]

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared([block_M, block_N], dtype, scope="shared.rsram")
            B_shared = T.alloc_shared([block_M, block_N], dtype, scope="shared.rsram")
            T.copy(A[by * block_M, bx * block_N], A_shared)

            T.comm.broadcast(A_shared, B_shared, (1, 2), direction="all")

    mod = tvm.IRModule({"main": main})
    target = determine_target("Sunmmio", return_object=True)
    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tilelang.transform.LowerTileOp()(mod)
        assert _broadcast_lines(mod.script()) == expected, "The generated script does not match the expected output."


@pytest.mark.parametrize(
    "M, N, block_M, block_N, dtype",
    [
        (1024, 1024, 128, 128, "float16"),
    ],
)
def test_comm_broadcast_lower_custom_mesh(M, N, block_M, block_N, dtype):
    expected = [
        "T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(B_shared[0, 0], 2, 128, 128), 1, T.int64(3), 0, 5)",
        "T.broadcast_(T.region(B_shared[0, 0], 1, 128, 128), T.region(B_shared[0, 0], 2, 128, 128), 0, T.int64(7), 0, 2)",
        "T.broadcast_(T.region(B_shared[0, 0], 1, 128, 128), T.region(B_shared[0, 0], 2, 128, 128), 0, T.int64(7), 0, 5)",
    ]

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared([block_M, block_N], dtype, scope="shared.rsram")
            B_shared = T.alloc_shared([block_M, block_N], dtype, scope="shared.rsram")
            T.copy(A[by * block_M, bx * block_N], A_shared)

            T.evaluate(
                tvm.tir.call_intrin(
                    "handle",
                    tvm.tir.op.Op.get("tl.tileop.comm_broadcast"),
                    A_shared[0:128, 0:128],
                    B_shared[0:128, 0:128],
                    -1,
                    5,
                    2,
                )
            )

    mod = tvm.IRModule({"main": main})
    target = determine_target("llvm -mcpu=sunmmio-a4e -mattr=device_mesh_nrow_2,device_mesh_ncol_3", return_object=True)
    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tilelang.transform.LowerTileOp()(mod)
        assert _broadcast_lines(mod.script()) == expected, "The generated script does not match the expected output."


@pytest.mark.parametrize(
    "M, N, block_M, block_N, dtype, accum_dtype",
    [
        (1024, 1024, 128, 128, "float16", "float"),
    ],
)
def test_comm_put_lower(M, N, block_M, block_N, dtype, accum_dtype):
    expected = [
        "T.broadcast_(T.region(A_shared[0, 0], 1, 128, 128), T.region(B_shared[0, 0], 2, 128, 128), 1, T.int64(4), 0, 6)",
        "T.broadcast_(T.region(B_shared[0, 0], 1, 128, 128), T.region(B_shared[0, 0], 2, 128, 128), 0, T.int64(8), 0, 10)",
    ]

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared([block_M, block_N], dtype, scope="shared.rsram")
            B_shared = T.alloc_shared([block_M, block_N], dtype, scope="shared.rsram")
            T.copy(A[by * block_M, bx * block_N], A_shared)

            T.comm.put(A_shared, B_shared, (1, 2), (2, 3))

    mod = tvm.IRModule({"main": main})
    target = determine_target("Sunmmio", return_object=True)
    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tilelang.transform.LowerTileOp()(mod)
        assert _broadcast_lines(mod.script()) == expected, "The generated script does not match the expected output."


@pytest.mark.parametrize(
    "M, N, block_M, block_N, dtype, accum_dtype",
    [
        (1024, 1024, 128, 128, "float16", "float"),
    ],
)
def test_comm_all_gather_lower(M, N, block_M, block_N, dtype, accum_dtype):
    expected = [
        _broadcast_line_no_core(
            "T.region(A_shared[0, 0], 1, 128, 128)",
            "T.region(C_shared[bx, 0, 0], 2, 1, 128, 128)",
            1,
            _COL_MASK_BX,
        )
    ]
    for row in range(4):
        slot = "bx % 4" if row == 0 else f"bx % 4 + {row * 4}"
        region = f"T.region(C_shared[{slot}, 0, 0], {{access}}, 1, 128, 128)"
        expected.append(_broadcast_line_no_core(region.format(access=1), region.format(access=2), 0, _ROW_MASK_BX))

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared([block_M, block_N], dtype, scope="shared.rsram")
            C_shared = T.alloc_shared([16, block_M, block_N], dtype, scope="shared.rsram")
            T.copy(A[by * block_M, bx * block_N], A_shared)

            T.comm.all_gather(A_shared, C_shared, direction="all")

    mod = tvm.IRModule({"main": main})
    target = determine_target("Sunmmio", return_object=True)
    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tilelang.transform.LowerTileOp()(mod)
        assert _broadcast_lines(mod.script()) == expected, "The generated script does not match the expected output."


@pytest.mark.parametrize(
    "M, N, block_M, block_N, dtype",
    [
        (1024, 1024, 128, 128, "float16"),
    ],
)
def test_comm_all_gather_axis0_all_lower(M, N, block_M, block_N, dtype):
    expected = _expected_axis0_all_lines("R_shared")

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared([block_M, block_N], dtype, scope="shared.rsram")
            # 4x4 mesh -> 16 cores gathered along axis=0: [16 * block_M, block_N]
            R_shared = T.alloc_shared([16 * block_M, block_N], dtype, scope="shared.rsram")
            T.copy(A[by * block_M, bx * block_N], A_shared)

            T.comm.all_gather(A_shared, R_shared, direction="all", axis=0)

    mod = tvm.IRModule({"main": main})
    target = determine_target("Sunmmio", return_object=True)
    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tilelang.transform.LowerTileOp()(mod)
        assert _broadcast_lines(mod.script()) == expected, "The generated script does not match the expected output."


@pytest.mark.parametrize(
    "M, N, block_M, block_N, dtype",
    [
        (1024, 1024, 128, 128, "float16"),
    ],
)
def test_comm_all_gather_axis_last_horizontal_lower(M, N, block_M, block_N, dtype):
    expected = _expected_axis_last_horizontal_lines("R_shared")

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared([block_M, block_N], dtype, scope="shared.rsram")
            # Horizontal allgather (K = mesh_ncol = 4) along last axis: [block_M, 4 * block_N]
            R_shared = T.alloc_shared([block_M, 4 * block_N], dtype, scope="shared.rsram")
            T.copy(A[by * block_M, bx * block_N], A_shared)

            T.comm.all_gather(A_shared, R_shared, direction="horizontal", axis=-1)

    mod = tvm.IRModule({"main": main})
    target = determine_target("Sunmmio", return_object=True)
    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tilelang.transform.LowerTileOp()(mod)
        assert _broadcast_lines(mod.script()) == expected, "The generated script does not match the expected output."


@pytest.mark.parametrize(
    "M, N, block_M, block_N, dtype",
    [
        (1024, 1024, 128, 128, "float16"),
    ],
)
def test_comm_all_gather_axis_last_all_lower(M, N, block_M, block_N, dtype):
    expected = _expected_axis_last_all_lines("R_shared")

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared([block_M, block_N], dtype, scope="shared.rsram")
            # All direction (K = 16) gathered along last axis: [block_M, 16 * block_N]
            R_shared = T.alloc_shared([block_M, 16 * block_N], dtype, scope="shared.rsram")
            T.copy(A[by * block_M, bx * block_N], A_shared)

            T.comm.all_gather(A_shared, R_shared, direction="all", axis=-1)

    mod = tvm.IRModule({"main": main})
    target = determine_target("Sunmmio", return_object=True)
    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tilelang.transform.LowerTileOp()(mod)
        assert _broadcast_lines(mod.script()) == expected, "The generated script does not match the expected output."


'''
@pytest.mark.parametrize(
    "M, N, block_M, block_N, dtype, accum_dtype",
    [
        (1024 * 128, 1024 * 128, 1024, 1024, "float16", "float"),
    ],
)
def test_comm_all_reduce_lower(M, N, block_M, block_N, dtype, accum_dtype):
    func_str = """
                for i in T.parallel(1024):
                    for j in T.parallel(256):
                        for vec in T.vectorized(4):
                            A_shared[i * 8 + (j * 4 + vec) // 512 * 4 + (j * 4 + vec) % 4] = T.Cast("float32", A[by * 1024 + i, bx * 1024 + (j * 4 + vec)])
                for i in T.unroll(1024, annotations={"pragma_unroll_explicit": T.bool(False)}):
                    buffer_2[i] = T.float32(0.0)
                    for rv in T.unroll(8, annotations={"pragma_unroll_explicit": T.bool(False)}):
                        buffer_2[i] = buffer_2[i] + A_shared[i * 8 + rv % 2 * 4 + rv // 2]
                    buffer_2[i] = T.call_extern("float32", "tl::AllReduce<tl::SumOp, 128, 1, 0>::run", buffer_2[i], T.tvm_access_ptr(T.type_annotation("float32"), workspace.data, 0, 128, 2))
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer.data, 0, 1024, 2), 1024, 0, 0)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer.data, 1024, 1024, 2), 1024, 1, 0)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer.data, 2048, 1024, 2), 1024, 2, 0)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer.data, 3072, 1024, 2), 1024, 3, 0)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer.data, 0, 1024, 2), 1024, 4, 0)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer.data, 1024, 1024, 2), 1024, 5, 0)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer.data, 2048, 1024, 2), 1024, 6, 0)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer.data, 3072, 1024, 2), 1024, 7, 0)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer.data, 0, 1024, 2), 1024, 8, 0)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer.data, 1024, 1024, 2), 1024, 9, 0)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer.data, 2048, 1024, 2), 1024, 10, 0)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer.data, 3072, 1024, 2), 1024, 11, 0)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer.data, 0, 1024, 2), 1024, 12, 0)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer.data, 1024, 1024, 2), 1024, 13, 0)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer.data, 2048, 1024, 2), 1024, 14, 0)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer.data, 3072, 1024, 2), 1024, 15, 0)
                for i in T.unroll(1024, annotations={"pragma_unroll_explicit": T.bool(False)}):
                    buffer_2[i] = T.float32(0.0)
                    for rv in T.unroll(4, annotations={"pragma_unroll_explicit": T.bool(False)}):
                        buffer_2[i] = buffer_2[i] + buffer[rv * 8 + i // 512 * 4 + i % 4]
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer_1.data, 0, 1024, 2), 1024, 0, 1)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer_1.data, 1024, 1024, 2), 1024, 4, 1)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer_1.data, 2048, 1024, 2), 1024, 8, 1)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer_1.data, 3072, 1024, 2), 1024, 12, 1)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer_1.data, 0, 1024, 2), 1024, 1, 1)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer_1.data, 1024, 1024, 2), 1024, 5, 1)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer_1.data, 2048, 1024, 2), 1024, 9, 1)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer_1.data, 3072, 1024, 2), 1024, 13, 1)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer_1.data, 0, 1024, 2), 1024, 2, 1)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer_1.data, 1024, 1024, 2), 1024, 6, 1)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer_1.data, 2048, 1024, 2), 1024, 10, 1)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer_1.data, 3072, 1024, 2), 1024, 14, 1)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer_1.data, 0, 1024, 2), 1024, 3, 1)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer_1.data, 1024, 1024, 2), 1024, 7, 1)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer_1.data, 2048, 1024, 2), 1024, 11, 1)
                T.broadcast_(T.tvm_access_ptr(T.type_annotation("float32"), buffer_2.data, 0, 1024, 1), T.tvm_access_ptr(T.type_annotation("float32"), buffer_1.data, 3072, 1024, 2), 1024, 15, 1)
                E_shared_clear = T.allocate([1024], "float32", "local")
                for i in T.unroll(1024, annotations={"pragma_unroll_explicit": T.bool(False)}):
                    E_shared_clear_1 = T.Buffer((1024,), data=E_shared_clear, scope="local")
                    E_shared_clear_1[i] = T.float32(0.0)
                    for rv in T.unroll(4, annotations={"pragma_unroll_explicit": T.bool(False)}):
                        E_shared_clear_1[i] = E_shared_clear_1[i] + buffer_1[rv * 8 + i // 512 * 4 + i % 4]
                    E_shared[i] = E_shared[i] + E_shared_clear_1[i]

# Metadata omitted. Use show_meta=True in script() method to show it.""".strip()

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared([block_M, block_N], dtype, scope="shared.rsram")
            E_shared = T.alloc_shared([block_M], dtype, scope="shared.rsram")
            T.copy(A[by * block_M, bx * block_N], A_shared)

            T.comm.all_reduce(A_shared, E_shared, "sum", "all", dim=-1, clear=False)

    mod = tvm.IRModule({"main": main})
    target = determine_target("Sunmmio", return_object=True)
    with tvm.target.Target(target):
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tilelang.transform.LayoutInference()(mod)
        mod = tilelang.transform.LowerTileOp()(mod)
        assert mod.script()[-len(func_str):] == func_str, "The generated script does not match the expected output."
'''

if __name__ == "__main__":
    test_comm_python_api(1024, 1024, 128, 128, "float16", "float")
    test_comm_broadcast_lower(1024, 1024, 128, 128, "float16", "float")
    test_comm_put_lower(1024, 1024, 128, 128, "float16", "float")
    test_comm_all_gather_lower(1024, 1024, 128, 128, "float16", "float")
    test_comm_all_gather_axis0_all_lower(1024, 1024, 128, 128, "float16")
    test_comm_all_gather_axis_last_horizontal_lower(1024, 1024, 128, 128, "float16")
    test_comm_all_gather_axis_last_all_lower(1024, 1024, 128, 128, "float16")
    # test_comm_all_reduce_lower(1024 * 128, 1024 * 128, 1024, 1024, "float16", "float")
