"""Small integration checks for scheduling, physical layouts and kernel splitting.

Pass-local contracts are tested in the neighboring files with explicit IR.
Only these tests deliberately exercise multiple compiler stages.
"""

import pytest
import tilelang
import tilelang.testing
import tilelang.ascend.language as T
from tilelang import tvm
from tvm import tirx
from testing.ascend._ir import nodes, calls, allocated_buffer

TILE = 16
STEPS = 4


def _make_shared_l0_ring_program(k_tiles):
    k_extent = k_tiles * TILE

    @T.prim_func
    def main(
        A: T.Tensor((2 * TILE, k_extent), "bfloat16"),
        B: T.Tensor((TILE, k_extent), "bfloat16"),
        C: T.Tensor((2, TILE, TILE), "float32"),
    ):
        with T.Kernel(1):
            l1a = T.alloc_l1((2 * TILE, k_extent), "bfloat16")
            l1b = T.alloc_l1((TILE, k_extent), "bfloat16")
            l0a = T.alloc_l0a((TILE, TILE), "bfloat16")
            l0b = T.alloc_l0b((TILE, TILE), "bfloat16")
            l0c = T.alloc_l0c((TILE, TILE), "float32")
            T.annotate_buffer_versions({l0a: 2, l0b: 2})
            T.copy(A, l1a)
            T.copy(B, l1b)
            # Two sibling reductions share the same physical operand buffers.
            for owner in T.Unroll(2, explicit=True):
                for k in T.Pipelined(k_tiles, num_stages=2):
                    T.copy(l1a[owner * TILE : (owner + 1) * TILE, k * TILE : (k + 1) * TILE], l0a)
                    T.copy(l1b[:, k * TILE : (k + 1) * TILE], l0b)
                    T.gemm(l0a, l0b, l0c, transpose_B=True, clear_accum=k == 0)
                T.copy(l0c, C[owner, :, :])

    return main


def _make_counter_l1_layout_program():
    tile = 64

    @T.prim_func
    def main(A: T.Tensor((4 * tile, tile), "bfloat16")):
        with T.Kernel(1):
            l1 = T.alloc_l1((tile, tile), "bfloat16")
            l0 = T.alloc_l0a((tile, tile), "bfloat16")
            T.annotate_buffer_versions({l1: (3, "counter")})
            for i in T.Pipelined(
                4,
                num_stages=3,
                annotations={"multi_buffer_eligible": [l1]},
            ):
                T.copy(A[i * tile : (i + 1) * tile, :], l1)
                T.copy(l1, l0)

    return main


def _make_epilogue_program(sids: int):
    @T.prim_func
    def main(
        A: T.Tensor((STEPS, TILE, TILE), "bfloat16"),
        B: T.Tensor((STEPS, TILE, TILE), "bfloat16"),
        C: T.Tensor((STEPS, TILE, TILE), "float32"),
    ):
        with T.MixedKernel(1, sids=sids):
            a_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            b_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            accum = T.alloc_l0c((TILE, TILE), "float32")
            output = T.alloc_shared((TILE // sids, TILE), "float32")
            T.annotate_buffer_versions({output: 2})

            for step in T.Pipelined(STEPS, num_stages=2):
                T.copy(A[step, :, :], a_l1)
                T.copy(B[step, :, :], b_l1)
                T.gemm(
                    a_l1,
                    b_l1,
                    accum,
                    transpose_B=True,
                    clear_accum=True,
                    unit_flag_ctrl=3,
                )
                if sids == 1:
                    T.copy(accum, output, unit_flag_ctrl=3)
                    T.copy(output, C[step, :, :])
                else:
                    # Each AIV owns half of the output; launch-count coverage
                    # must not depend on synchronizing overlapping GM stores.
                    T.dual_copy(accum, output, unit_flag_ctrl=3)
                    T.dual_copy(output, C[step, :, :])

    return main


def make_multi_kernel_program():
    @T.prim_func
    def main(
        a: T.Tensor((64,), "float32"),
        b: T.Tensor((64,), "float32"),
        c: T.Tensor((64,), "float32"),
    ):
        with T.Kernel(1):
            T.annotate_unlimit_memory("shared")
            ub0 = T.alloc_shared((64,), "float32")
            T.copy(a, ub0)
            T.copy(ub0, c)

        with T.Kernel(1):
            T.annotate_unlimit_memory("shared.l1")
            ub1 = T.alloc_shared((64,), "float32")
            T.copy(c, ub1)
            T.copy(ub1, b)

    return main


@pytest.mark.parametrize("k_tiles", [1, 2], ids=["single-k-tile", "pipelined-k"])
def test_sibling_l0_rings_survive_layout_lowering(k_tiles):
    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name in ("tl.PrepareMultiBuffer", "tl.MaterializeMultiBuffer"):
                snapshots[info.name] = mod

    with tvm.transform.PassContext(instruments=[Capture()]):
        artifact = tilelang.lower(_make_shared_l0_ring_program(k_tiles), target="ascend")
    prepared = snapshots["tl.PrepareMultiBuffer"]
    a, b = allocated_buffer(prepared, "l0a"), allocated_buffer(prepared, "l0b")
    owners = [loop for loop in nodes(prepared, tirx.For) if a.data in loop.annotations.get("tl.multi_buffer_counter_map", {})]
    assert len(owners) == 2
    counters = [loop.annotations["tl.multi_buffer_counter_map"][a.data] for loop in owners]
    assert all(clock.same_as(counters[0]) for clock in counters)
    assert all(loop.annotations["tl.multi_buffer_counter_map"][b.data].same_as(counters[0]) for loop in owners)
    materialized = snapshots["tl.MaterializeMultiBuffer"]
    assert int(allocated_buffer(materialized, "l0a").shape[0]) == 2
    assert artifact.kernel_source and artifact.device_mod


def test_counter_slot_is_compatible_with_l1_layout_lowering():
    artifact = tilelang.lower(_make_counter_l1_layout_program(), target="ascend")
    assert artifact.kernel_source and artifact.device_mod


def _make_subcore_gemm(pattern, steps, unit_flag=0):
    @T.prim_func
    def main(
        A: T.Tensor((steps, 32, 32), "bfloat16"),
        B: T.Tensor((32, 32), "bfloat16"),
        C: T.Tensor((steps, 32, 32), "float32"),
    ):
        with T.MixedKernel(1) as (_, sid):
            a_l1 = T.alloc_l1((32, 32), "bfloat16")
            b_l1 = T.alloc_l1((32, 32), "bfloat16")
            accum = T.alloc_l0c((32, 32), "float32")
            dq = T.alloc_shared((32, 32), "bfloat16")
            half = T.alloc_shared((16, 32), "bfloat16")
            epi = T.alloc_shared((32, 32), "float32")
            T.annotate_buffer_versions({epi: (2, "counter")})
            T.copy(B, b_l1)
            for i in T.Pipelined(steps, num_stages=2):
                if pattern == "epilogue":
                    T.copy(A[i, :, :], a_l1)
                elif pattern == "halves":
                    T.copy(A[i, sid * 16 : (sid + 1) * 16, :], half)
                    with T.SimdVF():
                        value = T.simd.vld(half[0, 0])
                        T.simd.vsts(half[0, 0], value)
                    T.copy(half, a_l1[sid * 16 : (sid + 1) * 16, :])
                elif pattern == "fixed":
                    if sid == 0:
                        T.copy(A[i, :, :], dq)
                        with T.SimdVF():
                            value = T.simd.vld(dq[0, 0])
                            T.simd.vsts(dq[0, 0], value)
                        T.copy(dq, a_l1)
                else:
                    if sid == i % 2:
                        T.copy(A[i, :, :], dq)
                        with T.SimdVF():
                            value = T.simd.vld(dq[0, 0])
                            T.simd.vsts(dq[0, 0], value)
                        T.copy(dq, a_l1)
                T.gemm(a_l1, b_l1, accum, transpose_B=True, clear_accum=True, unit_flag_ctrl=unit_flag)
                if pattern == "fixed":
                    if sid == 1:
                        T.copy(accum, epi, unit_flag_ctrl=unit_flag)
                        with T.SimdVF():
                            value = T.simd.vld(epi[0, 0])
                            T.simd.vsts(epi[0, 0], value)
                        T.copy(epi, C[i, :, :])
                else:
                    if sid == (i + 1) % 2:
                        T.copy(accum, epi, unit_flag_ctrl=unit_flag)
                        with T.SimdVF():
                            value = T.simd.vld(epi[0, 0])
                            T.simd.vsts(epi[0, 0], value)
                        T.copy(epi, C[i, :, :])

    return main


@pytest.mark.parametrize("pattern", ["epilogue", "fixed", "halves", "switch"])
def test_subcore_roles_survive_complete_lowering(pattern):
    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name in ("tl.ResolveCore", "tl.MaterializeMultiBuffer"):
                snapshots[info.name] = mod

    with tvm.transform.PassContext(instruments=[Capture()]):
        artifact = tilelang.lower(_make_subcore_gemm(pattern, 5), target="ascend")
    assert artifact.kernel_source and artifact.device_mod
    resolved = snapshots["tl.ResolveCore"]
    copies = [
        task for task in nodes(resolved, tirx.AttrStmt) if task.attr_key == "tl.ascend_task" and calls(task.body, "tl.tileop.ascend_copy")
    ]
    assert {1, 2, 4} <= {int(task.node["core_mask"]) for task in copies}
    materialized = snapshots["tl.MaterializeMultiBuffer"]
    func = materialized["main"]
    output = func.buffer_map[func.params[2]]
    buffers = [
        call.args[0].args[0].buffer for call in calls(materialized, "tl.tileop.ascend_copy") if call.args[1].args[0].buffer.same_as(output)
    ]
    assert len(buffers) == (1 if pattern == "fixed" else 2)
    assert len({buffer.data for buffer in buffers}) == len(buffers)
    allocations = [buffer for block in nodes(materialized, tirx.SBlock) for buffer in block.alloc_buffers]
    assert all(buffer.scope() == "shared.dyn" and any(buffer.same_as(allocated) for allocated in allocations) for buffer in buffers)
    assert all(tuple(int(dim) for dim in buffer.shape) == (2, 32, 32) for buffer in buffers)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize(
    "pattern, steps, unit_flag",
    [
        ("epilogue", 1, 0),
        ("epilogue", 5, 0),
        ("fixed", 5, 0),
        ("halves", 5, 0),
        ("switch", 5, 0),
        ("epilogue", 5, 3),
        ("fixed", 5, 3),
    ],
)
def test_subcore_gemm_results(pattern, steps, unit_flag):
    import torch

    torch.manual_seed(23)
    a = torch.randn(steps, 32, 32, dtype=torch.bfloat16).npu()
    b = torch.randn(32, 32, dtype=torch.bfloat16).npu()
    output = tilelang.compile(_make_subcore_gemm(pattern, steps, unit_flag), target="ascend", out_idx=-1)(a, b)
    torch.testing.assert_close(output.cpu(), a.cpu().float() @ b.cpu().float().T, rtol=0.01, atol=0.02)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("rows, dual, unit_flag", [(16, True, 3), (32, True, 3), (16, False, 0)])
def test_subcore_fixpipe_synchronization(rows, dual, unit_flag):
    import torch

    outputs = 1 if dual else 2

    @T.prim_func
    def main(A: T.Tensor((rows, 32), "bfloat16"), B: T.Tensor((32, 32), "bfloat16"), C: T.Tensor((outputs, rows, 32), "float32")):
        with T.MixedKernel(1) as (_, sid):
            a = T.alloc_l1((rows, 32), "bfloat16")
            b = T.alloc_l1((32, 32), "bfloat16")
            accum = T.alloc_l0c((rows, 32), "float32")
            ub = T.alloc_shared((rows // 2 if dual else rows, 32), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.gemm(a, b, accum, transpose_B=True, clear_accum=True, unit_flag_ctrl=unit_flag)
            if dual:
                T.dual_copy(accum, ub, unit_flag_ctrl=unit_flag)
                with T.SimdVF():
                    value = T.simd.vld(ub[0, 0])
                    T.simd.vsts(ub[0, 0], value)
                T.copy(ub, C[0, sid * (rows // 2) : (sid + 1) * (rows // 2), :])
            else:
                T.copy(accum, ub, unit_flag_ctrl=unit_flag)
                T.copy(ub, C[sid, :, :])

    torch.manual_seed(37)
    a = torch.randn(rows, 32, dtype=torch.bfloat16).npu()
    b = torch.randn(32, 32, dtype=torch.bfloat16).npu()
    output = tilelang.compile(main, target="ascend", out_idx=-1)(a, b).cpu()
    expected = a.cpu().float() @ b.cpu().float().T
    torch.testing.assert_close(output, expected.expand(outputs, rows, 32), rtol=0.01, atol=0.02)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("loop_kind", ["pipelined", "while"])
def test_subcore_loops_can_have_different_extents(loop_kind):
    import torch

    @T.prim_func
    def main(A: T.Tensor((2, 3, 64), "float32"), C: T.Tensor((2, 3, 64), "float32")):
        with T.MixedKernel(1) as (_, sid):
            ub = T.alloc_shared((64,), "float32")
            T.annotate_buffer_versions({ub: (2, "counter")})
            if loop_kind == "while":
                i = T.alloc_var("int32", init=0)
                while i < sid + 2:
                    T.copy(A[sid, i, :], ub)
                    T.copy(ub, C[sid, i, :])
                    i += 1
            else:
                for i in T.Pipelined(sid + 2, num_stages=2):
                    T.copy(A[sid, i, :], ub)
                    T.copy(ub, C[sid, i, :])

    a = torch.randn(2, 3, 64).npu()
    output = torch.full_like(a, -1)
    tilelang.compile(main, target="ascend")(a, output)
    expected = a.cpu()
    expected[0, 2] = -1
    torch.testing.assert_close(output.cpu(), expected)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("break_sid", [0, 1])
def test_subcore_break_preserves_the_other_cube_stream(break_sid):
    import torch

    @T.prim_func
    def main(
        A: T.Tensor((4, 32, 32), "bfloat16"),
        B: T.Tensor((32, 32), "bfloat16"),
        V: T.Tensor((4, 32, 32), "float32"),
        C: T.Tensor((2, 4, 32, 32), "float32"),
    ):
        with T.MixedKernel(1) as (_, sid):
            a = T.alloc_l1((32, 32), "bfloat16")
            b = T.alloc_l1((32, 32), "bfloat16")
            acc = T.alloc_l0c((32, 32), "float32")
            ub = T.alloc_shared((32, 32), "float32")
            T.copy(B, b)
            for i in T.serial(4):
                if sid == break_sid and i == 1:
                    T.loop_break()
                if sid == 0:
                    T.copy(V[i, :, :], ub)
                else:
                    T.copy(A[i, :, :], a)
                    T.gemm(a, b, acc, transpose_B=True, clear_accum=True)
                    T.copy(acc, ub)
                T.copy(ub, C[sid, i, :, :])

    a = torch.ones((4, 32, 32), dtype=torch.bfloat16).npu()
    b = torch.ones((32, 32), dtype=torch.bfloat16).npu()
    v = torch.full((4, 32, 32), 7.0).npu()
    output = torch.full((2, 4, 32, 32), -1.0).npu()
    tilelang.compile(main, target="ascend")(a, b, v, output)
    expected = torch.stack((v.cpu(), torch.full_like(v.cpu(), 32.0)))
    expected[break_sid, 1:] = -1
    torch.testing.assert_close(output.cpu(), expected)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("owner", [0, 1])
@pytest.mark.parametrize("consumer", ["copy", "loop-bound", "loop-break"])
def test_subcore_scalar_loop_exit_reaches_later_cube_consumers(owner, consumer):
    import torch

    @T.prim_func
    def main(A: T.Tensor((4, 32, 32), "bfloat16"), B: T.Tensor((32, 32), "bfloat16"), C: T.Tensor((32, 32), "float32")):
        with T.MixedKernel(1) as (_, sid):
            a = T.alloc_l1((32, 32), "bfloat16")
            b = T.alloc_l1((32, 32), "bfloat16")
            acc = T.alloc_l0c((32, 32), "float32")
            count = T.alloc_var("int32", init=0)
            for i in T.serial(4):
                if sid == owner:
                    if i == 1:
                        T.loop_break()
                    count += 1
            if sid == owner:
                T.copy(B, b)
                for j in T.serial(count if consumer == "loop-bound" else (1 if consumer == "copy" else 4)):
                    if consumer == "loop-break" and j >= count:
                        T.loop_break()
                    T.copy(A[count - 1 if consumer == "copy" else j, :, :], a)
                    T.gemm(a, b, acc, transpose_B=True, clear_accum=True)
                    T.copy(acc, C)

    a = torch.arange(1, 5, dtype=torch.float32)[:, None, None].expand(4, 32, 32).contiguous().to(torch.bfloat16).npu()
    b = torch.ones((32, 32), dtype=torch.bfloat16).npu()
    output = torch.full((32, 32), -1.0).npu()
    tilelang.compile(main, target="ascend")(a, b, output)
    torch.testing.assert_close(output.cpu(), torch.full((32, 32), 32.0))


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("varying_stride", [False, True])
def test_subcore_strided_view_preserves_physical_storage(varying_stride):
    import torch

    @T.prim_func
    def main(
        A: T.Tensor((32, 32), "bfloat16"),
        B: T.Tensor((32, 32), "bfloat16"),
        C: T.Tensor((2, 32, 32), "float32"),
        Storage: T.Tensor((2, 32, 64), "float32"),
    ):
        with T.MixedKernel(1) as (_, sid):
            a = T.alloc_l1((32, 32), "bfloat16")
            b = T.alloc_l1((32, 32), "bfloat16")
            acc = T.alloc_l0c((32, 32), "float32")
            storage = T.alloc_shared((32, 64), "float32")
            view = T.StridedTensor(
                (32, 32), ((sid + 1) * 32 if varying_stride else 64, 1), "float32", data=storage.data, scope="shared.dyn"
            )
            T.fill(storage, -1.0)
            T.copy(A, a)
            T.copy(B, b)
            T.gemm(a, b, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, view)
            T.copy(view, C[sid, :, :])
            T.copy(storage, Storage[sid, :, :])

    a = torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32).to(torch.bfloat16).npu()
    b = torch.eye(32, dtype=torch.bfloat16).npu()
    output = torch.empty((2, 32, 32), dtype=torch.float32).npu()
    storage = torch.empty((2, 32, 64), dtype=torch.float32).npu()
    tilelang.compile(main, target="ascend")(a, b, output, storage)
    expected = a.cpu().float()
    expected_storage = torch.full((2, 32, 64), -1.0)
    for sid in range(2):
        row_stride = (sid + 1) * 32 if varying_stride else 64
        expected_storage[sid].as_strided((32, 32), (row_stride, 1)).copy_(expected)
    torch.testing.assert_close(output.cpu(), expected.expand(2, 32, 32), rtol=0, atol=0)
    torch.testing.assert_close(storage.cpu(), expected_storage, rtol=0, atol=0)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("routing", ["constant", "parameter", "local"])
@pytest.mark.parametrize("explicit", [False, True])
def test_per_core_fixpipe_dispatch_preserves_each_subcore_output(routing, explicit):
    import torch

    @T.prim_func
    def main(A: T.Tensor((4, 32, 32), "bfloat16"), B: T.Tensor((32, 32), "bfloat16"), C: T.Tensor((4, 2, 32, 32), "float32")):
        with T.MixedKernel(4) as (bx, sid):
            a = T.alloc_l1((32, 32), "bfloat16")
            b = T.alloc_l1((32, 32), "bfloat16")
            acc = T.alloc_l0c((32, 32), "float32")
            ub = T.alloc_shared((32, 32), "float32")
            T.fill(ub, 0.0)
            T.copy(A[bx, :, :], a)
            T.copy(B, b)
            T.gemm(a, b, acc, transpose_B=True, clear_accum=True)
            with T.PerCoreTask():
                if bx % 2 == 0:
                    if explicit:
                        with T.Task():
                            route = bx // 2 % 2
                            T.copy(acc, ub, sub_blockid=0 if routing == "constant" else (route if routing == "local" else bx // 2 % 2))
                    else:
                        route = bx // 2 % 2
                        T.copy(acc, ub, sub_blockid=0 if routing == "constant" else (route if routing == "local" else bx // 2 % 2))
                elif explicit:
                    with T.Task():
                        route = (bx // 2 + 1) % 2
                        T.copy(acc, ub, sub_blockid=1 if routing == "constant" else (route if routing == "local" else (bx // 2 + 1) % 2))
                else:
                    route = (bx // 2 + 1) % 2
                    T.copy(acc, ub, sub_blockid=1 if routing == "constant" else (route if routing == "local" else (bx // 2 + 1) % 2))
            T.copy(ub, C[bx, sid, :, :])

    a = torch.arange(1, 5, dtype=torch.float32)[:, None, None].expand(4, 32, 32).contiguous().to(torch.bfloat16).npu()
    b = torch.ones((32, 32), dtype=torch.bfloat16).npu()
    output = torch.full((4, 2, 32, 32), -1.0).npu()
    tilelang.compile(main, target="ascend")(a, b, output)
    expected = torch.zeros_like(output.cpu())
    for block in range(4):
        sid = block % 2 if routing == "constant" else (block // 2 + block % 2) % 2
        expected[block, sid] = 32.0 * (block + 1)
    torch.testing.assert_close(output.cpu(), expected)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("direction", ["load", "store"])
def test_per_core_vector_dispatch_preserves_each_subcore_output(direction):
    import torch

    @T.prim_func
    def main(A: T.Tensor((2, 2, 64), "float32"), C: T.Tensor((2, 2, 64), "float32")):
        with T.MixedKernel(2) as (bx, sid):
            ub = T.alloc_shared((64,), "float32")
            if direction == "load":
                with T.PerCoreTask():
                    if bx == 0:
                        with T.Task():
                            T.copy(A[0, sid, :], ub)
                    else:
                        with T.Task():
                            T.copy(A[1, sid, :], ub)
            else:
                T.copy(A[bx, sid, :], ub)
            if direction == "store":
                with T.PerCoreTask():
                    if bx == 0:
                        with T.Task():
                            T.copy(ub, C[0, sid, :])
                    else:
                        with T.Task():
                            T.copy(ub, C[1, sid, :])
                # Different participation forces subcore projection even
                # though the candidate selection itself is subcore-uniform.
                if sid == 0:
                    T.fill(ub, 0.0)
            else:
                if sid == 0:
                    T.copy(ub, C[bx, 0, :])
                else:
                    T.copy(ub, C[bx, 1, :])

    data = torch.arange(256, dtype=torch.float32).reshape(2, 2, 64).npu()
    output = torch.full_like(data, -1.0)
    tilelang.compile(main, target="ascend")(data, output)
    torch.testing.assert_close(output.cpu(), data.cpu(), rtol=0, atol=0)


def test_subcore_mutable_scalar_is_replicated_with_its_guard():
    from testing.ascend._ir import schedule

    @T.prim_func
    def main(C: T.Tensor((5, 16, 16), "float32")):
        with T.MixedKernel(1) as (_, sid):
            accum = T.alloc_l0c((16, 16), "float32")
            ub = T.alloc_shared((16, 16), "float32")
            progress = T.alloc_var("int32", init=0)
            for i in T.serial(5):
                if sid == i % 2 and progress < sid + 2:
                    T.copy(accum, ub)
                    T.copy(ub, C[i, :, :])
                    progress += 1

    resolved = schedule(main, through="ResolveCore", auto_schedule=False, infer_layouts=True)
    progress = [buffer for block in nodes(resolved, tirx.SBlock) for buffer in block.alloc_buffers if buffer.name.startswith("progress")]
    assert len(progress) == 2 and not progress[0].data.same_as(progress[1].data)
    masks = {
        int(task.node["core_mask"])
        for task in nodes(resolved, tirx.AttrStmt)
        if task.attr_key == "tl.ascend_task"
        and any(store.buffer.same_as(buffer) for store in nodes(task.body, tirx.BufferStore) for buffer in progress)
    }
    # C's shared storage epoch is the union of the two guards. Both scalar
    # states can be recomputed on all three cores, independently of UB data.
    assert masks == {7}


@pytest.mark.parametrize("sids", [1, 2])
def test_mixed_kernel_uses_requested_vector_count(sids):
    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name in ("tl.AssignCore", "tl.InsertSync", "tl.LowerScheduledTIR"):
                snapshots[info.name] = mod

    with tvm.transform.PassContext(instruments=[Capture()]):
        artifact = tilelang.lower(_make_epilogue_program(sids), target="ascend")
    vector_mask = 1 if sids == 1 else 5
    assigned = [task for task in nodes(snapshots["tl.AssignCore"], tirx.AttrStmt) if task.attr_key == "tl.ascend_task"]
    masks = {int(task.node["core_mask"]) for task in assigned}
    assert vector_mask in masks
    assert all(mask & (2 | vector_mask) == mask for mask in masks)

    # AIC flags address exactly the participating AIV banks. The uniform AIV
    # body and its synchronization stay shared when both subcores participate.
    cube_banks = set()
    analyzer = tvm.arith.Analyzer()
    for task in nodes(snapshots["tl.InsertSync"], tirx.AttrStmt):
        if task.attr_key != "tl.ascend_task":
            continue
        for operation in ("set", "wait"):
            for call in calls(task.body, "tl.ascend_cross_core_" + operation + "_flag"):
                if int(task.node["core_mask"]) == 2:
                    bound = analyzer.const_int_bound(call.args[2])
                    assert bound.min_value // 16 == bound.max_value // 16
                    cube_banks.add(bound.min_value // 16)
                else:
                    assert int(task.node["core_mask"]) == vector_mask
    assert cube_banks == set(range(sids))
    vectors = [block for block in nodes(snapshots["tl.LowerScheduledTIR"], tirx.SBlock) if block.name_hint == "VECTOR"]
    assert len(vectors) == 1 and int(vectors[0].annotations["vector_count"]) == sids
    assert "asc_sync_intra_arrive(" in artifact.kernel_source
    assert "asc_sync_intra_wait(" in artifact.kernel_source
    if sids == 1:
        assert "if (asc_get_sub_block_id() == 0)" in artifact.kernel_source


def test_every_kernel_is_scheduled_and_called_in_program_order():
    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name in ("tl.InsertSync", "tl.LowerScheduledTIR"):
                snapshots[info.name] = mod

    program = make_multi_kernel_program()
    with tvm.transform.PassContext(instruments=[Capture()]):
        artifact = tilelang.lower(program, target="ascend")
    for name, mod in snapshots.items():
        roots = [block for block in nodes(mod, tirx.SBlock) if block.name_hint == "tilelang_root"]
        assert len(roots) == 2
        if name == "tl.InsertSync":
            assert len([block for block in nodes(mod, tirx.SBlock) if "tl.buffer_alias_map" in block.annotations]) == 2
        else:
            assert len([node for node in nodes(mod, tirx.AttrStmt) if node.attr_key == "tl.buffer_alias_map"]) == 2
    lowered = snapshots["tl.LowerScheduledTIR"]
    assert not any(
        node.attr_key in ("tl.schedule_unit", "tl.ascend_task", "tl.ascend_per_core_task") for node in nodes(lowered, tirx.AttrStmt)
    )
    symbols = {gvar.name_hint for gvar in artifact.device_mod.functions}
    assert len(symbols) == 2
    (host,) = artifact.host_mod.functions.values()
    host_calls = [
        call for call in calls(host, "tirx.tvm_call_packed") if isinstance(call.args[0], tirx.StringImm) and call.args[0].value in symbols
    ]
    assert len(host_calls) == 2 and {call.args[0].value for call in host_calls} == symbols
    a, b, c = [program.buffer_map[param].data for param in program.params]
    for call, expected in zip(host_calls, ({a, c}, {b, c})):
        actual = {arg for arg in call.args if isinstance(arg, tirx.Var)} & {a, b, c}
        assert actual == expected


def _make_nested_loop_break_program():
    @T.prim_func
    def main(
        A: T.Tensor((64, 64), "bfloat16"),
        X: T.Tensor((64,), "float32"),
        Y: T.Tensor((64,), "float32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((64, 64), "bfloat16")
            x_ub = T.alloc_shared((64,), "float32")
            i = T.alloc_var("int32")
            i = 0
            while i < 2:
                T.copy(A, a_l1)
                for _ in T.serial(2):
                    T.copy(X, x_ub)
                    T.copy(x_ub, Y)
                i = i + 1

    return main


def test_nested_loop_break_is_emitted_on_every_core():
    source = tilelang.lower(_make_nested_loop_break_program(), target="ascend").kernel_source
    aic_start = source.index("if ASC_IS_AIC")
    aiv_start = source.index("if ASC_IS_AIV")
    assert "break;" in source[aic_start:aiv_start]
    assert "break;" in source[aiv_start:]


def test_single_version_sibling_protocol_survives_reuse_and_lowering():
    @T.prim_func
    def main(A: T.Tensor((8, 64), "float32"), C: T.Tensor((8, 64), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((64,), "float32")
            T.annotate_buffer_versions({ub: (1, "auto")})
            for i in T.Pipelined(4, num_stages=2, annotations={"enable_offset": True}):
                with T.Stage(0):
                    for _first in T.serial(1):
                        T.copy(A[2 * i, :], ub)
                        T.copy(ub, C[2 * i, :])
                with T.Stage(1):
                    for _second in T.serial(1):
                        T.copy(A[2 * i + 1, :], ub)
                        T.copy(ub, C[2 * i + 1, :])

    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name in ("tl.InsertSync", "tl.MaterializeMultiBuffer", "tl.LowerScheduledTIR", "tl.MergeUBAllocations"):
                snapshots[info.name] = mod

    # This integration boundary protects the single-version counter cleanup
    # together with the alias contract produced by InsertSync and consumed by packing.
    with tvm.transform.PassContext(instruments=[Capture()]):
        artifact = tilelang.lower(main, target="ascend")
    inserted = snapshots["tl.InsertSync"]
    ub = allocated_buffer(inserted, "ub")
    owners = [loop for loop in nodes(inserted, tirx.For) if ub.data in loop.annotations.get("tl.multi_buffer_counter_map", {})]
    assert len(owners) == 2
    counter = owners[0].annotations["tl.multi_buffer_counter_map"][ub.data]
    assert owners[1].annotations["tl.multi_buffer_counter_map"][ub.data].same_as(counter)
    for operation in ("set", "wait"):
        flags = [call for call in calls(inserted, "tl.ascend_" + operation + "_flag") if call.args[0].value == "MTE3_MTE2"]
        assert flags and all(isinstance(call.args[1], tirx.IntImm) for call in flags)
    materialized = snapshots["tl.MaterializeMultiBuffer"]
    assert tuple(int(dim) for dim in allocated_buffer(materialized, "ub").shape) == (64,)
    assert not any(buffer.data.same_as(counter.data) for block in nodes(materialized, tirx.SBlock) for buffer in block.alloc_buffers)
    assert not any(
        node.buffer.data.same_as(counter.data) for kind in (tirx.BufferLoad, tirx.BufferStore) for node in nodes(materialized, kind)
    )
    assert any("tl.buffer_alias_map" in block.annotations for block in nodes(materialized, tirx.SBlock))
    assert any(node.attr_key == "tl.buffer_alias_map" for node in nodes(snapshots["tl.LowerScheduledTIR"], tirx.AttrStmt))
    assert not any(node.attr_key == "tl.buffer_alias_map" for node in nodes(snapshots["tl.MergeUBAllocations"], tirx.AttrStmt))
    assert artifact.kernel_source and artifact.device_mod


def test_mmad_direction_reaches_only_cube_and_precedes_its_gemm():
    @T.prim_func
    def main(A: T.Tensor((16, 16), "bfloat16"), B: T.Tensor((16, 16), "bfloat16"), C: T.Tensor((2, 16, 16), "float32")):
        with T.MixedKernel(1, sids=1):
            a = T.alloc_l1((16, 16), "bfloat16")
            b = T.alloc_l1((16, 16), "bfloat16")
            a_l0 = T.alloc_l0a((16, 16), "bfloat16")
            b_l0 = T.alloc_l0b((16, 16), "bfloat16")
            c0 = T.alloc_l0c((16, 16), "float32")
            c1 = T.alloc_l0c((16, 16), "float32")
            ub = T.alloc_shared((16, 16), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.copy(a, a_l0)
            T.copy(b, b_l0)
            T.set_mmad_direction("n")
            T.gemm(a_l0, b_l0, c0, transpose_B=True, clear_accum=True)
            T.set_mmad_direction("m")
            T.gemm(a_l0, b_l0, c1, transpose_B=True, clear_accum=True)
            T.copy(c0, ub)
            T.copy(ub, C[0, :, :])
            T.copy(c1, ub)
            T.copy(ub, C[1, :, :])

    source = tilelang.lower(main, target="ascend").kernel_source
    aic, aiv = source.split("if ASC_IS_AIC", 1)[1].split("if ASC_IS_AIV", 1)
    assert aic.count("asc_set_mmad_direction_n();") == 1
    assert aic.count("asc_set_mmad_direction_m();") == 1
    assert aic.index("asc_set_mmad_direction_n();") < aic.index("asc_mmad(")
    assert aic.index("asc_mmad(") < aic.index("asc_set_mmad_direction_m();") < aic.rindex("asc_mmad(")
    assert "asc_set_mmad_direction" not in aiv


@pytest.mark.parametrize("versions", [1, 2], ids=["single-version", "ring"])
def test_sf_group_owners_survive_a_plain_gemm_sibling(versions):
    @T.prim_func
    def main(
        A: T.Tensor((16, 128), "float8_e4m3fn"),
        B: T.Tensor((16, 128), "float8_e4m3fn"),
        S: T.Tensor((2, 16), "uint16"),
        C: T.Tensor((2, 3, 16, 16), "float32"),
    ):
        with T.Kernel(1):
            a1 = T.alloc_l1((16, 128), "float8_e4m3fn")
            b1 = T.alloc_l1((16, 128), "float8_e4m3fn")
            s1 = T.alloc_l1((16, 2), "uint16")
            a0 = T.alloc_l0a((16, 128), "float8_e4m3fn")
            b0 = T.alloc_l0b((16, 128), "float8_e4m3fn")
            sa = T.alloc_l0a_sf(a0)
            sb = T.alloc_l0b_sf(b0)
            acc = T.alloc_l0c((16, 16), "float32")
            T.annotate_buffer_versions({a0: (versions, "auto"), b0: (versions, "auto")})
            T.copy(A, a1)
            T.copy(B, b1)
            T.copy(S, s1, transpose=True)
            for owner in T.Unroll(2, explicit=True):
                for i in T.Pipelined(3, num_stages=2):
                    T.copy(a1, a0)
                    T.copy(b1, b0)
                    if owner == 0:
                        T.copy(s1, sa)
                        T.copy(s1, sb)
                        T.gemm_blockscaled(a0, b0, acc, sa, sb, transpose_B=True, clear_accum=True)
                    else:
                        T.gemm(a0, b0, acc, transpose_B=True, clear_accum=True)
                    T.copy(acc, C[owner, i, :, :])

    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name in ("tl.InsertSync", "tl.MaterializeMultiBuffer"):
                snapshots[info.name] = mod

    # Owner and counter metadata must survive Prepare -> ResolveCore -> Sync
    # and reconstruct the same physical ring in MaterializeMultiBuffer.
    with tvm.transform.PassContext(instruments=[Capture()]):
        artifact = tilelang.lower(main, target="ascend")
    prepared = snapshots["tl.InsertSync"]
    for data_name, sf_name in [("a0", "sa"), ("b0", "sb")]:
        data, sf = allocated_buffer(prepared, data_name), allocated_buffer(prepared, sf_name)
        owners = [loop for loop in nodes(prepared, tirx.For) if data.data in loop.annotations.get("multi_buffer_eligible", [])]
        assert len(owners) == 2
        counters = [loop.annotations["tl.multi_buffer_counter_map"] for loop in owners]
        assert all(
            mapping[data.data].same_as(counters[0][data.data]) and mapping[sf.data].same_as(mapping[data.data]) for mapping in counters
        )
        for name in (data_name, sf_name):
            shape = allocated_buffer(snapshots["tl.MaterializeMultiBuffer"], name).shape
            assert len(shape) == (3 if versions > 1 else 2)
            if versions > 1:
                assert int(shape[0]) == versions
    assert artifact.kernel_source and artifact.device_mod


@tilelang.testing.requires_ascend
def test_scalar_snapshot_precedes_pipeline_update():
    import torch

    @T.prim_func
    def main(A: T.Tensor((16,), "float32"), O: T.Tensor((16,), "float32")):
        with T.Kernel(1):
            accum = T.alloc_var("float32", init=0)
            out = T.alloc_shared((16,), "float32")
            for i in T.Pipelined(16, num_stages=2):
                previous = accum
                for _ in T.serial(2):
                    accum += A[i]
                out[i] = previous
            T.copy(out, O)

    a = torch.arange(16, dtype=torch.float32)
    result = tilelang.compile(main, target="ascend", out_idx=-1)(a.npu())
    torch.testing.assert_close(result.cpu(), 2 * (a.cumsum(0) - a), rtol=0, atol=0)
