"""Small integration checks for scheduling, physical layouts and kernel splitting.

Pass-local contracts are tested in the neighboring files with explicit IR.
Only these tests deliberately exercise multiple compiler stages.
"""

import pytest
import tilelang
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
        C: T.Tensor((STEPS, TILE, TILE), "bfloat16"),
    ):
        with T.MixedKernel(1, sids=sids):
            a_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            b_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            accum = T.alloc_l0c((TILE, TILE), "float32")
            output = T.alloc_shared((TILE, TILE), "bfloat16")
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
                T.copy(accum, output, unit_flag_ctrl=3)
                T.copy(output, C[step, :, :])

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


@pytest.mark.parametrize("sids", [1, 2])
def test_mixed_kernel_uses_requested_vector_count(sids):
    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name == "tl.LowerScheduledTIR":
                snapshots[info.name] = mod

    with tvm.transform.PassContext(instruments=[Capture()]):
        artifact = tilelang.lower(_make_epilogue_program(sids), target="ascend")
    vectors = [block for block in nodes(snapshots["tl.LowerScheduledTIR"], tirx.SBlock) if block.name_hint == "VECTOR"]
    assert vectors and all(int(block.annotations["vector_count"]) == sids for block in vectors)
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
