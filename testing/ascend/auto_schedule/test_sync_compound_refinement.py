"""Conservative boundaries for refining a compound task's access hull."""

import pytest

import tilelang
import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.ascend import transform
from tilelang.backend.target import determine_target
from testing.ascend._ir import calls, copy, kernel, seq
from testing.ascend.auto_schedule._scheduled_ir import unit
from tvm import tirx


_CROSS_AIV_ERROR = "does not support synchronization between AIV0 and AIV1"


def _writers(*, stride=None, extra=None, hint=None, copies=1):
    params = []
    if stride == "dynamic":
        stride = tirx.Var("gm_stride", "int32")
        params.append(stride)
    state = tirx.decl_buffer((1,), "int32", name="state")
    if extra in ("mutable_index", "mutable_guard", "mutable_layout"):
        params.append(state)
    gm = tirx.decl_buffer(
        (640 if extra in ("alias", "different_iterations") else 512,),
        "float32",
        name="gm",
        strides=[] if stride is None else [stride],
        elem_offset=state[0] if extra == "mutable_layout" else 0,
    )
    ub = [tirx.decl_buffer((64,), "float32", name=f"ub{sid}", scope="shared.dyn") for sid in range(2)]
    tasks, leaf_regions = [], []
    for sid in range(2):
        stage = tirx.Var(f"stage{sid}", "int32")
        begin = 128 * stage + sid * 64
        if extra == "mutable_index":
            begin += 128 * state[0]
        if extra == "different_iterations" and sid == 1:
            begin = 128 * stage + 128
        leaf_regions.append(tirx.BufferRegion(gm, [tvm.ir.Range.from_min_extent(begin, 64)]))
        target = gm
        if extra == "alias" and sid == 1:
            target = tirx.decl_buffer((512,), "float32", name="view", data=gm.data, elem_offset=64)
        operation = copy(ub[sid], target, dst_indices=[begin], dst_shape=[64])
        if extra == "special_copy" and sid == 0:
            operation = copy(
                gm,
                ub[sid],
                src_indices=[begin],
                src_shape=[64],
                annotations={"data_select": tirx.IntImm("int32", 1), "pad_value": tirx.FloatImm("float32", 0)},
            )
        if extra == "mutable_guard":
            operation = tirx.IfThenElse(state[0] > 0, operation, None)
        operations = [operation] * copies
        if extra == "opaque" and sid == 1:
            # A copy does not describe everything an escaped pointer can access.
            operations.insert(0, tirx.Evaluate(tirx.call_extern("int32", "opaque_write", gm.data)))
        tasks.append(unit(tirx.For(stage, 0, 4, tirx.ForKind.SERIAL, seq(*operations)), core=1 if sid == 0 else 4))
    annotations = {"tl.split_subcores": True}
    hints = []
    if hint in ("storage_negative", "leaf_overrides_storage"):
        hints.append([gm.data, gm.data, tirx.IntImm("int32", 0), tirx.const(False, "bool")])
    if hint in ("leaf_positive", "leaf_overrides_storage"):
        # NormalizeConflictHints retains these original leaf regions, including
        # their internal induction variables. Match before renaming those vars.
        hints.append([leaf_regions[0], leaf_regions[1], tirx.IntImm("int32", 0), tirx.const(True, "bool")])
    if hint == "storage_positive":
        hints.append([gm.data, gm.data, tirx.IntImm("int32", 0), tirx.const(True, "bool")])
    if hints:
        annotations["tl.root_conflict_hints"] = hints
    return kernel(seq(*tasks), buffers=ub, params=[gm, *params], annotations=annotations)


@pytest.mark.parametrize("stride", [None, 1])
def test_plain_dense_interleaved_copies_are_disjoint(stride):
    after = transform.InsertSync()(_writers(stride=stride))
    assert len(calls(after, "tl.tileop.ascend_copy")) == 2
    assert not calls(after, "tl.ascend_cross_core_set_flag")


@pytest.mark.parametrize("stride", [0, "dynamic", 2])
def test_non_dense_or_unknown_layout_retains_dependency(stride):
    # Zero stride aliases all indices. An unknown stride can be zero. Other
    # non-dense layouts keep the previous conservative analysis as well.
    with pytest.raises(tvm.error.InternalError, match=_CROSS_AIV_ERROR):
        transform.InsertSync()(_writers(stride=stride))


@pytest.mark.parametrize("extra", ["opaque", "mutable_index", "mutable_guard", "mutable_layout", "special_copy", "alias"])
def test_incomplete_or_unsupported_footprint_retains_dependency(extra):
    with pytest.raises(tvm.error.InternalError, match=_CROSS_AIV_ERROR):
        transform.InsertSync()(_writers(extra=extra))


def test_independent_internal_iterations_can_overlap():
    with pytest.raises(tvm.error.InternalError, match=_CROSS_AIV_ERROR):
        transform.InsertSync()(_writers(extra="different_iterations"))


@pytest.mark.parametrize("hint", ["leaf_positive", "leaf_overrides_storage", "storage_positive"])
def test_positive_conflict_hint_precedes_refinement_and_negative_hint(hint):
    with pytest.raises(tvm.error.InternalError, match=_CROSS_AIV_ERROR):
        transform.InsertSync()(_writers(hint=hint))


def test_explicit_storage_no_conflict_keeps_its_scope():
    after = transform.InsertSync()(_writers(extra="opaque", hint="storage_negative"))
    assert not calls(after, "tl.ascend_cross_core_set_flag")


def test_positive_leaf_hint_survives_unsupported_operations():
    with pytest.raises(tvm.error.InternalError, match=_CROSS_AIV_ERROR):
        transform.InsertSync()(_writers(extra="opaque", hint="leaf_overrides_storage"))


@pytest.mark.parametrize("copies", [9, 17], ids=["proof-budget", "copy-budget"])
def test_refinement_budget_exhaustion_retains_dependency(copies):
    with pytest.raises(tvm.error.InternalError, match=_CROSS_AIV_ERROR):
        transform.InsertSync()(_writers(copies=copies))


def _strided_store_with_assumption(explicit):
    num_tokens = T.dynamic("num_tokens")

    if explicit:

        @T.prim_func
        def main(src: T.Tensor((512,), "float32"), dst: T.Tensor((num_tokens,), "float32")):
            with T.MixedKernel(1) as (_, sid):
                ub = T.alloc_shared((256,), "float32")
                T.assume(num_tokens <= 512)
                T.copy(src[sid * 256 : sid * 256 + 256], ub)
                count = T.ceildiv(num_tokens - sid * 64, 128)
                with T.PerCoreTask():  # noqa: SIM117
                    with T.Task():
                        for stage in T.serial(count):
                            begin = stage * 128 + sid * 64
                            size = T.min(num_tokens - begin, 64)
                            T.copy(ub[stage * 64 : stage * 64 + size], dst[begin : begin + size])

    else:

        @T.prim_func
        def main(src: T.Tensor((512,), "float32"), dst: T.Tensor((num_tokens,), "float32")):
            with T.MixedKernel(1) as (_, sid):
                ub = T.alloc_shared((256,), "float32")
                T.assume(num_tokens <= 512)
                T.copy(src[sid * 256 : sid * 256 + 256], ub)
                count = T.ceildiv(num_tokens - sid * 64, 128)
                with T.PerCoreTask():
                    for stage in T.serial(count):
                        begin = stage * 128 + sid * 64
                        size = T.min(num_tokens - begin, 64)
                        T.copy(ub[stage * 64 : stage * 64 + size], dst[begin : begin + size])

    return main


@pytest.mark.parametrize("target", ["ascend", "pto"])
@pytest.mark.parametrize("explicit", [False, True], ids=["implicit-task", "explicit-task"])
def test_full_lowering_preserves_assumption_diagnostic_guards(target, explicit):
    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name == "tl.InsertSync":
                snapshots[info.name] = mod

    with determine_target(target, return_object=True), tvm.transform.PassContext(instruments=[Capture()]):
        artifact = tilelang.lower(_strided_store_with_assumption(explicit), target=target, enable_device_compile=False)
    assert not calls(snapshots["tl.InsertSync"], "tl.ascend_cross_core_set_flag")
    assert artifact.kernel_source
