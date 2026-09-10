import tilelang.ascend.language as T
import tvm
from tilelang.engine.lower import lower
from tvm import tirx
from tvm.tirx.stmt_functor import post_order_visit


def _make_alias_claim_program():
    size = 16

    @T.prim_func
    def main(A: T.Buffer((size,), "float32"), C: T.Buffer((size,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((size,), "float32")
            alias = T.reshape(ub, (4, 4))
            T.annotate_buffer_versions({ub: 2})

            # Deliberately list two Buffer aliases. The annotation pass must
            # normalize this frontend form to one storage Var.
            for i in T.Pipelined(
                size,
                num_stages=2,
                annotations={"multi_buffer_eligible": [alias, ub]},
            ):
                ub[i] = A[i]
                C[i] = alias[i // 4, i % 4] + T.float32(1)

    return main


def _make_subview_alias_program():
    @T.prim_func
    def main(A: T.Buffer((8,), "float32"), C: T.Buffer((8,), "float32")):
        with T.Kernel(1):
            base = T.alloc_shared((16,), "float32")
            alias = T.Tensor((4,), "float32", base.data, scope="shared.dyn")
            T.annotate_buffer_versions({base: 2})

            for i in T.Pipelined(4, num_stages=2):
                base[0] = A[i]
                alias[0] = A[i + 4]
                C[i] = base[0]
                C[i + 4] = alias[0]

    return main


def _make_cross_dtype_alias_program():
    @T.prim_func
    def main(
        A: T.Buffer((4,), "float32"),
        B: T.Buffer((4,), "float16"),
        C: T.Buffer((4,), "float32"),
    ):
        with T.Kernel(1):
            base = T.alloc_shared((16,), "float32")
            alias = T.view(base, (32,), dtype="float16")
            T.annotate_buffer_versions({base: 2})

            for i in T.Pipelined(4, num_stages=2):
                base[0] = A[i]
                alias[1] = B[i]
                C[i] = base[0] + T.cast(alias[1], "float32")

    return main


def _collect_unique_buffers(stmt, storage):
    buffers = []

    def add(buffer):
        if not buffer.data.same_as(storage):
            return
        if not any(buffer.same_as(existing) for existing in buffers):
            buffers.append(buffer)

    def visit(node):
        if isinstance(node, (tirx.BufferLoad, tirx.BufferStore)):
            add(node.buffer)

    post_order_visit(stmt, visit)
    return buffers


def test_multi_buffer_contract_uses_storage_vars_for_aliases():
    snapshots = {}
    wanted = {
        "tl.AnnotateMultiBufferEligible",
        "tl.AutoSchedule",
        "tl.InsertSync",
        "tl.MaterializeMultiBuffer",
    }

    @tvm.ir.instrument.pass_instrument
    class CapturePassIR:
        def run_after_pass(self, mod, info):
            if info.name in wanted:
                snapshots[info.name] = mod

    with tvm.transform.PassContext(opt_level=3, instruments=[CapturePassIR()]):
        lower(_make_alias_claim_program(), target="ascend")

    claims = []

    def collect_claims(node):
        if not isinstance(node, tirx.For):
            return
        claim = node.annotations.get("multi_buffer_eligible")
        if claim:
            claims.append(claim)

    annotated = snapshots["tl.AnnotateMultiBufferEligible"]["main"]
    post_order_visit(annotated.body, collect_claims)
    assert len(claims) == 1
    assert len(claims[0]) == 1
    storage = claims[0][0]
    assert isinstance(storage, tirx.Var)

    selected_versions = []

    def collect_selected_versions(node):
        if not isinstance(node, tirx.SBlock):
            return
        versions = node.annotations.get("tl.buffer_versions_map")
        if versions:
            selected_versions.append(versions)

    scheduled = snapshots["tl.AutoSchedule"]["main"]
    post_order_visit(scheduled.body, collect_selected_versions)
    assert len(selected_versions) == 1
    assert len(selected_versions[0]) == 1
    selected_storage = next(iter(selected_versions[0]))
    assert selected_storage.same_as(storage)
    assert int(selected_versions[0][selected_storage]) == 2

    synchronized = snapshots["tl.InsertSync"]["main"]
    logical_buffers = _collect_unique_buffers(synchronized.body, storage)
    logical_shapes = {tuple(int(dim) for dim in buffer.shape) for buffer in logical_buffers}
    assert logical_shapes == {(16,), (4, 4)}

    materialized = snapshots["tl.MaterializeMultiBuffer"]["main"]
    versioned_buffers = _collect_unique_buffers(materialized.body, storage)
    versioned_shapes = {tuple(int(dim) for dim in buffer.shape) for buffer in versioned_buffers}
    assert versioned_shapes == {(2, 16), (2, 4, 4)}


def test_alias_versions_use_the_allocation_pitch():
    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class CapturePassIR:
        def run_after_pass(self, mod, info):
            if info.name == "tl.MaterializeMultiBuffer":
                snapshots[info.name] = mod

    with tvm.transform.PassContext(opt_level=3, instruments=[CapturePassIR()]):
        lower(_make_subview_alias_program(), target="ascend")

    materialized = snapshots["tl.MaterializeMultiBuffer"]["main"]
    storages = []

    def collect_storage(node):
        if isinstance(node, tirx.SBlock):
            storages.extend(node.alloc_buffers)

    post_order_visit(materialized.body, collect_storage)
    base = next(buffer for buffer in storages if buffer.name == "base")
    aliases = _collect_unique_buffers(materialized.body, base.data)
    assert {tuple(int(dim) for dim in buffer.shape) for buffer in aliases} == {
        (2, 16),
        (2, 4),
    }
    assert {int(buffer.strides[0]) for buffer in aliases} == {16}


def test_cross_dtype_alias_versions_convert_the_allocation_pitch():
    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class CapturePassIR:
        def run_after_pass(self, mod, info):
            if info.name == "tl.MaterializeMultiBuffer":
                snapshots[info.name] = mod

    with tvm.transform.PassContext(opt_level=3, instruments=[CapturePassIR()]):
        lower(_make_cross_dtype_alias_program(), target="ascend")

    materialized = snapshots["tl.MaterializeMultiBuffer"]["main"]
    storages = []

    def collect_storage(node):
        if isinstance(node, tirx.SBlock):
            storages.extend(node.alloc_buffers)

    post_order_visit(materialized.body, collect_storage)
    base = next(buffer for buffer in storages if buffer.name == "base")
    aliases = _collect_unique_buffers(materialized.body, base.data)
    strides_by_dtype = {str(buffer.dtype): int(buffer.strides[0]) for buffer in aliases}
    assert strides_by_dtype["float32"] == 16
    assert strides_by_dtype["float16"] == 32


if __name__ == "__main__":
    test_multi_buffer_contract_uses_storage_vars_for_aliases()
