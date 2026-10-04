"""Observable M1–M4 contracts of the Ascend AutoSimtVF Stage1 pass."""

import tilelang
import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.ascend import transform
from tilelang.ascend.pipeline import AscendPassPipelineBody
from tvm import tirx
from tvm.tirx.stmt_functor import post_order_visit


def _prepare(func):
    mod = tvm.IRModule.from_expr(func)
    mod = tirx.transform.BindTarget(tvm.target.Target("ascend"))(mod)
    return tilelang.transform.MaterializeKernelLaunch(
        lower_grid_binding=True,
        lower_thread_binding=False,
        default_threads=None,
        unsupported_annotations=["cluster_dims"],
        launch_dim_tags=["cthread"],
    )(mod)


def _apply(func):
    return transform.AutoSimtVF()(_prepare(func))["main"]


def _collect(func, kind):
    found = []
    post_order_visit(
        func.body,
        lambda node: found.append(node) if isinstance(node, kind) else None,
    )
    return found


def _vfs(func):
    return [block for block in _collect(func, tirx.SBlock) if block.name_hint == "SIMT_VF"]


def _thread_extents(vf):
    return {
        str(attr.node.thread_tag): int(attr.value)
        for attr in _collect(vf, tirx.AttrStmt)
        if attr.attr_key == "thread_extent" and isinstance(attr.node, tirx.IterVar)
    }


def _copy_calls(func):
    return [call for call in _collect(func, tirx.Call) if str(call.op.name) in {"tl.tileop.copy", "tl.tileop.ascend_copy"}]


def test_m1_dtype_changing_copy_is_a_seed():
    @T.prim_func
    def main(A: T.Tensor((8,), "float32"), B: T.Tensor((8,), "float16")):
        with T.Kernel(1):
            shared = T.alloc_shared((8,), "float16")
            T.copy(A, shared)
            T.copy(shared, B)

    result = _apply(main)
    assert len(_vfs(result)) == 1
    assert len(_copy_calls(result)) == 2
    assert len(_copy_calls(_vfs(result)[0])) == 1


def test_m1_dynamic_origin_whole_fragment_copy_is_fusible():
    n = T.dynamic("n")

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), B: T.Tensor((n, 2), "float32")):
        with T.Kernel(1):
            frag = T.alloc_fragment((2,), "float32")
            for i in T.serial(n - 1):
                T.copy(A[i : i + 2], frag[:])
                for j in T.Parallel(2):
                    B[i, j] = frag[j]

    result = _apply(main)
    assert len(_vfs(result)) == 1
    assert len(_copy_calls(_vfs(result)[0])) == 1


def test_m1_existing_gemm_and_stage_keep_their_boundaries():
    @T.prim_func
    def main(A: T.Tensor((8,), "float32"), B: T.Tensor((8,), "float32")):
        with T.Kernel(1):
            lhs = T.alloc_l1((16, 16), "float16")
            rhs = T.alloc_l1((16, 16), "float16")
            accum = T.alloc_l0c((16, 16), "float32")
            T.gemm(lhs, rhs, accum)
            with T.Stage(0):
                for i in T.Parallel(8):
                    B[i] = A[i] + 1

    result = _apply(main)
    assert len(_vfs(result)) == 1
    assert any(call.op.name == "tl.tileop.gemm" for call in _collect(result, tirx.Call))
    assert any(attr.attr_key == "tl.ascend_stage" for attr in _collect(result, tirx.AttrStmt))


def test_m1_fill_outside_parallel_creates_seed():
    @T.prim_func
    def main(A: T.Tensor((8,), "float32")):
        with T.Kernel(1):
            frag = T.alloc_fragment((8,), "float32")
            T.fill(frag, 0)
            T.copy(frag, A)

    result = _apply(main)
    assert len(_vfs(result)) == 1
    assert any(buffer.scope() == "local.fragment" for buffer in _vfs(result)[0].alloc_buffers)


def test_m1_dynamic_ub_fill_stays_outside_vf():
    seq_len = T.dynamic("seq_len")

    @T.prim_func
    def main(A: T.Tensor((seq_len, 32), "float32")):
        with T.Kernel(1):
            shared = T.alloc_shared((16, 32), "float32")
            for tile in T.serial(T.ceildiv(seq_len, 16)):
                T.copy(
                    A[tile * 16 : tile * 16 + T.min(16, seq_len - tile * 16), :],
                    shared[: T.min(16, seq_len - tile * 16), :],
                )
                if seq_len - tile * 16 < 16:
                    T.fill(shared[seq_len - tile * 16 : 16, :], 0)
                for col in T.Parallel(32):
                    shared[0, col] += 1

    result = _apply(main)
    fills = [call for call in _collect(result, tirx.Call) if call.op.name == "tl.tileop.fill"]
    assert len(fills) == 1
    assert all(not any(call.same_as(fills[0]) for call in _collect(vf, tirx.Call)) for vf in _vfs(result))
    target = tvm.target.Target("ascend")
    with target, tvm.transform.PassContext(config={tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True}):
        assert AscendPassPipelineBody(tvm.IRModule.from_expr(main), target)["main"] is not None


def test_m1_fill_value_read_participates_in_fragment_flow():
    @T.prim_func
    def main(A: T.Tensor((8,), "float32"), B: T.Tensor((8,), "float32")):
        with T.Kernel(1):
            src = T.alloc_fragment((8,), "float32")
            dst = T.alloc_fragment((8,), "float32")
            shared = T.alloc_shared((8,), "float32")
            T.fill(src, 1)
            T.copy(A, shared)
            T.fill(dst, src[0])
            for i in T.Parallel(8):
                B[i] = dst[i]

    result = _apply(main)
    fills = [call for call in _collect(result, tirx.Call) if call.op.name == "tl.tileop.fill"]
    assert any(isinstance(call.args[1], tirx.BufferLoad) and call.args[1].buffer.scope() == "shared.dyn" for call in fills)
    target = tvm.target.Target("ascend")
    with target:
        assert AscendPassPipelineBody(tvm.IRModule.from_expr(main), target)["main"] is not None


def test_m1_scalar_fill_and_bind_remain_boundaries():
    @T.prim_func
    def scalar_fill(B: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            scalar = T.alloc_var("float32")
            T.fill(scalar, 1)
            B[0] = scalar

    prepared = _prepare(scalar_fill)
    transformed = transform.AutoSimtVF()(prepared)
    tvm.ir.assert_structural_equal(next(iter(transformed.functions.values())), next(iter(prepared.functions.values())))

    @T.prim_func
    def bound_scale(A: T.Tensor((8,), "float32"), B: T.Tensor((8,), "float32")):
        with T.Kernel(1):
            scale = A[0]
            for i in T.Parallel(8):
                B[i] = A[i] * scale

    result = next(iter(transform.AutoSimtVF()(_prepare(bound_scale)).functions.values()))
    assert len(_vfs(result)) == 1
    assert not _collect(_vfs(result)[0], tirx.Bind)
    target = tvm.target.Target("ascend")
    with target:
        assert len(AscendPassPipelineBody(tvm.IRModule.from_expr(bound_scale), target).functions) == 1


def test_m1_gemm_without_autosimt_candidate_is_unchanged():
    @T.prim_func
    def main():
        with T.Kernel(1):
            lhs = T.alloc_l1((16, 16), "float16")
            rhs = T.alloc_l1((16, 16), "float16")
            accum = T.alloc_l0c((16, 16), "float32")
            T.gemm(lhs, rhs, accum)

    result = _apply(main)
    tvm.ir.assert_structural_equal(result, _prepare(main)["main"])


def test_m1_parallel_is_seed_and_plain_engine_copy_is_boundary():
    @T.prim_func
    def main(A: T.Tensor((8,), "float32"), B: T.Tensor((8,), "float32")):
        with T.Kernel(1):
            shared = T.alloc_shared((8,), "float32")
            T.copy(A, shared)
            for i in T.Parallel(8):
                B[i] = shared[i] + 1

    result = _apply(main)
    assert len(_vfs(result)) == 1
    assert len(_copy_calls(result)) == 1
    assert not _copy_calls(_vfs(result)[0])
    assert len(_collect(_vfs(result)[0].body, tirx.For)) == 1


def test_m2_ancestor_fragment_definer_moves_into_consumer_vf():
    @T.prim_func
    def main(A: T.Tensor((8,), "float32"), B: T.Tensor((2, 8), "float32")):
        with T.Kernel(1):
            frag = T.alloc_fragment((8,), "float32")
            T.copy(A, frag)
            for outer in T.serial(2):
                for i in T.Parallel(8):
                    B[outer, i] = frag[i] + 1

    result = _apply(main)
    assert len(_vfs(result)) == 1
    assert len(_copy_calls(_vfs(result)[0])) == 1
    assert len(_copy_calls(result)) == 1


def test_m2_fragment_copy_address_change_blocks_relocation():
    @T.prim_func
    def main(A: T.Tensor((16,), "float32"), B: T.Tensor((2, 8), "float32")):
        with T.Kernel(1):
            frag = T.alloc_fragment((8,), "float32")
            address = T.alloc_var("int32")
            address = 0
            T.copy(A[address : address + 8], frag)
            for outer in T.serial(2):
                address = 8
                for i in T.Parallel(8):
                    B[outer, i] = frag[i]

    result = _apply(main)
    serial = next(loop for loop in _collect(result, tirx.For) if loop.kind == tirx.ForKind.SERIAL)
    input_copies = [call for call in _copy_calls(result) if "A" in str(call.args[0])]
    assert len(input_copies) == 1
    assert not any(call.same_as(input_copies[0]) for call in _copy_calls(serial))
    assert any(buffer.scope() == "shared.dyn" for block in _collect(result, tirx.SBlock) for buffer in block.alloc_buffers)


def test_m2_tileop_write_and_copy_range_dependencies_are_preserved():
    @T.prim_func
    def main(A: T.Tensor((8,), "float32"), B: T.Tensor((8,), "float32"), C: T.Tensor((8,), "float32")):
        with T.Kernel(1):
            shared = T.alloc_shared((8,), "float32")
            frag = T.alloc_fragment((8,), "float32")
            T.copy(A, shared)
            T.copy(shared, frag)
            with T.SimtVF(threads=128):
                T.copy(B, shared)
            for _outer in T.serial(1):
                for i in T.Parallel(8):
                    C[i] = frag[i]

    result = _apply(main)
    serial = next(loop for loop in _collect(result, tirx.For) if loop.kind == tirx.ForKind.SERIAL)
    fragment_copy = next(call for call in _copy_calls(result) if "frag" in str(call.args[1]))
    assert not any(call.same_as(fragment_copy) for call in _copy_calls(serial))

    @T.prim_func
    def main(A: T.Tensor((16,), "float32"), B: T.Tensor((8,), "float32")):
        with T.Kernel(1):
            index = T.alloc_fragment((2,), "int32")
            frag = T.alloc_fragment((8,), "float32")
            shared = T.alloc_shared((16,), "float32")
            for i in T.Parallel(2):
                index[i] = 0
            T.copy(A, shared)
            T.copy(A[index[0] : index[0] + 8], frag)
            for i in T.Parallel(8):
                B[i] = frag[i]

    result = _apply(main)
    indexed_copy = next(call for call in _copy_calls(result) if "A" in str(call.args[0]) and "index" in str(call.args[0]))
    loads = []
    post_order_visit(indexed_copy.args[0], lambda node: loads.append(node) if isinstance(node, tirx.BufferLoad) else None)
    assert any(load.buffer.name.startswith("index_simtvf_state") and load.buffer.scope() == "shared.dyn" for load in loads)


def test_m2_reduce_after_boundary_forms_standalone_vf():
    @T.prim_func
    def main(A: T.Tensor((2, 8), "float32"), B: T.Tensor((2,), "float32")):
        with T.Kernel(1):
            src = T.alloc_fragment((2, 8), "float32")
            dst = T.alloc_fragment((2,), "float32")
            shared = T.alloc_shared((2, 8), "float32")
            for i in T.Parallel(2):
                for j in T.Parallel(8):
                    src[i, j] = A[i, j]
            T.copy(A, shared)
            T.reduce_sum(src, dst, dim=1)
            T.copy(dst, B)

    result = _apply(main)
    assert len(_vfs(result)) >= 1
    assert sum(call.op.name == "tl.tileop.reduce" for call in _collect(result, tirx.Call)) == 1
    assert sum(any(call.op.name == "tl.tileop.reduce" for call in _collect(vf, tirx.Call)) for vf in _vfs(result)) == 1
    target = tvm.target.Target("ascend")
    with target:
        lowered = AscendPassPipelineBody(tvm.IRModule.from_expr(main), target)
    assert not any(call.op.name == "tl.tileop.reduce" for call in _collect(lowered["main"], tirx.Call))
    assert "AscendAllReduce" in str(lowered["main"])


def test_m2_injected_assumes_preserve_parallel_planning():
    n = T.dynamic("n")

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            T.assume(n > 0)
            for i in T.Parallel(n):
                B[i] = A[i] + 1

    injected = tilelang.transform.InjectAssumes()(_prepare(main))
    result = transform.AutoSimtVF()(injected)["main"]
    attrs = {attr.attr_key for attr in _collect(result, tirx.AttrStmt)}
    assert {"tl.assume", "tl.assume_requires_runtime_check"} <= attrs
    assert len(_vfs(result)) == 1
    target = tvm.target.Target("ascend")
    with target:
        assert AscendPassPipelineBody(tvm.IRModule.from_expr(main), target)["main"] is not None


def test_m2_reduce_chain_fuses_through_intermediate_fragment():
    @T.prim_func
    def main():
        with T.Kernel(1):
            src = T.alloc_fragment((2, 4, 8), "float32")
            tmp = T.alloc_fragment((2, 4), "float32")
            dst = T.alloc_fragment((2,), "float32")
            T.fill(src, 0)
            T.reduce_sum(src, tmp, dim=2)
            T.reduce_sum(tmp, dst, dim=1)

    result = _apply(main)
    assert len(_vfs(result)) == 1
    assert sum(call.op.name == "tl.tileop.reduce" for call in _collect(_vfs(result)[0], tirx.Call)) == 2


def test_m3_cross_region_fragment_uses_shared_state():
    @T.prim_func
    def main(A: T.Tensor((8,), "float32"), B: T.Tensor((8,), "float32")):
        with T.Kernel(1):
            frag = T.alloc_fragment((8,), "float32")
            shared = T.alloc_shared((8,), "float32")
            for i in T.Parallel(8):
                frag[i] = A[i]
            T.copy(A, shared)
            for i in T.Parallel(8):
                frag[i] = frag[i] + shared[i]
                B[i] = frag[i]

    result = _apply(main)
    assert len(_vfs(result)) == 2
    assert any(buffer.scope() == "shared.dyn" for block in _collect(result, tirx.SBlock) for buffer in block.alloc_buffers)
    assert any(buffer.scope() == "local.fragment" for vf in _vfs(result) for buffer in vf.alloc_buffers)
    assert len(_copy_calls(result)) == 5  # Current update classification reloads both regions.


def test_m3_read_only_reduce_source_reloads_vf_local_fragment():
    @T.prim_func
    def main(B: T.Tensor((2,), "float32"), A: T.Tensor((2, 8), "float32")):
        with T.Kernel(1):
            src = T.alloc_fragment((2, 8), "float32")
            dst = T.alloc_fragment((2,), "float32")
            shared = T.alloc_shared((2, 8), "float32")
            T.fill(src, 1)
            T.copy(A, shared)
            T.reduce_sum(src, dst, dim=1)
            for i in T.Parallel(2):
                B[i] = dst[i]

    result = _apply(main)
    assert len(_vfs(result)) == 2
    reduce = next(call for call in _collect(_vfs(result)[1], tirx.Call) if call.op.name == "tl.tileop.reduce")
    source = reduce.args[0].args[0].buffer
    assert source.scope() == "local.fragment"
    assert any(
        call.args[0].args[0].buffer.scope() == "shared.dyn" and call.args[1].args[0].buffer.same_as(source)
        for call in _copy_calls(_vfs(result)[1])
    )
    target = tvm.target.Target("ascend")
    with target:
        lowered = AscendPassPipelineBody(tvm.IRModule.from_expr(main), target)
    assert lowered["main"] is not None


def test_m3_strided_state_output_stays_in_vf():
    @T.prim_func
    def main(A: T.Tensor((2, 2, 8), "float32"), B: T.Tensor((2, 4, 16), "float32")):
        with T.Kernel(1):
            frag = T.alloc_fragment((2, 2, 8), "float32")
            shared = T.alloc_shared((2, 2, 8), "float32")
            T.fill(frag, 1)
            T.copy(A, shared)
            T.copy(frag, B[:, 1:3, 4:12])

    result = _apply(main)
    assert len(_vfs(result)) == 2
    assert any("B" in str(call.args[1]) for call in _copy_calls(_vfs(result)[1]))
    target = tvm.target.Target("ascend")
    with target:
        assert AscendPassPipelineBody(tvm.IRModule.from_expr(main), target)["main"] is not None


def test_m4_explicit_vf_threads_do_not_change_generated_vf_threads():
    @T.prim_func
    def main(A: T.Tensor((8,), "float32"), B: T.Tensor((8,), "float32"), C: T.Tensor((8,), "float32")):
        with T.Kernel(1):
            with T.SimtVF(threads=64):
                for i in T.Parallel(8):
                    B[i] = A[i]
            for i in T.Parallel(8):
                C[i] = A[i]

    result = _apply(main)
    assert len(_vfs(result)) == 2
    extents = [_thread_extents(vf)["threadIdx.x"] for vf in _vfs(result)]
    assert extents == [64, 128]
    repeated = transform.AutoSimtVF()(tvm.IRModule.from_expr(result))["main"]
    tvm.ir.assert_structural_equal(result, repeated)
