import pytest

from tilelang import tvm as tvm
import tilelang as tl
from tilelang.backend.target import determine_target
import tilelang.language as T
from tvm.tirx.stmt_functor import post_order_visit

auto_target = tvm.target.Target(determine_target("auto"))
sm80_target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})
sm90_target = tvm.target.Target({"kind": "cuda", "arch": "sm_90a"})
sm100_target = tvm.target.Target({"kind": "cuda", "arch": "sm_100"})


def _check(original, transformed, target=auto_target):
    func = original
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    mod = tl.transform.IfStmtBinding()(mod)
    mod = tl.transform.PipelinePlanning()(mod)
    mod = tl.transform.Simplify()(mod)
    transformed = tvm.IRModule.from_expr(transformed.with_attr("global_symbol", "main"))
    transformed = tvm.tirx.transform.BindTarget(target)(transformed)
    tvm.ir.assert_structural_equal(mod["main"], transformed["main"], True)


def _collect_pipeline_loop_annotations(func):
    annos = []

    def _visit(node):
        if isinstance(node, tvm.tirx.For) and "software_pipeline_stage" in node.annotations:
            annos.append(node.annotations)

    post_order_visit(func.body, _visit)
    return annos


def _collect_allocated_buffer_shapes(func):
    shapes = {}

    def _visit(node):
        if not isinstance(node, tvm.tirx.SBlock):
            return
        for buffer in node.alloc_buffers:
            shapes.setdefault(buffer.name, set()).add(tuple(int(dim) for dim in buffer.shape))

    post_order_visit(func.body, _visit)
    return shapes


def _run_pipeline_planning(func, target=auto_target):
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    mod = tl.transform.IfStmtBinding()(mod)
    mod = tl.transform.PipelinePlanning()(mod)
    return mod


def test_pipeline_planning_before_after_copy_gemm_num_stages_plan():
    @T.prim_func
    def before(
        A: T.Tensor((1024, 32), T.float32),
        B: T.Tensor((32, 1024), T.float32),
        C: T.Tensor((1024, 1024), T.float32),
    ):
        with T.Kernel(8, 8, threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 32), T.float32)
            B_shared = T.alloc_shared((32, 128), T.float32)
            C_local = T.alloc_fragment((128, 128), T.float32)

            T.clear(C_local)

            for ko in T.Pipelined(32, num_stages=3):
                T.copy(A[by * 128, ko * 32], A_shared)
                T.copy(B[ko * 32, bx * 128], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * 128, bx * 128])

    @T.prim_func
    def after(
        A: T.Tensor((1024, 32), T.float32),
        B: T.Tensor((32, 1024), T.float32),
        C: T.Tensor((1024, 1024), T.float32),
    ):
        with T.Kernel(8, 8, threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 32), T.float32)
            B_shared = T.alloc_shared((32, 128), T.float32)
            C_local = T.alloc_fragment((128, 128), T.float32)

            T.clear(C_local)

            for ko in T.serial(
                32,
                annotations={
                    "software_pipeline_async_producer_groups": [
                        T.int32(0),
                        T.int32(0),
                        T.int32(-1),
                    ],
                    "software_pipeline_async_producers": [
                        T.int32(1),
                        T.int32(1),
                        T.int32(0),
                    ],
                    "software_pipeline_async_stages": [T.int32(0)],
                    "software_pipeline_order": [
                        T.int32(0),
                        T.int32(1),
                        T.int32(2),
                    ],
                    "software_pipeline_stage": [
                        T.int32(0),
                        T.int32(0),
                        T.int32(2),
                    ],
                    "tl_pipelined_num_stages": T.int32(3),
                },
            ):
                T.copy(A[by * 128, ko * 32], A_shared)
                T.copy(B[ko * 32, bx * 128], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * 128, bx * 128])

    _check(before, after, target=sm80_target)


def test_pipeline_planning_before_after_copy_consumer_num_stages_plan():
    @T.prim_func
    def before(A: T.Tensor((64,), T.float16), C: T.Tensor((64,), T.float16)):
        S = T.alloc_buffer((16,), dtype=T.float16, scope="shared")
        for i in T.Pipelined(4, num_stages=2):
            T.copy(A[i * 16], S)
            T.copy(S, C[i * 16])

    @T.prim_func
    def after(A: T.Tensor((64,), T.float16), C: T.Tensor((64,), T.float16)):
        S = T.alloc_buffer((16,), dtype=T.float16, scope="shared")
        for i in T.serial(
            4,
            annotations={
                "software_pipeline_async_producer_groups": [
                    T.int32(0),
                    T.int32(-1),
                ],
                "software_pipeline_async_producers": [T.int32(1), T.int32(0)],
                "software_pipeline_async_stages": [T.int32(0)],
                "software_pipeline_order": [T.int32(0), T.int32(1)],
                "software_pipeline_stage": [T.int32(0), T.int32(1)],
                "tl_pipelined_num_stages": T.int32(2),
            },
        ):
            T.copy(A[i * 16], S)
            T.copy(S, C[i * 16])

    _check(before, after, target=sm80_target)


def test_pipeline_planning_before_after_loop_local_alloc_plan():
    @T.prim_func
    def before(A: T.Tensor((64,), T.float16), C: T.Tensor((64,), T.float16)):
        for i in T.Pipelined(4, num_stages=2):
            S = T.alloc_buffer((16,), dtype=T.float16, scope="shared")
            T.copy(A[i * 16], S)
            T.copy(S, C[i * 16])

    @T.prim_func
    def after(A: T.Tensor((64,), T.float16), C: T.Tensor((64,), T.float16)):
        for i in T.serial(
            4,
            annotations={
                "software_pipeline_async_producer_groups": [
                    T.int32(0),
                    T.int32(-1),
                ],
                "software_pipeline_async_producers": [T.int32(1), T.int32(0)],
                "software_pipeline_async_stages": [T.int32(0)],
                "software_pipeline_order": [T.int32(0), T.int32(1)],
                "software_pipeline_stage": [T.int32(0), T.int32(1)],
                "tl_pipelined_num_stages": T.int32(2),
            },
        ):
            S = T.alloc_buffer((16,), dtype=T.float16, scope="shared")
            T.copy(A[i * 16], S)
            T.copy(S, C[i * 16])

    _check(before, after, target=sm80_target)

    mod = _run_pipeline_planning(before, sm80_target)
    tl.transform.InjectSoftwarePipeline()(mod)


def test_pipeline_planning_schedules_guarded_body_after_replayable_bind_inline():
    @T.prim_func
    def before(
        KV: T.Tensor((4, 4), T.bfloat16),
        ids: T.Tensor((4,), T.int32),
        C: T.Tensor((4,), T.bfloat16),
    ):
        with T.Kernel(4, threads=1):
            A = T.alloc_shared((4,), T.bfloat16)

            for i in T.Pipelined(4, num_stages=2):
                if i > 1:
                    idx = ids[i]
                    T.copy(KV[idx, :], A)
                    T.copy(A, C)

    mod = _run_pipeline_planning(before, sm80_target)
    annos = _collect_pipeline_loop_annotations(mod["main"])
    assert len(annos) == 1
    anno = annos[0]
    assert [int(v) for v in anno["software_pipeline_stage"]] == [0, 1]
    assert [int(v) for v in anno["software_pipeline_order"]] == [0, 1]
    assert "software_pipeline_replayable_scalar_binds" not in anno
    tl.transform.InjectSoftwarePipeline()(mod)


def test_pipeline_planning_before_after_mbarrier_arrive_wait_plan():
    @T.prim_func
    def before(A: T.Tensor((64,), T.float16), C: T.Tensor((64,), T.float16)):
        with T.Kernel(1, threads=1):
            S = T.alloc_shared((16,), T.float16)
            mbar = T.alloc_barrier(1)
            for i in T.Pipelined(4, num_stages=2):
                T.copy(A[i * 16], S)
                T.mbarrier_arrive(mbar)
                T.mbarrier_wait_parity(mbar, i % 2)
                T.copy(S, C[i * 16])

    @T.prim_func
    def after(A: T.Tensor((64,), T.float16), C: T.Tensor((64,), T.float16)):
        with T.Kernel(1, threads=1):
            S = T.alloc_shared((16,), T.float16)
            mbar = T.alloc_barrier(1)
            for i in T.serial(
                4,
                annotations={
                    "software_pipeline_async_producer_groups": [
                        T.int32(0),
                        T.int32(-1),
                        T.int32(-1),
                        T.int32(-1),
                    ],
                    "software_pipeline_async_producers": [
                        T.int32(1),
                        T.int32(0),
                        T.int32(0),
                        T.int32(0),
                    ],
                    "software_pipeline_async_stages": [T.int32(0)],
                    "software_pipeline_order": [
                        T.int32(0),
                        T.int32(1),
                        T.int32(2),
                        T.int32(3),
                    ],
                    "software_pipeline_stage": [
                        T.int32(0),
                        T.int32(1),
                        T.int32(1),
                        T.int32(1),
                    ],
                    "tl_pipelined_num_stages": T.int32(2),
                },
            ):
                T.copy(A[i * 16], S)
                T.mbarrier_arrive(mbar)
                T.mbarrier_wait_parity(mbar, i % 2)
                T.copy(S, C[i * 16])

    _check(before, after, target=sm80_target)


def test_pipeline_planning_before_after_tma_copy_plan():
    @T.prim_func
    def before(A: T.Tensor((64,), T.float16), C: T.Tensor((64,), T.float16)):
        with T.Kernel(1, threads=1):
            S = T.alloc_shared((16,), T.float16)
            mbar = T.alloc_barrier(1)
            for i in T.Pipelined(4, num_stages=2):
                T.tma_copy(A[i * 16], S, barrier=mbar)
                T.barrier_arrive(mbar)
                T.mbarrier_wait_parity(mbar, i % 2)
                T.copy(S, C[i * 16])

    @T.prim_func
    def after(A: T.Tensor((64,), T.float16), C: T.Tensor((64,), T.float16)):
        with T.Kernel(1, threads=1):
            S = T.alloc_shared((16,), T.float16)
            mbar = T.alloc_barrier(1)
            for i in T.serial(
                4,
                annotations={
                    "software_pipeline_async_producer_groups": [
                        T.int32(0),
                        T.int32(-1),
                        T.int32(-1),
                        T.int32(-1),
                    ],
                    "software_pipeline_async_producers": [
                        T.int32(1),
                        T.int32(0),
                        T.int32(0),
                        T.int32(0),
                    ],
                    "software_pipeline_async_stages": [T.int32(0)],
                    "software_pipeline_order": [
                        T.int32(0),
                        T.int32(1),
                        T.int32(2),
                        T.int32(3),
                    ],
                    "software_pipeline_stage": [
                        T.int32(0),
                        T.int32(1),
                        T.int32(1),
                        T.int32(1),
                    ],
                    "tl_pipelined_num_stages": T.int32(2),
                },
            ):
                T.tma_copy(A[i * 16], S, barrier=mbar)
                T.barrier_arrive(mbar)
                T.mbarrier_wait_parity(mbar, i % 2)
                T.copy(S, C[i * 16])

    _check(before, after, target=sm90_target)


def test_pipeline_planning_before_after_wgmma_gemm_plan():
    @T.prim_func
    def before(
        A: T.Tensor((64, 16), T.float16),
        B: T.Tensor((16, 64), T.float16),
        D: T.Tensor((64, 64), T.float16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((64, 16), T.float16)
            B_shared = T.alloc_shared((16, 64), T.float16)
            C_local = T.alloc_fragment((64, 64), T.float16)

            for _i in T.Pipelined(4, num_stages=3):
                T.copy(A[0:64, 0:16], A_shared)
                T.copy(B[0:16, 0:64], B_shared)
                T.wgmma_gemm(A_shared, B_shared, C_local, clear_accum=True)
                T.wait_wgmma(0)
                T.copy(C_local, D[0:64, 0:64])

    @T.prim_func
    def after(
        A: T.Tensor((64, 16), T.float16),
        B: T.Tensor((16, 64), T.float16),
        D: T.Tensor((64, 64), T.float16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((64, 16), T.float16)
            B_shared = T.alloc_shared((16, 64), T.float16)
            C_local = T.alloc_fragment((64, 64), T.float16)

            for _i in T.serial(
                4,
                annotations={
                    "software_pipeline_async_producer_groups": [
                        T.int32(0),
                        T.int32(0),
                        T.int32(-1),
                        T.int32(-1),
                        T.int32(-1),
                    ],
                    "software_pipeline_async_producers": [
                        T.int32(1),
                        T.int32(1),
                        T.int32(0),
                        T.int32(0),
                        T.int32(0),
                    ],
                    "software_pipeline_async_stages": [T.int32(0)],
                    "software_pipeline_order": [
                        T.int32(1),
                        T.int32(2),
                        T.int32(0),
                        T.int32(3),
                        T.int32(4),
                    ],
                    "software_pipeline_stage": [
                        T.int32(0),
                        T.int32(0),
                        T.int32(3),
                        T.int32(3),
                        T.int32(3),
                    ],
                    "tl_pipelined_num_stages": T.int32(3),
                },
            ):
                T.copy(A[0:64, 0:16], A_shared)
                T.copy(B[0:16, 0:64], B_shared)
                T.wgmma_gemm(A_shared, B_shared, C_local, clear_accum=True)
                T.wait_wgmma(0)
                T.copy(C_local, D[0:64, 0:64])

    _check(before, after, target=sm90_target)

    mod = _run_pipeline_planning(before, sm90_target)
    mod = tl.transform.InjectSoftwarePipeline()(mod)
    shapes = _collect_allocated_buffer_shapes(mod["main"])
    assert (3, 64, 16) in shapes["A_shared"]
    assert (3, 16, 64) in shapes["B_shared"]
    assert (64, 64) in shapes["C_local"]
    assert all(len(shape) == 2 for shape in shapes["C_local"])


def test_pipeline_planning_before_after_tcgen05_gemm_plan():
    @T.prim_func
    def before(
        A: T.Tensor((128, 128), T.bfloat16),
        B: T.Tensor((128, 128), T.bfloat16),
        D: T.Tensor((128, 128), T.bfloat16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((128, 128), T.bfloat16)
            B_shared = T.alloc_shared((128, 128), T.bfloat16)
            C_tmem = T.alloc_tmem((128, 128), T.float32)
            C_local = T.alloc_fragment((128, 128), T.float32)
            C_shared = T.alloc_shared((128, 128), T.bfloat16)
            mbar = T.alloc_barrier(1)

            for i in T.Pipelined(2, num_stages=2):
                T.copy(A[0:128, 0:128], A_shared)
                T.copy(B[0:128, 0:128], B_shared)
                T.tcgen05_gemm(
                    A_shared,
                    B_shared,
                    C_tmem,
                    transpose_B=True,
                    mbar=mbar,
                    clear_accum=i == 0,
                )
                T.mbarrier_wait_parity(mbar, i % 2)

            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, D[0:128, 0:128])

    @T.prim_func
    def after(
        A: T.Tensor((128, 128), T.bfloat16),
        B: T.Tensor((128, 128), T.bfloat16),
        D: T.Tensor((128, 128), T.bfloat16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((128, 128), T.bfloat16)
            B_shared = T.alloc_shared((128, 128), T.bfloat16)
            C_tmem = T.alloc_tmem((128, 128), T.float32)
            C_local = T.alloc_fragment((128, 128), T.float32)
            C_shared = T.alloc_shared((128, 128), T.bfloat16)
            mbar = T.alloc_barrier(1)

            for i in T.serial(
                2,
                annotations={
                    "software_pipeline_async_producer_groups": [
                        T.int32(0),
                        T.int32(0),
                        T.int32(-1),
                        T.int32(-1),
                    ],
                    "software_pipeline_async_producers": [
                        T.int32(1),
                        T.int32(1),
                        T.int32(0),
                        T.int32(0),
                    ],
                    "software_pipeline_async_stages": [T.int32(0)],
                    "software_pipeline_order": [
                        T.int32(0),
                        T.int32(1),
                        T.int32(2),
                        T.int32(3),
                    ],
                    "software_pipeline_stage": [
                        T.int32(0),
                        T.int32(0),
                        T.int32(1),
                        T.int32(1),
                    ],
                    "tl_pipelined_num_stages": T.int32(2),
                },
            ):
                T.copy(A[0:128, 0:128], A_shared)
                T.copy(B[0:128, 0:128], B_shared)
                T.tcgen05_gemm(
                    A_shared,
                    B_shared,
                    C_tmem,
                    transpose_B=True,
                    mbar=mbar,
                    clear_accum=i == 0,
                )
                T.mbarrier_wait_parity(mbar, i)

            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, D[0:128, 0:128])

    _check(before, after, target=sm100_target)

    mod = _run_pipeline_planning(before, sm100_target)
    mod = tl.transform.InjectSoftwarePipeline()(mod)
    shapes = _collect_allocated_buffer_shapes(mod["main"])
    assert (2, 128, 128) in shapes["A_shared"]
    assert (2, 128, 128) in shapes["B_shared"]
    assert (128, 128) in shapes["C_tmem"]
    assert all(len(shape) == 2 for shape in shapes["C_tmem"])


def test_pipeline_planning_before_after_explicit_plan_drops_replayable_bind_slot():
    @T.prim_func
    def before(A: T.Tensor((64,), T.float16), C: T.Tensor((64,), T.float16)):
        S = T.alloc_buffer((16,), dtype=T.float16, scope="shared")
        for i in T.Pipelined(4, order=[1, 0], stage=[0, 1]):
            base: T.int32 = i * 16
            T.copy(A[base], S)
            T.copy(S, C[base])

    @T.prim_func
    def after(A: T.Tensor((64,), T.float16), C: T.Tensor((64,), T.float16)):
        S = T.alloc_buffer((16,), dtype=T.float16, scope="shared")
        for i in T.serial(
            4,
            annotations={
                "software_pipeline_async_producer_groups": [
                    T.int32(0),
                    T.int32(-1),
                ],
                "software_pipeline_async_producers": [T.int32(1), T.int32(0)],
                "software_pipeline_async_stages": [T.int32(0)],
                "software_pipeline_order": [T.int32(1), T.int32(0)],
                "software_pipeline_replayable_scalar_binds": [
                    T.int32(1),
                    T.int32(0),
                    T.int32(0),
                ],
                "software_pipeline_stage": [T.int32(0), T.int32(1)],
            },
        ):
            T.copy(A[i * 16], S)
            T.copy(S, C[i * 16])

    _check(before, after, target=sm80_target)


def test_pipeline_planning_before_after_explicit_plan_keeps_pipeline_buffer_bind_slot():
    @T.prim_func
    def before(A: T.Tensor((64,), T.float16), C: T.Tensor((64,), T.float16)):
        S = T.alloc_buffer((16,), dtype=T.float16, scope="shared")
        for i in T.Pipelined(4, order=[0, 1, 2], stage=[0, 1, 1]):
            base: T.int32 = i * 16
            T.copy(A[base], S)
            value: T.float16 = S[0]
            C[base] = value

    @T.prim_func
    def after(A: T.Tensor((64,), T.float16), C: T.Tensor((64,), T.float16)):
        S = T.alloc_buffer((16,), dtype=T.float16, scope="shared")
        for i in T.serial(
            4,
            annotations={
                "software_pipeline_async_producer_groups": [
                    T.int32(0),
                    T.int32(-1),
                    T.int32(-1),
                ],
                "software_pipeline_async_producers": [
                    T.int32(1),
                    T.int32(0),
                    T.int32(0),
                ],
                "software_pipeline_async_stages": [T.int32(0)],
                "software_pipeline_order": [
                    T.int32(0),
                    T.int32(1),
                    T.int32(2),
                ],
                "software_pipeline_replayable_scalar_binds": [
                    T.int32(1),
                    T.int32(0),
                    T.int32(0),
                    T.int32(0),
                ],
                "software_pipeline_stage": [
                    T.int32(0),
                    T.int32(1),
                    T.int32(1),
                ],
            },
        ):
            T.copy(A[i * 16], S)
            value: T.float16 = S[0]
            C[i * 16] = value

    _check(before, after, target=sm80_target)


def test_simple_pipeline():
    @T.prim_func
    def before(A: T.Tensor((1024, 32), T.float32), B: T.Tensor((32, 1024), T.float32), C: T.Tensor((1024, 1024), T.float32)):
        with T.Kernel(8, 8, threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 32), T.float32)
            B_shared = T.alloc_shared((32, 128), T.float32)
            C_local = T.alloc_fragment((128, 128), T.float32)

            T.clear(C_local)

            for ko in T.Pipelined(32, num_stages=3):
                T.copy(A[by * 128, ko * 32], A_shared)
                T.copy(B[ko * 32, bx * 128], B_shared)

                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * 128, bx * 128])

    func = before
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tirx.transform.BindTarget(auto_target)(mod)
    mod = tl.transform.IfStmtBinding()(mod)
    mod = tl.transform.PipelinePlanning()(mod)
    mod = tl.transform.Simplify()(mod)

    annos = _collect_pipeline_loop_annotations(mod["main"])
    assert len(annos) == 1
    anno = annos[0]
    assert "software_pipeline_stage" in anno
    assert "software_pipeline_order" in anno
    assert "tl_pipelined_num_stages" in anno
    stages = [int(s) for s in anno["software_pipeline_stage"]]
    orders = [int(o) for o in anno["software_pipeline_order"]]
    assert stages == [0, 0, 2]
    assert orders == [0, 1, 2]
    assert int(anno["tl_pipelined_num_stages"]) == 3
    # tma_copies annotation depends on target TMA capability
    if "software_pipeline_tma_copies" in anno:
        tma_copies = [int(t) for t in anno["software_pipeline_tma_copies"]]
        # On TMA-capable targets, copies are marked as TMA-eligible
        assert tma_copies[2] == 0  # gemm is never TMA


def test_pipeline_planning_recognizes_parallel_bufferstore_copy_stages():
    @T.prim_func
    def before(
        A: T.Tensor((1024, 32), T.float32),
        B: T.Tensor((32, 1024), T.float32),
        C: T.Tensor((1024, 1024), T.float32),
    ):
        with T.Kernel(8, 8, threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 32), T.float32)
            B_shared = T.alloc_shared((32, 128), T.float32)
            C_local = T.alloc_fragment((128, 128), T.float32)

            T.clear(C_local)

            for ko in T.Pipelined(32, num_stages=3):
                for i, k in T.Parallel(128, 32):
                    A_shared[i, k] = A[by * 128 + i, ko * 32 + k]
                for k, j in T.Parallel(32, 128):
                    B_shared[k, j] = B[ko * 32 + k, bx * 128 + j]
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * 128, bx * 128])

    mod = _run_pipeline_planning(before, sm80_target)
    annos = _collect_pipeline_loop_annotations(mod["main"])
    assert annos, "Expected at least one loop annotated by PipelinePlanning"
    anno = annos[0]
    stages = [int(v) for v in anno["software_pipeline_stage"]]
    orders = [int(v) for v in anno["software_pipeline_order"]]
    async_producers = [int(v) for v in anno["software_pipeline_async_producers"]]
    async_groups = [int(v) for v in anno["software_pipeline_async_producer_groups"]]
    assert stages == [0, 0, 2]
    assert orders == [0, 1, 2]
    assert async_producers == [1, 1, 0]
    assert async_groups == [0, 0, -1]


def test_pipeline_planning_marks_async_producers_per_statement():
    @T.prim_func
    def before(A: T.Tensor((1024, 32), T.float32), B: T.Tensor((32, 1024), T.float32), C: T.Tensor((1024, 1024), T.float32)):
        with T.Kernel(8, 8, threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 32), T.float32)
            B_shared = T.alloc_shared((32, 128), T.float32)
            C_local = T.alloc_fragment((128, 128), T.float32)

            T.clear(C_local)

            for ko in T.Pipelined(32, num_stages=3):
                T.copy(A[by * 128, ko * 32], A_shared)
                T.copy(B[ko * 32, bx * 128], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * 128, bx * 128])

    mod = _run_pipeline_planning(before, sm80_target)
    annos = _collect_pipeline_loop_annotations(mod["main"])
    assert annos, "Expected at least one loop annotated by PipelinePlanning"
    anno = annos[0]
    assert "software_pipeline_async_producers" in anno
    assert "software_pipeline_async_producer_groups" in anno
    assert "software_pipeline_async_stages" in anno
    async_producers = [int(v) for v in anno["software_pipeline_async_producers"]]
    async_groups = [int(v) for v in anno["software_pipeline_async_producer_groups"]]
    async_stages = [int(v) for v in anno["software_pipeline_async_stages"]]
    assert async_producers == [1, 1, 0]
    assert async_groups == [0, 0, -1]
    assert async_stages == [0]


def test_pipeline_planning_keeps_plain_hopper_pipeline_copies_sync():
    hopper_target = tvm.target.Target({"kind": "cuda", "arch": "sm_90a"})

    @T.prim_func
    def before(
        A: T.Tensor((1024, 32), T.float32),
        B: T.Tensor((32, 1024), T.float32),
        C: T.Tensor((1024, 1024), T.float32),
    ):
        with T.Kernel(8, 8, threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 32), T.float32)
            B_shared = T.alloc_shared((32, 128), T.float32)
            C_local = T.alloc_fragment((128, 128), T.float32)
            T.clear(C_local)
            for k in T.Pipelined(32, num_stages=2):
                T.copy(A[by * 128, k * 32], A_shared)
                T.copy(B[k * 32, bx * 128], B_shared)
                T.gemm(A_shared, B_shared, C_local)

    mod = _run_pipeline_planning(before, hopper_target)

    annos = _collect_pipeline_loop_annotations(mod["main"])
    assert annos, "Expected at least one loop annotated by PipelinePlanning"
    anno = annos[0]
    assert "software_pipeline_tma_copies" not in anno


def test_pipeline_planning_stages_bind_with_dependent_copy():
    @T.prim_func
    def before(
        KV: T.Tensor((4, 4), T.float16),
        ids: T.Tensor((4,), T.int32),
        C: T.Tensor((4,), T.float16),
    ):
        with T.Kernel(1, threads=1):
            A = T.alloc_shared((4,), T.float16)
            for i in T.Pipelined(4, num_stages=2):
                _id = ids[i]
                T.copy(KV[_id, :], A)
                T.copy(A, C)

    mod = _run_pipeline_planning(before, sm80_target)
    annos = _collect_pipeline_loop_annotations(mod["main"])
    assert annos, "Expected at least one loop annotated by PipelinePlanning"
    stages = [int(v) for v in annos[0]["software_pipeline_stage"]]
    orders = [int(v) for v in annos[0]["software_pipeline_order"]]
    async_producers = [int(v) for v in annos[0]["software_pipeline_async_producers"]]
    replayable_binds = [int(v) for v in annos[0]["software_pipeline_replayable_scalar_binds"]]

    assert len(stages) == 2, f"Expected copy and consumer stages, got {stages}"
    assert stages == [0, 1]
    assert orders == [0, 1]
    assert async_producers == [1, 0]
    assert replayable_binds == [1, 0, 0]


def test_pipeline_planning_mixed_buffer_scalar_producer_chain():
    @T.prim_func
    def before(
        A: T.Tensor((64,), T.float16),
        B: T.Tensor((64,), T.float16),
    ):
        with T.Kernel(1, threads=256):
            index_shared = T.alloc_shared((1,), T.int32)
            A_shared = T.alloc_shared((32,), T.float16)
            for k in T.Pipelined(2, num_stages=2):
                index_shared[0] = k * 32
                offset = index_shared[0]
                T.copy(A[offset], A_shared)
                for i in T.Parallel(32):
                    B[k * 32 + i] = A_shared[i]

    mod = _run_pipeline_planning(before, sm80_target)
    annos = _collect_pipeline_loop_annotations(mod["main"])
    assert len(annos) == 1
    anno = annos[0]
    stages = [int(v) for v in anno["software_pipeline_stage"]]
    orders = [int(v) for v in anno["software_pipeline_order"]]

    assert stages == [0, 0, 0, 1]
    assert orders == [0, 1, 2, 3]
    tl.transform.InjectSoftwarePipeline()(mod)


def test_pipeline_planning_tracks_buffer_dependencies_in_copy_indices():
    """Regression for #1792: copy indices may load a local.var buffer."""

    @T.prim_func
    def before(
        A: T.Tensor((64,), T.float16),
        B: T.Tensor((64,), T.float16),
    ):
        with T.Kernel(1, threads=32):
            offset = T.alloc_var(T.int32)
            A_shared = T.alloc_shared((16,), T.float16)
            for k in T.Pipelined(4, num_stages=2):
                offset = k * 16
                T.copy(A[offset], A_shared)
                T.copy(A_shared, B[k * 16])

    mod = _run_pipeline_planning(before, sm80_target)
    annos = _collect_pipeline_loop_annotations(mod["main"])
    assert len(annos) == 1
    anno = annos[0]
    stages = [int(v) for v in anno["software_pipeline_stage"]]
    orders = [int(v) for v in anno["software_pipeline_order"]]

    assert stages == [0, 0, 1]
    assert orders == [0, 1, 2]
    tl.transform.InjectSoftwarePipeline()(mod)


def test_pipeline_planning_preserves_read_before_shared_overwrite():
    """Regression for #2668: a WAR dependency must not cross stages."""

    @T.prim_func
    def before(
        A: T.Tensor((8, 32), T.int32),
        S_init: T.Tensor((32,), T.int32),
        B: T.Tensor((32,), T.int32),
    ):
        with T.Kernel(1, threads=32):
            S = T.alloc_shared((32,), T.int32)
            acc = T.alloc_fragment((32,), T.int32)
            T.clear(acc)
            T.copy(S_init, S)
            for k in T.Pipelined(8, num_stages=2):
                for i in T.Parallel(32):
                    acc[i] += S[i]
                T.copy(A[k, :], S)
            T.copy(acc, B)

    mod = _run_pipeline_planning(before, sm80_target)
    annos = _collect_pipeline_loop_annotations(mod["main"])
    assert len(annos) == 1
    anno = annos[0]
    stages = [int(v) for v in anno["software_pipeline_stage"]]
    orders = [int(v) for v in anno["software_pipeline_order"]]

    # The consume and following overwrite form a loop-carried buffer
    # lifecycle.  Keeping them in one stage preserves the serial read-before-
    # write order; separating them would let a future async copy race the read.
    assert stages[0] == stages[1]
    assert orders == [0, 1]
    assert "software_pipeline_async_producers" not in anno
    tl.transform.InjectSoftwarePipeline()(mod)


def test_pipeline_planning_rejects_aliased_buffer_views():
    """PipelinePlanning must reject multiple Buffer views of one data Var."""

    @T.prim_func
    def before(
        A: T.Tensor((64,), T.float16),
        B: T.Tensor((64,), T.float16),
    ):
        with T.Kernel(1, threads=16):
            shared = T.alloc_shared((16,), T.float16)
            shared_view = T.decl_buffer(
                (4, 4),
                T.float16,
                data=shared.data,
                scope="shared.dyn",
            )
            for k in T.Pipelined(4, num_stages=2):
                T.copy(A[k * 16], shared)
                for i, j in T.Parallel(4, 4):
                    B[k * 16 + i * 4 + j] = shared_view[i, j]

    with pytest.raises(tvm.TVMError, match="does not support aliased Buffer views"):
        _run_pipeline_planning(before, sm80_target)


def test_pipeline_planning_rejects_aliased_buffer_views_with_explicit_schedule():
    """Explicit pipeline annotations must not bypass alias validation."""

    @T.prim_func
    def before(
        A: T.Tensor((64,), T.float16),
        B: T.Tensor((64,), T.float16),
    ):
        with T.Kernel(1, threads=16):
            shared = T.alloc_shared((16,), T.float16)
            shared_view = T.decl_buffer(
                (4, 4),
                T.float16,
                data=shared.data,
                scope="shared.dyn",
            )
            for k in T.Pipelined(4, order=[0, 1], stage=[0, 1]):
                T.copy(A[k * 16], shared)
                for i, j in T.Parallel(4, 4):
                    B[k * 16 + i * 4 + j] = shared_view[i, j]

    with pytest.raises(tvm.TVMError, match="does not support aliased Buffer views"):
        _run_pipeline_planning(before, sm80_target)


def test_pipeline_planning_preserves_guarded_control_before_copy():
    """A guarded opaque control call must retain its dependency on a copy."""

    @T.prim_func
    def before(
        A: T.Tensor((64,), T.float16),
        B: T.Tensor((64,), T.float16),
    ):
        with T.Kernel(1, threads=1):
            shared = T.alloc_shared((16,), T.float16)
            barrier = T.alloc_barrier(1)
            for k in T.Pipelined(4, num_stages=2):
                if k > 0:
                    T.mbarrier_wait_parity(barrier, k % 2)
                T.copy(A[k * 16], shared)
                T.copy(shared, B[k * 16])

    mod = _run_pipeline_planning(before, sm80_target)
    annos = _collect_pipeline_loop_annotations(mod["main"])
    assert len(annos) == 1
    anno = annos[0]
    stages = [int(v) for v in anno["software_pipeline_stage"]]
    orders = [int(v) for v in anno["software_pipeline_order"]]

    assert stages[0] <= stages[1], "The guarded control statement must not execute in a later logical iteration than the copy it precedes"
    if stages[0] == stages[1]:
        assert orders[0] < orders[1]


def test_pipeline_planning_places_dependency_sinks_in_last_stage():
    @T.prim_func
    def before(
        A: T.Tensor((64,), T.float16),
        B: T.Tensor((64,), T.float16),
    ):
        with T.Kernel(1, threads=16):
            unused = T.alloc_shared((1,), T.int32)
            A_shared = T.alloc_shared((16,), T.float16)
            for k in T.Pipelined(4, num_stages=3):
                unused[0] = k
                T.copy(A[k * 16], A_shared)
                T.copy(A_shared, B[k * 16])

    mod = _run_pipeline_planning(before, sm80_target)
    annos = _collect_pipeline_loop_annotations(mod["main"])
    assert len(annos) == 1
    anno = annos[0]
    stages = [int(v) for v in anno["software_pipeline_stage"]]
    orders = [int(v) for v in anno["software_pipeline_order"]]

    assert stages == [2, 0, 2]
    assert orders == [1, 0, 2]
    tl.transform.InjectSoftwarePipeline()(mod)


def test_pipeline_planning_places_guarded_sink_in_last_stage():
    @T.prim_func
    def before(
        A: T.Tensor((64,), T.float16),
        B: T.Tensor((64,), T.float16),
    ):
        with T.Kernel(1, threads=16):
            unused = T.alloc_shared((1,), T.int32)
            A_shared = T.alloc_shared((16,), T.float16)
            for k in T.Pipelined(4, num_stages=3):
                if k >= 0:
                    unused[0] = k
                T.copy(A[k * 16], A_shared)
                T.copy(A_shared, B[k * 16])

    mod = _run_pipeline_planning(before, sm80_target)
    annos = _collect_pipeline_loop_annotations(mod["main"])
    assert len(annos) == 1
    anno = annos[0]
    stages = [int(v) for v in anno["software_pipeline_stage"]]
    orders = [int(v) for v in anno["software_pipeline_order"]]

    # Source-order stages are intentionally non-monotonic. The guarded store
    # is an independent sink and moves to the terminal stage, while dependency
    # paths and the final execution order remain valid.
    assert stages == [2, 0, 2]
    assert orders == [1, 0, 2]
    tl.transform.InjectSoftwarePipeline()(mod)


def test_pipeline_planning_keeps_final_stage_with_internal_buffer_producer():
    @T.prim_func
    def before(
        A: T.Tensor((64,), T.float16),
        B: T.Tensor((4,), T.float16),
    ):
        with T.Kernel(1, threads=16):
            first = T.alloc_shared((16,), T.float16)
            second = T.alloc_shared((16,), T.float16)
            scalar = T.alloc_local((1,), T.float16)
            for k in T.Pipelined(4, num_stages=2):
                T.copy(A[k * 16], first)
                T.copy(first, second)
                scalar[0] = second[0]
                B[k] = scalar[0]

    mod = _run_pipeline_planning(before, sm80_target)
    annos = _collect_pipeline_loop_annotations(mod["main"])
    assert len(annos) == 1
    stages = [int(v) for v in annos[0]["software_pipeline_stage"]]
    orders = [int(v) for v in annos[0]["software_pipeline_order"]]

    # The zero-weight scalar store and its consumer are coalesced into the
    # provisional final stage.  Since the store still produces an internal
    # Buffer value, this is real work rather than a consumer-only endpoint and
    # must prevent terminal-stage retiming.
    assert stages == [0, 2, 2, 2]
    assert orders == [1, 0, 2, 3]
    tl.transform.InjectSoftwarePipeline()(mod)


def test_pipeline_planning_does_not_compact_num_stages_one():
    @T.prim_func
    def before(
        A: T.Tensor((64,), T.float16),
        B: T.Tensor((64,), T.float16),
    ):
        with T.Kernel(1, threads=16):
            shared = T.alloc_shared((16,), T.float16)
            for k in T.Pipelined(4, num_stages=1):
                T.copy(A[k * 16], shared)
                T.copy(shared, B[k * 16])

    mod = _run_pipeline_planning(before, sm80_target)
    annos = _collect_pipeline_loop_annotations(mod["main"])
    assert len(annos) == 1
    stages = [int(v) for v in annos[0]["software_pipeline_stage"]]

    assert stages == [0, 1]
    tl.transform.InjectSoftwarePipeline()(mod)


def test_pipeline_planning_can_disable_terminal_stage_compaction():
    @T.prim_func
    def before(
        A: T.Tensor((64,), T.float16),
        B: T.Tensor((64,), T.float16),
    ):
        with T.Kernel(1, threads=16):
            shared = T.alloc_shared((16,), T.float16)
            for k in T.Pipelined(4, num_stages=3):
                T.copy(A[k * 16], shared)
                T.copy(shared, B[k * 16])

    @T.prim_func
    def before_uncompacted(
        A: T.Tensor((64,), T.float16),
        B: T.Tensor((64,), T.float16),
    ):
        with T.Kernel(1, threads=16):
            shared = T.alloc_shared((16,), T.float16)
            for k in T.Pipelined(4, num_stages=3, compact_terminal_stage=False):
                T.copy(A[k * 16], shared)
                T.copy(shared, B[k * 16])

    default_mod = _run_pipeline_planning(before, sm80_target)
    default_annos = _collect_pipeline_loop_annotations(default_mod["main"])
    assert len(default_annos) == 1
    assert [int(v) for v in default_annos[0]["software_pipeline_stage"]] == [
        0,
        2,
    ]

    mod = _run_pipeline_planning(before_uncompacted, sm80_target)

    annos = _collect_pipeline_loop_annotations(mod["main"])
    assert len(annos) == 1
    stages = [int(v) for v in annos[0]["software_pipeline_stage"]]
    orders = [int(v) for v in annos[0]["software_pipeline_order"]]

    assert stages == [0, 3]
    assert orders == [0, 1]
    assert "tl_pipeline_compact_terminal_stage" not in annos[0]
    tl.transform.InjectSoftwarePipeline()(mod)


def test_pipeline_planning_num_stages_zero_keeps_plain_loop():
    @T.prim_func
    def before(
        A: T.Tensor((16,), T.float16),
        B: T.Tensor((16,), T.float16),
    ):
        for i in T.Pipelined(16, num_stages=0):
            B[i] = A[i]

    mod = _run_pipeline_planning(before, sm80_target)
    assert not _collect_pipeline_loop_annotations(mod["main"])
    assert "num_stages" not in mod["main"].script()


def test_pipeline_planning_manual_ws_uses_ring_buffer_stage_count():
    @T.prim_func
    def before(
        A: T.Tensor((64,), T.float16),
        B: T.Tensor((64,), T.float16),
    ):
        with T.Kernel(1, threads=256):
            A_shared = T.alloc_shared((16,), T.float16)
            for k in T.Pipelined(4, num_stages=2):
                with T.ws(0):
                    T.copy(A[k * 16], A_shared)
                with T.ws(1):
                    T.copy(A_shared, B[k * 16])

    mod = _run_pipeline_planning(before, sm80_target)
    annos = _collect_pipeline_loop_annotations(mod["main"])
    assert len(annos) == 1
    stages = [int(v) for v in annos[0]["software_pipeline_stage"]]
    assert stages == [0, 1]
    tl.transform.InjectSoftwarePipeline()(mod)


def test_pipeline_planning_accepts_explicit_bind_free_annotations():
    @T.prim_func
    def before(
        A: T.Tensor((64,), T.float16),
        B: T.Tensor((64,), T.float16),
        C: T.Tensor((64,), T.float16),
    ):
        with T.Kernel(1, threads=16):
            A_shared = T.alloc_shared((16,), T.float16)
            for i in T.Pipelined(4, order=[1, 0], stage=[0, 1]):
                base: T.int32 = i * 16
                T.copy(A[base], A_shared)
                T.copy(A_shared, C[base])

    mod = _run_pipeline_planning(before, sm80_target)
    annos = _collect_pipeline_loop_annotations(mod["main"])
    assert annos, "Expected at least one loop annotated by PipelinePlanning"
    anno = annos[0]
    stages = [int(v) for v in anno["software_pipeline_stage"]]
    orders = [int(v) for v in anno["software_pipeline_order"]]
    async_producers = [int(v) for v in anno["software_pipeline_async_producers"]]
    async_groups = [int(v) for v in anno["software_pipeline_async_producer_groups"]]
    replayable_binds = [int(v) for v in anno["software_pipeline_replayable_scalar_binds"]]

    assert stages == [0, 1]
    assert orders == [1, 0]
    assert async_producers == [1, 0]
    assert async_groups == [0, -1]
    assert replayable_binds == [1, 0, 0]


def test_pipeline_planning_keeps_bind_that_reads_pipeline_written_buffer():
    @T.prim_func
    def before(
        A: T.Tensor((64,), T.float16),
        C: T.Tensor((64,), T.float16),
    ):
        with T.Kernel(1, threads=16):
            A_shared = T.alloc_shared((16,), T.float16)
            for i in T.Pipelined(4, order=[0, 1, 2], stage=[0, 1, 1]):
                base: T.int32 = i * 16
                T.copy(A[base], A_shared)
                value: T.float16 = A_shared[0]
                C[base] = value

    mod = _run_pipeline_planning(before, sm80_target)
    annos = _collect_pipeline_loop_annotations(mod["main"])
    assert annos, "Expected at least one loop annotated by PipelinePlanning"
    anno = annos[0]
    stages = [int(v) for v in anno["software_pipeline_stage"]]
    orders = [int(v) for v in anno["software_pipeline_order"]]
    replayable_binds = [int(v) for v in anno["software_pipeline_replayable_scalar_binds"]]

    assert stages == [0, 1, 1]
    assert orders == [0, 1, 2]
    assert replayable_binds == [1, 0, 0, 0]


def test_pipeline_planning_keeps_bind_that_reads_atomic_target():
    @T.prim_func
    def before(
        counter: T.Tensor((1,), T.int32),
        out: T.Tensor((8,), T.int32),
    ):
        for i in T.Pipelined(
            4,
            order=[0, 1, 2, 3],
            stage=[0, 0, 1, 1],
        ):
            snapshot: T.int32 = counter[0]
            pos: T.int32 = T.atomic_add(counter[0], 1, return_prev=True)
            out[i * 2] = snapshot
            out[i * 2 + 1] = pos

    mod = _run_pipeline_planning(before, sm80_target)
    annos = _collect_pipeline_loop_annotations(mod["main"])
    assert annos, "Expected at least one loop annotated by PipelinePlanning"
    anno = annos[0]
    stages = [int(v) for v in anno["software_pipeline_stage"]]
    orders = [int(v) for v in anno["software_pipeline_order"]]

    assert stages == [0, 0, 1, 1]
    assert orders == [0, 1, 2, 3]
    assert "software_pipeline_replayable_scalar_binds" not in anno
