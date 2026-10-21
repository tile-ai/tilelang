from tilelang import tvm as tvm
import tilelang as tl
import pytest
import tilelang.testing
from tilelang.testing.ir import collect_nodes
from tilelang.backend.target import determine_target
import tilelang.language as T
from tvm.tirx.stmt_functor import post_order_visit

auto_target = tvm.target.Target(determine_target("auto"))
sm80_target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})
sm90_target = tvm.target.Target({"kind": "cuda", "arch": "sm_90a"})
sm100_target = tvm.target.Target({"kind": "cuda", "arch": "sm_100"})


def _supports_software_pipelining(target) -> bool:
    # PipelinePlanning strips the pipeline annotations on ROCm targets other than
    # gfx950, which fall back to a plain sequential loop.
    # See the TargetIsRocm/TargetIsGfx950 guard in src/transform/pipeline_planning.cc.
    if target.kind.name == "hip":
        from tilelang.rocm.target import target_is_gfx950

        return target_is_gfx950(target)
    return True


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
                        T.int32(1),
                        T.int32(2),
                        T.int32(0),
                        T.int32(3),
                    ],
                    "software_pipeline_stage": [
                        T.int32(0),
                        T.int32(0),
                        T.int32(2),
                        T.int32(2),
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
    if not _supports_software_pipelining(auto_target):
        assert len(annos) == 0
        return
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


@tilelang.testing.requires_cuda
def test_pipeline_planning_propagates_buffer_producer_through_bind():
    """A copy's scalar index can depend transitively on a shared-buffer store."""

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
                offset: T.int32 = index_shared[0]
                T.copy(A[offset], A_shared)
                for i in T.Parallel(32):
                    B[k * 32 + i] = A_shared[i]

    mod = _run_pipeline_planning(before, sm80_target)
    anno = _collect_pipeline_loop_annotations(mod["main"])[0]
    stages = [int(value) for value in anno["software_pipeline_stage"]]
    orders = [int(value) for value in anno["software_pipeline_order"]]
    assert stages == [0, 0, 0, 1]
    assert orders == [0, 1, 2, 3]
    tl.transform.InjectSoftwarePipeline()(mod)


@tilelang.testing.requires_cuda
def test_pipeline_planning_does_not_propagate_disjoint_buffer_region():
    """A store to another region of the same buffer is not a copy producer."""

    @T.prim_func
    def before(
        A: T.Tensor((64,), T.float16),
        B: T.Tensor((64,), T.float16),
    ):
        with T.Kernel(1, threads=256):
            index_shared = T.alloc_shared((2,), T.int32)
            A_shared = T.alloc_shared((32,), T.float16)
            for k in T.Pipelined(2, num_stages=2):
                index_shared[1] = k
                index_shared[0] = k * 32
                offset: T.int32 = index_shared[0]
                T.copy(A[offset], A_shared)
                for i in T.Parallel(32):
                    B[k * 32 + i] = A_shared[i]

    mod = _run_pipeline_planning(before, sm80_target)
    anno = _collect_pipeline_loop_annotations(mod["main"])[0]
    stages = [int(value) for value in anno["software_pipeline_stage"]]
    orders = [int(value) for value in anno["software_pipeline_order"]]
    assert stages == [1, 0, 0, 0, 1]
    assert orders == [3, 0, 1, 2, 4]
    tl.transform.InjectSoftwarePipeline()(mod)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("threads_x,threads_y", [(32, 1), (8, 4)])
def test_pipeline_planning_keeps_disjoint_thread_copy_regions(threads_x, threads_y):
    @T.prim_func
    def before(A: T.Tensor((128,), T.float32), B: T.Tensor((128,), T.float32)):
        with T.Kernel(1, threads=(threads_x, threads_y)):
            tx = T.get_thread_binding(0)
            ty = T.get_thread_binding(1)
            shared = T.alloc_shared((64,), T.float32)
            for k in T.Pipelined(2, num_stages=2):
                shared[tx + threads_x * ty] = A[k * 64 + tx + threads_x * ty]
                shared[tx + threads_x * ty + 32] = A[k * 64 + tx + threads_x * ty + 32]
                B[k * 64 + tx + threads_x * ty] = shared[tx + threads_x * ty]
                B[k * 64 + tx + threads_x * ty + 32] = shared[tx + threads_x * ty + 32]

    mod = tl.transform.MaterializeKernelLaunch(lower_thread_binding=True)(tvm.IRModule.from_expr(before))
    mod = _run_pipeline_planning(mod["before"], sm80_target)
    anno = _collect_pipeline_loop_annotations(mod["main"])[0]
    stages = [int(value) for value in anno["software_pipeline_stage"]]
    assert stages[:2] == [0, 0]
    assert all(stage > 0 for stage in stages[2:])
    tl.transform.InjectSoftwarePipeline()(mod)
    with sm80_target:
        tl.lower(before, target=sm80_target)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("materialize_launch", [False, True])
def test_pipeline_planning_tracks_cross_thread_index_producers(materialize_launch):
    @T.prim_func
    def before(A: T.Tensor((64,), T.float32), B: T.Tensor((64,), T.float32)):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            indices = T.alloc_shared((64,), T.int32)
            data = T.alloc_shared((32,), T.float32)
            for k in T.Pipelined(2, num_stages=2):
                indices[tx] = k * 32
                indices[tx + 32] = k * 32
                offset: T.int32 = indices[tx + 1]
                data[tx] = A[offset + tx]
                B[k * 32 + tx] = data[tx]

    if materialize_launch:
        before = tl.transform.MaterializeKernelLaunch(lower_thread_binding=True)(tvm.IRModule.from_expr(before))["before"]
    mod = _run_pipeline_planning(before, sm80_target)
    anno = _collect_pipeline_loop_annotations(mod["main"])[0]
    assert [int(value) for value in anno["software_pipeline_stage"]] == [0, 0, 0, 0, 1]
    assert [int(value) for value in anno["software_pipeline_order"]] == [0, 1, 2, 3, 4]
    tl.transform.InjectSoftwarePipeline()(mod)


@tilelang.testing.requires_cuda
def test_pipeline_planning_reaches_mixed_dependency_fixed_point():
    """Alternating scalar and buffer edges propagate to arbitrary depth."""

    @T.prim_func
    def before(
        A: T.Tensor((64,), T.float16),
        B: T.Tensor((64,), T.float16),
    ):
        with T.Kernel(1, threads=256):
            index_0 = T.alloc_shared((1,), T.int32)
            index_1 = T.alloc_shared((1,), T.int32)
            A_shared = T.alloc_shared((32,), T.float16)
            for k in T.Pipelined(2, num_stages=2):
                index_0[0] = k * 32
                base: T.int32 = index_0[0]
                index_1[0] = base
                offset: T.int32 = index_1[0]
                T.copy(A[offset], A_shared)
                for i in T.Parallel(32):
                    B[k * 32 + i] = A_shared[i]

    mod = _run_pipeline_planning(before, sm80_target)
    anno = _collect_pipeline_loop_annotations(mod["main"])[0]
    stages = [int(value) for value in anno["software_pipeline_stage"]]
    orders = [int(value) for value in anno["software_pipeline_order"]]
    assert stages == [0, 0, 0, 0, 0, 1]
    assert orders == [0, 1, 2, 3, 4, 5]
    tl.transform.InjectSoftwarePipeline()(mod)


@tilelang.testing.requires_cuda
def test_pipeline_planning_redirects_copy_anchor_through_copy_producer():
    """A copy feeding another copy's index joins the downstream copy group."""

    @T.prim_func
    def before(
        indices: T.Tensor((2,), T.int32),
        A: T.Tensor((64,), T.float16),
        B: T.Tensor((64,), T.float16),
    ):
        with T.Kernel(1, threads=256):
            index_shared = T.alloc_shared((1,), T.int32)
            A_shared = T.alloc_shared((32,), T.float16)
            for k in T.Pipelined(2, num_stages=2):
                T.copy(indices[k], index_shared)
                offset: T.int32 = index_shared[0]
                T.copy(A[offset], A_shared)
                for i in T.Parallel(32):
                    B[k * 32 + i] = A_shared[i]

    mod = _run_pipeline_planning(before, sm80_target)
    anno = _collect_pipeline_loop_annotations(mod["main"])[0]
    stages = [int(value) for value in anno["software_pipeline_stage"]]
    orders = [int(value) for value in anno["software_pipeline_order"]]
    assert stages == [0, 0, 0, 1]
    assert orders == [0, 1, 2, 3]
    async_groups = [int(value) for value in anno["software_pipeline_async_producer_groups"]]
    assert async_groups[0] >= 0
    assert async_groups[2] >= 0
    assert async_groups[0] != async_groups[2]
    tl.transform.InjectSoftwarePipeline()(mod)


@tilelang.testing.requires_cuda
def test_pipeline_planning_resolves_each_multi_root_anchor_before_minimum():
    """A shared producer uses the earliest resolved sink across copy groups."""

    @T.prim_func
    def before(
        indices: T.Tensor((64,), T.int32),
        A: T.Tensor((64,), T.float16),
        C: T.Tensor((64,), T.float16),
        B: T.Tensor((128,), T.float16),
    ):
        with T.Kernel(1, threads=256):
            base_shared = T.alloc_shared((1,), T.int32)
            A_shared = T.alloc_shared((32,), T.float16)
            next_shared = T.alloc_shared((1,), T.int32)
            C_shared = T.alloc_shared((32,), T.float16)
            for k in T.Pipelined(2, num_stages=2):
                base_shared[0] = k * 32
                base: T.int32 = base_shared[0]
                T.copy(A[base], A_shared)
                T.copy(indices[base], next_shared)
                next_base: T.int32 = next_shared[0]
                for i in T.Parallel(32):
                    B[k * 64 + i] = A_shared[i]
                T.copy(C[next_base], C_shared)
                for i in T.Parallel(32):
                    B[k * 64 + 32 + i] = C_shared[i]

    mod = _run_pipeline_planning(before, sm80_target)
    anno = _collect_pipeline_loop_annotations(mod["main"])[0]
    stages = [int(value) for value in anno["software_pipeline_stage"]]
    orders = [int(value) for value in anno["software_pipeline_order"]]
    assert stages == [0, 0, 0, 0, 0, 2, 0, 2]
    assert orders[1] < orders[2]
    assert orders[1] < orders[3]
    tl.transform.InjectSoftwarePipeline()(mod)


@tilelang.testing.requires_cuda
def test_pipeline_planning_skips_transitive_tma_pipeline():
    """Sequential lowering provides the wait before the dependent copy."""
    N, C, H, W, KH = 1, 32, 8, 8, 3
    stride, dilation, padding = 1, 1, 1

    @T.prim_func
    def before(
        data: T.Tensor((N, H, W, C), T.float16),
        A: T.Tensor((64,), T.float16),
        B: T.Tensor((64,), T.float16),
    ):
        with T.Kernel(1, threads=128):
            data_shared = T.alloc_shared((16, 32), T.float16)
            A_shared = T.alloc_shared((32,), T.float16)
            for k in T.Pipelined(2, num_stages=2):
                T.im2col(
                    data,
                    data_shared,
                    0,
                    k,
                    KH,
                    stride,
                    dilation,
                    padding,
                )
                offset: T.int32 = T.cast(data_shared[0, 0], "int32")
                T.copy(A[offset], A_shared)
                for i in T.Parallel(32):
                    B[k * 32 + i] = A_shared[i]

    mod = _run_pipeline_planning(before, sm90_target)
    assert not _collect_pipeline_loop_annotations(mod["main"])
    loops = collect_nodes(mod["main"], tvm.tirx.For)
    assert all("num_stages" not in loop.annotations for loop in loops)
    tl.transform.InjectSoftwarePipeline()(mod)


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


@tilelang.testing.requires_cuda
def test_pipeline_planning_waits_before_dependent_copy():
    @T.prim_func
    def before(Ids: T.Tensor((2,), T.int32), A: T.Tensor((64,), T.float32), B: T.Tensor((64,), T.float32)):
        with T.Kernel(1, threads=32):
            index = T.alloc_shared((1,), T.int32)
            shared = T.alloc_shared((32,), T.float32)
            for k in T.Pipelined(2, num_stages=2):
                T.copy(Ids[k], index)
                T.copy(A[index[0]], shared)
                for i in T.Parallel(32):
                    B[k * 32 + i] = shared[i]

    mod = _run_pipeline_planning(before, sm80_target)
    anno = _collect_pipeline_loop_annotations(mod["main"])[0]
    assert [int(v) for v in anno["software_pipeline_async_producers"]] == [1, 0, 0]
    tl.transform.InjectSoftwarePipeline()(mod)
