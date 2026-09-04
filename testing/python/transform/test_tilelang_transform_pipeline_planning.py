from tilelang import tvm as tvm
import pytest
import tilelang as tl
import tilelang.testing
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


def _conditional_copy_gemm_kernel(num_stages):
    """Pipelined GEMM whose copy+gemm body is guarded by a runtime condition.

    Before the fix, PipelineStageAnalyzer rejected conditionally executed copy
    stages from async, so the copies stayed synchronous. The copies should be
    eligible for cp.async: commit is emitted outside the user branch, so group
    counting stays consistent for skipped iterations.
    """
    M = N = K = 256
    BM, BN, BK = 64, 64, 32

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float16),
        limit: T.int32,
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
            a_sh = T.alloc_shared((BM, BK), T.float16)
            b_sh = T.alloc_shared((BK, BN), T.float16)
            c_loc = T.alloc_fragment((BM, BN), T.float32)
            T.clear(c_loc)
            for k in T.Pipelined(T.ceildiv(K, BK), num_stages=num_stages):
                if k < limit:
                    T.copy(A[by * BM : (by + 1) * BM, k * BK : (k + 1) * BK], a_sh)
                    T.copy(B[k * BK : (k + 1) * BK, bx * BN : (bx + 1) * BN], b_sh)
                    T.gemm(a_sh, b_sh, c_loc)
            T.copy(c_loc, C[by * BM : (by + 1) * BM, bx * BN : (bx + 1) * BN])

    return kernel


def _line_after_matching_brace(src, needle):
    """Return the first statement line after the closing brace of the if block
    that starts with `needle`."""
    start = src.index(needle)
    depth = 0
    i = src.index("{", start)
    while True:
        ch = src[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    for line in src[i + 1 :].splitlines():
        line = line.strip()
        if line and not line.startswith("#line"):
            return line
    raise AssertionError("no statement found after " + needle)


@pytest.mark.parametrize("num_stages", [2, 3])
def test_pipeline_planning_conditional_copy_emits_cp_async(num_stages):
    from tilelang.tools.compile_only import compile_kernel_source

    kernel = _conditional_copy_gemm_kernel(num_stages)
    src = compile_kernel_source(kernel, {"kind": "cuda", "arch": "sm_80"})

    # The conditional copy stage must lower to cp.async instead of falling
    # back to a synchronous vector copy.
    assert "tl::cp_async" in src
    assert "tl::cp_async_commit();" in src
    # Steady-state wait depth and the final drain.
    assert f"tl::cp_async_wait<{num_stages - 1}>();" in src
    assert "tl::cp_async_wait<0>();" in src

    # Commit must be emitted outside the producer branch: the statement right
    # after the steady-state `if ((k + n) < limit) { ... }` block is the
    # commit, so every iteration commits exactly one (possibly empty) group
    # and wait<N> group counting stays consistent for skipped iterations.
    producer_if = f"if ((k + {num_stages - 1}) < limit)"
    next_stmt = _line_after_matching_brace(src, producer_if)
    assert next_stmt == "tl::cp_async_commit();", next_stmt


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
def test_pipeline_planning_conditional_copy_cp_async_numerics():
    """Device-side numerics for the conditional async pipeline.

    Sweeps the runtime limit so skipped iterations exercise the empty-commit
    path, on the cp.async pipeline (warp specialization disabled so Hopper
    does not reroute the copies through WS/TMA, which is classified
    independently of this change).
    """
    import torch

    M = N = K = 256
    BK = 32
    num_iters = K // BK

    kernel = tilelang.compile(
        _conditional_copy_gemm_kernel(num_stages=2),
        out_idx=[2],
        pass_configs={tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )

    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    for limit in (0, 1, 3, num_iters, num_iters + 2):
        c = kernel(a, b, limit)
        ref = torch.zeros(M, N, device="cuda", dtype=torch.float32)
        for k in range(min(limit, num_iters)):
            ref += a[:, k * BK : (k + 1) * BK].float() @ b[k * BK : (k + 1) * BK, :].float()
        torch.testing.assert_close(c.float(), ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    tilelang.testing.main()
