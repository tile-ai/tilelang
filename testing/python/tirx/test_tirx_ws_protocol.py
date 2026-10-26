from tilelang.transform import _tirx_passes
import pytest
from tilelang._compat import ir
from tvm_tirx.target import Target

import tilelang
from tilelang import language as T
from tilelang._compat import stmt_functor


pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def nodes(value, kind):
    result = []
    stmt_functor.post_order_visit(value, lambda node: result.append(node) if isinstance(node, kind) else None)
    return result


def prepare(function):
    module = ir.IRModule({"main": function.with_attr("global_symbol", "main")})
    module = _tirx_passes.BindTarget(Target({"kind": "cuda", "arch": "sm_100a"}))(module)
    return tilelang.transform.MaterializeKernelLaunch()(module)


def materialize(function):
    module = prepare(function)
    with ir.transform.PassContext(config={"tl.enable_auto_warp_specialization": "role_based"}):
        module = tilelang.cuda.transform.AutoWarpSpecialization()(module)
    assert any("tl.ws_schedule" in node.attrs for node in nodes(module, ir.RegionStmt))
    module = tilelang.cuda.transform.MaterializeWSSchedule()(module)
    assert ir.check_well_formed(module)
    ir.assert_structural_equal(module, ir.load_json(ir.save_json(module)))
    return module


def persistent_copy(trips=3, thread_dependent=False):
    @T.prim_func
    def kernel(source: T.Tensor((4, 64, 64), T.float16), output: T.Tensor((4, 64, 64), T.float16)):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((64, 64), T.float16)
            fragment = T.alloc_fragment((64, 64), T.float16)
            wave = T.alloc_local((1,), T.int32)
            wave[0] = 0
            while wave[0] < trips:
                T.copy(source[wave[0], 0, 0], shared)
                T.copy(shared, fragment)
                T.copy(fragment, output[wave[0], 0, 0])
                if thread_dependent:
                    wave[0] = wave[0] + 1 + T.get_thread_binding()
                else:
                    wave[0] = wave[0] + 1

    return kernel


def mixed_raw_copy():
    @T.prim_func
    def kernel(source: T.Tensor((4, 128), T.float32), raw: T.Tensor((4, 128), T.float32), output: T.Tensor((4, 128), T.float32)):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((128,), T.float32)
            raw_shared = T.alloc_shared((128,), T.float32)
            thread = T.get_thread_binding()
            for step in T.Pipelined(4, num_stages=2):
                T.ptx_cp_async(T.access_ptr(raw_shared[thread], "w", 1), T.access_ptr(raw[step, thread], "r", 1), 1)
                T.copy(source[step, 0], shared)
                for index in T.Parallel(128):
                    output[step, index] = raw_shared[index] + shared[index]

    return kernel


def test_raw_copy_is_committed_and_waited_by_its_automatic_pipeline():
    module = materialize(mixed_raw_copy())
    calls = nodes(module, ir.Call)
    assert any(call.op.name == "tl.ptx_cp_async" for call in calls)
    assert any(call.op.name == "tirx.ptx.cp_async_commit_group" for call in calls)
    assert any(call.op.name == "tl.ptx_cp_async_barrier_noinc" for call in calls)
    assert any(call.op.name == "tl.mbarrier_wait_parity" for call in calls)


@pytest.mark.gpu
def test_mixed_raw_copy_cuda_numerics():
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Requires CUDA with TMA support")
    kernel = tilelang.compile(mixed_raw_copy(), target="cuda", out_idx=[2])
    source = torch.arange(512, device="cuda", dtype=torch.float32).reshape(4, 128)
    raw = source * 2
    torch.testing.assert_close(kernel(source, raw), source + raw, rtol=0, atol=0)


def test_pipeline_depth_annotations_attach_to_their_own_loop():
    @T.prim_func
    def kernel(source: T.Tensor((2, 2, 64, 64), T.float16), output: T.Tensor((2, 2, 64, 64), T.float16)):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((64, 64), T.float16)
            fragment = T.alloc_fragment((64, 64), T.float16)
            for wave in T.Pipelined(2, num_stages=1):
                T.annotate_ws_pipeline_depth({shared: 3})
                for step in T.Pipelined(2, num_stages=2):
                    T.annotate_ws_pipeline_depth({shared: 4})
                    T.copy(source[wave, step, 0, 0], shared)
                    T.copy(shared, fragment)
                    T.copy(fragment, output[wave, step, 0, 0])

    module = prepare(kernel)
    loops = nodes(module, ir.For)
    assert sorted(int(next(iter(loop.annotations["tl.ws_pipeline_depth"].values()))) for loop in loops) == [3, 4]
    assert not any("tl.ws_pipeline_depth" in region.attrs for region in nodes(module, ir.RegionStmt))


def test_while_roles_share_one_control_state_protocol():
    module = materialize(persistent_copy())
    loops = nodes(module, ir.While)
    assert len(loops) == 2
    for loop in loops:
        condition = loop.condition
        assert isinstance(condition, ir.prim.LT)
        state = condition.a.source
        stores = [node for node in nodes(loop, ir.TensorStore) if node.dest.same_as(state)]
        assert len(stores) == 1
        assert int(stores[0].value.b) == 1
    barriers = [node for node in nodes(module, ir.Bind) if isinstance(node.value, ir.Call) and "barrier_init" in (node.value.attrs or {})]
    assert len(barriers) == 1
    assert [int(value) for value in barriers[0].value.attrs["barrier_init"]] == [32, 128]


def test_while_rejects_thread_dependent_control_updates():
    with pytest.raises(ValueError, match="role-uniform"):
        materialize(persistent_copy(thread_dependent=True))


@pytest.mark.gpu
@pytest.mark.parametrize("trips", [0, 1, 3, 4])
def test_automatic_persistent_copy_cuda_numerics(trips):
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Requires CUDA with TMA support")
    kernel = tilelang.compile(persistent_copy(trips), target="cuda", pass_configs={"tl.enable_auto_warp_specialization": "role_based"})
    source = torch.randn((4, 64, 64), dtype=torch.float16, device="cuda")
    output = torch.full_like(source, -2)
    kernel(source, output)
    torch.cuda.synchronize()
    expected = torch.full_like(source, -2)
    expected[:trips].copy_(source[:trips])
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def manual_staged_while(trips, depth=2, thread_dependent=False):
    @T.prim_func
    def kernel(source: T.Tensor((2, 5, 64, 64), T.float16), output: T.Tensor((2, 5, 64, 64), T.float16)):
        with T.Kernel(1, threads=128):
            shared_a = T.alloc_shared((64, 64), T.float16)
            shared_b = T.alloc_shared((64, 64), T.float16)
            fragment_a = T.alloc_fragment((64, 64), T.float16)
            fragment_b = T.alloc_fragment((64, 64), T.float16)
            wave = T.alloc_local((1,), T.int32)
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[T.WSRole("Load", warps_lo=0, warps_hi=1, max_nreg=40), T.WSRole("Worker", warps_lo=4, warps_hi=8, max_nreg=224)],
                    pipelines=[T.WSPipeline("a", [shared_a], depth=depth), T.WSPipeline("b", [shared_b], depth=depth)],
                    scopes=[
                        T.WSScope(T.WSScope.ROOT, {"Load": ["init", "wave"], "Worker": ["init", "wave"]}),
                        T.WSScope(
                            "wave",
                            {
                                "Load": [
                                    T.WSSync.producer_acquire("a"),
                                    "load_a",
                                    T.WSSync.producer_commit("a"),
                                    T.WSSync.producer_acquire("b"),
                                    "load_b",
                                    T.WSSync.producer_commit("b"),
                                    "next",
                                ],
                                "Worker": [
                                    T.WSSync.consumer_wait("a"),
                                    "read_a",
                                    T.WSSync.consumer_release("a"),
                                    "store_a",
                                    T.WSSync.consumer_wait("b", stage=1),
                                    "read_b",
                                    T.WSSync.consumer_release("b", stage=1),
                                    "store_b",
                                    "next",
                                ],
                            },
                        ),
                    ],
                )
            )
            with T.ws_op("init"):
                wave[0] = 0
            with T.ws_op("wave"):
                while wave[0] < trips:
                    T.copy(source[0, wave[0], 0, 0], shared_a, annotations={T.WSID: "load_a"})
                    T.copy(source[1, wave[0], 0, 0], shared_b, annotations={T.WSID: "load_b"})
                    T.copy(shared_a, fragment_a, annotations={T.WSID: "read_a"})
                    T.copy(shared_b, fragment_b, annotations={T.WSID: "read_b"})
                    T.copy(fragment_a, output[0, wave[0], 0, 0], annotations={T.WSID: "store_a"})
                    T.copy(fragment_b, output[1, wave[0], 0, 0], annotations={T.WSID: "store_b"})
                    with T.ws_op("next"):
                        if thread_dependent:
                            wave[0] = wave[0] + 1 + T.get_thread_binding()
                        else:
                            wave[0] = wave[0] + 1

    return kernel


def manual(function):
    module = tilelang.cuda.transform.MaterializeWSSchedule()(prepare(function))
    assert ir.check_well_formed(module)
    ir.assert_structural_equal(module, ir.load_json(ir.save_json(module)))
    return module


def test_staged_while_preserves_local_control_and_native_binding_scopes():
    module = manual(manual_staged_while(4))
    assert len(nodes(module, ir.While)) == 2
    delayed = [binding for binding in nodes(module, ir.Bind) if binding.var.name.startswith("ws_state_wave_Worker_1_")]
    assert len(delayed) == 1
    assert delayed[0].var.ty.scope.name == "local"


def test_staged_while_rejects_role_dependent_control():
    with pytest.raises(ValueError, match="role-uniform"):
        manual(manual_staged_while(4, thread_dependent=True))


@pytest.mark.gpu
@pytest.mark.parametrize("depth", [2, 3])
@pytest.mark.parametrize("trips", [0, 1, 4])
def test_staged_while_cuda_numerics(depth, trips):
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Requires CUDA with TMA support")
    kernel = tilelang.compile(manual_staged_while(trips, depth), target="cuda")
    source = torch.randn((2, 5, 64, 64), dtype=torch.float16, device="cuda")
    output = torch.full_like(source, -2)
    kernel(source, output)
    torch.cuda.synchronize()
    expected = torch.full_like(source, -2)
    expected[:, :trips].copy_(source[:, :trips])
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def alias_copy(use_view):
    @T.prim_func
    def kernel(source: T.Tensor((3, 64, 64), T.float16), output: T.Tensor((3, 32, 128), T.float16)):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((64, 64), T.float16)
            alias = T.view(shared, (32, 128)) if use_view else T.reshape(shared, (32, 128))
            fragment = T.alloc_fragment((32, 128), T.float16)
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[T.WSRole("Load", warps_lo=0, warps_hi=1, max_nreg=40), T.WSRole("Worker", warps_lo=4, warps_hi=8, max_nreg=224)],
                    pipelines=[T.WSPipeline("data", [shared], depth=2)],
                    scopes=[
                        T.WSScope(T.WSScope.ROOT, {"Load": ["wave"], "Worker": ["wave"]}),
                        T.WSScope(
                            "wave",
                            {
                                "Load": [T.WSSync.producer_acquire("data"), "load", T.WSSync.producer_commit("data")],
                                "Worker": [T.WSSync.consumer_wait("data"), "read", T.WSSync.consumer_release("data"), "store"],
                            },
                        ),
                    ],
                )
            )
            for wave in T.serial(3, annotations={T.WSID: "wave"}):
                T.copy(source[wave, 0, 0], shared, annotations={T.WSID: "load"})
                T.copy(alias, fragment, annotations={T.WSID: "read"})
                T.copy(fragment, output[wave, 0, 0], annotations={T.WSID: "store"})

    return kernel


@pytest.mark.gpu
@pytest.mark.parametrize("use_view", [False, True])
def test_versioned_shared_alias_cuda_numerics(use_view):
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Requires CUDA with TMA support")
    kernel = tilelang.compile(alias_copy(use_view), target="cuda")
    source = torch.randn((3, 64, 64), dtype=torch.float16, device="cuda")
    output = torch.full((3, 32, 128), -2, dtype=torch.float16, device="cuda")
    kernel(source, output)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, source.reshape_as(output), rtol=0, atol=0)


def manual_staged_scalar(trips):
    @T.prim_func
    def kernel(
        source: T.Tensor((2, 5, 64, 64), T.float16),
        choices: T.Tensor((5,), T.int32),
        output: T.Tensor((2, 5, 64, 64), T.float16),
    ):
        with T.Kernel(1, threads=128):
            shared_a = T.alloc_shared((64, 64), T.float16)
            shared_b = T.alloc_shared((64, 64), T.float16)
            fragment_a = T.alloc_fragment((64, 64), T.float16)
            fragment_b = T.alloc_fragment((64, 64), T.float16)
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[T.WSRole("Load", warps_lo=0, warps_hi=1, max_nreg=40), T.WSRole("Worker", warps_lo=4, warps_hi=8, max_nreg=224)],
                    pipelines=[T.WSPipeline("a", [shared_a], depth=2), T.WSPipeline("b", [shared_b], depth=2)],
                    scopes=[
                        T.WSScope(T.WSScope.ROOT, {"Load": ["wave"], "Worker": ["wave"]}),
                        T.WSScope(
                            "wave",
                            {
                                "Load": [
                                    "select",
                                    T.WSSync.producer_acquire("a"),
                                    "load_a",
                                    T.WSSync.producer_commit("a"),
                                    T.WSSync.producer_acquire("b", stage=1),
                                    "load_b",
                                    T.WSSync.producer_commit("b", stage=1),
                                ],
                                "Worker": [
                                    T.WSSync.consumer_wait("a"),
                                    "read_a",
                                    T.WSSync.consumer_release("a"),
                                    "store_a",
                                    T.WSSync.consumer_wait("b"),
                                    "read_b",
                                    T.WSSync.consumer_release("b"),
                                    "store_b",
                                ],
                            },
                        ),
                    ],
                )
            )
            for wave in T.serial(trips, annotations={T.WSID: "wave"}):
                with T.ws_op("select"):
                    choice = choices[wave]
                T.copy(source[0, choice, 0, 0], shared_a, annotations={T.WSID: "load_a"})
                T.copy(source[1, wave, 0, 0], shared_b, annotations={T.WSID: "load_b"})
                T.copy(shared_a, fragment_a, annotations={T.WSID: "read_a"})
                T.copy(shared_b, fragment_b, annotations={T.WSID: "read_b"})
                T.copy(fragment_a, output[0, wave, 0, 0], annotations={T.WSID: "store_a"})
                T.copy(fragment_b, output[1, wave, 0, 0], annotations={T.WSID: "store_b"})

    return kernel


def test_simplify_keeps_schedule_references_until_materialization():
    module = tilelang.transform.Simplify()(prepare(manual_staged_scalar(0)))
    assert any(loop.annotations.get(T.WSID) == "wave" for loop in nodes(module, ir.For))
    module = tilelang.cuda.transform.MaterializeWSSchedule()(module)
    module = tilelang.transform.Simplify()(module)
    assert ir.check_well_formed(module)
    assert not any(call.op.name == "tl.tileop.copy" for call in nodes(module, ir.Call))


@pytest.mark.gpu
@pytest.mark.parametrize("trips", [0, 1, 4])
def test_staged_scalar_binding_cuda_numerics(trips):
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Requires CUDA with TMA support")
    kernel = tilelang.compile(manual_staged_scalar(trips), target="cuda")
    source = torch.randn((2, 5, 64, 64), dtype=torch.float16, device="cuda")
    choices = torch.tensor([3, 0, 4, 1, 2], dtype=torch.int32, device="cuda")
    output = torch.full_like(source, -2)
    kernel(source, choices, output)
    torch.cuda.synchronize()
    expected = torch.full_like(source, -2)
    expected[0, :trips].copy_(source[0, choices[:trips].long()])
    expected[1, :trips].copy_(source[1, :trips])
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
