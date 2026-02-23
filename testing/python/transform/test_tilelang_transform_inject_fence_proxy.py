# ruff: noqa
from tilelang import tvm as tvm
import tilelang as tl
from tilelang.utils.target import determine_target
import tilelang.language as T
import tilelang.testing
from tvm import tir

auto_target = tvm.target.Target(determine_target("auto"))


def _check(original, transformed):
    func = original
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(auto_target)(mod)
    mod = tl.transform.InjectFenceProxy()(mod)
    mod = tir.transform.LowerOpaqueBlock()(mod)
    transformed = tvm.IRModule.from_expr(transformed.with_attr("global_symbol", "main"))
    transformed = tvm.tir.transform.BindTarget(auto_target)(transformed)
    transformed = tir.transform.LowerOpaqueBlock()(transformed)

    tvm.ir.assert_structural_equal(mod["main"], transformed["main"], True)


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_lower_fence_proxy():
    @T.prim_func
    def before():
        with T.Kernel(8):
            A_shared = T.decl_buffer((1, 8, 256), T.float16, scope="shared.dyn")
            B_shared = T.decl_buffer((1, 4, 512), T.float16, scope="shared.dyn")
            C_local = T.decl_buffer((32,), scope="local")
            for i in T.unroll(16):
                C_local[i * 2 : i * 2 + 2] = T.Broadcast(T.float32(0), 2)
            # A shared-memory generic store should trigger a fence before the
            # following async-proxy GEMM on Hopper (SM90+).
            A_shared[0, 0, 0] = T.float16(0)
            T.call_intrin(
                "handle",
                tir.op.Op.get("tl.tl_gemm"),
                "tl::gemm_ss<128, 128, 32, 4, 1, 0, 0, 0, 32, 128, 0, 0, true>",
                T.tvm_access_ptr(T.type_annotation(T.float16), A_shared.data, 0, 2048, 1),
                T.tvm_access_ptr(T.type_annotation(T.float16), B_shared.data, 0, 2048, 1),
                T.tvm_access_ptr(T.type_annotation(T.float32), C_local.data, 0, 32, 3),
            )

    @T.prim_func
    def after():
        with T.Kernel(8):
            A_shared = T.decl_buffer((1, 8, 256), T.float16, scope="shared.dyn")
            B_shared = T.decl_buffer((1, 4, 512), T.float16, scope="shared.dyn")
            C_local = T.decl_buffer((32,), scope="local")
            for i in T.unroll(16):
                C_local[i * 2 : i * 2 + 2] = T.Broadcast(T.float32(0), 2)
            A_shared[0, 0, 0] = T.float16(0)
            T.fence_proxy_async()
            T.call_intrin(
                "handle",
                tir.op.Op.get("tl.tl_gemm"),
                "tl::gemm_ss<128, 128, 32, 4, 1, 0, 0, 0, 32, 128, 0, 0, true>",
                T.tvm_access_ptr(T.type_annotation(T.float16), A_shared.data, 0, 2048, 1),
                T.tvm_access_ptr(T.type_annotation(T.float16), B_shared.data, 0, 2048, 1),
                T.tvm_access_ptr(T.type_annotation(T.float32), C_local.data, 0, 32, 3),
            )

    _check(before, after)


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_async_to_generic_no_double_fence():
    @T.prim_func
    def before():
        with T.Kernel(8):
            A_shared = T.decl_buffer((1024,), T.uint8, scope="shared.dyn")
            B_shared = T.decl_buffer((1024,), T.uint8, scope="shared.dyn")
            T.ptx_cp_async(
                T.tvm_access_ptr(T.type_annotation(T.uint8), A_shared.data, 0, 16, 2),
                T.tvm_access_ptr(T.type_annotation(T.uint8), B_shared.data, 0, 16, 1),
                16,
            )
            T.fence_proxy_async()
            T.call_extern("handle", "generic_op")

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(auto_target)(mod)
    mod = tl.transform.InjectFenceProxy()(mod)

    def _count_fences(stmt):
        count = 0

        def visit(node):
            nonlocal count
            if isinstance(node, tir.Evaluate):
                call = node.value
                if isinstance(call, tir.Call):
                    op = call.op
                    name = getattr(op, "name", None)
                    if name == "tl.fence_proxy_async":
                        count += 1

        tir.stmt_functor.post_order_visit(stmt, visit)
        return count

    assert _count_fences(mod["main"].body) == 1


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_cp_async_then_wgmma_injects_fence_proxy():
    """cp.async is treated as generic proxy traffic for fence injection."""

    @T.prim_func
    def before():
        with T.Kernel(1):
            A_shared = T.decl_buffer((1024,), T.uint8, scope="shared.dyn")
            B_global = T.decl_buffer((1024,), T.uint8, scope="global")
            desc_a = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            desc_b = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            C_local = T.decl_buffer((32,), T.float16, scope="local")
            T.ptx_cp_async(
                T.tvm_access_ptr(T.type_annotation(T.uint8), A_shared.data, 0, 16, 2),
                T.tvm_access_ptr(T.type_annotation(T.uint8), B_global.data, 0, 16, 1),
                16,
            )
            T.warpgroup_arrive()
            T.ptx_wgmma_ss(
                T.float16,
                "m64n64k16",
                T.bool(True),
                T.bool(True),
                "fp16",
                "fp16",
                "fp16",
                desc_a.data,
                T.int32(0),
                desc_b.data,
                T.int32(0),
                C_local.data,
                T.int32(0),
                T.bool(True),
                1,
                1,
            )

    @T.prim_func
    def after():
        with T.Kernel(1):
            A_shared = T.decl_buffer((1024,), T.uint8, scope="shared.dyn")
            B_global = T.decl_buffer((1024,), T.uint8, scope="global")
            desc_a = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            desc_b = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            C_local = T.decl_buffer((32,), T.float16, scope="local")
            T.ptx_cp_async(
                T.tvm_access_ptr(T.type_annotation(T.uint8), A_shared.data, 0, 16, 2),
                T.tvm_access_ptr(T.type_annotation(T.uint8), B_global.data, 0, 16, 1),
                16,
            )
            T.warpgroup_arrive()
            T.fence_proxy_async()
            T.ptx_wgmma_ss(
                T.float16,
                "m64n64k16",
                T.bool(True),
                T.bool(True),
                "fp16",
                "fp16",
                "fp16",
                desc_a.data,
                T.int32(0),
                desc_b.data,
                T.int32(0),
                C_local.data,
                T.int32(0),
                T.bool(True),
                1,
                1,
            )

    _check(before, after)


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_unknown_extern_default_is_none():
    @T.prim_func
    def before():
        with T.Kernel(1):
            smem = T.decl_buffer((1,), T.float16, scope="shared")
            smem[0] = T.float16(0)
            T.evaluate(T.call_extern("handle", "custom_op"))

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(auto_target)(mod)
    mod = tl.transform.InjectFenceProxy()(mod)

    def _count_fences(stmt):
        count = 0

        def visit(node):
            nonlocal count
            if isinstance(node, tir.Evaluate):
                call = node.value
                if isinstance(call, tir.Call):
                    name = getattr(call.op, "name", None)
                    if name == "tl.fence_proxy_async":
                        count += 1

        tir.stmt_functor.post_order_visit(stmt, visit)
        return count

    assert _count_fences(mod["main"].body) == 0


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_unknown_extern_shared_store_then_wgmma_injects_fence_proxy():
    """Opaque calls that may write shared memory must be treated as generic."""

    @T.prim_func
    def before():
        with T.Kernel(1):
            smem = T.decl_buffer((256,), T.float16, scope="shared")
            desc_a = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            desc_b = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            C_local = T.decl_buffer((32,), T.float16, scope="local")
            T.evaluate(
                T.call_extern(
                    "handle",
                    "custom_smem_store",
                    T.tvm_access_ptr(T.type_annotation(T.float16), smem.data, 0, 16, 2),
                )
            )
            T.warpgroup_arrive()
            T.ptx_wgmma_ss(
                T.float16,
                "m64n64k16",
                T.bool(True),
                T.bool(True),
                "fp16",
                "fp16",
                "fp16",
                desc_a.data,
                T.int32(0),
                desc_b.data,
                T.int32(0),
                C_local.data,
                T.int32(0),
                T.bool(True),
                1,
                1,
            )

    @T.prim_func
    def after():
        with T.Kernel(1):
            smem = T.decl_buffer((256,), T.float16, scope="shared")
            desc_a = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            desc_b = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            C_local = T.decl_buffer((32,), T.float16, scope="local")
            T.evaluate(
                T.call_extern(
                    "handle",
                    "custom_smem_store",
                    T.tvm_access_ptr(T.type_annotation(T.float16), smem.data, 0, 16, 2),
                )
            )
            T.warpgroup_arrive()
            T.fence_proxy_async()
            T.ptx_wgmma_ss(
                T.float16,
                "m64n64k16",
                T.bool(True),
                T.bool(True),
                "fp16",
                "fp16",
                "fp16",
                desc_a.data,
                T.int32(0),
                desc_b.data,
                T.int32(0),
                C_local.data,
                T.int32(0),
                T.bool(True),
                1,
                1,
            )

    _check(before, after)


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_unknown_extern_address_of_shared_then_wgmma_injects_fence_proxy():
    @T.prim_func
    def before():
        with T.Kernel(1):
            smem = T.decl_buffer((256,), T.float16, scope="shared")
            desc_a = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            desc_b = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            C_local = T.decl_buffer((32,), T.float16, scope="local")
            T.evaluate(T.call_extern("handle", "custom_ptr_store", T.address_of(smem[0])))
            T.warpgroup_arrive()
            T.ptx_wgmma_ss(
                T.float16,
                "m64n64k16",
                T.bool(True),
                T.bool(True),
                "fp16",
                "fp16",
                "fp16",
                desc_a.data,
                T.int32(0),
                desc_b.data,
                T.int32(0),
                C_local.data,
                T.int32(0),
                T.bool(True),
                1,
                1,
            )

    @T.prim_func
    def after():
        with T.Kernel(1):
            smem = T.decl_buffer((256,), T.float16, scope="shared")
            desc_a = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            desc_b = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            C_local = T.decl_buffer((32,), T.float16, scope="local")
            T.evaluate(T.call_extern("handle", "custom_ptr_store", T.address_of(smem[0])))
            T.warpgroup_arrive()
            T.fence_proxy_async()
            T.ptx_wgmma_ss(
                T.float16,
                "m64n64k16",
                T.bool(True),
                T.bool(True),
                "fp16",
                "fp16",
                "fp16",
                desc_a.data,
                T.int32(0),
                desc_b.data,
                T.int32(0),
                C_local.data,
                T.int32(0),
                T.bool(True),
                1,
                1,
            )

    _check(before, after)


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_inject_fence_proxy_does_not_inject_tma_store_sync():
    @T.prim_func
    def before():
        with T.Kernel(8):
            A_global = T.decl_buffer((128,), T.float16, scope="global")
            T.evaluate(T.call_intrin("handle", tir.op.Op.get("tl.tma_store"), A_global.data))

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(auto_target)(mod)
    mod = tl.transform.InjectFenceProxy()(mod)

    arrives = 0
    waits = 0

    def visit(node):
        nonlocal arrives, waits
        if isinstance(node, tir.Evaluate):
            call = node.value
            if isinstance(call, tir.Call):
                name = getattr(call.op, "name", None)
                if name == "tl.tma_store_arrive":
                    arrives += 1
                elif name in ("tl.tma_store_wait", "tl.tma_store_wait<0>"):
                    waits += 1

    tir.stmt_functor.post_order_visit(mod["main"].body, visit)
    assert arrives == 0
    assert waits == 0


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_wgmma_marked_async():
    @T.prim_func
    def before():
        with T.Kernel(1):
            A_shared = T.decl_buffer((1,), T.float16, scope="shared")
            desc_a = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            desc_b = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            C_local = T.decl_buffer((32,), T.float16, scope="local")
            A_shared[0] = T.float16(0)
            T.warpgroup_arrive()
            T.ptx_wgmma_ss(
                T.float16,
                "m64n64k16",
                T.bool(True),
                T.bool(True),
                "fp16",
                "fp16",
                "fp16",
                desc_a.data,
                T.int32(0),
                desc_b.data,
                T.int32(0),
                C_local.data,
                T.int32(0),
                T.bool(True),
                1,
                1,
            )

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(auto_target)(mod)
    mod = tl.transform.InjectFenceProxy()(mod)
    order = []

    def visit(node):
        if isinstance(node, tir.Evaluate):
            call = node.value
            if isinstance(call, tir.Call):
                order.append(getattr(call.op, "name", ""))

    tir.stmt_functor.post_order_visit(mod["main"].body, visit)

    assert "tl.ptx_wgmma_ss" in order
    assert "tl.fence_proxy_async" in order
    assert order.index("tl.fence_proxy_async") < order.index("tl.ptx_wgmma_ss")


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_shared_barrier_ops_do_not_trigger_fence_proxy():
    @T.prim_func
    def before(A_desc: T.handle("uint8x128", "grid_constant")):
        with T.Kernel(1):
            smem = T.decl_buffer((256,), T.uint8, scope="shared.dyn")
            mbarrier = T.decl_buffer((1,), T.uint64, scope="shared.barrier")
            desc_a = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            desc_b = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            C_local = T.decl_buffer((32,), T.float16, scope="local")

            # Local stores should not be treated as generic proxy traffic.
            C_local[0] = T.float16(0)

            # Descriptor initialization is metadata only and should not be
            # treated as generic proxy traffic.
            T.initialize_wgmma_descriptor(desc_a, T.uint64(0), 2, 1, 32)
            T.initialize_wgmma_descriptor(desc_b, T.uint64(0), 2, 1, 32)

            if T.shuffle_elect(0):
                T.ptx_init_barrier_thread_count(mbarrier[0], 128)
            T.ptx_fence_barrier_init()
            T.tvm_storage_sync("shared")

            # Barrier ops should not be classified as generic proxy traffic.
            T.mbarrier_wait_parity(mbarrier[0], 0)
            T.mbarrier_expect_tx(mbarrier[0], 16)
            T.tma_load(
                A_desc,
                mbarrier[0],
                T.tvm_access_ptr(T.type_annotation(T.uint8), smem.data, 0, 16, 2),
                0,
                0,
                0,
            )

            # Another async proxy op after barrier ops.
            T.warpgroup_arrive()
            T.ptx_wgmma_ss(
                T.float16,
                "m64n64k16",
                T.bool(True),
                T.bool(True),
                "fp16",
                "fp16",
                "fp16",
                desc_a.data,
                T.int32(0),
                desc_b.data,
                T.int32(0),
                C_local.data,
                T.int32(0),
                T.bool(True),
                1,
                1,
            )

    @T.prim_func
    def after(A_desc: T.handle("uint8x128", "grid_constant")):
        with T.Kernel(1):
            smem = T.decl_buffer((256,), T.uint8, scope="shared.dyn")
            mbarrier = T.decl_buffer((1,), T.uint64, scope="shared.barrier")
            desc_a = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            desc_b = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            C_local = T.decl_buffer((32,), T.float16, scope="local")

            C_local[0] = T.float16(0)

            T.initialize_wgmma_descriptor(desc_a, T.uint64(0), 2, 1, 32)
            T.initialize_wgmma_descriptor(desc_b, T.uint64(0), 2, 1, 32)

            if T.shuffle_elect(0):
                T.ptx_init_barrier_thread_count(mbarrier[0], 128)
            T.ptx_fence_barrier_init()
            T.tvm_storage_sync("shared")

            T.mbarrier_wait_parity(mbarrier[0], 0)
            T.mbarrier_expect_tx(mbarrier[0], 16)
            T.tma_load(
                A_desc,
                mbarrier[0],
                T.tvm_access_ptr(T.type_annotation(T.uint8), smem.data, 0, 16, 2),
                0,
                0,
                0,
            )
            T.warpgroup_arrive()
            T.ptx_wgmma_ss(
                T.float16,
                "m64n64k16",
                T.bool(True),
                T.bool(True),
                "fp16",
                "fp16",
                "fp16",
                desc_a.data,
                T.int32(0),
                desc_b.data,
                T.int32(0),
                C_local.data,
                T.int32(0),
                T.bool(True),
                1,
                1,
            )

    _check(before, after)


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_regression_0219_fence_no_fence_inserted():
    """Regression test copied from `debug/0219_fence/fence.py`.

    This kernel mixes:
    - shared.barrier initialization + sync
    - producer TMA loads (async proxy)
    - consumer WGMMA (async proxy)

    There is no generic shared-memory traffic that should force a
    generic->async proxy switch, so InjectFenceProxy must be a no-op.
    """

    @T.prim_func
    def before(
        A_desc: T.handle("uint8x128", "grid_constant"),
        B_desc: T.handle("uint8x128", "grid_constant"),
        C: T.handle("float16", "global"),
    ):
        T.func_attr(
            {
                "target": T.target(
                    {
                        "arch": "sm_90a",
                        "keys": ["cuda", "gpu"],
                        "kind": "cuda",
                        "max_num_threads": 1024,
                        "tag": "",
                        "thread_warp_size": 32,
                    }
                ),
                "tir.is_global_func": True,
                "tir.noalias": True,
                "tl.non_restrict_params": [],
                "tl.readonly_param_indices": [0, 1],
            }
        )
        bx = T.launch_thread("blockIdx.x", 8)
        buf_dyn_shmem = T.allocate([49152], "uint8", "shared.dyn")
        C_local = T.allocate([128], "float32", "local")
        desc_a = T.allocate([1], "uint64", "local.descriptor.wgmma")
        desc_b = T.allocate([1], "uint64", "local.descriptor.wgmma")
        C_local_cast = T.allocate([2], "float16", "local")
        by = T.launch_thread("blockIdx.y", 8)
        tx = T.launch_thread("threadIdx.x", 256)
        mbarrier = T.decl_buffer((6,), "uint64", scope="shared.barrier")
        if T.shuffle_elect(0):
            T.call_extern("handle", "tl::prefetch_tma_descriptor", A_desc)
            T.call_extern("handle", "tl::prefetch_tma_descriptor", B_desc)
            T.ptx_init_barrier_thread_count(mbarrier[0], 128)
            T.ptx_init_barrier_thread_count(mbarrier[1], 128)
            T.ptx_init_barrier_thread_count(mbarrier[2], 128)
            T.ptx_init_barrier_thread_count(mbarrier[3], 128)
            T.ptx_init_barrier_thread_count(mbarrier[4], 128)
            T.ptx_init_barrier_thread_count(mbarrier[5], 128)
        T.ptx_fence_barrier_init()
        T.tvm_storage_sync("shared")
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        T.attr([128, 128], "kWarpSpecializationScope", 0)
        if tx >= 128:
            for ko in range(32):
                T.mbarrier_wait_parity(mbarrier[ko % 3 + 3], T.bitwise_xor(ko % 6 // 3, 1))
                if T.shuffle_elect(128):
                    T.mbarrier_expect_tx(mbarrier[ko % 3], 8192)
                    T.tma_load(
                        A_desc,
                        mbarrier[ko % 3],
                        T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, ko % 3 * 4096, 4096, 2),
                        ko * 32,
                        by * 128,
                        0,
                    )
                if T.shuffle_elect(128):
                    T.mbarrier_expect_tx(mbarrier[ko % 3], 8192)
                    T.tma_load(
                        B_desc,
                        mbarrier[ko % 3],
                        T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, 12288 + ko % 3 * 4096, 2048, 2),
                        bx * 128,
                        ko * 32,
                        0,
                    )
                    T.tma_load(
                        B_desc,
                        mbarrier[ko % 3],
                        T.tvm_access_ptr(
                            T.type_annotation("float16"),
                            buf_dyn_shmem,
                            12288 + (ko % 3 * 4096 + 2048),
                            2048,
                            2,
                        ),
                        bx * 128 + 64,
                        ko * 32,
                        0,
                    )
                T.ptx_arrive_barrier(mbarrier[ko % 3])
        else:
            C_local_2 = T.Buffer((128,), data=C_local, scope="local")
            for i in T.unroll(32):
                C_local_2[i * 4 : i * 4 + 4] = T.Broadcast(T.float32(0.0), 4)
            for ko in range(32):
                T.mbarrier_wait_parity(mbarrier[ko % 3], ko % 6 // 3)
                desc_a_2 = T.Buffer((1,), "uint64", data=desc_a, scope="local.descriptor.wgmma")
                T.initialize_wgmma_descriptor(
                    desc_a_2[0],
                    T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, ko % 3 * 4096, 4096, 1),
                    2,
                    1,
                    32,
                )
                desc_b_2 = T.Buffer((1,), "uint64", data=desc_b, scope="local.descriptor.wgmma")
                T.initialize_wgmma_descriptor(
                    desc_b_2[0],
                    T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, 12288 + ko % 3 * 4096, 4096, 1),
                    1,
                    256,
                    64,
                )
                T.warpgroup_fence_operand("float32", C_local, 0, 128)
                T.warpgroup_arrive()
                for i in T.unroll(2):
                    for ki in T.unroll(2):
                        T.ptx_wgmma_ss(
                            "float32",
                            "m64n128k16",
                            T.bool(True),
                            T.bool(False),
                            "fp16",
                            "fp16",
                            "fp32",
                            desc_a,
                            T.shift_right(i * 4096 + ki * 32, 4),
                            desc_b,
                            T.shift_right(ki * 2048, 4),
                            C_local,
                            i * 64,
                            1,
                            1,
                            1,
                        )
                T.warpgroup_commit_batch()
                T.warpgroup_wait(0)
                T.warpgroup_fence_operand("float32", C_local, 0, 128)
                T.ptx_arrive_barrier(mbarrier[ko % 3 + 3])
            for i in T.unroll(128):
                C_local_2[i] = T.max(C_local_2[i], T.float32(0.0))
            for i in T.unroll(64):
                C_local_cast_2 = T.Buffer((2,), "float16", data=C_local_cast, scope="local")
                C_local_cast_2[0:2] = T.Cast("float16x2", C_local_2[i * 2 : i * 2 + 2])
                C_2 = T.Buffer((1048576,), "float16", data=C)
                C_2[
                    by * 131072
                    + i // 32 * 65536
                    + tx // 32 * 16384
                    + i % 2 * 8192
                    + tx % 32 // 4 * 1024
                    + bx * 128
                    + i % 32 // 2 * 8
                    + tx % 4 * 2 : by * 131072
                    + i // 32 * 65536
                    + tx // 32 * 16384
                    + i % 2 * 8192
                    + tx % 32 // 4 * 1024
                    + bx * 128
                    + i % 32 // 2 * 8
                    + tx % 4 * 2
                    + 2
                ] = C_local_cast_2[0:2]

    @T.prim_func
    def after(
        A_desc: T.handle("uint8x128", "grid_constant"),
        B_desc: T.handle("uint8x128", "grid_constant"),
        C: T.handle("float16", "global"),
    ):
        T.func_attr(
            {
                "target": T.target(
                    {
                        "arch": "sm_90a",
                        "keys": ["cuda", "gpu"],
                        "kind": "cuda",
                        "max_num_threads": 1024,
                        "tag": "",
                        "thread_warp_size": 32,
                    }
                ),
                "tir.is_global_func": True,
                "tir.noalias": True,
                "tl.non_restrict_params": [],
                "tl.readonly_param_indices": [0, 1],
            }
        )
        bx = T.launch_thread("blockIdx.x", 8)
        buf_dyn_shmem = T.allocate([49152], "uint8", "shared.dyn")
        C_local = T.allocate([128], "float32", "local")
        desc_a = T.allocate([1], "uint64", "local.descriptor.wgmma")
        desc_b = T.allocate([1], "uint64", "local.descriptor.wgmma")
        C_local_cast = T.allocate([2], "float16", "local")
        by = T.launch_thread("blockIdx.y", 8)
        tx = T.launch_thread("threadIdx.x", 256)
        mbarrier = T.decl_buffer((6,), "uint64", scope="shared.barrier")
        if T.shuffle_elect(0):
            T.call_extern("handle", "tl::prefetch_tma_descriptor", A_desc)
            T.call_extern("handle", "tl::prefetch_tma_descriptor", B_desc)
            T.ptx_init_barrier_thread_count(mbarrier[0], 128)
            T.ptx_init_barrier_thread_count(mbarrier[1], 128)
            T.ptx_init_barrier_thread_count(mbarrier[2], 128)
            T.ptx_init_barrier_thread_count(mbarrier[3], 128)
            T.ptx_init_barrier_thread_count(mbarrier[4], 128)
            T.ptx_init_barrier_thread_count(mbarrier[5], 128)
        T.ptx_fence_barrier_init()
        T.tvm_storage_sync("shared")
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        T.attr([128, 128], "kWarpSpecializationScope", 0)
        if tx >= 128:
            for ko in range(32):
                T.mbarrier_wait_parity(mbarrier[ko % 3 + 3], T.bitwise_xor(ko % 6 // 3, 1))
                if T.shuffle_elect(128):
                    T.mbarrier_expect_tx(mbarrier[ko % 3], 8192)
                    T.tma_load(
                        A_desc,
                        mbarrier[ko % 3],
                        T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, ko % 3 * 4096, 4096, 2),
                        ko * 32,
                        by * 128,
                        0,
                    )
                if T.shuffle_elect(128):
                    T.mbarrier_expect_tx(mbarrier[ko % 3], 8192)
                    T.tma_load(
                        B_desc,
                        mbarrier[ko % 3],
                        T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, 12288 + ko % 3 * 4096, 2048, 2),
                        bx * 128,
                        ko * 32,
                        0,
                    )
                    T.tma_load(
                        B_desc,
                        mbarrier[ko % 3],
                        T.tvm_access_ptr(
                            T.type_annotation("float16"),
                            buf_dyn_shmem,
                            12288 + (ko % 3 * 4096 + 2048),
                            2048,
                            2,
                        ),
                        bx * 128 + 64,
                        ko * 32,
                        0,
                    )
                T.ptx_arrive_barrier(mbarrier[ko % 3])
        else:
            C_local_2 = T.Buffer((128,), data=C_local, scope="local")
            for i in T.unroll(32):
                C_local_2[i * 4 : i * 4 + 4] = T.Broadcast(T.float32(0.0), 4)
            for ko in range(32):
                T.mbarrier_wait_parity(mbarrier[ko % 3], ko % 6 // 3)
                desc_a_2 = T.Buffer((1,), "uint64", data=desc_a, scope="local.descriptor.wgmma")
                T.initialize_wgmma_descriptor(
                    desc_a_2[0],
                    T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, ko % 3 * 4096, 4096, 1),
                    2,
                    1,
                    32,
                )
                desc_b_2 = T.Buffer((1,), "uint64", data=desc_b, scope="local.descriptor.wgmma")
                T.initialize_wgmma_descriptor(
                    desc_b_2[0],
                    T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, 12288 + ko % 3 * 4096, 4096, 1),
                    1,
                    256,
                    64,
                )
                T.warpgroup_fence_operand("float32", C_local, 0, 128)
                T.warpgroup_arrive()
                for i in T.unroll(2):
                    for ki in T.unroll(2):
                        T.ptx_wgmma_ss(
                            "float32",
                            "m64n128k16",
                            T.bool(True),
                            T.bool(False),
                            "fp16",
                            "fp16",
                            "fp32",
                            desc_a,
                            T.shift_right(i * 4096 + ki * 32, 4),
                            desc_b,
                            T.shift_right(ki * 2048, 4),
                            C_local,
                            i * 64,
                            1,
                            1,
                            1,
                        )
                T.warpgroup_commit_batch()
                T.warpgroup_wait(0)
                T.warpgroup_fence_operand("float32", C_local, 0, 128)
                T.ptx_arrive_barrier(mbarrier[ko % 3 + 3])
            for i in T.unroll(128):
                C_local_2[i] = T.max(C_local_2[i], T.float32(0.0))
            for i in T.unroll(64):
                C_local_cast_2 = T.Buffer((2,), "float16", data=C_local_cast, scope="local")
                C_local_cast_2[0:2] = T.Cast("float16x2", C_local_2[i * 2 : i * 2 + 2])
                C_2 = T.Buffer((1048576,), "float16", data=C)
                C_2[
                    by * 131072
                    + i // 32 * 65536
                    + tx // 32 * 16384
                    + i % 2 * 8192
                    + tx % 32 // 4 * 1024
                    + bx * 128
                    + i % 32 // 2 * 8
                    + tx % 4 * 2 : by * 131072
                    + i // 32 * 65536
                    + tx // 32 * 16384
                    + i % 2 * 8192
                    + tx % 32 // 4 * 1024
                    + bx * 128
                    + i % 32 // 2 * 8
                    + tx % 4 * 2
                    + 2
                ] = C_local_cast_2[0:2]

    _check(before, after)


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_ldmatrix_then_wgmma_does_not_inject_fence_proxy():
    """Shared-memory loads (including ldmatrix) do not trigger fence injection."""

    @T.prim_func
    def before():
        with T.Kernel(1):
            smem = T.decl_buffer((256,), T.float16, scope="shared")
            regs = T.decl_buffer((16,), T.float16, scope="local")
            desc_a = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            desc_b = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            C_local = T.decl_buffer((32,), T.float16, scope="local")
            T.call_intrin(
                "handle",
                tir.op.Op.get("tl.ptx_ldmatrix"),
                T.int32(0),
                1,
                T.tvm_access_ptr(T.type_annotation(T.float16), smem.data, 0, 16, 1),
                regs.data,
            )
            T.warpgroup_arrive()
            T.ptx_wgmma_ss(
                T.float16,
                "m64n64k16",
                T.bool(True),
                T.bool(True),
                "fp16",
                "fp16",
                "fp16",
                desc_a.data,
                T.int32(0),
                desc_b.data,
                T.int32(0),
                C_local.data,
                T.int32(0),
                T.bool(True),
                1,
                1,
            )

    _check(before, before)


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_stmatrix_then_wgmma_injects_fence_proxy():
    @T.prim_func
    def before():
        with T.Kernel(1):
            smem = T.decl_buffer((256,), T.float16, scope="shared")
            desc_a = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            desc_b = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            C_local = T.decl_buffer((32,), T.float16, scope="local")
            T.call_intrin(
                "handle",
                tir.op.Op.get("tl.ptx_stmatrix"),
                T.int32(0),
                1,
                T.tvm_access_ptr(T.type_annotation(T.float16), smem.data, 0, 16, 2),
                T.int32(0),
            )
            T.warpgroup_arrive()
            T.ptx_wgmma_ss(
                T.float16,
                "m64n64k16",
                T.bool(True),
                T.bool(True),
                "fp16",
                "fp16",
                "fp16",
                desc_a.data,
                T.int32(0),
                desc_b.data,
                T.int32(0),
                C_local.data,
                T.int32(0),
                T.bool(True),
                1,
                1,
            )

    @T.prim_func
    def after():
        with T.Kernel(1):
            smem = T.decl_buffer((256,), T.float16, scope="shared")
            desc_a = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            desc_b = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            C_local = T.decl_buffer((32,), T.float16, scope="local")
            T.call_intrin(
                "handle",
                tir.op.Op.get("tl.ptx_stmatrix"),
                T.int32(0),
                1,
                T.tvm_access_ptr(T.type_annotation(T.float16), smem.data, 0, 16, 2),
                T.int32(0),
            )
            T.warpgroup_arrive()
            T.fence_proxy_async()
            T.ptx_wgmma_ss(
                T.float16,
                "m64n64k16",
                T.bool(True),
                T.bool(True),
                "fp16",
                "fp16",
                "fp16",
                desc_a.data,
                T.int32(0),
                desc_b.data,
                T.int32(0),
                C_local.data,
                T.int32(0),
                T.bool(True),
                1,
                1,
            )

    _check(before, after)


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_if_merge_may_be_generic_then_async_injects_fence_proxy():
    @T.prim_func
    def before(flag: T.int32):
        with T.Kernel(1):
            smem = T.decl_buffer((1,), T.float16, scope="shared")
            desc_a = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            desc_b = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            C_local = T.decl_buffer((32,), T.float16, scope="local")
            if flag == 1:
                smem[0] = T.float16(0)
            T.warpgroup_arrive()
            T.ptx_wgmma_ss(
                T.float16,
                "m64n64k16",
                T.bool(True),
                T.bool(True),
                "fp16",
                "fp16",
                "fp16",
                desc_a.data,
                T.int32(0),
                desc_b.data,
                T.int32(0),
                C_local.data,
                T.int32(0),
                T.bool(True),
                1,
                1,
            )

    @T.prim_func
    def after(flag: T.int32):
        with T.Kernel(1):
            smem = T.decl_buffer((1,), T.float16, scope="shared")
            desc_a = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            desc_b = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            C_local = T.decl_buffer((32,), T.float16, scope="local")
            if flag == 1:
                smem[0] = T.float16(0)
            T.warpgroup_arrive()
            T.fence_proxy_async()
            T.ptx_wgmma_ss(
                T.float16,
                "m64n64k16",
                T.bool(True),
                T.bool(True),
                "fp16",
                "fp16",
                "fp16",
                desc_a.data,
                T.int32(0),
                desc_b.data,
                T.int32(0),
                C_local.data,
                T.int32(0),
                T.bool(True),
                1,
                1,
            )

    _check(before, after)


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_hoist_fence_proxy_out_of_if():
    """Hoist a single fence out of a pure-async if-then-else region."""

    @T.prim_func
    def before(flag: T.int32):
        with T.Kernel(1):
            smem = T.decl_buffer((1,), T.float16, scope="shared")
            desc_a = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            desc_b = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            C_local = T.decl_buffer((32,), T.float16, scope="local")
            smem[0] = T.float16(0)
            T.warpgroup_arrive()
            if flag == 1:
                T.ptx_wgmma_ss(
                    T.float16,
                    "m64n64k16",
                    T.bool(True),
                    T.bool(True),
                    "fp16",
                    "fp16",
                    "fp16",
                    desc_a.data,
                    T.int32(0),
                    desc_b.data,
                    T.int32(0),
                    C_local.data,
                    T.int32(0),
                    T.bool(True),
                    1,
                    1,
                )
            else:
                T.ptx_wgmma_ss(
                    T.float16,
                    "m64n64k16",
                    T.bool(True),
                    T.bool(True),
                    "fp16",
                    "fp16",
                    "fp16",
                    desc_a.data,
                    T.int32(0),
                    desc_b.data,
                    T.int32(0),
                    C_local.data,
                    T.int32(0),
                    T.bool(True),
                    1,
                    1,
                )

    @T.prim_func
    def after(flag: T.int32):
        with T.Kernel(1):
            smem = T.decl_buffer((1,), T.float16, scope="shared")
            desc_a = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            desc_b = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            C_local = T.decl_buffer((32,), T.float16, scope="local")
            smem[0] = T.float16(0)
            T.warpgroup_arrive()
            T.fence_proxy_async()
            if flag == 1:
                T.ptx_wgmma_ss(
                    T.float16,
                    "m64n64k16",
                    T.bool(True),
                    T.bool(True),
                    "fp16",
                    "fp16",
                    "fp16",
                    desc_a.data,
                    T.int32(0),
                    desc_b.data,
                    T.int32(0),
                    C_local.data,
                    T.int32(0),
                    T.bool(True),
                    1,
                    1,
                )
            else:
                T.ptx_wgmma_ss(
                    T.float16,
                    "m64n64k16",
                    T.bool(True),
                    T.bool(True),
                    "fp16",
                    "fp16",
                    "fp16",
                    desc_a.data,
                    T.int32(0),
                    desc_b.data,
                    T.int32(0),
                    C_local.data,
                    T.int32(0),
                    T.bool(True),
                    1,
                    1,
                )

    _check(before, after)


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_hoist_fence_proxy_out_of_unrolled_loop():
    """Prefer a single preheader fence over per-iteration fences.

    If the loop body performs async-proxy ops but never performs generic shared
    traffic, a possibly-generic entry state should be resolved by inserting a
    single fence before the loop, rather than inside the loop body.
    """

    @T.prim_func
    def before():
        with T.Kernel(1):
            smem = T.decl_buffer((1,), T.float16, scope="shared")
            desc_a = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            desc_b = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            C_local = T.decl_buffer((32,), T.float16, scope="local")

            smem[0] = T.float16(0)
            T.warpgroup_arrive()
            for _ in T.unroll(12):
                T.ptx_wgmma_ss(
                    T.float16,
                    "m64n64k16",
                    T.bool(True),
                    T.bool(True),
                    "fp16",
                    "fp16",
                    "fp16",
                    desc_a.data,
                    T.int32(0),
                    desc_b.data,
                    T.int32(0),
                    C_local.data,
                    T.int32(0),
                    T.bool(True),
                    1,
                    1,
                )

    @T.prim_func
    def after():
        with T.Kernel(1):
            smem = T.decl_buffer((1,), T.float16, scope="shared")
            desc_a = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            desc_b = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            C_local = T.decl_buffer((32,), T.float16, scope="local")

            smem[0] = T.float16(0)
            T.warpgroup_arrive()
            T.fence_proxy_async()
            for _ in T.unroll(12):
                T.ptx_wgmma_ss(
                    T.float16,
                    "m64n64k16",
                    T.bool(True),
                    T.bool(True),
                    "fp16",
                    "fp16",
                    "fp16",
                    desc_a.data,
                    T.int32(0),
                    desc_b.data,
                    T.int32(0),
                    C_local.data,
                    T.int32(0),
                    T.bool(True),
                    1,
                    1,
                )

    _check(before, after)


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_hoist_fence_proxy_out_of_while_loop():
    """Hoist a single fence out of a pure-async while-loop body."""

    @T.prim_func
    def before():
        with T.Kernel(1):
            smem = T.decl_buffer((1,), T.float16, scope="shared")
            counter = T.decl_buffer((1,), T.int32, scope="local")
            desc_a = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            desc_b = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            C_local = T.decl_buffer((32,), T.float16, scope="local")
            smem[0] = T.float16(0)
            counter[0] = 0
            while counter[0] < T.int32(12):
                T.warpgroup_arrive()
                T.ptx_wgmma_ss(
                    T.float16,
                    "m64n64k16",
                    T.bool(True),
                    T.bool(True),
                    "fp16",
                    "fp16",
                    "fp16",
                    desc_a.data,
                    T.int32(0),
                    desc_b.data,
                    T.int32(0),
                    C_local.data,
                    T.int32(0),
                    T.bool(True),
                    1,
                    1,
                )
                counter[0] = counter[0] + 1

    @T.prim_func
    def after():
        with T.Kernel(1):
            smem = T.decl_buffer((1,), T.float16, scope="shared")
            counter = T.decl_buffer((1,), T.int32, scope="local")
            desc_a = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            desc_b = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            C_local = T.decl_buffer((32,), T.float16, scope="local")
            smem[0] = T.float16(0)
            counter[0] = 0
            T.fence_proxy_async()
            while counter[0] < T.int32(12):
                T.warpgroup_arrive()
                T.ptx_wgmma_ss(
                    T.float16,
                    "m64n64k16",
                    T.bool(True),
                    T.bool(True),
                    "fp16",
                    "fp16",
                    "fp16",
                    desc_a.data,
                    T.int32(0),
                    desc_b.data,
                    T.int32(0),
                    C_local.data,
                    T.int32(0),
                    T.bool(True),
                    1,
                    1,
                )
                counter[0] = counter[0] + 1

    _check(before, after)


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_loop_carried_generic_then_async_injects_fence_proxy():
    """Generic proxy traffic at the end of an iteration may affect the next iteration."""

    @T.prim_func
    def before():
        with T.Kernel(1):
            smem = T.decl_buffer((1,), T.float16, scope="shared")
            desc_a = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            desc_b = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            C_local = T.decl_buffer((32,), T.float16, scope="local")
            for _ in range(2):
                T.warpgroup_arrive()
                T.ptx_wgmma_ss(
                    T.float16,
                    "m64n64k16",
                    T.bool(True),
                    T.bool(True),
                    "fp16",
                    "fp16",
                    "fp16",
                    desc_a.data,
                    T.int32(0),
                    desc_b.data,
                    T.int32(0),
                    C_local.data,
                    T.int32(0),
                    T.bool(True),
                    1,
                    1,
                )
                smem[0] = T.float16(0)

    @T.prim_func
    def after():
        with T.Kernel(1):
            smem = T.decl_buffer((1,), T.float16, scope="shared")
            desc_a = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            desc_b = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            C_local = T.decl_buffer((32,), T.float16, scope="local")
            for _ in range(2):
                T.warpgroup_arrive()
                T.fence_proxy_async()
                T.ptx_wgmma_ss(
                    T.float16,
                    "m64n64k16",
                    T.bool(True),
                    T.bool(True),
                    "fp16",
                    "fp16",
                    "fp16",
                    desc_a.data,
                    T.int32(0),
                    desc_b.data,
                    T.int32(0),
                    C_local.data,
                    T.int32(0),
                    T.bool(True),
                    1,
                    1,
                )
                smem[0] = T.float16(0)

    _check(before, after)


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_shared_load_does_not_trigger_fence_proxy():
    """Shared loads are not treated as generic proxy traffic for fence injection."""

    @T.prim_func
    def before():
        with T.Kernel(1):
            smem = T.decl_buffer((1,), T.float16, scope="shared")
            desc_a = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            desc_b = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            C_local = T.decl_buffer((32,), T.float16, scope="local")
            x = smem[0]
            C_local[0] = x
            T.warpgroup_arrive()
            T.ptx_wgmma_ss(
                T.float16,
                "m64n64k16",
                T.bool(True),
                T.bool(True),
                "fp16",
                "fp16",
                "fp16",
                desc_a.data,
                T.int32(0),
                desc_b.data,
                T.int32(0),
                C_local.data,
                T.int32(0),
                T.bool(True),
                1,
                1,
            )

    @T.prim_func
    def after():
        with T.Kernel(1):
            smem = T.decl_buffer((1,), T.float16, scope="shared")
            desc_a = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            desc_b = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            C_local = T.decl_buffer((32,), T.float16, scope="local")
            x = smem[0]
            C_local[0] = x
            T.warpgroup_arrive()
            T.ptx_wgmma_ss(
                T.float16,
                "m64n64k16",
                T.bool(True),
                T.bool(True),
                "fp16",
                "fp16",
                "fp16",
                desc_a.data,
                T.int32(0),
                desc_b.data,
                T.int32(0),
                C_local.data,
                T.int32(0),
                T.bool(True),
                1,
                1,
            )

    _check(before, after)


if __name__ == "__main__":
    tilelang.testing.main()
