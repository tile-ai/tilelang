# from tvm.script import tir as T

@T.prim_func
def cast_to_fp8_e4m3(X: T.Buffer((132, 4096), "float32"), X_fp8: T.Buffer((132, 4096), "e4m3_float8"), scale_inv: T.Buffer((1,), "float32"), scale: T.Buffer((132,), "float32")):
    # with T.block("root"):
    sm_id = T.launch_thread("blockIdx.x", 132)
    tx = T.launch_thread("threadIdx.x", 512)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    with T.block("tilelang_root"):
        X_1d = T.Buffer((540672,), data=X.data)
        X_fp8_1d = T.Buffer((540672,), "e4m3_float8", data=X_fp8.data)
        T.reads(X_1d[sm_id * 4096], scale[T.min(tx, 0):T.min(tx, 0) + (T.max(tx, 0) + 1 - T.min(tx, 0))], X_fp8_1d[sm_id * 4096])
        T.writes(scale[T.min(sm_id, 0):T.min(sm_id, 0) + (T.max(sm_id, 0) + 1 - T.min(sm_id, 0))], scale_inv[0])
        y_shared = T.alloc_buffer((4096,), scope="shared.dyn")
        y_local = T.alloc_buffer((4096,), scope="local.fragment")
        y_shared_fp8 = T.alloc_buffer((4096,), "e4m3_float8", scope="shared.dyn")
        scale_shared = T.alloc_buffer((512,), scope="shared.dyn")
        scale_local = T.alloc_buffer((512,), scope="local.fragment")
        local_max = T.alloc_buffer((1,), scope="local.fragment")
        shared_max = T.alloc_buffer((1,), scope="local.fragment")
        global_max = T.alloc_buffer((1,), scope="local.fragment")
        rescale_factor = T.alloc_buffer((1,), scope="local.fragment")
        T.fill(T.tvm_access_ptr(T.type_annotation("float32"), shared_max.data, 0, 1, 2), 0)
        for i in range(1):
            T.copy(T.region(X_1d[i * 540672 + sm_id * 4096], 1, 4096), T.region(y_shared[0], 2, 4096))
            T.copy(T.region(y_shared[0], 1, 4096), T.region(y_local[0], 2, 4096))
            T.reduce(T.tvm_access_ptr(T.type_annotation("float32"), y_local.data, 0, 4096, 1), T.tvm_access_ptr(T.type_annotation("float32"), local_max.data, 0, 1, 2), "absmax", 0, T.bool(True))
            for j in T.parallel(1):
                shared_max[j] = T.max(shared_max[j], local_max[j])
        scale[sm_id] = shared_max[0]
        scale_shared[tx] = T.if_then_else(tx < 132, scale[tx], T.float32(0))
        T.copy(T.region(scale_shared[0], 1, 512), T.region(scale_local[0], 2, 512))
        T.reduce(T.tvm_access_ptr(T.type_annotation("float32"), scale_local.data, 0, 512, 1), T.tvm_access_ptr(T.type_annotation("float32"), global_max.data, 0, 1, 2), "max", 0, T.bool(True))
        scale[0] = global_max[0]
        scale_inv[0] = global_max[0] / T.float32(448)
        rescale_factor[0] = T.float32(448) / scale[0]
        for i in range(1):
            T.copy(T.region(X_1d[i * 540672 + sm_id * 4096], 1, 4096), T.region(y_shared[0], 2, 4096))
            T.copy(T.region(y_shared[0], 1, 4096), T.region(y_local[0], 2, 4096))
            for j in T.parallel(4096):
                y_shared_fp8[j] = T.Cast("e4m3_float8", y_local[j] * rescale_factor[0])
            T.copy(T.region(y_shared_fp8[0], 1, 4096), T.region(X_fp8_1d[i * 540672 + sm_id * 4096], 2, 4096))
