import tilelang
import tilelang.language as T
import tilelang.testing


@T.prim_func
def _producer_bind_kernel(
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


def test_pipeline_planning_tracks_transitive_copy_index_producer():
    target = tilelang.tvm.target.Target({"kind": "cuda", "arch": "sm_80"})
    with target:
        tilelang.lower(_producer_bind_kernel, target=target)


if __name__ == "__main__":
    tilelang.testing.main()
