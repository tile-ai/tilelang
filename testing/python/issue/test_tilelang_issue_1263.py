import tilelang.testing
import tilelang.language as T


def _test_kernel(M, N):
    dtype = "bfloat16"

    @T.prim_func
    def fwd_main(
        KV: T.Tensor((M, N), dtype),
        ids: T.Tensor((4,), "int32"),
        ids2: T.Tensor((4,), "int32"),
    ):
        with T.Kernel(4, threads=1):
            A = T.alloc_shared([N], dtype)
            B = T.alloc_shared([N], dtype)

            for i in T.Pipelined(4, num_stages=1):
                id = ids[i]
                id2 = ids2[id]
                T.copy(KV[id2, :], A)
                T.clear(B)

    return fwd_main


def _test_kernel_if_cond(M, N):
    dtype = "bfloat16"

    @T.prim_func
    def fwd_main(
        KV: T.Tensor((M, N), dtype),
        ids: T.Tensor((4,), "int32"),
        ids2: T.Tensor((4,), "int32"),
    ):
        with T.Kernel(4, threads=1):
            A = T.alloc_shared([N], dtype)
            B = T.alloc_shared([N], dtype)

            for i in T.Pipelined(4, num_stages=1):
                id = ids[i]
                id2 = ids2[id]
                if id2 > 1:
                    T.copy(KV[id2, :], A)
                    T.clear(B)

    return fwd_main


def test_issue_1263_pipeline_no_consumer():
    tilelang.compile(_test_kernel(1024, 1024))
    tilelang.compile(
        _test_kernel(1024, 1024),
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        },
    )
    tilelang.compile(_test_kernel_if_cond(1024, 1024))
    tilelang.compile(
        _test_kernel_if_cond(1024, 1024),
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        },
    )


if __name__ == "__main__":
    tilelang.testing.main()
