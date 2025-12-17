import tilelang
import tilelang.language as T
import tilelang.testing


def _make_kernel(M, N):
    dtype = T.bfloat16

    @T.prim_func
    def fwd_main(KV: T.Tensor((M, N), dtype), ids: T.Tensor((4,), T.int32)):
        with T.Kernel(4, threads=1):
            A = T.alloc_shared([N], dtype)
            B = T.alloc_shared([N], dtype)

            # Regression for a bug where InjectSoftwarePipeline left the loop
            # variable as a free var, causing MakePackedAPI to fail
            for i in T.Pipelined(4, num_stages=1):
                _id = ids[i]
                T.copy(KV[_id, :], A)
                T.clear(B)

    return fwd_main


def test_make_packed_api_no_free_loop_var():
    func = _make_kernel(4, 4)
    # Keep warp-specialization/TMA disabled to match the original repro
    cfg = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }
    tilelang.compile(func, pass_configs=cfg)


if __name__ == "__main__":
    tilelang.testing.main()
