"""Regression test for issue #3653.

A shared-memory sync required inside a `T.Parallel` guard (`tx < 448`, 14 whole
warps) could not be placed when the enclosing `if` condition was written with
`&` instead of `T.And`. The condition itself is block-uniform; the bitwise
spelling only reached `Analyzer::CanProve` as a boolean `bitwise_and` call,
which the constraint translation did not model, so the warp-uniformity proof
failed and ThreadSync rejected otherwise valid code.

The kernel only needs to lower; it is not executed.
"""

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm

BI, D, W4 = 64, 448, 112


def _lower(guard):
    @T.prim_func
    def main(
        Q: T.Tensor([64, D], "bfloat16"),
        Kp: T.Tensor([BI, W4], "uint32"),
        Sc: T.Tensor([BI, 7], "uint32"),
        Len: T.Tensor([4], "int32"),
        O: T.Tensor([4, 10, 64, BI], "float32"),
    ):
        with T.Kernel(10, 4, threads=512) as (bx, by):
            Q_f = T.alloc_fragment([64, D], "bfloat16")
            K_s = T.alloc_shared([BI, W4], "uint32")
            sc = T.alloc_fragment([BI, 7], "uint32")
            KV_s = T.alloc_shared([BI, D], "bfloat16")
            acc = T.alloc_fragment([64, BI], "float32")
            T.clear(acc)
            tk = Len[by]
            if guard(bx, tk):
                T.copy(Q, Q_f)
                T.copy(Kp, K_s)
                T.copy(Sc, sc)
                for _ in T.serial(2):
                    for i, d in T.Parallel(BI, D):
                        b = (K_s[i, d // 4] >> T.Cast("uint32", d % 4 * 8)) & T.uint32(0xFF)
                        KV_s[i, d] = T.reinterpret("bfloat16", T.Cast("uint16", (b << T.uint32(8)) | sc[i, d // 64]))
                    T.gemm(Q_f, KV_s, acc, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            T.copy(acc, O[by, bx, :, :])

    cfg = {
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_89"})
    with target, tvm.transform.PassContext(config=cfg):
        tilelang.lower(main, target=target, enable_device_compile=False)


@tilelang.testing.requires_cuda
def test_thread_sync_accepts_bitwise_and_guard():
    _lower(lambda bx, tk: (bx % 10 < 2) & (tk > 0))


@tilelang.testing.requires_cuda
def test_thread_sync_accepts_logical_and_guard():
    _lower(lambda bx, tk: T.And(bx % 10 < 2, tk > 0))


if __name__ == "__main__":
    tilelang.testing.main()
