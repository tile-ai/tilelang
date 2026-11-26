import pytest

import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.engine.lower import lower

N = 128


def _kernel():
    @T.prim_func
    def main(A: T.Tensor((N,), "int32"), B: T.Tensor((N,), "int32")):
        with T.Kernel(1, threads=128):
            buf = T.alloc_fragment((N,), "int32")
            T.copy(A, buf)
            for i in T.Parallel(N):
                B[i] = buf[i] + buf[(i + 1) % N]

    return main


@tilelang.testing.requires_cuda
def test_fragment_two_indices_reports_user_error():
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_89"})
    with target, pytest.raises(ValueError, match="multiple distinct indices"):
        lower(_kernel().with_attr("global_symbol", "main"), target=target)


if __name__ == "__main__":
    tilelang.testing.main()
