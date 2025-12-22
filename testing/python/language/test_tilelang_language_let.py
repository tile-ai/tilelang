import tilelang.testing
from tilelang import tvm as tvm
from tilelang import language as T


def test_let_vectorize_load():
    @T.prim_func
    def main(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (16, 16), dtype=T.float32, align=16)

        for _blockIdx in T.thread_binding(1, thread="blockIdx.x"):
            for _threadIdx in T.thread_binding(128, thread="threadIdx.x"):
                b = A[0, 0:4]
                A[0, 4:8] = b

    mod = tvm.IRModule({"main": main})
    mod = tvm.compile(mod, target="cuda")
    assert "float4 b" in mod.mod.imports[0].inspect_source()


if __name__ == "__main__":
    tilelang.testing.main()
