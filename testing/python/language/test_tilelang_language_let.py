import tilelang.testing
from tilelang import tvm as tvm
from tvm import IRModule
from tilelang import language as T
from tilelang.utils.tensor import map_torch_type

def test_let_vectorize_load():
    @T.prim_func
    def main(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (16, 16), dtype="float32", align=16)

        for blockIdx in T.thread_binding(1, thread="blockIdx.x"):
            for threadIdx in T.thread_binding(128, thread="threadIdx.x"):
                b: T.float32x4 = A[0, 0:4]
                A[0, 4:8] = b

    mod = tvm.IRModule({"main": main})
    mod = tvm.compile(mod, target="cuda")
    assert "float4 b" in mod.mod.imported_modules[0].get_source()
    

if __name__ == "__main__":
    tilelang.testing.main()
