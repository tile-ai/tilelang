import tilelang
import tilelang.ascend.language as T
import tilelang.testing


@tilelang.jit()
def get_kernel():

    @T.prim_func
    def program(global_buf: T.Buffer((1), "int32")):
        with T.Kernel(2) as bx:
            shared_buf = T.alloc_shared([64], T.float32)
            with T.SimdVF():
                mask_f32 = T.simd.pset(32)
                for _ in T.serial(T.min(1, bx)):
                    v = T.simd.vld(shared_buf[0])
                    T.simd.vsts(shared_buf[0], v, mask_f32)
            global_buf[0] = bx

    return program


def test_simdvf_blockidx_capture():
    source = get_kernel().get_kernel_source()
    print(source)


if __name__ == "__main__":
    tilelang.testing.main()
