"""Legal scalar-SS/VF neighbours used to distinguish validation from scheduling."""

import tilelang
from tilelang.ascend import language as T


def test_ss_scalar_alias_remains_legal():
    @T.prim_func
    def main(out: T.Tensor((2,), "uint64")):
        with T.MixedKernel(1) as (_, sid):
            owner = T.alloc_ss((2,), "uint64")
            alias = T.view(owner, shape=(2,), dtype="uint64")
            owner[sid] = T.uint64(7)
            out[sid] = alias[sid]

    result = tilelang.lower(main, target="ascend", enable_device_compile=False)
    assert "volatile __ssbuf__ uint64_t" in result.kernel_source
    assert "SIMT" not in result.kernel_source


def test_scalar_ss_and_disjoint_simt_global_output():
    @T.prim_func
    def main(out: T.Tensor((256,), "uint32"), scalar_out: T.Tensor((2,), "uint32")):
        with T.MixedKernel(1) as (_, sid):
            words = T.alloc_ss((2,), "uint32")
            words[sid] = T.uint32(1)
            scalar_out[sid] = words[sid]
            with T.SimtVF(threads=128):
                tx = T.get_thread_binding()
                out[tx + sid * 128] = T.uint32(2)

    result = tilelang.lower(main, target="ascend", enable_device_compile=False)
    assert "volatile __ssbuf__" in result.kernel_source
    assert "__simt_vf__" in result.kernel_source


def test_simt_ub_access_remains_legal():
    @T.prim_func
    def main(inp: T.Tensor((128,), "float32"), out: T.Tensor((128,), "float32")):
        with T.Kernel(1):
            values = T.alloc_shared((128,), "float32")
            T.copy(inp, values)
            with T.SimtVF(threads=128):
                tx = T.get_thread_binding()
                out[tx] = values[tx]

    result = tilelang.lower(main, target="ascend", enable_device_compile=False)
    assert "__simt_vf__" in result.kernel_source
    assert "__ubuf__" in result.kernel_source
