import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


def test_merge_shared_memory_fp4_non_alias_offset_units():
    @T.prim_func
    def before(
        A: T.Tensor((16,), T.float4_e2m1fn),
        B: T.Tensor((16,), T.float4_e2m1fn),
    ):
        with T.Kernel(1, threads=32):
            a = T.alloc_shared((3,), T.float4_e2m1fn)
            b = T.alloc_shared((4,), T.float4_e2m1fn)
            a[0] = A[0]
            b[0] = A[1]
            B[0] = a[0]
            B[1] = b[0]

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main").with_attr("target", tvm.target.Target("webgpu")))
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.MergeSharedMemoryAllocations()(mod)

    src = mod.script()
    # The first FP4 allocation has three logical values, i.e. two physical
    # bytes. After 16-byte alignment, the second allocation starts at byte 16,
    # which is logical FP4 element offset 32 in the non-alias rewrite path.
    assert "b[32]" in src
    assert "b[16]" not in src


if __name__ == "__main__":
    tilelang.testing.main()
