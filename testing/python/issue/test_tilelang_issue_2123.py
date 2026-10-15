import tilelang
import tilelang.testing
from tilelang.testing.ir import assert_call_count, collect_calls
import tilelang.language as T
from tilelang import tvm
from tilelang.cuda.pipeline import CUDAPassPipelineBodyPrologue
from tilelang.transform import LowerAccessPtr


def issue_2123_atomic_load_repro(num_tiles, threads=32):
    @T.prim_func
    def kernel(status: T.Tensor((num_tiles,), T.int32), out: T.Tensor((1,), T.int32)):
        with T.Kernel(num_tiles, threads=threads) as tile:
            look = T.alloc_var(T.int32)
            state = T.alloc_var(T.int32)
            done = T.alloc_var(T.bool)
            tx = T.get_thread_binding()
            if tx == 0:
                look = tile - 1
                done = look < 0
                state = 0
                while not done:
                    state = T.atomic_load(status[look], memory_order="acquire")
                    if state != 0:
                        done = True
                    else:
                        look -= 1
                        done = look < 0
                if tile == num_tiles - 1:
                    out[0] = state

    return kernel


def _assert_access_ptr_lowered(mod):
    assert collect_calls(mod["main"], op="tirx.tvm_access_ptr")
    assert_call_count(mod["main"], op="tl.access_ptr", count=0)


def test_issue_2123_atomic_load_lower_access_ptr_direct():
    func = issue_2123_atomic_load_repro(4).with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(func)

    lowered = LowerAccessPtr()(mod)

    _assert_access_ptr_lowered(lowered)


@tilelang.testing.requires_cuda
def test_issue_2123_atomic_load_lower_access_ptr_pipeline():
    target = tvm.target.Target("cuda", host="llvm")
    func = issue_2123_atomic_load_repro(4).with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(func)

    lowered = CUDAPassPipelineBodyPrologue(mod, target)

    _assert_access_ptr_lowered(lowered)


if __name__ == "__main__":
    tilelang.testing.main()
