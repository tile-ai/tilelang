import tilelang
import tilelang.testing
from tilelang import language as T


@tilelang.jit
def _tilelang_issue_layout_free_inference_choose_smallest_replication():
    @T.prim_func
    def main(A: T.Tensor((128, 4), T.float), B: T.Tensor((128, 4), T.float)):
        with T.Kernel(1, threads=128) as _:
            A_frag = T.alloc_fragment((128, 4), T.float)
            B_frag = T.alloc_fragment((128, 4), T.float)
            S_frag = T.alloc_fragment((4,), T.float)
            T.annotate_layout(
                {
                    A_frag: T.Fragment(A_frag.shape, lambda i, j: (i, j)),
                }
            )
            for i, j in T.Parallel(128, 4):
                A_frag[i, j] = S_frag[j]
            for i, j in T.Parallel(128, 4):
                B_frag[i, j] = S_frag[j]

    return main


def test_tilelang_issue_layout_free_inference_choose_smallest_replication():
    kernel = _tilelang_issue_layout_free_inference_choose_smallest_replication()
    source = kernel.get_kernel_source()
    assert "float S_frag[4];" in source, "S_frag is not in the source"
    assert "float B_frag[4];" in source, "B_frag is not in the source"
    assert "float A_frag[4];" in source, "A_frag is not in the source"


if __name__ == "__main__":
    tilelang.testing.main()
