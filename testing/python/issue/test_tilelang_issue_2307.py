import tilelang
import tilelang.language as T
import tilelang.testing
from tvm.target import Target


def _get_issue_2307_source() -> str:
    @T.prim_func
    def main(score: T.Tensor((16,), T.float32)):
        with T.Kernel(1, threads=32):
            score_fragment = T.alloc_fragment(32, T.float32)
            for i in T.Parallel(32):
                if i < 16:
                    score_fragment[i] = score[i]
                else:
                    score_fragment[i] = T.infinity(T.float32)
            for i in T.Parallel(32):
                if i < 16:
                    T.device_assert(T.isfinite(score_fragment[i]))

    target = Target("cuda")
    with target:
        artifact = tilelang.lower(main, target=target)
    return artifact.kernel_source


@tilelang.testing.requires_cuda
def test_fragment_read_assert_parallel_loop_is_partitioned():
    src = _get_issue_2307_source()
    lines = src.splitlines()
    assert "for (int i = 0; i < 32; ++i)" not in src

    assert_idx = next(i for i, line in enumerate(lines) if "device_assert_with_msg" in line)
    context = "\n".join(lines[max(0, assert_idx - 4) : assert_idx + 1])
    assert "threadIdx.x) < 16" in context
    assert "isfinite(score_fragment[0])" in context


if __name__ == "__main__":
    tilelang.testing.main()
