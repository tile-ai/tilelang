import tilelang
import tilelang.testing
from tilelang import language as T


def test_issue_1728():
    @tilelang.jit()
    def get_qwq(hidden: int):
        num_tokens = T.dynamic("num_tokens")
        num_sms = num_tokens

        @T.prim_func
        def qwq(A: T.Tensor[(num_tokens,)]):
            with T.Kernel(num_sms) as sm_id:
                stop = sm_id + 1
                for block_idx in T.serial(sm_id, stop):
                    _pid_x, _pid_y = (block_idx, hidden)

        return qwq

    get_qwq(1)


if __name__ == "__main__":
    tilelang.testing.main()
