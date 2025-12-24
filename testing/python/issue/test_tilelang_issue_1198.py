import tilelang.testing
import tilelang.language as T


def test_issue_1198():
    @T.prim_func
    def foo(
        x: T.Buffer(
            [
                32,
            ],
            T.int32,
        ),
    ):
        pass


if __name__ == "__main__":
    tilelang.testing.main()
