import tilelang.ascend.language as T
from tilelang.backend.target import determine_target


def main():
    target = determine_target("ascend", return_object=True)

    with target:
        try:

            @T.prim_func
            def _bad_kernel(a: T.Buffer((1,), "float32")):
                with T.Kernel(1, threads=128):
                    a[0] = T.float32(1)
        except ValueError as err:
            msg = str(err)
            assert "threads" in msg and "ascend" in msg.lower(), f"unexpected ValueError message: {msg}"
        else:
            raise AssertionError("Ascend should reject explicit threads= in T.Kernel")

    print("[ok] Ascend T.Kernel rejects explicit threads= as expected.")


if __name__ == "__main__":
    main()
