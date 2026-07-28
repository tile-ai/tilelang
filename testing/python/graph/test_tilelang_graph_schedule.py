from tilelang import tvm
from tilelang.graph.schedule import canonicalize_scheduled_tir
from tvm.script import tirx as T


@T.prim_func
def mixed_width_block_binding(n: T.int64, output: T.Buffer((1,), "float32")):
    with T.sblock("root"):
        T.reads()
        T.writes()
        with T.sblock("update"):
            index = T.axis.spatial(T.int64(16), n)
            if index == T.int64(0):
                output[0] = T.float32(1)


def test_canonicalize_converts_block_bindings_before_narrowing():
    """An int64 binding must be substituted before its block var is narrowed."""
    mod = tvm.IRModule({"main": mixed_width_block_binding})

    canonicalized = canonicalize_scheduled_tir(mod)

    assert "n == T.int64(0)" in canonicalized.script()


if __name__ == "__main__":
    tvm.testing.main()
