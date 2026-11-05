import tvm_tirx as tvm
import pytest
from tvm_tirx import ir, tirx
from tilelang.testing.ir import collect_nodes
import tilelang as tl
import tilelang.language as T
import tilelang.testing


def vectorize_access_legalize(M: int = 64, N: int = 64):
    dtype = T.float32
    vec_len = 8

    @T.prim_func
    def main(
        A: T.Tensor((M, N, vec_len), dtype=T.float32),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            A_shared = T.alloc_shared((M, N, vec_len), dtype=dtype)
            tid = T.get_thread_binding()
            for j in T.serial(N):
                for v in T.vectorized(vec_len):
                    A_shared[tid, j, v] = A[tid, j, v]

    @T.prim_func
    def expected(
        A: T.Tensor((M, N, vec_len), dtype=T.float32),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            A_shared = T.alloc_shared((M, N, vec_len), dtype=dtype)
            tid = T.get_thread_binding()
            for j, v_2 in T.grid(M, vec_len // 4):
                for vec in T.vectorized(4):
                    A_shared[tid, j, v_2 * 4 + vec] = A[tid, j, v_2 * 4 + vec]

    return main, expected


def _assert_function_equal(actual, expected):
    # Bind corresponding parameters by comparing complete native Functions.
    # Normalize pure arithmetic on both sides: (2 + outer*2) + lane and
    # 2 + (outer*2 + lane) describe the same fixed-width address expression.
    # The loops, types, layouts and memory operations still compare structurally.
    def canonical(function):
        module = tvm.IRModule({"main": function.without_attr("global_symbol")})
        return tirx.transform.StmtSimplify()(module)["main"]

    tvm.ir.assert_structural_equal(canonical(actual), canonical(expected))


def assert_vectorize_access(M: int = 64, N: int = 64):
    func, expected = vectorize_access_legalize(M, N)
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    with tvm.target.Target("cuda"):
        transformed = tl.transform.LegalizeVectorizedLoop()(mod)
    _assert_function_equal(transformed["main"], expected)


def test_vectorize_access():
    assert_vectorize_access(64, 64)


@tilelang.testing.requires_cuda
def test_vectorized_nonzero_min_uses_aligned_vector_width():
    @T.prim_func
    def before(
        A: T.Tensor((8,), T.int32),
        B: T.Tensor((8,), T.int32),
    ):
        for value in T.vectorized(2, 6):
            B[value] = A[value] + 1

    @T.prim_func
    def expected(
        A: T.Tensor((8,), T.int32),
        B: T.Tensor((8,), T.int32),
    ):
        for value_outer in T.serial(2):
            for vec in T.vectorized(2):
                B[2 + (value_outer * 2 + vec)] = A[2 + (value_outer * 2 + vec)] + 1

    mod = tvm.IRModule.from_expr(before)
    with tvm.target.Target("cuda"):
        transformed = tl.transform.LegalizeVectorizedLoop()(mod)
    _assert_function_equal(transformed["before"], expected)


@tilelang.testing.requires_cuda
def test_vectorized_split_preserves_large_int64_extent():
    start, extent = 2, 1 << 32
    a = ir.Var("A", tirx.TensorType((start + extent,), "int32", layout="default"))
    i = ir.Var("i", "int64")
    body = ir.TensorStore(a, [i], ir.TensorLoad(a, [i]) + 1)
    loop = ir.For(i, ir.const(start, "int64"), ir.const(start + extent, "int64"), ir.const(1, "int64"), ir.ForKind.VECTORIZED, [body])
    func = tirx.Function([a], [loop], ir.PrimType("void"))
    with tvm.target.Target("cuda"):
        mod = tl.transform.LegalizeVectorizedLoop()(tvm.IRModule({"main": func}))
    loops = collect_nodes(mod["main"], ir.For)
    assert len(loops) == 2
    assert all(loop.var.ty.dtype == "int64" for loop in loops)
    assert int(loops[0].end - loops[0].begin) * int(loops[1].end - loops[1].begin) == extent


def _target_width_function():
    @T.prim_func
    def target_width(A: T.Tensor((16,), "float32")):
        for i in T.vectorized(16):
            A[i] = A[i] + 1

    return target_width


def _assert_target_width(function, expected_width):
    transformed = tl.transform.LegalizeVectorizedLoop()(tvm.IRModule({"main": function}))["main"]
    loops = collect_nodes(transformed, ir.For)
    (vector,) = [loop for loop in loops if loop.kind == ir.ForKind.VECTORIZED]
    (outer,) = [loop for loop in loops if loop.kind != ir.ForKind.VECTORIZED]
    assert int(vector.end - vector.begin) == expected_width
    assert int(outer.end - outer.begin) * expected_width == 16
    assert int(vector.step) == int(outer.step) == 1
    return transformed


@pytest.mark.parametrize("target,width", [("ascend", 2), ({"kind": "cuda", "arch": "sm_80"}, 4)])
def test_vectorize_uses_bound_function_target_without_context(target, width):
    assert tvm.target.Target.current() is None
    function = _target_width_function().with_attr("target", tvm.target.Target(target))
    _assert_target_width(function, width)
    assert tvm.target.Target.current() is None


@pytest.mark.parametrize(
    "bound,ambient,width",
    [
        ("ascend", {"kind": "cuda", "arch": "sm_80"}, 2),
        ({"kind": "cuda", "arch": "sm_80"}, "ascend", 4),
    ],
)
def test_vectorize_function_target_wins_over_different_context(bound, ambient, width):
    target = tvm.target.Target(bound)
    function = _target_width_function().with_attr("target", target)
    with tvm.target.Target(ambient) as outer:
        transformed = _assert_target_width(function, width)
        assert tvm.target.Target.current().same_as(outer)
        assert transformed.attrs["target"].same_as(target)
    assert tvm.target.Target.current() is None


@pytest.mark.parametrize("target,width", [("ascend", 2), ({"kind": "cuda", "arch": "sm_80"}, 4)])
def test_vectorize_unbound_function_retains_explicit_context_fallback(target, width):
    with tvm.target.Target(target):
        _assert_target_width(_target_width_function(), width)
    assert tvm.target.Target.current() is None


def test_vectorize_missing_both_target_sources_is_an_error():
    assert tvm.target.Target.current() is None
    with pytest.raises(RuntimeError, match="Target context required"):
        tl.transform.LegalizeVectorizedLoop()(tvm.IRModule({"main": _target_width_function()}))


def test_vectorize_no_vector_loop_does_not_require_a_target():
    @T.prim_func
    def serial_only(A: T.Tensor((16,), "float32")):
        for i in T.serial(16):
            A[i] = A[i] + 1

    assert tvm.target.Target.current() is None
    result = tl.transform.LegalizeVectorizedLoop()(tvm.IRModule({"main": serial_only}))["main"]
    tvm.ir.assert_structural_equal(result, serial_only)


if __name__ == "__main__":
    tilelang.testing.main()
