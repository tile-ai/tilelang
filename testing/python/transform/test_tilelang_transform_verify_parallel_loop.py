import tilelang as tl
import tilelang.testing
from tilelang import tvm


def _make_parallel_store(
    *,
    injective_indices: bool,
    varying_value: bool = True,
    value_lanes: int = 1,
):
    output_dtype = "int32" if value_lanes == 1 else f"int32x{value_lanes}"
    output = tvm.tirx.decl_buffer((32, 128), output_dtype, name="output")
    ti = tvm.tirx.Var("ti", "int32")
    tj = tvm.tirx.Var("tj", "int32")
    i = tvm.tirx.Var("i", "int32")
    j = tvm.tirx.Var("j", "int32")

    indices = [i, j] if injective_indices else [0, 0]
    value = ti * 128 + tj if varying_value else 1
    if value_lanes > 1:
        value = tvm.tirx.Broadcast(value, value_lanes)
    store = tvm.tirx.BufferStore(output, value, indices)
    body = tvm.tirx.SeqStmt([tvm.tirx.Bind(i, ti), tvm.tirx.Bind(j, tj), store])
    body = tvm.tirx.For(tj, 0, 128, tvm.tirx.ForKind.PARALLEL, body)
    body = tvm.tirx.For(ti, 0, 32, tvm.tirx.ForKind.PARALLEL, body)
    return tvm.IRModule.from_expr(tvm.tirx.PrimFunc([output], body))


def _verify(mod, capfd):
    tl.transform.VerifyParallelLoop()(mod)
    return capfd.readouterr().err


def test_flat_bind_indices_are_proven_race_free(capfd):
    mod = _make_parallel_store(injective_indices=True)

    assert "Data race detected" not in _verify(mod, capfd)


def test_irrelevant_flat_binds_do_not_hide_race(capfd):
    mod = _make_parallel_store(injective_indices=False)

    assert "Data race detected" in _verify(mod, capfd)


def test_same_value_parallel_store_is_allowed(capfd):
    mod = _make_parallel_store(injective_indices=False, varying_value=False)

    assert "Data race detected" not in _verify(mod, capfd)


def test_vector_value_parallel_store_is_supported(capfd):
    mod = _make_parallel_store(injective_indices=True, value_lanes=2)

    assert "Data race detected" not in _verify(mod, capfd)


def test_vector_value_race_reaches_the_prover(capfd):
    # Non-injective indices defeat the rewrite simplifier, so the condition
    # reaches the SMT prover -- which aborts on any vector node.
    mod = _make_parallel_store(injective_indices=False, value_lanes=2)

    assert "Data race detected" in _verify(mod, capfd)


def test_same_vector_value_parallel_store_is_allowed(capfd):
    mod = _make_parallel_store(injective_indices=False, varying_value=False, value_lanes=2)

    assert "Data race detected" not in _verify(mod, capfd)


if __name__ == "__main__":
    tilelang.testing.main()
