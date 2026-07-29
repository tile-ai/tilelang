import tilelang as tl
import tilelang.testing
from tilelang import ir as tl_ir
from tilelang import tvm


def _make_parallel_store(*, injective_indices: bool, varying_value: bool = True):
    output = tvm.tirx.decl_buffer((32, 128), "int32", name="output")
    ti = tvm.tirx.Var("ti", "int32")
    tj = tvm.tirx.Var("tj", "int32")
    i = tvm.tirx.Var("i", "int32")
    j = tvm.tirx.Var("j", "int32")

    indices = [i, j] if injective_indices else [0, 0]
    value = ti * 128 + tj if varying_value else 1
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


def _make_two_racy_stores():
    out_a = tvm.tirx.decl_buffer((32,), "int32", name="out_a")
    out_b = tvm.tirx.decl_buffer((32,), "int32", name="out_b")
    ti = tvm.tirx.Var("ti", "int32")
    store_a = tvm.tirx.BufferStore(out_a, ti, [0])
    store_b = tvm.tirx.BufferStore(out_b, ti, [0])
    body = tvm.tirx.SeqStmt([store_a, store_b])
    body = tvm.tirx.For(ti, 0, 32, tvm.tirx.ForKind.PARALLEL, body)
    mod = tvm.IRModule.from_expr(tvm.tirx.PrimFunc([out_a, out_b], body))
    return mod, store_a, store_b


def test_data_race_diagnostic_includes_span(capfd):
    mod, store_a, store_b = _make_two_racy_stores()
    tl_ir.set_stmt_span(store_a, tl_ir.make_span("kernel.py", 21))
    tl_ir.set_stmt_span(store_b, tl_ir.make_span("kernel.py", 22))

    err = _verify(mod, capfd)
    assert err.count("Data race detected") == 1
    assert "[1]" in err
    assert "--> kernel.py:21:1" in err
    assert "[2]" in err
    assert "--> kernel.py:22:1" in err


if __name__ == "__main__":
    tilelang.testing.main()
